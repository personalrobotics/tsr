# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy
import numpy.random
from functools import reduce
from numpy import pi

from .utils import EPSILON, geodesic_distance, wrap_to_interval

NANBW = numpy.ones(6)*float('nan')


class TSR:
    """
    Core Task Space Region (TSR) class — geometry-only, robot-agnostic.

    A TSR is defined by a transform T0_w to the TSR frame, a transform Tw_e
    from the TSR frame to the end-effector, and a bounding box Bw over 6 DoFs.
    """

    def __init__(self, T0_w=None, Tw_e=None, Bw=None):
        if T0_w is None:
            T0_w = numpy.eye(4)
        if Tw_e is None:
            Tw_e = numpy.eye(4)
        if Bw is None:
            Bw = numpy.zeros((6, 2))

        self.T0_w = numpy.array(T0_w)
        self.Tw_e = numpy.array(Tw_e)
        self.Bw = numpy.array(Bw)

        if numpy.any(self.Bw[0:3, 0] > self.Bw[0:3, 1]):
            raise ValueError('Bw translation bounds must be [min, max]', Bw)

        # We will now create a continuous version of the bound to maintain:
        # 1. Bw[i,1] > Bw[i,0] which is necessary for LBFGS-B
        # 2. signed rotations, necessary for expressiveness
        Bw_cont = numpy.copy(self.Bw)

        Bw_interval = Bw_cont[3:6, 1] - Bw_cont[3:6, 0]
        Bw_interval = numpy.minimum(Bw_interval, 2*pi)

        Bw_cont[3:6, 0] = wrap_to_interval(Bw_cont[3:6, 0])
        Bw_cont[3:6, 1] = Bw_cont[3:6, 0] + Bw_interval

        self._Bw_cont = Bw_cont

    @staticmethod
    def rot_to_rpy(rot):
        """
        Converts a rotation matrix to one valid rpy
        @param rot 3x3 rotation matrix
        @return rpy (3,) rpy
        """
        rpy = numpy.zeros(3)
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            p = -numpy.arcsin(rot[2, 0])
            rpy[0] = numpy.arctan2((rot[2, 1]/numpy.cos(p)),
                                   (rot[2, 2]/numpy.cos(p)))
            rpy[1] = p
            rpy[2] = numpy.arctan2((rot[1, 0]/numpy.cos(p)),
                                   (rot[0, 0]/numpy.cos(p)))
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = numpy.arctan2(rot[0, 1], rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = pi/2
                rpy[2] = 0.
            else:
                r_offset = numpy.arctan2(-rot[0, 1], -rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = -pi/2
                rpy[2] = 0.
        return rpy

    @staticmethod
    def trans_to_xyzrpy(trans):
        """
        Converts a transformation matrix to one valid xyzrpy
        @param trans 4x4 transformation matrix
        @return xyzrpy 6x1 xyzrpy
        """
        xyz, rot = trans[0:3, 3], trans[0:3, 0:3]
        rpy = TSR.rot_to_rpy(rot)
        return numpy.hstack((xyz, rpy))

    @staticmethod
    def rpy_to_rot(rpy):
        """
        Converts an rpy to a rotation matrix
        @param rpy (3,) rpy
        @return rot 3x3 rotation matrix
        """
        rot = numpy.zeros((3, 3))
        r, p, y = rpy[0], rpy[1], rpy[2]
        rot[0][0] = numpy.cos(p)*numpy.cos(y)
        rot[1][0] = numpy.cos(p)*numpy.sin(y)
        rot[2][0] = -numpy.sin(p)
        rot[0][1] = (numpy.sin(r)*numpy.sin(p)*numpy.cos(y) -
                     numpy.cos(r)*numpy.sin(y))
        rot[1][1] = (numpy.sin(r)*numpy.sin(p)*numpy.sin(y) +
                     numpy.cos(r)*numpy.cos(y))
        rot[2][1] = numpy.sin(r)*numpy.cos(p)
        rot[0][2] = (numpy.cos(r)*numpy.sin(p)*numpy.cos(y) +
                     numpy.sin(r)*numpy.sin(y))
        rot[1][2] = (numpy.cos(r)*numpy.sin(p)*numpy.sin(y) -
                     numpy.sin(r)*numpy.cos(y))
        rot[2][2] = numpy.cos(r)*numpy.cos(p)
        return rot

    @staticmethod
    def xyzrpy_to_trans(xyzrpy):
        """
        Converts an xyzrpy to a transformation matrix
        @param xyzrpy 6x1 xyzrpy vector
        @return trans 4x4 transformation matrix
        """
        trans = numpy.zeros((4, 4))
        trans[3][3] = 1.0
        xyz, rpy = xyzrpy[0:3], xyzrpy[3:6]
        trans[0:3, 3] = xyz
        rot = TSR.rpy_to_rot(rpy)
        trans[0:3, 0:3] = rot
        return trans

    @staticmethod
    def xyz_within_bounds(xyz, Bw):
        """
        Checks whether an xyz value is within a given xyz bounds.
        Main issue: dealing with roundoff issues for zero bounds
        @param xyz a (3,) xyz value
        @param Bw bounds on xyz
        @return check a (3,) vector of True if within and False if outside
        """
        # Check bounds condition on XYZ component.
        xyzcheck = []
        for i, x in enumerate(xyz):
            x_val = x.item() if hasattr(x, 'item') else float(x)  # Convert to scalar
            xyzcheck.append(((x_val + EPSILON) >= Bw[i, 0]) and
                           ((x_val - EPSILON) <= Bw[i, 1]))
        return xyzcheck

    @staticmethod
    def rpy_within_bounds(rpy, Bw):
        """
        Checks whether an rpy value is within a given rpy bounds.
        Assumes all values in the bounds are [-pi, pi]
        Two main issues: dealing with roundoff issues for zero bounds and
        Wraparound for rpy.
        @param rpy a (3,) rpy value
        @param Bw bounds on rpy
        @return check a (3,) vector of True if within and False if outside
        """
        # Unwrap rpy to Bw_cont.
        rpy = wrap_to_interval(rpy, lower=Bw[:3, 0])

        # Check bounds condition on RPY component.
        rpycheck = [False] * 3
        for i in range(0, 3):
            if (Bw[i, 0] > Bw[i, 1] + EPSILON):
                # An outer interval
                rpycheck[i] = (((rpy[i] + EPSILON) >= Bw[i, 0]) or
                               ((rpy[i] - EPSILON) <= Bw[i, 1]))
            else:
                # An inner interval
                rpycheck[i] = (((rpy[i] + EPSILON) >= Bw[i, 0]) and
                               ((rpy[i] - EPSILON) <= Bw[i, 1]))
        return rpycheck

    @staticmethod
    def rot_within_rpy_bounds(rot, Bw):
        """
        Checks whether a rotation matrix is within a given rpy bounds.
        Assumes all values in the bounds are [-pi, pi]
        Two main challenges with rpy:
            (1) Usually, two rpy solutions for each rot.
            (2) 1D subspace of degenerate solutions at singularities.
        Based on: http://staff.city.ac.uk/~sbbh653/publications/euler.pdf
        @param rot 3x3 rotation matrix
        @param Bw bounds on rpy
        @return check a (3,) vector of True if within and False if outside
        @return rpy the rpy consistent with the bound or None if nothing is
        """
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            # Not a singularity. Two pitch solutions
            psol = -numpy.arcsin(rot[2, 0])
            for p in [psol, (pi - psol)]:
                rpy = numpy.zeros(3)
                rpy[0] = numpy.arctan2((rot[2, 1]/numpy.cos(p)),
                                       (rot[2, 2]/numpy.cos(p)))
                rpy[1] = p
                rpy[2] = numpy.arctan2((rot[1, 0]/numpy.cos(p)),
                                       (rot[0, 0]/numpy.cos(p)))
                rpycheck = TSR.rpy_within_bounds(rpy, Bw)
                if all(rpycheck):
                    return rpycheck, rpy
            return rpycheck, None
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = numpy.arctan2(rot[0, 1], rot[0, 2])
                # Valid rotation: [y + r_offset, pi/2, y]
                # check the four r-y Bw corners
                rpy_list = []
                rpy_list.append([Bw[2, 0] + r_offset, pi/2, Bw[2, 0]])
                rpy_list.append([Bw[2, 1] + r_offset, pi/2, Bw[2, 1]])
                rpy_list.append([Bw[0, 0], pi/2, Bw[0, 0] - r_offset])
                rpy_list.append([Bw[0, 1], pi/2, Bw[0, 1] - r_offset])
                for rpy in rpy_list:
                    rpycheck = TSR.rpy_within_bounds(rpy, Bw)
                    # No point checking anything if pi/2 not in Bw
                    if (rpycheck[1] is False):
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            else:
                r_offset = numpy.arctan2(-rot[0, 1], -rot[0, 2])
                # Valid rotation: [-y + r_offset, -pi/2, y]
                # check the four r-y Bw corners
                rpy_list = []
                rpy_list.append([-Bw[2, 0] + r_offset, -pi/2, Bw[2, 0]])
                rpy_list.append([-Bw[2, 1] + r_offset, -pi/2, Bw[2, 1]])
                rpy_list.append([Bw[0, 0], -pi/2, -Bw[0, 0] + r_offset])
                rpy_list.append([Bw[0, 1], -pi/2, -Bw[0, 1] + r_offset])
                for rpy in rpy_list:
                    rpycheck = TSR.rpy_within_bounds(rpy, Bw)
                    # No point checking anything if -pi/2 not in Bw
                    if (rpycheck[1] is False):
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
        return rpycheck, None

    def to_transform(self, xyzrpy):
        """
        Converts a [x y z roll pitch yaw] into an
        end-effector transform.

        @param  xyzrpy [x y z roll pitch yaw]
        @return trans 4x4 transform
        """
        if len(xyzrpy) != 6:
            raise ValueError('xyzrpy must be of length 6')
        if not all(self.is_valid(xyzrpy)):
            raise ValueError('Invalid xyzrpy', xyzrpy)
        Tw = TSR.xyzrpy_to_trans(xyzrpy)
        trans = reduce(numpy.dot, [self.T0_w, Tw, self.Tw_e])
        return trans

    def to_xyzrpy(self, trans):
        """
        Converts an end-effector transform to xyzrpy values
        @param  trans  4x4 transform
        @return xyzrpy 6x1 vector of Bw values
        """
        Tw = reduce(numpy.dot, [numpy.linalg.inv(self.T0_w),
                                trans,
                                numpy.linalg.inv(self.Tw_e)])
        xyz, rot = Tw[0:3, 3], Tw[0:3, 0:3]
        rpycheck, rpy = TSR.rot_within_rpy_bounds(rot, self._Bw_cont)
        if not all(rpycheck):
            rpy = TSR.rot_to_rpy(rot)
        return numpy.hstack((xyz, rpy))

    def is_valid(self, xyzrpy, ignoreNAN=False):
        """
        Checks if a xyzrpy is a valid sample from the TSR.
        Two main issues: dealing with roundoff issues for zero bounds and
        Wraparound for rpy.
        @param xyzrpy 6x1 vector of Bw values
        @param ignoreNAN (optional, defaults to False) ignore NaN xyzrpy
        @return a 6x1 vector of True if bound is valid and False if not
        """
        # Extract XYZ and RPY components of input and TSR.
        Bw_xyz, Bw_rpy = self._Bw_cont[0:3, :], self._Bw_cont[3:6, :]
        xyz, rpy = xyzrpy[0:3], xyzrpy[3:6]

        # Check bounds condition on XYZ component.
        xyzcheck = TSR.xyz_within_bounds(xyz, Bw_xyz)

        # Check bounds condition on RPY component.
        rpycheck = TSR.rpy_within_bounds(rpy, Bw_rpy)

        # Concatenate the XYZ and RPY components of the check.
        check = numpy.hstack((xyzcheck, rpycheck))

        # If ignoreNAN, components with NaN values are always OK.
        if ignoreNAN:
            check |= numpy.isnan(xyzrpy)

        return check

    def contains(self, trans):
        """
        Checks if the TSR contains the transform
        @param  trans 4x4 transform
        @return a 6x1 vector of True if bound is valid and False if not
        """
        # Extract XYZ and rot components of input and TSR.
        Bw_xyz, Bw_rpy = self._Bw_cont[0:3, :], self._Bw_cont[3:6, :]
        xyz, rot = trans[0:3, 3], trans[0:3, 0:3]  # Extract translation vector
        # Check bounds condition on XYZ component.
        xyzcheck = TSR.xyz_within_bounds(xyz, Bw_xyz)
        # Check bounds condition on rot component.
        rotcheck, rpy = TSR.rot_within_rpy_bounds(rot, Bw_rpy)

        return all(numpy.hstack((xyzcheck, rotcheck)))

    def _displacement_to_tsr(self, trans):
        """
        Compute the displacement vector from a transform to the TSR.

        Implements Section 4.2 of Berenson et al. 2011:
            T0_s' = T0_s * (Tw_e)^-1        (Equation 5)
            Tw_s' = (T0_w)^-1 * T0_s'       (Equation 6)
            dw = [translation; RPY]         (Equation 7)
            Δx_i = displacement to bounds   (Equation 8)

        @param trans 4x4 transform (T0_s - end-effector pose in world frame)
        @return dx 6x1 displacement vector to TSR
        @return dw 6x1 displacement vector in w frame (for computing bwopt)
        """
        # Equation 5: T0_s' = T0_s * (Tw_e)^-1
        T0_s_prime = numpy.dot(trans, numpy.linalg.inv(self.Tw_e))

        # Equation 6: Tw_s' = (T0_w)^-1 * T0_s'
        Tw_s_prime = numpy.dot(numpy.linalg.inv(self.T0_w), T0_s_prime)

        # Equation 7: Convert to displacement vector [xyz, rpy]
        xyz = Tw_s_prime[0:3, 3]
        rot = Tw_s_prime[0:3, 0:3]
        rpy = TSR.rot_to_rpy(rot)
        dw = numpy.hstack((xyz, rpy))

        # Handle RPY redundancy - find the RPY representation that minimizes distance
        # The paper mentions checking equivalent rotations {x4 ± π, −x5 ± π, x6 ± π}
        best_dx = None
        best_dw = None
        best_norm = float('inf')

        # Generate candidate RPY values (original + 8 equivalent representations)
        rpy_candidates = [rpy]

        # Add equivalent RPY representations due to Euler angle redundancy
        # When pitch = ±π/2, there's a singularity with infinite solutions
        # Otherwise, there are generally 2 solutions: (r, p, y) and (r±π, π-p, y±π)
        r, p, y = rpy
        rpy_candidates.append(numpy.array([r + pi, pi - p, y + pi]))
        rpy_candidates.append(numpy.array([r + pi, pi - p, y - pi]))
        rpy_candidates.append(numpy.array([r - pi, pi - p, y + pi]))
        rpy_candidates.append(numpy.array([r - pi, pi - p, y - pi]))
        rpy_candidates.append(numpy.array([r + pi, -pi - p, y + pi]))
        rpy_candidates.append(numpy.array([r + pi, -pi - p, y - pi]))
        rpy_candidates.append(numpy.array([r - pi, -pi - p, y + pi]))
        rpy_candidates.append(numpy.array([r - pi, -pi - p, y - pi]))

        for rpy_cand in rpy_candidates:
            # Wrap RPY to bounds interval for comparison
            rpy_wrapped = wrap_to_interval(rpy_cand, lower=self._Bw_cont[3:6, 0])
            dw_cand = numpy.hstack((xyz, rpy_wrapped))

            # Equation 8: Compute displacement to bounds
            dx = numpy.zeros(6)
            for i in range(6):
                if dw_cand[i] < self._Bw_cont[i, 0]:
                    dx[i] = dw_cand[i] - self._Bw_cont[i, 0]
                elif dw_cand[i] > self._Bw_cont[i, 1]:
                    dx[i] = dw_cand[i] - self._Bw_cont[i, 1]
                # else: dx[i] = 0 (already initialized)

            norm = numpy.linalg.norm(dx)
            if norm < best_norm:
                best_norm = norm
                best_dx = dx
                best_dw = dw_cand

        return best_dx, best_dw

    def distance(self, trans, rotation_weight=1.0):
        """
        Computes the distance from the TSR to a transform.

        Implements the closed-form distance calculation from Section 4.2 of
        Berenson et al. 2011. Translation is in meters, rotation in radians.

        @param trans 4x4 transform
        @param rotation_weight weight for rotation vs translation (default 1.0)
                               Higher values penalize rotation errors more.
        @return dist Distance to TSR (0 if transform is inside TSR)
        @return bwopt Closest Bw value to trans (6x1 xyzrpy)
        """
        # Fast path: if transform is contained, distance is 0
        if self.contains(trans):
            return 0., self.to_xyzrpy(trans)

        # Compute displacement using closed-form formula from paper
        dx, dw = self._displacement_to_tsr(trans)

        # Apply rotation weight (paper mentions translation/rotation can be weighted)
        dx_weighted = dx.copy()
        dx_weighted[3:6] *= rotation_weight

        dist = numpy.linalg.norm(dx_weighted)

        # Compute bwopt: the closest point in the TSR bounds
        bwopt = numpy.clip(dw, self._Bw_cont[:, 0], self._Bw_cont[:, 1])
        # Wrap RPY back to [-pi, pi]
        bwopt[3:6] = wrap_to_interval(bwopt[3:6])

        return dist, bwopt

    def distance_optimize(self, trans):
        """
        Computes the Geodesic Distance from the TSR to a transform using
        numerical optimization. This is slower but may be more accurate for
        complex cases.

        @param trans 4x4 transform
        @return dist Geodesic distance to TSR
        @return bwopt Closest Bw value to trans
        """
        if self.contains(trans):
            return 0., self.to_xyzrpy(trans)

        import scipy.optimize

        def objective(bw):
            bwtrans = self.to_transform(bw)
            return geodesic_distance(bwtrans, trans)

        bwinit = (self._Bw_cont[:, 0] + self._Bw_cont[:, 1])/2
        bwbounds = [(self._Bw_cont[i, 0], self._Bw_cont[i, 1])
                    for i in range(6)]

        bwopt, dist, info = scipy.optimize.fmin_l_bfgs_b(
                                objective, bwinit, fprime=None,
                                args=(),
                                bounds=bwbounds, approx_grad=True)
        return dist, bwopt

    def sample_xyzrpy(self, xyzrpy=NANBW):
        """
        Samples from Bw to generate an xyzrpy sample
        Can specify some values optionally as NaN.

        @param xyzrpy   (optional) a 6-vector of Bw with float('nan') for
                        dimensions to sample uniformly.
        @return         an xyzrpy sample
        """
        check = self.is_valid(xyzrpy, ignoreNAN=True)
        if not all(check):
            raise ValueError('xyzrpy must be within bounds', check)

        Bw_sample = numpy.array([self._Bw_cont[i, 0] +
                                (self._Bw_cont[i, 1] - self._Bw_cont[i, 0]) *
                                numpy.random.random_sample()
                                if numpy.isnan(x) else x
                                for i, x in enumerate(xyzrpy)])
        # Unwrap rpy to [-pi, pi]
        Bw_sample[3:6] = wrap_to_interval(Bw_sample[3:6])
        return Bw_sample

    def sample(self, xyzrpy=NANBW):
        """
        Samples from Bw to generate an end-effector transform.
        Can specify some Bw values optionally.

        @param xyzrpy   (optional) a 6-vector of Bw with float('nan') for
                        dimensions to sample uniformly.
        @return         4x4 transform
        """
        return self.to_transform(self.sample_xyzrpy(xyzrpy))

    def to_dict(self):
        """ Convert this TSR to a python dict. """
        return {
            'T0_w': self.T0_w.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist(),
        }

    @staticmethod
    def from_dict(x):
        """ Construct a TSR from a python dict. """
        return TSR(
            T0_w=numpy.array(x['T0_w']),
            Tw_e=numpy.array(x['Tw_e']),
            Bw=numpy.array(x['Bw']),
        )

    def to_json(self):
        """ Convert this TSR to a JSON string. """
        import json
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSR from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json
        x_dict = json.loads(x, *args, **kw_args)
        return TSR.from_dict(x_dict)

    def to_yaml(self):
        """ Convert this TSR to a YAML string. """
        import yaml
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x, *args, **kw_args):
        """
        Construct a TSR from a YAML string.

        This method internally forwards all arguments to `yaml.safe_load`.
        """
        import yaml
        x_dict = yaml.safe_load(x, *args, **kw_args)
        return TSR.from_dict(x_dict)
