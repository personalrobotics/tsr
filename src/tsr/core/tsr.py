# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy as np
import numpy.random as npr
from numpy import pi
from typing import Optional, Tuple
import scipy.optimize

from tsr.core.utils import wrap_to_interval, EPSILON

NANBW = np.ones(6) * float("nan")


class TSR:
    """
    Core Task Space Region (TSR) class — geometry-only, robot-agnostic.

    A TSR is defined by a transform T0_w to the TSR frame, a transform Tw_e
    from the TSR frame to the end-effector, and a bounding box Bw over 6 DoFs.
    """

    def __init__(self, T0_w=None, Tw_e=None, Bw=None):
        if T0_w is None:
            T0_w = np.eye(4)
        if Tw_e is None:
            Tw_e = np.eye(4)
        if Bw is None:
            Bw = np.zeros((6, 2))

        self.T0_w = np.array(T0_w)
        self.Tw_e = np.array(Tw_e)
        self.Bw = np.array(Bw)

        if np.any(self.Bw[0:3, 0] > self.Bw[0:3, 1]):
            raise ValueError("Bw translation bounds must be [min, max]", Bw)

        # Continuous wrap-safe version of Bw
        Bw_cont = np.copy(self.Bw)
        Bw_interval = Bw_cont[3:6, 1] - Bw_cont[3:6, 0]
        Bw_interval = np.minimum(Bw_interval, 2 * pi)

        Bw_cont[3:6, 0] = wrap_to_interval(Bw_cont[3:6, 0])
        Bw_cont[3:6, 1] = Bw_cont[3:6, 0] + Bw_interval
        self._Bw_cont = Bw_cont

    @staticmethod
    def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
        """Convert [roll, pitch, yaw] to 3×3 rotation matrix."""
        r, p, y = rpy
        rot = np.zeros((3, 3))
        rot[0][0] = np.cos(p) * np.cos(y)
        rot[1][0] = np.cos(p) * np.sin(y)
        rot[2][0] = -np.sin(p)
        rot[0][1] = np.sin(r) * np.sin(p) * np.cos(y) - np.cos(r) * np.sin(y)
        rot[1][1] = np.sin(r) * np.sin(p) * np.sin(y) + np.cos(r) * np.cos(y)
        rot[2][1] = np.sin(r) * np.cos(p)
        rot[0][2] = np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.sin(y)
        rot[1][2] = np.cos(r) * np.sin(p) * np.sin(y) - np.sin(r) * np.cos(y)
        rot[2][2] = np.cos(r) * np.cos(p)
        return rot

    @staticmethod
    def xyzrpy_to_trans(xyzrpy: np.ndarray) -> np.ndarray:
        """Convert xyz+rpy (6-vector) to a 4×4 transform."""
        xyz, rpy = xyzrpy[:3], xyzrpy[3:]
        trans = np.eye(4)
        trans[:3, :3] = TSR.rpy_to_rot(rpy)
        trans[:3, 3] = xyz
        return trans

    @staticmethod
    def trans_to_xyzrpy(trans: np.ndarray) -> np.ndarray:
        """Convert a 4×4 transform to xyz+rpy (6-vector)."""
        xyz = trans[:3, 3]
        rot = trans[:3, :3]
        rpy = TSR.rot_to_rpy(rot)
        return np.concatenate([xyz, rpy])

    @staticmethod
    def xyz_within_bounds(xyz: np.ndarray, Bw: np.ndarray) -> list:
        """Check if xyz values are within bounds."""
        xyzcheck = []
        for i, x in enumerate(xyz):
            x_val = x.item() if hasattr(x, 'item') else float(x)
            xyzcheck.append(((x_val + EPSILON) >= Bw[i, 0]) and
                           ((x_val - EPSILON) <= Bw[i, 1]))
        return xyzcheck

    @staticmethod
    def rpy_within_bounds(rpy: np.ndarray, Bw: np.ndarray) -> list:
        """Check if rpy values are within bounds."""
        # Unwrap rpy to Bw bounds
        rpy = wrap_to_interval(rpy, lower=Bw[:3, 0])
        
        # Check bounds condition on RPY component
        rpycheck = [False] * 3
        for i in range(3):
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
    def rot_within_rpy_bounds(rot: np.ndarray, Bw: np.ndarray) -> tuple:
        """Check if rotation matrix is within RPY bounds."""
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            # Not a singularity. Two pitch solutions
            psol = -np.arcsin(rot[2, 0])
            for p in [psol, (pi - psol)]:
                rpy = np.zeros(3)
                rpy[0] = np.arctan2((rot[2, 1]/np.cos(p)), (rot[2, 2]/np.cos(p)))
                rpy[1] = p
                rpy[2] = np.arctan2((rot[1, 0]/np.cos(p)), (rot[0, 0]/np.cos(p)))
                rpycheck = TSR.rpy_within_bounds(rpy, Bw)
                if all(rpycheck):
                    return rpycheck, rpy
            return rpycheck, None
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                # Valid rotation: [y + r_offset, pi/2, y]
                rpy_list = []
                rpy_list.append([Bw[2, 0] + r_offset, pi/2, Bw[2, 0]])
                rpy_list.append([Bw[2, 1] + r_offset, pi/2, Bw[2, 1]])
                rpy_list.append([Bw[0, 0], pi/2, Bw[0, 0] - r_offset])
                rpy_list.append([Bw[0, 1], pi/2, Bw[0, 1] - r_offset])
                for rpy in rpy_list:
                    rpycheck = TSR.rpy_within_bounds(rpy, Bw)
                    if not rpycheck[1]:  # No point checking if pi/2 not in Bw
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                # Valid rotation: [-y + r_offset, -pi/2, y]
                rpy_list = []
                rpy_list.append([-Bw[2, 0] + r_offset, -pi/2, Bw[2, 0]])
                rpy_list.append([-Bw[2, 1] + r_offset, -pi/2, Bw[2, 1]])
                rpy_list.append([Bw[0, 0], -pi/2, -Bw[0, 0] + r_offset])
                rpy_list.append([Bw[0, 1], -pi/2, -Bw[0, 1] + r_offset])
                for rpy in rpy_list:
                    rpycheck = TSR.rpy_within_bounds(rpy, Bw)
                    if not rpycheck[1]:  # No point checking if -pi/2 not in Bw
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            return [False, False, False], None

    def to_transform(self, xyzrpy: np.ndarray) -> np.ndarray:
        """Convert xyzrpy into world-frame pose using T0_w * T * Tw_e."""
        if len(xyzrpy) != 6:
            raise ValueError("xyzrpy must be length 6")
        if not self.is_valid(xyzrpy):
            raise ValueError("Invalid xyzrpy", xyzrpy)
        return self.T0_w @ TSR.xyzrpy_to_trans(xyzrpy) @ self.Tw_e

    def sample_xyzrpy(self, xyzrpy: np.ndarray = NANBW) -> np.ndarray:
        """Sample from the bounds Bw, optionally fixing some dimensions."""
        Bw_sample = np.array([
            self._Bw_cont[i, 0] + (self._Bw_cont[i, 1] - self._Bw_cont[i, 0]) * npr.random_sample()
            if np.isnan(x) else x
            for i, x in enumerate(xyzrpy)
        ])
        Bw_sample[3:6] = wrap_to_interval(Bw_sample[3:6])
        return Bw_sample

    def sample(self, xyzrpy: np.ndarray = NANBW) -> np.ndarray:
        """Sample a 4×4 world-frame transform from this TSR."""
        return self.to_transform(self.sample_xyzrpy(xyzrpy))

    def distance(self, trans: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the geodesic distance from a transform to this TSR using numerical optimization.
        
        This method uses scipy.optimize to find the minimum geodesic distance
        over all valid poses in the TSR.
        
        Args:
            trans: 4x4 transform matrix
            
        Returns:
            distance: geodesic distance to TSR
            bwopt: closest Bw value to trans
        """
        if self.contains(trans):
            return 0.0, self.to_xyzrpy(trans)

        def objective(bw):
            bwtrans = self.to_transform(bw)
            from tsr.core.utils import geodesic_distance
            return geodesic_distance(bwtrans, trans)

        # Initialize optimization at center of bounds
        bwinit = (self._Bw_cont[:, 0] + self._Bw_cont[:, 1]) / 2
        
        # Set bounds for optimization
        bwbounds = [(self._Bw_cont[i, 0], self._Bw_cont[i, 1]) for i in range(6)]

        # Run optimization
        bwopt, dist, info = scipy.optimize.fmin_l_bfgs_b(
            objective, bwinit, fprime=None, args=(),
            bounds=bwbounds, approx_grad=True)
        
        return dist, bwopt

    def contains(self, trans: np.ndarray) -> bool:
        """
        Check if a transform is within this TSR.
        
        This method works directly on the world-frame transform without applying
        TSR transforms, matching the legacy implementation.
        """
        # Extract XYZ and rot components directly from input transform
        xyz = trans[0:3, 3]
        rot = trans[0:3, 0:3]
        
        # Check bounds condition on XYZ component
        xyzcheck = []
        for i, x in enumerate(xyz):
            x_val = x.item() if hasattr(x, 'item') else float(x)
            xyzcheck.append(((x_val + EPSILON) >= self.Bw[i, 0]) and
                           ((x_val - EPSILON) <= self.Bw[i, 1]))
        
        # Check bounds condition on rotation component
        rotcheck, rpy = self._rot_within_rpy_bounds(rot, self.Bw[3:6, :])
        
        return all(xyzcheck + rotcheck)
    
    def _rot_within_rpy_bounds(self, rot: np.ndarray, Bw: np.ndarray) -> tuple:
        """
        Check whether a rotation matrix is within given RPY bounds.
        
        Args:
            rot: 3x3 rotation matrix
            Bw: bounds on RPY (3x2 array)
            
        Returns:
            check: 3-element list of booleans
            rpy: RPY angles or None
        """
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            # Not a singularity. Two pitch solutions
            psol = -np.arcsin(rot[2, 0])
            for p in [psol, (pi - psol)]:
                rpy = np.zeros(3)
                rpy[0] = np.arctan2((rot[2, 1]/np.cos(p)), (rot[2, 2]/np.cos(p)))
                rpy[1] = p
                rpy[2] = np.arctan2((rot[1, 0]/np.cos(p)), (rot[0, 0]/np.cos(p)))
                rpycheck = self._rpy_within_bounds(rpy, Bw)
                if all(rpycheck):
                    return rpycheck, rpy
            return rpycheck, None
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                # Valid rotation: [y + r_offset, pi/2, y]
                rpy_list = []
                rpy_list.append([Bw[2, 0] + r_offset, pi/2, Bw[2, 0]])
                rpy_list.append([Bw[2, 1] + r_offset, pi/2, Bw[2, 1]])
                rpy_list.append([Bw[0, 0], pi/2, Bw[0, 0] - r_offset])
                rpy_list.append([Bw[0, 1], pi/2, Bw[0, 1] - r_offset])
                for rpy in rpy_list:
                    rpycheck = self._rpy_within_bounds(rpy, Bw)
                    if not rpycheck[1]:  # No point checking if pi/2 not in Bw
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                # Valid rotation: [-y + r_offset, -pi/2, y]
                rpy_list = []
                rpy_list.append([-Bw[2, 0] + r_offset, -pi/2, Bw[2, 0]])
                rpy_list.append([-Bw[2, 1] + r_offset, -pi/2, Bw[2, 1]])
                rpy_list.append([Bw[0, 0], -pi/2, -Bw[0, 0] + r_offset])
                rpy_list.append([Bw[0, 1], -pi/2, -Bw[0, 1] + r_offset])
                for rpy in rpy_list:
                    rpycheck = self._rpy_within_bounds(rpy, Bw)
                    if not rpycheck[1]:  # No point checking if -pi/2 not in Bw
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            return [False, False, False], None
    
    def _rpy_within_bounds(self, rpy: np.ndarray, Bw: np.ndarray) -> list:
        """
        Check whether RPY values are within given bounds.
        
        Args:
            rpy: 3-element RPY array
            Bw: bounds on RPY (3x2 array)
            
        Returns:
            check: 3-element list of booleans
        """
        # Unwrap RPY to Bw bounds
        rpy = wrap_to_interval(rpy, lower=Bw[:3, 0])
        
        # Check bounds condition on RPY component
        rpycheck = [False] * 3
        for i in range(3):
            if (Bw[i, 0] > Bw[i, 1] + EPSILON):
                # An outer interval
                rpycheck[i] = (((rpy[i] + EPSILON) >= Bw[i, 0]) or
                               ((rpy[i] - EPSILON) <= Bw[i, 1]))
            else:
                # An inner interval
                rpycheck[i] = (((rpy[i] + EPSILON) >= Bw[i, 0]) and
                               ((rpy[i] - EPSILON) <= Bw[i, 1]))
        return rpycheck

    def is_valid(self, xyzrpy: np.ndarray, ignoreNAN: bool = False) -> bool:
        """
        Check if xyzrpy is within the bounds of this TSR.
        
        Args:
            xyzrpy: 6-vector [x, y, z, roll, pitch, yaw]
            ignoreNAN: If True, ignore NaN values in xyzrpy
        """
        if len(xyzrpy) != 6:
            return False
        
        for i in range(6):
            if ignoreNAN and np.isnan(xyzrpy[i]):
                continue
            
            if xyzrpy[i] < self.Bw[i, 0] or xyzrpy[i] > self.Bw[i, 1]:
                return False
        
        return True

    def to_xyzrpy(self, trans: np.ndarray) -> np.ndarray:
        """Convert a world-frame transform to xyzrpy in TSR frame."""
        # Compute TSR-frame transform: T = inv(T0_w) * trans * inv(Tw_e)
        T = np.linalg.inv(self.T0_w) @ trans @ np.linalg.inv(self.Tw_e)
        
        # Extract translation
        xyz = T[:3, 3]
        
        # Extract rotation and convert to RPY
        rot = T[:3, :3]
        rpy = TSR.rot_to_rpy(rot)
        
        return np.concatenate([xyz, rpy])

    @staticmethod
    def rot_to_rpy(rot: np.ndarray) -> np.ndarray:
        """Convert 3×3 rotation matrix to [roll, pitch, yaw]."""
        rpy = np.zeros(3)
        
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            p = -np.arcsin(rot[2, 0])
            rpy[0] = np.arctan2((rot[2, 1]/np.cos(p)), (rot[2, 2]/np.cos(p)))
            rpy[1] = p
            rpy[2] = np.arctan2((rot[1, 0]/np.cos(p)), (rot[0, 0]/np.cos(p)))
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = pi/2
                rpy[2] = 0.
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = -pi/2
                rpy[2] = 0.
        
        return rpy

    def to_dict(self) -> dict:
        """Convert TSR to dictionary representation."""
        return {
            'T0_w': self.T0_w.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist()
        }

    @staticmethod
    def from_dict(data: dict) -> 'TSR':
        """Create TSR from dictionary representation."""
        return TSR(
            T0_w=np.array(data['T0_w']),
            Tw_e=np.array(data['Tw_e']),
            Bw=np.array(data['Bw'])
        )

    def to_json(self) -> str:
        """Convert TSR to JSON string."""
        import json
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(json_str: str) -> 'TSR':
        """Create TSR from JSON string."""
        import json
        data = json.loads(json_str)
        return TSR.from_dict(data)

    def to_yaml(self) -> str:
        """Convert TSR to YAML string."""
        import yaml
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(yaml_str: str) -> 'TSR':
        """Create TSR from YAML string."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return TSR.from_dict(data)