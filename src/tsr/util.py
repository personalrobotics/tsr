# Geodesic Distance
# wrap_to_interval
# GetManipulatorIndex

import logging
import math
import numpy
import scipy.misc
import scipy.optimize
import threading
import time
import warnings



def wrap_to_interval(angles, lower=-numpy.pi):
    """
    Wraps an angle into a semi-closed interval of width 2*pi.

    By default, this interval is `[-pi, pi)`.  However, the lower bound of the
    interval can be specified to wrap to the interval `[lower, lower + 2*pi)`.
    If `lower` is an array the same length as angles, the bounds will be
    applied element-wise to each angle in `angles`.

    See: http://stackoverflow.com/a/32266181

    @param angles an angle or 1D array of angles to wrap
    @type  angles float or numpy.array
    @param lower optional lower bound on wrapping interval
    @type  lower float or numpy.array
    """
    return (angles - lower) % (2 * numpy.pi) + lower


def GeodesicError(t1, t2):
    """
    Computes the error in global coordinates between two transforms.

    @param t1 current transform
    @param t2 goal transform
    @return a 4-vector of [dx, dy, dz, solid angle]
    """
    trel = numpy.dot(numpy.linalg.inv(t1), t2)
    trans = numpy.dot(t1[0:3, 0:3], trel[0:3, 3])
    angle,direction,point = (trel)
    return numpy.hstack((trans, angle))



def GeodesicDistance(t1, t2, r=1.0):
    """
    Computes the geodesic distance between two transforms

    @param t1 current transform
    @param t2 goal transform
    @param r in units of meters/radians converts radians to meters
    """
    error = GeodesicError(t1, t2)
    error[3] = r * error[3]
    return numpy.linalg.norm(error)