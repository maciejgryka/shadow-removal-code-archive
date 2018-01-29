from __future__ import print_function

import sys

import numpy as np
cimport numpy as np

from numpy.linalg import det


DTYPE = np.double
    
ctypedef np.double_t DTYPE_t


def get_best_channel_scaling(np.ndarray shad, np.ndarray matte_r):
    """
    Find scaling factors for green and blue channels of the matte that result
    in the best unshadowed result. The coefficients are found with brute-force
    2D search.

    """
    assert shad.dtype == DTYPE and matte_r.dtype == DTYPE
    cdef int h = shad.shape[0]
    cdef int w = shad.shape[1]

    scaling_range = np.arange(0.9, 1.2, 0.01)
    cdef double min_error = 1000000.0
    cdef double error
    cdef double s_g_best = 1.0
    cdef double s_b_best = 1.0

    # declare vars
    cdef int index_b
    cdef int index_g
    cdef double s_g
    cdef double s_b
    cdef int x
    cdef int y
    cdef int n_steps = len(scaling_range)
    cdef np.ndarray matte = np.dstack([matte_r, matte_r, matte_r])
    cdef np.ndarray unshad = np.array(shad, dtype=DTYPE)
    cdef np.ndarray colors = np.zeros([3, shad.shape[0] * shad.shape[1]], dtype=DTYPE)
    
    for index_b in xrange(n_steps):
        s_b = scaling_range[index_b]
        # modify the blue channel of the matte using s_b
        matte[:,:,2] = (matte_r - 1.0) / s_b + 1.0
        for index_g in xrange(n_steps):
            s_g = scaling_range[index_g]
            # modify the green channel of the matte using s_g
            matte[:,:,1] = (matte_r - 1.0) / s_g + 1.0
            # get unshadowed image
            unshad = shad / matte
            # extract all the pixels from the image
            for y in xrange(h):
                for x in xrange(w):
                    colors[0, y * w + x] = unshad[y, x, 0]
                    colors[1, y * w + x] = unshad[y, x, 1]
                    colors[2, y * w + x] = unshad[y, x, 2]
            # the error is proportional to the determinant of the covariance
            # of all the pixel colors
            error = np.log(det(np.cov(colors)))
            if error < min_error:
                min_error = error
                s_g_best = s_g
                s_b_best = s_b
    return s_g_best, s_b_best
