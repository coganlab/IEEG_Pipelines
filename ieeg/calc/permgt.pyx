import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport qsort

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t INTLONG_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t INTP_t
cnp.import_array()

# cdef DTYPE_t[:] arr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline cnp.ndarray permgt1d(cnp.ndarray diff):
    cdef int n = diff.shape[0], m = diff.shape[0] - 1
    cdef cnp.ndarray[INTLONG_t, ndim=1] sorted_indices = diff.argsort()
    cdef cnp.ndarray[DTYPE_t, ndim=1] proportions = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t i = 0
    for i in range(n):
        proportions[sorted_indices[i]] = i * 1.0 / m
    return proportions

cpdef cnp.ndarray permgtnd(cnp.ndarray diff, int axis=0):
    cdef cnp.ndarray arr_in

    if diff.ndim == 1:
        return permgt1d(diff)
    elif diff.ndim == 0:
        raise ValueError("Cannot apply perm_gt to a 0-dimensional array")

    if axis != -1 or axis != diff.ndim - 1:
        arr_in = np.swapaxes(diff, axis, -1)
    else:
        arr_in = diff

    if diff.ndim == 1:
        arr_in = permgt1d(arr_in)
    elif diff.ndim > 1:
        for i in range(arr_in.shape[0]):
            arr_in[i] = permgtnd(arr_in[i], -1)
    else:
        raise ValueError("Cannot apply perm_gt to a 0-dimensional array")
    
    if axis != -1 or axis != diff.ndim - 1:
        return np.swapaxes(arr_in, -1, axis)
    else:
        return arr_in