cimport numpy as cnp
import cython
import numpy as np
from libc.math cimport NAN

cnp.import_array()

cpdef cnp.ndarray nan_concatinate(list[cnp.ndarray] arrs, int axis=0):

    cdef int ax, i, n_arr = len(arrs), size = 0, dim, n_dim = 0
    cdef cnp.ndarray arr_out
    cdef list out_shape

    # Get the shapes of the arrays

    for i in range(n_arr):
        if arrs[i].ndim > n_dim:
            n_dim = arrs[i].ndim

    if n_dim == 0:
        return np.concatenate(arrs)

    assert n_dim > axis, "Axis out of bounds."

    # out_shape = <int *>malloc(n_dim * sizeof(int))
    out_shape = [0] * n_dim
    for i in range(n_dim):
        # out_shape[i] = 0
        for j in range(n_arr):
            if i < arrs[j].ndim:
                dim = arrs[j].shape[i]
            else:
                dim = 1

            if i == axis:
                out_shape[i] += dim
            elif out_shape[i] < dim:
                out_shape[i] = dim

    # Create the output array
    arr_out = np.empty(out_shape, arrs[0].dtype)

    # Copy the arrays into the output array
    for i in range(n_arr):
        if n_dim == 1:
            fill1d(arrs[i], arr_out, size)
        elif n_dim == 2:
            fill2d(arrs[i], arr_out, size, axis)
        elif n_dim == 3:
            fill3d(arrs[i], arr_out, size, axis)
        elif n_dim == 4:
            fill4d(arrs[i], arr_out, size, axis)
        elif n_dim == 5:
            fill5d(arrs[i], arr_out, size, axis)
        elif n_dim == 6:
            fill6d(arrs[i], arr_out, size, axis)
        elif n_dim == 7:
            fill7d(arrs[i], arr_out, size, axis)
        else:
            raise ValueError("Only up to 7D arrays are supported.")
        size += arrs[i].shape[axis]

    return arr_out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill1d(double[::1] arr, double[::1] arr_out, const int start) noexcept nogil:
    cdef int i

    for i in range(arr.shape[0]):
        if i < arr.shape[0]:
            arr_out[start + i] = arr[i]
        else:
            arr_out[start + i] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill2d(double[:, ::1] arr, double[:, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, istart
    cdef Py_ssize_t[2] max_shape

    max_shape[0] = max(arr_out.shape[0], arr.shape[0])
    max_shape[1] = max(arr_out.shape[1], arr.shape[1])

    if axis == 0:
        for i in range(arr.shape[0]):
            istart = start + i
            for j in range(arr.shape[1]):
                arr_out[istart, j] = arr[i, j]
            for j in range(arr.shape[1], max_shape[1]):
                arr_out[istart, j] = NAN
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                istart = start + j
                arr_out[i, istart] = arr[i, j]
            for j in range(arr.shape[1], max_shape[0]):
                arr_out[i, j] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill3d(double[:, :, ::1] arr, double[:, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, istart
    cdef Py_ssize_t[3] max_shape

    max_shape[0] = max(arr_out.shape[0], arr.shape[0])
    max_shape[1] = max(arr_out.shape[1], arr.shape[1])
    max_shape[2] = max(arr_out.shape[2], arr.shape[2])

    if axis == 0:
        for i in range(arr.shape[0]):
            istart = start + i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    arr_out[istart, j, k] = arr[i, j, k]
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    arr_out[istart, j, k] = NAN
            for j in range(max_shape[0]):
                for k in range(arr.shape[2], max_shape[2]):
                    arr_out[istart, j, k] = NAN
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                istart = start + j
                for k in range(arr.shape[2]):
                    arr_out[i, istart, k] = arr[i, j, k]
                for k in range(arr.shape[2], max_shape[2]):
                    arr_out[i, istart, k] = NAN
            for j in range(arr.shape[1], max_shape[0]):
                for k in range(max_shape[1]):
                    arr_out[j, i, k] = NAN
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    istart = start + k
                    arr_out[i, j, istart] = arr[i, j, k]
                for k in range(arr.shape[2], max_shape[1]):
                    arr_out[i, j, k] = NAN
            for k in range(arr.shape[1], max_shape[0]):
                for j in range(max_shape[1]):
                    arr_out[k, j, i] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill4d(double[:, :, :, ::1] arr, double[:, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, istart
    cdef Py_ssize_t[4] max_shape

    max_shape[0] = max(arr_out.shape[0], arr.shape[0])
    max_shape[1] = max(arr_out.shape[1], arr.shape[1])
    max_shape[2] = max(arr_out.shape[2], arr.shape[2])
    max_shape[3] = max(arr_out.shape[3], arr.shape[3])

    if axis == 0:
        for i in range(arr.shape[0]):
            istart = start + i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        arr_out[istart, j, k, l] = arr[i, j, k, l]
                    for l in range(arr.shape[3], max_shape[3]):
                        arr_out[istart, j, k, l] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        arr_out[istart, j, k, l] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[2]):
                    for l in range(max_shape[3]):
                        arr_out[istart, j, k, l] = NAN
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                istart = start + j
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        arr_out[i, istart, k, l] = arr[i, j, k, l]
                    for l in range(arr.shape[3], max_shape[3]):
                        arr_out[i, istart, k, l] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        arr_out[i, istart, k, l] = NAN
            for j in range(arr.shape[1], max_shape[0]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        arr_out[j, i, k, l] = NAN
    elif axis == 2:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    istart = start + k
                    for l in range(arr.shape[3]):
                        arr_out[i, j, istart, l] = arr[i, j, k, l]
                    for l in range(arr.shape[3], max_shape[3]):
                        arr_out[i, j, istart, l] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        arr_out[i, j, k, l] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        arr_out[j, k, i, l] = NAN
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        istart = start + l
                        arr_out[i, j, k, istart] = arr[i, j, k, l]
                    for l in range(arr.shape[3], max_shape[3]):
                        arr_out[i, j, k, l] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        arr_out[i, j, k, l] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        arr_out[j, k, i, l] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill5d(double[:, :, :, :, ::1] arr, double[:, :, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, m, istart
    cdef Py_ssize_t[5] max_shape

    max_shape[0] = max(arr_out.shape[0], arr.shape[0])
    max_shape[1] = max(arr_out.shape[1], arr.shape[1])
    max_shape[2] = max(arr_out.shape[2], arr.shape[2])
    max_shape[3] = max(arr_out.shape[3], arr.shape[3])
    max_shape[4] = max(arr_out.shape[4], arr.shape[4])

    if axis == 0:
        for i in range(arr.shape[0]):
            istart = start + i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            arr_out[istart, j, k, l, m] = arr[i, j, k, l, m]
                        for m in range(arr.shape[4], max_shape[4]):
                            arr_out[istart, j, k, l, m] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[istart, j, k, l, m] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[istart, j, k, l, m] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[istart, j, k, l, m] = NAN
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                istart = start + j
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            arr_out[i, istart, k, l, m] = arr[i, j, k, l, m]
                        for m in range(arr.shape[4], max_shape[4]):
                            arr_out[i, istart, k, l, m] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, istart, k, l, m] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, istart, k, l, m] = NAN
            for j in range(arr.shape[1], max_shape[0]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            arr_out[j, i, k, l, m] = NAN
    elif axis == 2:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    istart = start + k
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            arr_out[i, j, istart, l, m] = arr[i, j, k, l, m]
                        for m in range(arr.shape[4], max_shape[4]):
                            arr_out[i, j, istart, l, m] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, j, istart, l, m] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, j, k, l, m] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            arr_out[j, k, i, l, m] = NAN
    elif axis == 3:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        istart = start + l
                        for m in range(arr.shape[4]):
                            arr_out[i, j, k, istart, m] = arr[i, j, k, l, m]
                        for m in range(arr.shape[4], max_shape[4]):
                            arr_out[i, j, k, istart, m] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, j, k, l, m] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, j, k, l, m] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            arr_out[j, k, i, l, m] = NAN
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            istart = start + m
                            arr_out[i, j, k, l, istart] = arr[i, j, k, l, m]
                        for m in range(arr.shape[4], max_shape[4]):
                            arr_out[i, j, k, l, m] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, j, k, l, m] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            arr_out[i, j, k, l, m] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            arr_out[j, k, i, l, m] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill6d(double[:, :, :, :, :, ::1] arr, double[:, :, :, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, m, n, istart
    cdef Py_ssize_t[6] max_shape

    max_shape[0] = max(arr_out.shape[0], arr.shape[0])
    max_shape[1] = max(arr_out.shape[1], arr.shape[1])
    max_shape[2] = max(arr_out.shape[2], arr.shape[2])
    max_shape[3] = max(arr_out.shape[3], arr.shape[3])
    max_shape[4] = max(arr_out.shape[4], arr.shape[4])
    max_shape[5] = max(arr_out.shape[5], arr.shape[5])

    if axis == 0:
        for i in range(arr.shape[0]):
            istart = start + i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                arr_out[istart, j, k, l, m, n] = arr[i, j, k, l, m, n]
                            for n in range(arr.shape[5], max_shape[5]):
                                arr_out[istart, j, k, l, m, n] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[istart, j, k, l, m, n] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[istart, j, k, l, m, n] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[istart, j, k, l, m, n] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[istart, j, k, l, m, n] = NAN
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                istart = start + j
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                arr_out[i, istart, k, l, m, n] = arr[i, j, k, l, m, n]
                            for n in range(arr.shape[5], max_shape[5]):
                                arr_out[i, istart, k, l, m, n] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, istart, k, l, m, n] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, istart, k, l, m, n] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, istart, k, l, m, n] = NAN
            for j in range(arr.shape[1], max_shape[0]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                arr_out[j, i, k, l, m, n] = NAN
    elif axis == 2:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    istart = start + k
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                arr_out[i, j, istart, l, m, n] = arr[i, j, k, l, m, n]
                            for n in range(arr.shape[5], max_shape[5]):
                                arr_out[i, j, istart, l, m, n] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, istart, l, m, n] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, istart, l, m, n] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                arr_out[j, k, i, l, m, n] = NAN
    elif axis == 3:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        istart = start + l
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                arr_out[i, j, k, istart, m, n] = arr[i, j, k, l, m, n]
                            for n in range(arr.shape[5], max_shape[5]):
                                arr_out[i, j, k, istart, m, n] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, istart, m, n] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                arr_out[j, k, i, l, m, n] = NAN
    elif axis == 4:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            istart = start + m
                            for n in range(arr.shape[5]):
                                arr_out[i, j, k, l, istart, n] = arr[i, j, k, l, m, n]
                            for n in range(arr.shape[5], max_shape[5]):
                                arr_out[i, j, k, l, istart, n] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                arr_out[j, k, i, l, m, n] = NAN
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                istart = start + n
                                arr_out[i, j, k, l, m, istart] = arr[i, j, k, l, m, n]
                            for n in range(arr.shape[5], max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                arr_out[i, j, k, l, m, n] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                arr_out[j, k, i, l, m, n] = NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill7d(double[:, :, :, :, :, :, ::1] arr, double[:, :, :, :, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, m, n, o, istart
    cdef Py_ssize_t[7] max_shape

    max_shape[0] = max(arr_out.shape[0], arr.shape[0])
    max_shape[1] = max(arr_out.shape[1], arr.shape[1])
    max_shape[2] = max(arr_out.shape[2], arr.shape[2])
    max_shape[3] = max(arr_out.shape[3], arr.shape[3])
    max_shape[4] = max(arr_out.shape[4], arr.shape[4])
    max_shape[5] = max(arr_out.shape[5], arr.shape[5])
    max_shape[6] = max(arr_out.shape[6], arr.shape[6])

    if axis == 0:
        for i in range(arr.shape[0]):
            istart = start + i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                for o in range(arr.shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[istart, j, k, l, m, n, o] = NAN
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                istart = start + j
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                for o in range(arr.shape[6]):
                                    arr_out[i, istart, k, l, m, n, o] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[i, istart, k, l, m, n, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, istart, k, l, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, istart, k, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, istart, k, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, istart, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[0]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                for o in range(max_shape[5]):
                                    arr_out[j, i, k, l, m, n, o] = NAN
    elif axis == 2:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    istart = start + k
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                for o in range(arr.shape[6]):
                                    arr_out[i, j, istart, l, m, n, o] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[i, j, istart, l, m, n, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, istart, l, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, istart, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, istart, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                for o in range(max_shape[5]):
                                    arr_out[j, k, i, l, m, n, o] = NAN
    elif axis == 3:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        istart = start + l
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                for o in range(arr.shape[6]):
                                    arr_out[i, j, k, istart, m, n, o] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[i, j, k, istart, m, n, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, istart, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                for o in range(max_shape[5]):
                                    arr_out[j, k, i, l, m, n, o] = NAN
    elif axis == 4:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            istart = start + m
                            for n in range(arr.shape[5]):
                                for o in range(arr.shape[6]):
                                    arr_out[i, j, k, l, istart, n, o] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[i, j, k, l, istart, n, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                for o in range(max_shape[5]):
                                    arr_out[j, k, i, l, m, n, o] = NAN
    elif axis == 5:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                istart = start + n
                                for o in range(arr.shape[6]):
                                    arr_out[i, j, k, l, m, istart, o] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[i, j, k, l, m, istart, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                for o in range(max_shape[5]):
                                    arr_out[j, k, i, l, m, n, o] = NAN
    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                istart = start + o
                                arr_out[i, j, k, l, m, n, istart] = arr[i, j, k, l, m, n, o]
                                for o in range(arr.shape[6], max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                            for n in range(arr.shape[5], max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                        for m in range(arr.shape[4], max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                    for l in range(arr.shape[3], max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
                for k in range(arr.shape[2], max_shape[2]):
                    for l in range(max_shape[3]):
                        for m in range(max_shape[4]):
                            for n in range(max_shape[5]):
                                for o in range(max_shape[6]):
                                    arr_out[i, j, k, l, m, n, o] = NAN
            for j in range(arr.shape[1], max_shape[1]):
                for k in range(max_shape[1]):
                    for l in range(max_shape[2]):
                        for m in range(max_shape[3]):
                            for n in range(max_shape[4]):
                                for o in range(max_shape[5]):
                                    arr_out[j, k, i, l, m, n, o] = NAN

