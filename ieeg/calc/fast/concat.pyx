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
    cdef Py_ssize_t i

    for i in range(arr.shape[0]):
        arr_out[start + i] = arr[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill2d(double[:, ::1] arr, double[:, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j

    if axis == 0:
        for i in range(arr.shape[0]):
            fill1d(arr[i], arr_out[start + i], 0)
            for j in range(arr.shape[1], arr_out.shape[1]):
                arr_out[start + i, j] = NAN

    else:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr_out[i, start + j] = arr[i, j]
        for i in range(arr.shape[0], arr_out.shape[0]):
            for j in range(arr.shape[1]):
                arr_out[i, start + j] = NAN


cdef void _3shape(const Py_ssize_t shape1, const Py_ssize_t shape2, const Py_ssize_t shape3,
                  const Py_ssize_t shape4, Py_ssize_t[3][4] shapes) noexcept nogil:

    shapes[0][0] = shape1; shapes[0][1] = shape2
    shapes[0][2] = shape3; shapes[0][3] = shape4

    shapes[1][0] = 0; shapes[1][1] = shape1
    shapes[1][2] = shape3; shapes[1][3] = shape4

    shapes[2][0] = shape1; shapes[2][1] = shape2
    shapes[2][2] = 0; shapes[2][3] = shape3


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill3d(double[:, :, ::1] arr, double[:, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, istart, x
    cdef Py_ssize_t[3][4] shapes
    cdef Py_ssize_t[4] sh

    istart = arr.shape[axis] + start

    if axis == 0:
        _3shape(arr.shape[1], arr_out.shape[1], arr.shape[2], arr_out.shape[2], shapes)
        for i in range(start, istart):
            fill2d(arr[i - start], arr_out[i], 0, 1)
            for x in range(3):
                for j in range(shapes[x][0], shapes[x][1]):
                    for k in range(shapes[x][2], shapes[x][3]):
                        arr_out[i, j, k] = NAN
    elif axis == 1:
        _3shape(arr.shape[0], arr_out.shape[0], arr.shape[2], arr_out.shape[2], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                fill1d(arr[i, j], arr_out[i, start + j], 0)
        for x in range(3):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(start, istart):
                    for k in range(shapes[x][2], shapes[x][3]):
                        arr_out[i, j, k] = NAN
    else:
        _3shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    arr_out[i, j, start + k] = arr[i, j, k]
        for x in range(3):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(start, istart):
                        arr_out[i, j, start + k] = NAN


cdef void _4shape(const Py_ssize_t shape1, const Py_ssize_t shape2, const Py_ssize_t shape3,
                  const Py_ssize_t shape4, const Py_ssize_t shape5, const Py_ssize_t shape6,
                  Py_ssize_t[4][6] shapes) noexcept nogil:

    shapes[0][0] = shape1; shapes[0][1] = shape2
    shapes[0][2] = shape3; shapes[0][3] = shape4
    shapes[0][4] = shape5; shapes[0][5] = shape6

    shapes[1][0] = 0; shapes[1][1] = shape1
    shapes[1][2] = shape3; shapes[1][3] = shape4
    shapes[1][4] = shape5; shapes[1][5] = shape6

    shapes[2][0] = shape1; shapes[2][1] = shape2
    shapes[2][2] = 0; shapes[2][3] = shape3
    shapes[2][4] = shape5; shapes[2][5] = shape6

    shapes[3][0] = shape1; shapes[3][1] = shape2
    shapes[3][2] = shape3; shapes[3][3] = shape4
    shapes[3][4] = 0; shapes[3][5] = shape6


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill4d(double[:, :, :, ::1] arr, double[:, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, x, istart
    cdef Py_ssize_t[4][6] shapes
    
    istart = arr.shape[axis] + start

    if axis == 0:
        _4shape(arr.shape[1], arr_out.shape[1], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], shapes)
        for i in range(start, istart):
            fill3d(arr[i - start], arr_out[i], 0, 1)
            for x in range(4):
                for j in range(shapes[x][0], shapes[x][1]):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            arr_out[i, j, k, l] = NAN
    elif axis == 1:
        _4shape(arr.shape[0], arr_out.shape[0], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                fill2d(arr[i, j], arr_out[i, start + j], 0, 1)
        for x in range(4):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(start, istart):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            arr_out[i, j, k, l] = NAN
    elif axis == 2:
        _4shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[3], arr_out.shape[3], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    fill1d(arr[i, j, k], arr_out[i, j, start + k], 0)
        for x in range(4):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(start, istart):
                        for l in range(shapes[x][4], shapes[x][5]):
                            arr_out[i, j, k, l] = NAN
    else:
        _4shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        arr_out[i, j, k, start + l] = arr[i, j, k, l]
        for x in range(4):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(start, istart):
                            arr_out[i, j, k, l] = NAN


cdef void _5shape(const Py_ssize_t shape1, const Py_ssize_t shape2, const Py_ssize_t shape3,
                  const Py_ssize_t shape4, const Py_ssize_t shape5, const Py_ssize_t shape6,
                  const Py_ssize_t shape7, const Py_ssize_t shape8, Py_ssize_t[5][8] shapes) noexcept nogil:

    shapes[0][0] = shape1; shapes[0][1] = shape2
    shapes[0][2] = shape3; shapes[0][3] = shape4
    shapes[0][4] = shape5; shapes[0][5] = shape6
    shapes[0][6] = shape7; shapes[0][7] = shape8

    shapes[1][0] = 0; shapes[1][1] = shape1
    shapes[1][2] = shape3; shapes[1][3] = shape4
    shapes[1][4] = shape5; shapes[1][5] = shape6
    shapes[1][6] = shape7; shapes[1][7] = shape8

    shapes[2][0] = shape1; shapes[2][1] = shape2
    shapes[2][2] = 0; shapes[2][3] = shape3
    shapes[2][4] = shape5; shapes[2][5] = shape6
    shapes[2][6] = shape7; shapes[2][7] = shape8

    shapes[3][0] = shape1; shapes[3][1] = shape2
    shapes[3][2] = shape3; shapes[3][3] = shape4
    shapes[3][4] = 0; shapes[3][5] = shape5
    shapes[3][6] = shape7; shapes[3][7] = shape8

    shapes[4][0] = shape1; shapes[4][1] = shape2
    shapes[4][2] = shape3; shapes[4][3] = shape4
    shapes[4][4] = shape5; shapes[4][5] = shape6
    shapes[4][6] = 0; shapes[4][7] = shape7


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill5d(double[:, :, :, :, ::1] arr, double[:, :, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, m, x, istart
    cdef Py_ssize_t[5][8] shapes
    
    istart = arr.shape[axis] + start
    
    if axis == 0:
        _5shape(arr.shape[1], arr_out.shape[1], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4], shapes)
        for i in range(start, istart):
            fill4d(arr[i - start], arr_out[i], 0, 1)
            for x in range(5):
                for j in range(shapes[x][0], shapes[x][1]):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                arr_out[i, j, k, l, m] = NAN
    elif axis == 1:
        _5shape(arr.shape[0], arr_out.shape[0], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                fill3d(arr[i, j], arr_out[i, start + j], 0, 1)
        for x in range(5):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(start, istart):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                arr_out[i, j, k, l, m] = NAN
    elif axis == 2:
        _5shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    fill2d(arr[i, j, k], arr_out[i, j, start + k], 0, 1)
        for x in range(5):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(start, istart):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                arr_out[i, j, k, l, m] = NAN
    elif axis == 3:
        _5shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[4], arr_out.shape[4], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        fill1d(arr[i, j, k, l], arr_out[i, j, k, start + l], 0)
        for x in range(5):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(start, istart):
                            for m in range(shapes[x][6], shapes[x][7]):
                                arr_out[i, j, k, l, m] = NAN
    else:
        _5shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[3], arr_out.shape[3], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            arr_out[i, j, k, l, start + m] = arr[i, j, k, l, m]
        for x in range(5):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(shapes[x][6], shapes[x][7]):
                            for m in range(start, istart):
                                arr_out[i, j, k, l, m] = NAN


cdef void _6shape(const Py_ssize_t shape1, const Py_ssize_t shape2, const Py_ssize_t shape3,
                  const Py_ssize_t shape4, const Py_ssize_t shape5, const Py_ssize_t shape6,
                  const Py_ssize_t shape7, const Py_ssize_t shape8, const Py_ssize_t shape9,
                  const Py_ssize_t shape10, Py_ssize_t[6][10] shapes) noexcept nogil:

    shapes[0][0] = shape1; shapes[0][1] = shape2
    shapes[0][2] = shape3; shapes[0][3] = shape4
    shapes[0][4] = shape5; shapes[0][5] = shape6
    shapes[0][6] = shape7; shapes[0][7] = shape8
    shapes[0][8] = shape9; shapes[0][9] = shape10

    shapes[1][0] = 0; shapes[1][1] = shape1
    shapes[1][2] = shape3; shapes[1][3] = shape4
    shapes[1][4] = shape5; shapes[1][5] = shape6
    shapes[1][6] = shape7; shapes[1][7] = shape8
    shapes[1][8] = shape9; shapes[1][9] = shape10

    shapes[2][0] = shape1; shapes[2][1] = shape2
    shapes[2][2] = 0; shapes[2][3] = shape3
    shapes[2][4] = shape5; shapes[2][5] = shape6
    shapes[2][6] = shape7; shapes[2][7] = shape8
    shapes[2][8] = shape9; shapes[2][9] = shape10

    shapes[3][0] = shape1; shapes[3][1] = shape2
    shapes[3][2] = shape3; shapes[3][3] = shape4
    shapes[3][4] = 0; shapes[3][5] = shape5
    shapes[3][6] = shape7; shapes[3][7] = shape8
    shapes[3][8] = shape9; shapes[3][9] = shape10

    shapes[4][0] = shape1; shapes[4][1] = shape2
    shapes[4][2] = shape3; shapes[4][3] = shape4
    shapes[4][4] = shape5; shapes[4][5] = shape6
    shapes[4][6] = 0; shapes[4][7] = shape7
    shapes[4][8] = shape9; shapes[4][9] = shape10
    
    shapes[5][0] = shape1; shapes[5][1] = shape2
    shapes[5][2] = shape3; shapes[5][3] = shape4
    shapes[5][4] = shape5; shapes[5][5] = shape6
    shapes[5][6] = shape7; shapes[5][7] = shape8
    shapes[5][8] = 0; shapes[5][9] = shape9


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill6d(double[:, :, :, :, :, ::1] arr, double[:, :, :, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, m, n, x, istart
    cdef Py_ssize_t[6][10] shapes
    
    istart = arr.shape[axis] + start
    
    if axis == 0:
        _6shape(arr.shape[1], arr_out.shape[1], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], shapes)
        for i in range(start, istart):
            fill5d(arr[i - start], arr_out[i], 0, 1)
            for x in range(6):
                for j in range(shapes[x][0], shapes[x][1]):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    arr_out[i, j, k, l, m, n] = NAN
    elif axis == 1:
        _6shape(arr.shape[0], arr_out.shape[0], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                fill4d(arr[i, j], arr_out[i, start + j], 0, 1)
        for x in range(6):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(start, istart):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    arr_out[i, j, k, l, m, n] = NAN
    elif axis == 2:
        _6shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    fill3d(arr[i, j, k], arr_out[i, j, start + k], 0, 1)
        for x in range(6):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(start, istart):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    arr_out[i, j, k, l, m, n] = NAN
    elif axis == 3:
        _6shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        fill2d(arr[i, j, k, l], arr_out[i, j, k, start + l], 0, 1)
        for x in range(6):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(start, istart):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    arr_out[i, j, k, l, m, n] = NAN
    elif axis == 4:
        _6shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[3], arr_out.shape[3],
                arr.shape[5], arr_out.shape[5], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            fill1d(arr[i, j, k, l, m], arr_out[i, j, k, l, start + m], 0)
        for x in range(6):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(shapes[x][6], shapes[x][7]):
                            for m in range(start, istart):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    arr_out[i, j, k, l, m, n] = NAN
    else:
        _6shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[3], arr_out.shape[3],
                arr.shape[4], arr_out.shape[4], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                arr_out[i, j, k, l, m, start + n] = arr[i, j, k, l, m, n]
        for x in range(6):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(shapes[x][6], shapes[x][7]):
                            for m in range(shapes[x][8], shapes[x][9]):
                                for n in range(start, istart):
                                    arr_out[i, j, k, l, m, n] = NAN
                                    
                                    
cdef void _7shape(const Py_ssize_t shape1, const Py_ssize_t shape2, const Py_ssize_t shape3,
                    const Py_ssize_t shape4, const Py_ssize_t shape5, const Py_ssize_t shape6,
                    const Py_ssize_t shape7, const Py_ssize_t shape8, const Py_ssize_t shape9,
                    const Py_ssize_t shape10, const Py_ssize_t shape11, const Py_ssize_t shape12,
                    Py_ssize_t[7][12] shapes) noexcept nogil:
    
        shapes[0][0] = shape1; shapes[0][1] = shape2
        shapes[0][2] = shape3; shapes[0][3] = shape4
        shapes[0][4] = shape5; shapes[0][5] = shape6
        shapes[0][6] = shape7; shapes[0][7] = shape8
        shapes[0][8] = shape9; shapes[0][9] = shape10
        shapes[0][10] = shape11; shapes[0][11] = shape12
    
        shapes[1][0] = 0; shapes[1][1] = shape1
        shapes[1][2] = shape3; shapes[1][3] = shape4
        shapes[1][4] = shape5; shapes[1][5] = shape6
        shapes[1][6] = shape7; shapes[1][7] = shape8
        shapes[1][8] = shape9; shapes[1][9] = shape10
        shapes[1][10] = shape11; shapes[1][11] = shape12
    
        shapes[2][0] = shape1; shapes[2][1] = shape2
        shapes[2][2] = 0; shapes[2][3] = shape3
        shapes[2][4] = shape5; shapes[2][5] = shape6
        shapes[2][6] = shape7; shapes[2][7] = shape8
        shapes[2][8] = shape9; shapes[2][9] = shape10
        shapes[2][10] = shape11; shapes[2][11] = shape12
    
        shapes[3][0] = shape1; shapes[3][1] = shape2
        shapes[3][2] = shape3; shapes[3][3] = shape4
        shapes[3][4] = 0; shapes[3][5] = shape5
        shapes[3][6] = shape7; shapes[3][7] = shape8
        shapes[3][8] = shape9; shapes[3][9] = shape10
        shapes[3][10] = shape11; shapes[3][11] = shape12

        shapes[4][0] = shape1; shapes[4][1] = shape2
        shapes[4][2] = shape3; shapes[4][3] = shape4
        shapes[4][4] = shape5; shapes[4][5] = shape6
        shapes[4][6] = 0; shapes[4][7] = shape7
        shapes[4][8] = shape9; shapes[4][9] = shape10
        shapes[4][10] = shape11; shapes[4][11] = shape12
        
        shapes[5][0] = shape1; shapes[5][1] = shape2
        shapes[5][2] = shape3; shapes[5][3] = shape4
        shapes[5][4] = shape5; shapes[5][5] = shape6
        shapes[5][6] = shape7; shapes[5][7] = shape8
        shapes[5][8] = 0; shapes[5][9] = shape9
        shapes[5][10] = shape11; shapes[5][11] = shape12
        
        shapes[6][0] = shape1; shapes[6][1] = shape2
        shapes[6][2] = shape3; shapes[6][3] = shape4
        shapes[6][4] = shape5; shapes[6][5] = shape6
        shapes[6][6] = shape7; shapes[6][7] = shape8
        shapes[6][8] = shape9; shapes[6][9] = shape10
        shapes[6][10] = 0; shapes[6][11] = shape11


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill7d(double[:, :, :, :, :, :, ::1] arr, double[:, :, :, :, :, :, ::1] arr_out, const int start, int axis) noexcept nogil:
    cdef Py_ssize_t i, j, k, l, m, n, o, x, istart
    cdef Py_ssize_t[7][12] shapes

    istart = arr.shape[axis] + start

    if axis == 0:
        _7shape(arr.shape[1], arr_out.shape[1], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], arr.shape[6], arr_out.shape[6], shapes)
        for i in range(start, istart):
            fill6d(arr[i - start], arr_out[i], 0, 1)
            for x in range(7):
                for j in range(shapes[x][0], shapes[x][1]):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    for o in range(shapes[x][10], shapes[x][11]):
                                        arr_out[i, j, k, l, m, n, o] = NAN
    elif axis == 1:
        _7shape(arr.shape[0], arr_out.shape[0], arr.shape[2], arr_out.shape[2],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], arr.shape[6], arr_out.shape[6], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                fill5d(arr[i, j], arr_out[i, start + j], 0, 1)
        for x in range(7):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(start, istart):
                    for k in range(shapes[x][2], shapes[x][3]):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    for o in range(shapes[x][10], shapes[x][11]):
                                        arr_out[i, j, k, l, m, n, o] = NAN
    elif axis == 2:
        _7shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[3], arr_out.shape[3], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], arr.shape[6], arr_out.shape[6], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    fill4d(arr[i, j, k], arr_out[i, j, start + k], 0, 1)
        for x in range(7):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(start, istart):
                        for l in range(shapes[x][4], shapes[x][5]):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    for o in range(shapes[x][10], shapes[x][11]):
                                        arr_out[i, j, k, l, m, n, o] = NAN
    elif axis == 3:
        _7shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[4], arr_out.shape[4],
                arr.shape[5], arr_out.shape[5], arr.shape[6], arr_out.shape[6], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        fill3d(arr[i, j, k, l], arr_out[i, j, k, start + l], 0, 1)
        for x in range(7):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(start, istart):
                            for m in range(shapes[x][6], shapes[x][7]):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    for o in range(shapes[x][10], shapes[x][11]):
                                        arr_out[i, j, k, l, m, n, o] = NAN
    elif axis == 4:
        _7shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[3], arr_out.shape[3],
                arr.shape[5], arr_out.shape[5], arr.shape[6], arr_out.shape[6], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            fill2d(arr[i, j, k, l, m], arr_out[i, j, k, l, start + m], 0, 1)
        for x in range(7):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(shapes[x][6], shapes[x][7]):
                            for m in range(start, istart):
                                for n in range(shapes[x][8], shapes[x][9]):
                                    for o in range(shapes[x][10], shapes[x][11]):
                                        arr_out[i, j, k, l, m, n, o] = NAN
    elif axis == 5:
        _7shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[3], arr_out.shape[3],
                arr.shape[4], arr_out.shape[4], arr.shape[6], arr_out.shape[6], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                fill1d(arr[i, j, k, l, m, n], arr_out[i, j, k, l, m, start + n], 0)
        for x in range(7):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(shapes[x][6], shapes[x][7]):
                            for m in range(shapes[x][8], shapes[x][9]):
                                for n in range(start, istart):
                                    for o in range(shapes[x][10], shapes[x][11]):
                                        arr_out[i, j, k, l, m, n, o] = NAN
    else:
        _7shape(arr.shape[0], arr_out.shape[0], arr.shape[1], arr_out.shape[1],
                arr.shape[2], arr_out.shape[2], arr.shape[3], arr_out.shape[3],
                arr.shape[4], arr_out.shape[4], arr.shape[5], arr_out.shape[5], shapes)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    for l in range(arr.shape[3]):
                        for m in range(arr.shape[4]):
                            for n in range(arr.shape[5]):
                                for o in range(arr.shape[6]):
                                    arr_out[i, j, k, l, m, n, start + o] = arr[i, j, k, l, m, n, o]
        for x in range(7):
            for i in range(shapes[x][0], shapes[x][1]):
                for j in range(shapes[x][2], shapes[x][3]):
                    for k in range(shapes[x][4], shapes[x][5]):
                        for l in range(shapes[x][6], shapes[x][7]):
                            for m in range(shapes[x][8], shapes[x][9]):
                                for n in range(shapes[x][10], shapes[x][11]):
                                    for o in range(start, istart):
                                        arr_out[i, j, k, l, m, n, o] = NAN

