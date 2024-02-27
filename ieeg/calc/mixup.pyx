import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t INTP_t
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline void mixup2d(DTYPE_t[:, ::1] arr, DTYPE_t alpha=1.0):
    cdef Py_ssize_t i, n_nan, row, j
    cdef cnp.ndarray[BOOL_t, ndim=1, cast=True] wh
    cdef cnp.ndarray[INTP_t, ndim=2] vectors
    cdef INTP_t [::1] non_nan_rows, nan_rows
    cdef DTYPE_t [::1] lam

    # Get indices of rows with NaN values
    wh = np.isnan(arr).any(axis=1)
    non_nan_rows = np.nonzero(~wh)[0]
    nan_rows = np.nonzero(wh)[0]
    n_nan = nan_rows.shape[0]

    # Construct an array of 2-length vectors for each NaN row
    vectors = np.empty((n_nan, 2)).astype(np.intp)

    # The two elements of each vector are different indices of non-NaN rows
    for i in range(n_nan):
        vectors[i, :] = np.random.choice(non_nan_rows, 2, replace=False)

    # get beta distribution parameters
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=n_nan).astype(DTYPE)
    else:
        lam = np.ones(n_nan, dtype=DTYPE)

    with nogil:
        for i in range(n_nan):
            row = nan_rows[i]
            for j in range(arr.shape[1]):
                arr[row, j] = lam[i] * arr[vectors[i, 0], j] + (1 - lam[i]) * arr[vectors[i, 1], j]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mixup3d(DTYPE_t[:,:,::1] arr, float alpha):
    for i in range(arr.shape[0]):
        mixup2d(arr[i, :, :], alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mixup4d(DTYPE_t[:,:,:,::1] arr, float alpha):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            mixup2d(arr[i, j, :, :], alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mixup5d(DTYPE_t[:,:,:,:,::1] arr, float alpha):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                mixup2d(arr[i, j, k, :, :], alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mixupnd(cnp.ndarray[DTYPE_t] arr, float alpha):
    if arr.ndim < 2:
        raise ValueError("Cannot apply mixup to a 1-dimensional array")  
    elif arr.ndim == 2:
        mixup2d(arr, alpha)
    else:
        for i in range(arr.shape[0]):
            # Ensure that the last two dimensions are free
            mixupnd(arr[i], alpha)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline void norm1d(DTYPE_t [:] arr):
    cdef Py_ssize_t i
    cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] wh
    cdef INTP_t[::1] non_nan_rows
    cdef DTYPE_t mean = 0, std = 0

    # Get indices of rows with NaN values
    wh = np.isnan(arr)
    non_nan_rows = np.flatnonzero(~wh)

    # Check if there are at least two non-NaN rows
    if non_nan_rows.shape[0] < 1:
        raise ValueError("No test data to fit distribution")

    # Calculate mean and standard deviation for each column
    for i in non_nan_rows:
        mean += arr[i]
    mean /= non_nan_rows.shape[0]
    for i in non_nan_rows:
        std += (arr[i] - mean) * (arr[i] - mean)
    std /= (non_nan_rows.shape[0] - 1)
    std = sqrt(std)

    # Get the normal distribution of each timepoint
    for i in np.flatnonzero(wh):
        arr[i] = np.random.normal(mean, std)


@cython.boundscheck(False)
cpdef void norm2d(DTYPE_t[:,::1] arr):
    for i in range(arr.shape[0]):
        norm1d(arr[i, :])


@cython.boundscheck(False)
cpdef void norm3d(DTYPE_t[:,:,::1] arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            norm1d(arr[i, j, :])


@cython.boundscheck(False)
cpdef void norm4d(DTYPE_t[:,:,:,::1] arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                norm1d(arr[i, j, k, :])


@cython.boundscheck(False)
cpdef void norm5d(DTYPE_t[:,:,:,:,::1] arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                for l in range(arr.shape[3]):
                    norm1d(arr[i, j, k, l, :])


@cython.boundscheck(False)
cpdef void normnd(cnp.ndarray[DTYPE_t] arr):
    if arr.ndim == 1:
        norm1d(arr)
    else:
        for i in range(arr.shape[0]):
            # Ensure that the last two dimensions are free
            normnd(arr[i])
