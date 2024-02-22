import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t INTP_t
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void mixup2d(DTYPE_t[:, ::1] arr, DTYPE_t alpha=1.0):
    cdef Py_ssize_t i, n_nan, row, j
    cdef cnp.ndarray[BOOL_t, ndim=1, cast=True] wh
    cdef INTP_t [::1] non_nan_rows, nan_rows
    cdef DTYPE_t [::1] lam
    cdef INTP_t x1, x2

    # Get indices of rows with NaN values
    wh = np.isnan(arr).any(axis=1)
    non_nan_rows = np.nonzero(~wh)[0]
    nan_rows = np.nonzero(wh)[0]
    n_nan = arr.shape[0] - non_nan_rows.shape[0]

    # get beta distribution parameters
    if alpha >= 1. :
        lam = np.array([rand() / float(RAND_MAX) for _ in range(n_nan)]).astype(DTYPE)
    elif 1. > alpha > 0.:
        lam = np.random.beta(alpha, alpha, size=n_nan).astype(DTYPE)
    else:
        lam = np.ones(n_nan, dtype=DTYPE)

    with nogil:
        for i in range(n_nan):
            row = nan_rows[i]
            x1 = non_nan_rows[rand() % non_nan_rows.shape[0]]
            x2 = non_nan_rows[rand() % non_nan_rows.shape[0]]
            while x1 == x2:
                x2 = non_nan_rows[rand() % non_nan_rows.shape[0]]
            for j in range(arr.shape[1]):
                arr[row, j] = lam[i] * arr[x1, j] + (1 - lam[i]) * arr[x2, j]


@cython.boundscheck(False)
cpdef void mixupnd(cnp.ndarray arr, int obs_axis, float alpha=1.0):
    cdef cnp.ndarray arr_in

    # create a view of the array with the observation axis in the second to
    # last position
    if obs_axis != -2:
        arr_in = np.swapaxes(arr, obs_axis, -2)
    else:
        arr_in = arr

    if arr.ndim == 2:
        mixup2d(arr_in, alpha)
    elif arr.ndim > 2:
        for i in range(arr_in.shape[0]):
            # Ensure that the last two dimensions are free
            mixupnd(arr_in[i], -2, alpha)
    else:
        raise ValueError("Cannot apply mixup to a 1-dimensional array")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void norm1d(DTYPE_t [:] arr):
    cdef Py_ssize_t i
    cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] wh
    cdef DTYPE_t mean, std

    # Get indices of rows with NaN values
    wh = np.isnan(arr)

    # Check if there are at least two non-NaN rows
    if np.sum(~wh) < 1:
        raise ValueError("No test data to fit distribution")

    # Calculate mean and standard deviation for each column
    mean = np.mean(arr[~wh])
    std = np.std(arr[~wh])

    # Get the normal distribution of each timepoint
    for i in np.flatnonzero(wh):
        arr[i] = np.random.normal(mean, std)

@cython.boundscheck(False)
cpdef void normnd(cnp.ndarray arr, int obs_axis=-1):
    cdef cnp.ndarray arr_in

    # create a view of the array with the observation axis in the last position
    if obs_axis != -1:
        arr_in = np.swapaxes(arr, obs_axis, -1)
    else:
        arr_in = arr

    if arr.ndim == 1:
        norm1d(arr_in)
    elif arr.ndim > 1:
        for i in range(arr_in.shape[0]):
            # Ensure that the last two dimensions are free
            normnd(arr_in[i], -1)
    else:
        raise ValueError("Cannot apply norm to a 0-dimensional array")
