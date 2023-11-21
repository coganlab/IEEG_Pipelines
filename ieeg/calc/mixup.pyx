import numpy as np
cimport numpy as np
from libc.stdlib cimport rand
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def mixup2d(np.ndarray[DTYPE_t, ndim=2] arr, float alpha=1.0):
    cdef Py_ssize_t i, n_nan
    cdef np.ndarray[DTYPE_t, ndim=1] lam
    cdef np.ndarray[DTYPE_t, ndim=2] x1, x2
    cdef np.intp_t [::1] non_nan_rows
    cdef np.intp_t [:, ::1] vectors
    cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] wh

    # Get indices of rows with NaN values
    wh = np.isnan(arr).any(axis=1)
    non_nan_rows = np.flatnonzero(~wh)
    n_nan = arr.shape[0] - non_nan_rows.shape[0]

    # Construct an array of 2-length vectors for each NaN row
    vectors = np.empty((n_nan, 2), dtype=np.intp)

    # The two elements of each vector are different indices of non-NaN rows
    for i in range(n_nan):
        (vectors[i, 0], vectors[i, 1]) = np.random.choice(non_nan_rows, 2, replace=False)

    # get beta distribution parameters
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha, size=n_nan).astype(DTYPE)
    else:
        lam = np.ones(n_nan, dtype=DTYPE)

    x1 = arr[vectors[:, 0]]
    x2 = arr[vectors[:, 1]]

    arr[wh] = lam[:, None] * x1 + (np.ones(n_nan, dtype=DTYPE) - lam)[:, None] * x2
