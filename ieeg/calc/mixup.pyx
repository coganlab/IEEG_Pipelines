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

    # Get indices of rows with NaN values
    non_nan_rows = np.flatnonzero(~np.isnan(arr).any(axis=1))
    n_nan = non_nan_rows.shape[0]

    # Construct an array of 2-length vectors for each NaN row
    vectors = np.empty((n_nan, 2), dtype=np.intp)

    # The two elements of each vector are different indices of non-NaN rows
    for i in range(n_nan):
        (vectors[i, 0], vectors[i, 1]) = list(np.random.choice(non_nan_rows, 2, replace=False))

    # get beta distribution parameters
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha, size=n_nan).astype(DTYPE)
    else:
        lam = np.ones(n_nan, dtype=DTYPE)

    x1 = arr[vectors[:, 0]]
    x2 = arr[vectors[:, 1]]

    arr[non_nan_rows] = lam[:, None] * x1 + (np.ones(n_nan, dtype=DTYPE) - lam)[:, None] * x2


# Define a fused type that can hold int, double or float[^1^][1]
ctypedef fused numeric:
    int
    double
    float

# Declare a function that takes two memoryviews of the same type
def elem_mult(numeric[::1] vector_1, numeric[::1] vector_2):
    # Check that the vectors have the same length
    assert vector_1.shape[0] == vector_2.shape[0]
    # Get the length of the vectors
    cdef Py_ssize_t n = vector_1.shape[0]
    # Create an empty NumPy array of the same type and length
    cdef numeric[:] result = np.empty(n, dtype=vector_1.dtype)
    # Use a loop to multiply the elements
    cdef Py_ssize_t i
    for i in range(n):
        result[i] = vector_1[i] * vector_2[i]
    # Return the result as a NumPy array
    return np.asarray(result)
