import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, srand, malloc
from libc.math cimport sqrt, isnan
from numpy.random cimport bitgen_t
from numpy.random import SFC64, Generator
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer


cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void mixup2d(double[:, :] arr, const Py_ssize_t x, const Py_ssize_t y,
                         double[::1] lam, const Py_ssize_t n_nan) noexcept nogil:
    cdef Py_ssize_t i, row, j, k = 0, x1, x2, *non_nan_rows, *nan_rows
    cdef int *wh = <int *>malloc(x * sizeof(int))

    non_nan_rows = <Py_ssize_t*>malloc((x - n_nan) * sizeof(Py_ssize_t))
    nan_rows = <Py_ssize_t*>malloc(n_nan * sizeof(Py_ssize_t))

    # Get indices of rows with NaN values
    for i in range(x):
        for j in range(y):
            if isnan(arr[i, j]):
                wh[i] = 1
                nan_rows[i - k] = i
                break
        else:
            wh[i] = 0
            non_nan_rows[k] = i
            k += 1

    for i in range(n_nan):
        row = nan_rows[i]
        x1 = non_nan_rows[rand() % k]
        x2 = non_nan_rows[rand() % k]
        while x1 == x2:
            x2 = non_nan_rows[rand() % k]
        for j in range(y):
            arr[row, j] = lam[i] * arr[x1, j] + (1 - lam[i]) * arr[x2, j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void mixup3d(double[:, :, :] arr, const Py_ssize_t x, const Py_ssize_t y,
                         const Py_ssize_t z, double[:, ::1] lam, Py_ssize_t[::1] n_nan) noexcept nogil:
    for i in range(x):
        mixup2d(arr[i], y, z, lam[i, :n_nan[i]], n_nan[i])

@cython.boundscheck(False)
cpdef void mixupnd(cnp.ndarray arr, int obs_axis, float alpha=1.0,
                   int seed=-1):
    cdef cnp.ndarray arr_in
    cdef RNG rng
    cdef double[:, ::1] lam
    cdef Py_ssize_t[::1] n_nan
    cdef Py_ssize_t i
    global rng

    # create a view of the array with the observation axis in the second to
    # last position
    if obs_axis != -2:
        arr_in = np.swapaxes(arr, obs_axis, -2)
    else:
        arr_in = arr

    if seed >= 0:
        rng = RNG(seed)
        srand(seed)

    if arr.ndim == 2:
        # get beta distribution parameters
        n_nan = np.isnan(arr_in).any(axis=1).sum(dtype=np.intp)[None]
        if alpha > 0:
            lam = rng.generator.beta(alpha, alpha, size=(1, n_nan[0]))
        else:
            lam = np.ones((1, n_nan))
        mixup2d(arr_in, arr_in.shape[0], arr_in.shape[1], lam[0], n_nan[0])
    elif arr.ndim == 3:
        n_nan = np.isnan(arr_in).any(axis=2).sum(axis=1, dtype=np.intp)
        if alpha > 0:
            lam = rng.generator.beta(alpha, alpha, size=(arr_in.shape[0], np.max(n_nan)))
        else:
            lam = np.ones((arr_in.shape[0], np.max(n_nan)))

        mixup3d(arr_in, arr_in.shape[0], arr_in.shape[1], arr_in.shape[2], lam, n_nan)
    elif arr.ndim > 3:
        for i in range(arr_in.shape[0]):
            # Ensure that the last two dimensions are free
            mixupnd(arr_in[i], -2, alpha, -2)
    else:
        raise ValueError("Cannot apply mixup to a 1-dimensional array")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void norm1d(double[:] arr):
    cdef Py_ssize_t i
    cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] wh
    cdef Py_ssize_t[::1] wh_iter
    cdef double mean, std, sum, var

    # Get indices of rows with NaN values
    wh = np.isnan(arr)
    wh_iter = np.flatnonzero(~wh)

    # Check if there are at least two non-NaN rows
    if wh_iter.shape[0] < 1:
        raise ValueError("No test data to fit distribution")

    # Calculate mean and standard deviation for each column
    sum = 0
    var = 0
    for i in wh_iter:
        sum += arr[i]
    mean = sum / wh_iter.shape[0]
    for i in wh_iter:
        var += (arr[i] - mean) ** 2
    std = sqrt(var / wh_iter.shape[0])

    # Get the normal distribution of each timepoint
    for i in range(arr.shape[0]):
        if wh[i]:
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


cdef class RNG:
    cdef bitgen_t *rng
    cdef object bit_generator
    cdef object generator

    def __cinit__(self, int seed=-1):
        cdef const char *capsule_name = "BitGenerator"
        if seed == -1:
            self.bit_generator = SFC64()
        else:
            self.bit_generator = SFC64(seed=seed)
        self.generator = Generator(self.bit_generator)
        capsule = self.bit_generator.capsule
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        self.rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    def lock(self):
        return self.bit_generator.lock

    def spawn(self, int n):
        for i in range(n):
            yield RNG(seed=i)
