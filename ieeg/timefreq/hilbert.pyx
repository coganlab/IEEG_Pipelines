import numpy as np
from numpy.fft import fft, ifft
cimport numpy as cnp
from libc.math cimport sqrtf, log10f
cimport cython

cnp.import_array()
ctypedef float DTYPE_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.complex64_t DTYPE_C_t

cpdef tuple filterbank_hilbert_first_half_wrapper(cnp.ndarray[DTYPE_t, ndim=2] x, int fs, DTYPE_t minf, DTYPE_t maxf):
    return filterbank_hilbert_first_half_inner(x, fs, minf, maxf)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef tuple filterbank_hilbert_first_half_inner(cnp.ndarray[DTYPE_t, ndim=2] x, int fs, DTYPE_t minf, DTYPE_t maxf):
    cdef DTYPE_t f0 = 0.018, octSpace = 1./7
    cdef DTYPE_t[::1] a = np.array([log10f(0.39), 0.5], dtype='float32')
    cdef DTYPE_t sigma_f = 0.39 * sqrtf(f0)
    cdef cnp.ndarray[DTYPE_C_t, ndim=1] h
    cdef cnp.ndarray[DTYPE_C_t, ndim=2] h_T
    cdef cnp.ndarray[DTYPE_t, ndim=1] cfs, exponent, sigma_fs, sds, freqs
    cdef cnp.ndarray[DTYPE_C_t, ndim=2] Xf
    cdef int N = x.shape[0]
    cdef Py_ssize_t len_cfs = 1, i = 1
    
    while f0 < maxf:
        if f0 < 4:
            f0 += sigma_f
            sigma_f = 0.39 * sqrtf(f0)
        else:
            f0 *= 2**octSpace
        len_cfs += 1
    f0 = 0.018
    sigma_f = 0.39 * np.sqrt(f0)
    cfs = np.zeros(len_cfs, dtype='float32')
    cfs[0] = f0
    while f0 < maxf:
        if f0 < 4:
            f0 += sigma_f
            sigma_f = 0.39 * sqrtf(f0)
        else:
            f0 *= 2**octSpace
        cfs[i] = f0
        i += 1

    if len_cfs == 1 or cfs[len_cfs - 2] < minf:
        raise ValueError(
            (f'Frequency band [{minf}, {maxf}] is too narrow, so no filters in'
             ' filterbank are placed inside. Try a wider frequency band.'))
    cfs = cfs[np.logical_and(cfs >= minf, cfs <= maxf)]

    exponent = np.concatenate(
        (np.ones((len(cfs), 1), dtype='float32'), np.log10(cfs)[:, np.newaxis]), axis=1) @ a
    sigma_fs = np.power(10, exponent)
    sds = sigma_fs * sqrtf(2)
    freqs = (np.arange(0, N//2+1)*(fs*1.0/N)).astype('float32')
    Xf = fft(x, N, axis=0).astype('complex64')

    h = np.zeros(N, dtype='complex64')
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    
    h_T = h[(slice(None), np.newaxis)]
    return Xf, freqs, cfs, N, sds, h_T

cpdef cnp.ndarray[DTYPE_t] extract_channel_wrapper(cnp.ndarray[DTYPE_C_t, ndim=1] Xf, cnp.ndarray[DTYPE_t, ndim=1] freqs, cnp.ndarray[DTYPE_t, ndim=1] cfs, int N, cnp.ndarray[DTYPE_t, ndim=1] sds, cnp.ndarray[DTYPE_C_t, ndim=2] h, DTYPE_t minf, DTYPE_t maxf):
    return extract_channel_inner(Xf, freqs, cfs, N, sds, h, minf, maxf)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[DTYPE_t] extract_channel_inner(cnp.ndarray[DTYPE_C_t, ndim=1] Xf, cnp.ndarray[DTYPE_t, ndim=1] freqs, cnp.ndarray[DTYPE_t, ndim=1] cfs, int N, cnp.ndarray[DTYPE_t, ndim=1] sds, cnp.ndarray[DTYPE_C_t, ndim=2] h, DTYPE_t minf, DTYPE_t maxf):
    cdef int n_freqs = len(freqs)
    cdef cnp.ndarray[DTYPE_t, ndim=2] k = freqs.reshape(-1, 1) - cfs.reshape(1, -1)
    cdef cnp.ndarray[DTYPE_C_t, ndim=2] H = np.zeros((N, len(cfs)), dtype='complex64')
    H[:n_freqs, :] = np.exp(-0.5 * np.divide(k, sds) ** 2).astype('complex64')
    H[n_freqs:, :] = H[1:(N+1)//2, :][::-1]
    H[0, :] = 0.
    H = np.multiply(H, h)
    cdef cnp.ndarray[DTYPE_C_t, ndim=2] hilb_channel = ifft(Xf[:, np.newaxis] * H, N, axis=0).astype('complex64')    
    cdef cnp.ndarray[BOOL_t, ndim=1, cast=True] band_locator = np.logical_and(cfs >= minf, cfs <= maxf)
    cdef cnp.ndarray[DTYPE_t, ndim=2] hilb_amp = np.abs(hilb_channel[:, band_locator])
    return hilb_amp