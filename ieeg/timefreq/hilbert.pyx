import numpy as np
from numpy.fft import fft, ifft
cimport numpy as cnp
cimport cython

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t INTP_t
ctypedef cnp.complex64_t CTYPE_t

cpdef tuple filterbank_hilbert_first_half_wrapper(cnp.ndarray x, int fs, DTYPE_t minf, DTYPE_t maxf):
    return filterbank_hilbert_first_half_inner(x, fs, minf, maxf)

cdef tuple filterbank_hilbert_first_half_inner(cnp.ndarray x, int fs, DTYPE_t minf, DTYPE_t maxf):
    cdef const DTYPE_t f0 = 0.018, octSpace = 1./7
    cdef cnp.ndarray[DTYPE_t, ndim=1] a = np.array([np.log10(0.39), 0.5])
    cdef cnp.ndarray[DTYPE_t] cfs = np.array([f0])
    cdef DTYPE_t sigma_f = 0.39 * np.sqrt(cfs[-1])
    cdef DTYPE_t cfs_end = f0
    cdef cnp.ndarray[DTYPE_t] exponent, sigma_fs, sds, freqs, Xf, h
    cdef int N

    if x.ndim == 1:
        x = x[:, np.newaxis]
    
    # determining size of cfs prior to assignment
    # could potentially get wrid of bounds check & negative indexing
    while cfs_end < maxf:
        if cfs_end < 4:
            cfs_end += sigma_f
        else:
            cfs_end *= 2**octSpace
        np.append(cfs, cfs_end)
        sigma_f = 10**(a[0]+a[1]*np.log10(cfs_end))

    if np.logical_and(cfs >= minf, cfs <= maxf).sum() == 0:
        raise ValueError(
            (f'Frequency band [{minf}, {maxf}] is too narrow, so no filters in'
             ' filterbank are placed inside. Try a wider frequency band.'))

    cfs = cfs[np.logical_and(cfs >= minf, cfs <= maxf)]

    exponent = np.concatenate(
        (np.ones((len(cfs), 1)), np.log10(cfs)[:, np.newaxis]), axis=1) @ a
    sigma_fs = 10**exponent
    sds = sigma_fs * np.sqrt(2)

    N = x.shape[0]
    freqs = np.arange(0, N//2+1)*(fs/N)

    Xf = fft(x, N, axis=0)

    h = np.zeros(N, dtype='float64')
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    
    ind = [np.newaxis] * x.ndim
    ind[0] = slice(None)
    h = h[tuple(ind)]
    return Xf, freqs, cfs, N, sds, h, x

cpdef tuple extract_channel_wrapper(cnp.ndarray Xf, cnp.ndarray freqs, cnp.ndarray cfs, int N, cnp.ndarray sds, cnp.ndarray h, cnp.ndarray x, tuple Wn):
    return extract_channel_inner(Xf, freqs, cfs, N, sds, h, x, Wn)

cdef tuple extract_channel_inner(cnp.ndarray Xf, cnp.ndarray freqs, cnp.ndarray cfs, int N, cnp.ndarray sds, cnp.ndarray h, cnp.ndarray x, tuple Wn):
    cdef double minf = Wn[0], maxf = Wn[1]
    cdef int n_freqs = len(freqs)
    cdef cnp.ndarray k = freqs.reshape(-1, 1) - cfs.reshape(1, -1)
    cdef cnp.ndarray H = np.zeros((N, len(cfs)), dtype='complex64')
    H[:n_freqs, :] = np.exp(-0.5 * np.divide(k, sds) ** 2)
    H[n_freqs:, :] = H[1:int(np.floor((N+1)/2)), :][::-1]
    H[0, :] = 0.
    H = np.multiply(H, h)
    cdef cnp.ndarray hilb_channel = ifft(Xf[:, np.newaxis] * H, N, axis=0).astype('complex64')    
    cdef cnp.ndarray band_locator = np.logical_and(cfs >= minf, cfs <= maxf)
    # phase not needed
    cdef cnp.ndarray hilb_phase = np.angle(hilb_channel[:, band_locator])
    cdef cnp.ndarray hilb_amp = np.abs(hilb_channel[:, band_locator])
    return hilb_phase, hilb_amp
    # return hilb_amp