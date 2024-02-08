import numpy as np
from numpy.fft import fft, ifft
cimport numpy as cnp
cimport cython

cnp.import_array()

cpdef tuple filterbank_hilbert_first_half_wrapper(cnp.ndarray x, int fs, list Wn=[70, 150]):
    return filterbank_hilbert_first_half_inner(x, fs, Wn)

cdef tuple filterbank_hilbert_first_half_inner(cnp.ndarray x, int fs, list Wn):
    cdef double minf = Wn[0], maxf = Wn[1]

    if minf >= maxf:
        raise ValueError(
            (f'Upper bound of frequency range must be greater than lower bound'
             f', but got lower bound of {minf} and upper bound of {maxf}'))

    if x.ndim != 1 and x.ndim != 2:
        raise ValueError(
            ('Input signal must be 1- or 2-dimensional but got input with'
             f'shape {x.shape}'))

    if x.ndim == 1:
        x = x[:, np.newaxis]
    
    cdef cnp.ndarray[cnp.double_t] a = np.array([np.log10(0.39), 0.5])
    cdef double f0 = 0.018, octSpace = 1./7
    cdef list cfs = [f0]
    cdef double sigma_f = 0.39 * np.sqrt(cfs[-1])

    while cfs[-1] < maxf:
        if cfs[-1] < 4:
            cfs.append(cfs[-1]+sigma_f)
        else:
            cfs.append(cfs[-1]*(2**(octSpace)))
        sigma_f = 10**(a[0]+a[1]*np.log10(cfs[-1]))

    cdef cnp.ndarray[cnp.double_t] cfs_nparray = np.array(cfs)
    if np.logical_and(cfs_nparray >= minf, cfs_nparray <= maxf).sum() == 0:
        raise ValueError(
            (f'Frequency band [{minf}, {maxf}] is too narrow, so no filters in'
             ' filterbank are placed inside. Try a wider frequency band.'))

    cfs_nparray = cfs_nparray[np.logical_and(cfs >= minf, cfs <= maxf)]

    cdef cnp.ndarray[cnp.double_t] exponent = np.concatenate(
        (np.ones((len(cfs), 1)), np.log10(cfs)[:, np.newaxis]), axis=1) @ a
    cdef cnp.ndarray[cnp.double_t] sigma_fs = 10**exponent
    cdef cnp.ndarray[cnp.double_t] sds = sigma_fs * np.sqrt(2)

    cdef int N = x.shape[0]
    cdef cnp.ndarray[cnp.double_t] freqs = np.arange(0, N//2+1)*(fs/N)

    x = x.astype('float32')
    cdef cnp.ndarray[cnp.double_t] Xf = fft(x, N, axis=0)

    cdef cnp.ndarray[cnp.double_t] h = np.zeros(N, dtype=Xf.dtype)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
    
    if x.ndim > 1:
        cdef list ind = [np.newaxis] * x.ndim
        ind[0] = slice(None)
        h = h[tuple(ind)]

    cpdef tuple extract_channel(cnp.ndarray Xf):
        return extract_channel_inner(Xf)

    cdef tuple extract_channel_inner(cnp.ndarray Xf):
        cdef int n_freqs = len(freqs)
        cdef cnp.ndarray[cnp.double_t] k = freqs.reshape(-1, 1) - cfs.reshape(1, -1)
        cdef cnp.ndarray[cnp.double_t] H = np.zeros((N, len(cfs)), dtype='float32')
        H[:n_freqs, :] = np.exp(-0.5 * np.divide(k, sds) ** 2)
        H[n_freqs:, :] = H[1:int(np.floor((N+1)/2)), :][::-1]
        H[0, :] = 0.
        H = np.multiply(H, h)
        cdef cnp.ndarray[cnp.double_t] hilb_channel = ifft(Xf[:, np.newaxis] * H, N, axis=0).astype('complex64')
        cdef cnp.ndarray[cnp.double_t] hilb_phase = np.zeros((x.shape[0], len(cfs)), dtype='float32')
        cdef cnp.ndarray[cnp.double_t] hilb_amp = np.zeros((x.shape[0], len(cfs)), dtype='float32')
    
        cdef cnp.ndarray band_locator = np.logical_and(cfs >= minf, cfs <= maxf)
        cdef cnp.ndarray[cnp.double_t] hilb_phase = np.angle(hilb_channel[:, band_locator])
        cdef cnp.ndarray[cnp.double_t] hilb_amp = np.abs(hilb_channel[:, band_locator])

        return hilb_phase, hilb_amp

    return Xf, x, cfs, extract_channel

