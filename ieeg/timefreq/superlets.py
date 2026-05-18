# Time-frequency analysis with superlets
# Based on 'Time-frequency super-resolution with superlets'
# by Moca et al., 2021 Nature Communications
#
# Implementation by Harald Bârzan and Richard Eugen Ardelean
#
# Updated to use adaptive_superlet_transform from https://github.com/irhum/superlets
# Converted from JAX to Numba/NumPy for optimization
# GPU Support: Uses CuPy, Numba CUDA, and Array API for GPU acceleration

import numpy as np
from scipy.signal import fftconvolve, convolve, oaconvolve
from ieeg import Signal
import mne
from joblib import Parallel, delayed
from typing import Union
import warnings

from tqdm import tqdm
from ieeg.arrays.api import (
    array_namespace, is_cupy
)
from ieeg.timefreq import _superlets_kernels

# CuPy imports for GPU support
try:
    import cupy as cp
    import cupyx.scipy.signal as cp_signal
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_signal = None


# spread, in units of standard deviation, of the Gaussian window of the
# Morlet wavelet
MORLET_SD_SPREAD = 6

# the length, in units of standard deviation, of the actual support window of
# the Morlet
MORLET_SD_FACTOR = 2.5


# ============================================================================
# Numba-optimized adaptive superlet transform implementation
# Based on https://github.com/irhum/superlets
# ============================================================================

def _cxmorelet_batch_cpu(freqs: np.ndarray, cycles: np.ndarray,
                         sampling_freq: float, wavelets_out: np.ndarray):
    """Batch computation of Morlet wavelets (CPU).

    Thin Python wrapper around the OpenMP C kernel in
    ``ieeg.timefreq._superlets_kernels.cxmorelet_batch``. Previously
    implemented via ``@numba.njit(parallel=True)``; the C kernel is
    byte-equivalent within float64 ULP and removes the runtime numba
    dependency for this code path.

    Parameters
    ----------
    freqs : np.ndarray (float64, 1-D)
        Frequencies for the wavelets.
    cycles : np.ndarray (float64, 1-D)
        Cycle counts for each order.
    sampling_freq : float
        Sampling frequency; ``wavelets_out.shape[2]`` must equal
        ``int(sampling_freq * 2)``.
    wavelets_out : np.ndarray (complex128, shape (max_order, n_freqs, n_samples))
        Pre-allocated C-contiguous output buffer; modified in place.
    """
    _superlets_kernels.cxmorelet_batch(
        np.ascontiguousarray(freqs, dtype=np.float64),
        np.ascontiguousarray(cycles, dtype=np.float64),
        float(sampling_freq),
        wavelets_out,
    )


# =========================================================================
# GPU launcher — library-agnostic via ``__cuda_array_interface__``.
#
# The CUDA-C source in ``_superlets_cuda_kernel.cu`` is JIT-compiled by
# nvrtc at first call (via the official NVIDIA ``cuda-python`` package)
# and launched through the CUDA driver API. The launcher accepts ANY
# object that exposes ``__cuda_array_interface__`` — cupy arrays,
# torch CUDA tensors, numba device arrays — extracting the raw device
# pointers from the protocol. No dependency on cupy or torch in the
# launcher itself.
# =========================================================================
import pathlib

_CUDA_KERNEL_PATH = pathlib.Path(__file__).parent / "_superlets_cuda_kernel.cu"
_CUDA_KERNEL_CACHE = {}


def _cuda_python_available() -> bool:
    """True iff ``cuda-python`` and a working CUDA driver are importable."""
    try:
        from cuda.bindings import nvrtc, driver  # noqa: F401
        return True
    except Exception:
        return False


def _check_cuda(err, lib_name="CUDA"):
    """Raise on a non-success CUDA driver/nvrtc error."""
    # cuda-python returns enum members for both nvrtc and driver paths.
    # Both have a ``value`` of 0 on success; non-zero is an error.
    if int(err) != 0:
        raise RuntimeError(f"{lib_name} call failed with code {int(err)}")


def _compile_cxmorelet_ptx():
    """Compile the .cu source to PTX bytes (cached after first call).

    PTX is portable across CUDA contexts on a single host install — the
    driver re-JITs to SASS for the active device at module-load time.
    So we compile once globally; we still load the module per CUDA
    context (see ``_get_cuda_cxmorelet_launcher``).
    """
    if "ptx" in _CUDA_KERNEL_CACHE:
        return _CUDA_KERNEL_CACHE["ptx"]

    from cuda.bindings import nvrtc

    if not _CUDA_KERNEL_PATH.exists():
        raise RuntimeError(
            f"superlets CUDA kernel source not found at {_CUDA_KERNEL_PATH}"
        )
    source = _CUDA_KERNEL_PATH.read_text().encode("utf-8")

    err, prog = nvrtc.nvrtcCreateProgram(
        source, b"_superlets_cuda_kernel.cu", 0, [], [])
    _check_cuda(err, "nvrtc")

    opts = [b"--gpu-architecture=compute_50",
            b"--use_fast_math"]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(err) != 0:
        err_log, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(
            f"nvrtc compile failed (code {int(err)}): "
            f"{log.decode('utf-8', errors='replace')}"
        )

    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    _check_cuda(err, "nvrtc")
    ptx = b" " * ptx_size
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    _check_cuda(err, "nvrtc")

    _CUDA_KERNEL_CACHE["ptx"] = bytes(ptx)
    return _CUDA_KERNEL_CACHE["ptx"]


def _get_cuda_cxmorelet_launcher():
    """Return a callable that launches ``cxmorelet_batch_kernel``.

    The returned callable signature is::

        launcher(freqs, cycles, sampling_freq, wavelets_out,
                 n_samples, max_order, n_freqs)

    where ``freqs``/``cycles``/``wavelets_out`` may be any objects
    providing ``__cuda_array_interface__`` (cupy arrays, torch CUDA
    tensors, numba device arrays, etc.). The launcher:

      - Compiles the CUDA source to PTX exactly once.
      - Loads the PTX into a CUDA module per CUDA context (cupy and
        torch may use different contexts even on the same device; CUDA
        modules are context-specific, so we maintain a small cache).
      - Launches via ``cuLaunchKernel`` against the device pointers
        extracted from each input's ``__cuda_array_interface__``.

    Raises ``RuntimeError`` if ``cuda-python`` is not installed.
    """
    if "fn" in _CUDA_KERNEL_CACHE:
        return _CUDA_KERNEL_CACHE["fn"]

    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError as e:
        raise RuntimeError(
            "cuda-python is required for the superlets GPU path. "
            "Install with `pip install cuda-python` (alongside a working "
            "CUDA toolkit)."
        ) from e

    # Compile to PTX (once).
    ptx = _compile_cxmorelet_ptx()

    # Initialise the driver. Cheap if already done.
    err, = cuda_driver.cuInit(0)
    _check_cuda(err, "cuInit")

    # Per-context module cache: {ctx_int: (module, kernel)}. CUDA
    # modules are bound to the context they were loaded in, so cupy
    # and torch can each end up with their own entry on first use.
    module_cache = {}

    def _get_kernel_for_current_ctx():
        err, ctx = cuda_driver.cuCtxGetCurrent()
        _check_cuda(err, "cuCtxGetCurrent")
        if int(ctx) == 0:
            # No context active on this thread. Retain the primary
            # context on device 0 and make it current. cupy and torch
            # also use primary contexts, so this is interoperable.
            err, dev = cuda_driver.cuDeviceGet(0)
            _check_cuda(err, "cuDeviceGet")
            err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
            _check_cuda(err, "cuDevicePrimaryCtxRetain")
            err, = cuda_driver.cuCtxSetCurrent(ctx)
            _check_cuda(err, "cuCtxSetCurrent")

        key = int(ctx)
        if key not in module_cache:
            err, module = cuda_driver.cuModuleLoadData(ptx)
            _check_cuda(err, "cuModuleLoadData")
            err, kernel = cuda_driver.cuModuleGetFunction(
                module, b"cxmorelet_batch_kernel")
            _check_cuda(err, "cuModuleGetFunction")
            module_cache[key] = (module, kernel)
        return module_cache[key][1]

    def _devptr(arr):
        try:
            return int(arr.__cuda_array_interface__["data"][0])
        except AttributeError as exc:
            raise TypeError(
                f"object of type {type(arr).__name__} does not expose "
                f"__cuda_array_interface__"
            ) from exc

    def launcher(freqs, cycles, sampling_freq, wavelets_out,
                 n_samples, max_order, n_freqs):
        """Launch the kernel against any ``__cuda_array_interface__`` inputs."""
        kernel = _get_kernel_for_current_ctx()

        freqs_p = np.array([_devptr(freqs)], dtype=np.uint64)
        cycles_p = np.array([_devptr(cycles)], dtype=np.uint64)
        wavelets_p = np.array([_devptr(wavelets_out)], dtype=np.uint64)

        # Pack scalar args into numpy buffers; cuLaunchKernel reads each
        # arg by pointer-to-value.
        sf = np.array([float(sampling_freq)], dtype=np.float64)
        ns = np.array([int(n_samples)], dtype=np.int32)
        mo = np.array([int(max_order)], dtype=np.int32)
        nf = np.array([int(n_freqs)], dtype=np.int32)

        args = np.array(
            [freqs_p.ctypes.data,
             cycles_p.ctypes.data,
             sf.ctypes.data,
             wavelets_p.ctypes.data,
             ns.ctypes.data,
             mo.ctypes.data,
             nf.ctypes.data],
            dtype=np.uint64,
        )

        threads = 256
        total = int(max_order) * int(n_freqs)
        blocks = (total + threads - 1) // threads

        err, = cuda_driver.cuLaunchKernel(
            kernel,
            blocks, 1, 1,
            threads, 1, 1,
            0,              # shared memory
            0,              # stream (default / current)
            args.ctypes.data,
            0,              # extra
        )
        _check_cuda(err, "cuLaunchKernel")

        err, = cuda_driver.cuCtxSynchronize()
        _check_cuda(err, "cuCtxSynchronize")

    _CUDA_KERNEL_CACHE["fn"] = launcher
    _CUDA_KERNEL_CACHE["module_cache"] = module_cache  # keep alive
    return launcher


# Backwards-compat: True iff the GPU kernel can be compiled and launched.
CUDA_KERNEL_AVAILABLE = _cuda_python_available()


def _cxmorelet(freq: float, cycles: float, sampling_freq: float, xp=np):
    """
    Create a complex Morlet wavelet (Array API compatible).
    
    Parameters
    ----------
    freq : float
        Center frequency in Hz.
    cycles : float
        Number of cycles.
    sampling_freq : float
        Sampling frequency in Hz.
    xp : module, optional
        Array module (np or cp). Defaults to np.
        
    Returns
    -------
    array
        Complex Morlet wavelet, normalized by sum of absolute values.
    """
    k_sd = 5.0
    n_samples = int(sampling_freq * 2)
    t = xp.linspace(-1.0, 1.0, n_samples)
    
    bc = cycles / (k_sd * freq)
    norm = 1.0 / (bc * xp.sqrt(2.0 * xp.pi))
    gauss = xp.exp(-t**2 / (2.0 * bc**2))
    sine = xp.exp(1j * 2.0 * xp.pi * freq * t)
    
    wavelet = norm * gauss * sine
    # Normalize by sum of absolute values (as in the original repo)
    return wavelet / xp.sum(xp.abs(wavelet))


def _order_to_cycles(base_cycle: float, max_order: int, mode: str, xp=np):
    """
    Convert orders to cycle counts (CPU/Numba version).
    
    Parameters
    ----------
    base_cycle : float
        Base number of cycles.
    max_order : int
        Maximum order.
    mode : str
        "add" for additive or "mul" for multiplicative.
        
    Returns
    -------
    np.ndarray
        Array of cycle counts.
    """
    if mode == "add":
        return xp.array([base_cycle + i for i in range(max_order)])
    else:  # mode == "mul"
        return xp.array([(i + 1) * base_cycle for i in range(max_order)])


def _compute_orders_vectorized(freqs, f_min: float, f_max: float, 
                                o_min: int, o_max: int, xp=np):
    """
    Vectorized computation of adaptive orders for all frequencies (Array API compatible).
    
    Parameters
    ----------
    freqs : array
        Array of frequencies (NumPy or CuPy).
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    o_min : int
        Minimum order.
    o_max : int
        Maximum order.
    xp : module, optional
        Array module (np or cp). Defaults to np.
        
    Returns
    -------
    array
        Array of orders in the appropriate namespace.
    """
    freqs = xp.asarray(freqs)
    
    # Handle edge case where f_max == f_min
    if f_max == f_min:
        return xp.full(len(freqs), o_min, dtype=xp.int32)
    
    # Vectorized computation: o_min + round((o_max - o_min) * (f - f_min) / (f_max - f_min))
    # Compute normalized position: (freqs - f_min) / (f_max - f_min)
    normalized = (freqs - f_min) / (f_max - f_min)
    
    # Scale by order range and add minimum order
    orders_float = o_min + (o_max - o_min) * normalized
    
    # Round to nearest integer and convert to int32
    orders = xp.round(orders_float).astype(xp.int32)
    
    return orders
    

def _precompute_wavelets(freqs, cycles, sampling_freq: float, xp=np):
    """
    Pre-compute all wavelets for all (freq, cycle) combinations (Array API compatible).
    
    Uses vectorized Numba ufunc for efficient batch computation.
    
    Parameters
    ----------
    freqs : array
        Frequencies array (NumPy or CuPy).
    cycles : array
        Cycles array (NumPy or CuPy).
    sampling_freq : float
        Sampling frequency.
    xp : module, optional
        Array module (np or cp). Defaults to np.
    
    Returns
    -------
    list
        List of lists containing wavelets in the appropriate namespace.
    """
    max_order = len(cycles)
    n_freqs = len(freqs)
    n_samples = int(sampling_freq * 2)
    
    # Convert to appropriate types
    freqs_arr = xp.asarray(freqs)
    cycles_arr = xp.asarray(cycles)
    
    # For CPU arrays, use vectorized Numba function
    if not is_cupy(xp) or not CUPY_AVAILABLE:
        # Convert to NumPy for Numba
        freqs_np = np.asarray(freqs_arr, dtype=np.float64)
        cycles_np = np.asarray(cycles_arr, dtype=np.float64)
        
        # Pre-allocate output array
        wavelets_array = np.zeros((max_order, n_freqs, n_samples), dtype=np.complex128)
        
        # Compute all wavelets in parallel using Numba
        _cxmorelet_batch_cpu(freqs_np, cycles_np, sampling_freq, wavelets_array)
        
        # Convert to list of lists format
        wavelets = []
        for i_order in range(max_order):
            freq_wavelets = []
            for i_freq in range(n_freqs):
                freq_wavelets.append(wavelets_array[i_order, i_freq, :])
            wavelets.append(freq_wavelets)
    else:
        # GPU path via a library-agnostic CUDA launcher (cuda-python).
        # The launcher accepts any __cuda_array_interface__ provider, so
        # this works for cupy, torch CUDA tensors, numba device arrays —
        # whichever namespace `xp` belongs to. Output is allocated in
        # the same namespace via xp.zeros so the rest of the pipeline
        # sees the user's preferred GPU backend.
        freqs_dev  = xp.ascontiguousarray(freqs_arr,  dtype=xp.float64)
        cycles_dev = xp.ascontiguousarray(cycles_arr, dtype=xp.float64)
        wavelets_array_gpu = xp.zeros(
            (max_order, n_freqs, n_samples), dtype=xp.complex128,
        )

        launcher = _get_cuda_cxmorelet_launcher()
        launcher(
            freqs_dev, cycles_dev, float(sampling_freq),
            wavelets_array_gpu,
            int(n_samples), int(max_order), int(n_freqs),
        )

        # Build the list-of-lists shape callers expect (same as the CPU path).
        wavelets = []
        for i_order in range(max_order):
            freq_wavelets = []
            for i_freq in range(n_freqs):
                freq_wavelets.append(wavelets_array_gpu[i_order, i_freq, :])
            wavelets.append(freq_wavelets)
    return wavelets


def _is_gpu_array(arr):
    """Check if array is a GPU array (CuPy) using Array API."""
    if not CUPY_AVAILABLE:
        return False
    xp = array_namespace(arr)
    return is_cupy(xp)


def _apply_mask_and_geomean_cpu(out: np.ndarray, orders: np.ndarray,
                                eps: float) -> np.ndarray:
    """Apply masking and compute geometric mean (CPU).

    Thin Python wrapper around the OpenMP C kernel in
    ``ieeg.timefreq._superlets_kernels.apply_mask_and_geomean``.
    Previously implemented via ``@numba.njit``; the C kernel is
    byte-equivalent within float64 ULP and removes the runtime numba
    dependency for this code path.

    Parameters
    ----------
    out : np.ndarray (float64, shape (max_order, n_freqs, n_times))
        Modified in place: per-frequency, entries with ``i_order + 1 >
        orders[i_freq]`` are set to 1.0.
    orders : np.ndarray (int32/int64, 1-D shape (n_freqs,))
        Adaptive orders per frequency.
    eps : float
        Numerical-stability epsilon for the ``log(x + eps)`` step.

    Returns
    -------
    np.ndarray (float64, shape (n_freqs, n_times))
        Geometric mean over the first ``orders[i_freq]`` orders.
    """
    max_order, n_freqs, n_times = out.shape
    result = np.zeros((n_freqs, n_times), dtype=np.float64)
    # The C kernel requires int64 orders. Production paths pass int32
    # (see _compute_orders_vectorized), so cast here once per call.
    _superlets_kernels.apply_mask_and_geomean(
        np.ascontiguousarray(out, dtype=np.float64),
        np.ascontiguousarray(orders, dtype=np.int64),
        float(eps),
        result,
    )
    return result


def _apply_mask_and_geomean_gpu(out, orders, eps: float):
    """GPU version of mask + geometric mean — backend-agnostic via ``xp``.

    Works for any array namespace that the array-API-compat layer
    recognises as GPU-capable (cupy arrays, torch CUDA tensors). No
    direct cupy or torch references; the launcher and downstream ops
    all flow through ``array_namespace(out)``.

    Parameters
    ----------
    out : array (GPU)
        Input/output (mutated in place) of shape
        ``(max_order, n_freqs, n_times)``.
    orders : array (GPU)
        1-D adaptive orders per frequency.
    eps : float
        Numerical-stability epsilon.

    Returns
    -------
    array (GPU)
        Result of shape ``(n_freqs, n_times)`` in the same namespace.
    """
    xp = array_namespace(out)
    max_order, n_freqs, n_times = out.shape
    result = xp.zeros((n_freqs, n_times), dtype=xp.float64)

    # Apply mask: set values to 1.0 where (i_order + 1) > order
    for i_freq in range(n_freqs):
        order = int(orders[i_freq])
        for i_order in range(max_order):
            if (i_order + 1) > order:
                out[i_order, i_freq, :] = 1.0

    # Compute geometric mean: exp(sum(log(X + eps)) / order)
    for i_freq in range(n_freqs):
        order = int(orders[i_freq])
        if order > 0:
            X = out[:order, i_freq, :]
            log_sum = xp.sum(xp.log(X + eps), axis=0)
            result[i_freq, :] = xp.exp(log_sum / order)
        else:
            result[i_freq, :] = out[0, i_freq, :]

    return result


def _apply_mask_and_geomean(out, orders, eps: float):
    """Wrapper that dispatches to CPU or GPU version based on array type."""
    if _is_gpu_array(out):
        return _apply_mask_and_geomean_gpu(out, orders, eps)
    else:
        return _apply_mask_and_geomean_cpu(out, orders, eps)


def adaptive_superlet_transform(signal: np.ndarray, freqs: np.ndarray, 
                                sampling_freq: float, base_cycle: float,
                                min_order: int, max_order: int, 
                                eps: float = 1e-12, mode: str = "mul",
                                use_gpu: bool = False) -> np.ndarray:
    """
    Compute the adaptive superlet transform of the provided signal.
    
    This is a Numba-optimized port of the JAX implementation from
    https://github.com/irhum/superlets
    
    Parameters
    ----------
    signal : np.ndarray or cp.ndarray
        1D array containing the signal data.
    freqs : np.ndarray or cp.ndarray
        1D sorted array containing the frequencies to compute the wavelets at.
    sampling_freq : float
        Sampling frequency of the signal.
    base_cycle : float
        The number of cycles corresponding to order=1.
    min_order : int
        The minimum upper limit of orders to be used for a frequency in the 
        adaptive superlet.
    max_order : int
        The maximum upper limit of orders to be used for a frequency in the 
        adaptive superlet.
    eps : float, optional
        Epsilon value to be used for numerical stability in the geometric mean. 
        Defaults to 1e-12.
    mode : str, optional
        "add" or "mul", corresponding to the use of additive or multiplicative 
        adaptive superlets. Defaults to "mul".
    use_gpu : bool, optional
        Force GPU usage if True. If False, auto-detect from input arrays.
        Defaults to False.
        
    Returns
    -------
    np.ndarray or cp.ndarray
        2D array (Frequency x Time) representing the computed scalogram.
    """
    # Detect GPU usage from input arrays or use_gpu flag using Array API
    xp = array_namespace(signal, freqs)
    
    # Convert to appropriate array type
    signal = xp.asarray(signal, dtype=xp.float64)
    freqs = xp.asarray(freqs, dtype=xp.float64)
    
    # Compute cycles for all orders
    cycles = _order_to_cycles(base_cycle, max_order, mode, xp=xp)
    
    # Compute adaptive orders for each frequency
    f_min = float(xp.min(freqs))
    f_max = float(xp.max(freqs))
    orders = _compute_orders_vectorized(freqs, f_min, f_max, min_order, max_order, xp=xp)
    
    n_freqs = len(freqs)
    n_times = len(signal)
    
    # Pre-compute all wavelets (outside hot loop)
    # Wavelets are computed in the appropriate namespace (CPU or GPU)
    wavelets = _precompute_wavelets(freqs, cycles, sampling_freq, xp=xp)
    
    # Pre-allocate output array
    # Shape: (max_order, n_freqs, n_times) to match original JAX implementation
    out = xp.zeros((max_order, n_freqs, n_times), dtype=xp.float64)
    
    # Get convolution function - use CuPy's scipy.signal directly
    if is_cupy(xp) and CUPY_AVAILABLE:
        conv_func = cp_signal.oaconvolve
    else:
        conv_func = oaconvolve
    
    # Compute all wavelet transforms
    for i_order in range(max_order):
        for i_freq in range(n_freqs):
            wavelet = wavelets[i_order][i_freq]
            # # Transfer wavelet to GPU if needed
            # if is_gpu and not _is_gpu_array(wavelet):
            #     wavelet = cp.asarray(wavelet)
            # Use FFT-based convolution (GPU-accelerated if using CuPy)
            conv_result = conv_func(signal, wavelet, mode='same')
            # Compute power (multiply by sqrt(2)^2 = 2 as in original)
            power = 2.0 * xp.abs(conv_result) ** 2
            out[i_order, i_freq, :] = power
    
    # Apply mask and compute geometric mean
    result = _apply_mask_and_geomean(out, orders, eps)
    
    return result


def computeWaveletSize(fc, nc, fs):
    """
    Compute the size in samples of a morlet wavelet.

    Parameters
    ----------
    fc : float
        Center frequency in Hz.
    nc : float
        Number of cycles.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    int
        Size of the wavelet in samples.
    """
    sd = (nc / 2) * (1 / np.abs(fc)) / MORLET_SD_FACTOR
    return int(2 * np.floor(np.round(sd * fs * MORLET_SD_SPREAD) / 2) + 1)


def computeLongestWaveletSize(fs, foi, c1, ord):
    """
    Estimates the size of the longest wavelet.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    foi : array_like
        Frequencies of interest in Hz.
    c1 : float
        Base number of cycles parameter.
    ord : tuple or list
        The order or order range for superlets.

    Returns
    -------
    int
        Size of the longest wavelet in samples.
    """
    # make order parameter
    if len(ord) == 1:
        ord = (ord, ord)
    # orders = np.linspace(start=ord[0], stop=ord[1], num=len(foi))
    orders = np.interp(foi, [min(foi), max(foi)], ord)
    # create wavelets
    max = 0
    for iFreq in range(len(foi)):
        centerFreq = foi[iFreq]
        nWavelets = int(np.ceil(orders[iFreq]))

        for iWave in range(nWavelets):
            # create morlet wavelet
            wlen = computeWaveletSize(centerFreq, fs, (iWave + 1) * c1)
            if wlen > max:
                max = wlen

    return max


def gausswin(size, alpha):
    """
    Create a Gaussian window.

    Parameters
    ----------
    size : int
        Size of the window in samples.
    alpha : float
        Parameter controlling the width of the window.

    Returns
    -------
    ndarray
        Gaussian window of specified size.
    """
    halfSize = int(np.floor(size / 2))
    idiv = alpha / halfSize

    t = (np.arange(size, dtype=np.float64) - halfSize) * idiv
    window = np.exp(-(t * t) * 0.5)

    return window


def morlet(fc, nc, fs):
    """
    Create an analytic Morlet wavelet.

    Parameters
    ----------
    fc : float
        Center frequency in Hz.
    nc : float
        Number of cycles.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    ndarray
        Complex Morlet wavelet.
    """
    size = computeWaveletSize(fc, nc, fs)
    half = int(np.floor(size / 2))
    gauss = gausswin(size, MORLET_SD_SPREAD / 2)
    igsum = 1 / gauss.sum()
    ifs = 1 / fs

    t = (np.arange(size, dtype=np.float64) - half) * ifs
    wavelet = gauss * np.exp(2 * np.pi * fc * t * 1j) * igsum

    return wavelet


def fractional(x):
    """
    Get the fractional part of the scalar value x.

    Parameters
    ----------
    x : float
        Input scalar value.

    Returns
    -------
    float
        Fractional part of x.
    """
    return x - int(x)


class SuperletTransform:
    """
    Class used to compute the Superlet Transform of input data.

    This class implements the superlet transform algorithm for time-frequency
     analysis as described in Moca et al., 2021.
     
    .. deprecated:: 
        This class is deprecated. Use `adaptive_superlet_transform` directly
        or the `superlets()` and `superlet_tfr()` functions instead.
    """

    def __init__(self,
                 inputSize,
                 samplingRate,
                 baseCycles,
                 superletOrders,
                 frequencyRange=None,
                 frequencyBins=None,
                 frequencies=None):
        """
        Initialize the superlet transform.

        Parameters
        ----------
        inputSize : int
            Size of the input in samples.
        samplingRate : float
            The sampling rate of the input signal in Hz.
        baseCycles : float
            Number of cycles of the smallest wavelet (c1 in the paper).
        superletOrders : tuple
            A tuple containing the range of superlet orders, linearly
             distributed along frequencyRange.
        frequencyRange : tuple
            Tuple of ascending frequency points, in Hz.
        frequencyBins : int
            Number of frequency bins to sample in the interval frequencyRange.
        frequencies : array_like, optional
            Specific list of frequencies - can be provided instead of
             frequencyRange (it is ignored in this case).
        """
        warnings.warn(
            "SuperletTransform is deprecated. Use adaptive_superlet_transform "
            "directly or the superlets() and superlet_tfr() functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # clear to reinit
        self.clear()

        # initialize containers
        if frequencies is not None:
            frequencyBins = len(frequencies)
            frequencyRange = [np.min(frequencies), np.max(frequencies)]
            self.frequencies = frequencies
        else:
            self.frequencies = np.linspace(start=frequencyRange[0],
                                           stop=frequencyRange[1],
                                           num=frequencyBins)

        self.inputSize = inputSize
        self.orders = np.interp(self.frequencies, frequencyRange,
                                superletOrders)
        # self.orders = np.linspace(start=superletOrders[0],
        #                           stop=superletOrders[1], num=frequencyBins)
        self.convBuffer = np.zeros(inputSize, dtype=np.complex128)
        self.poolBuffer = np.zeros(inputSize, dtype=np.float64)
        self.superlets = []

        # create wavelets
        for iFreq in range(frequencyBins):
            centerFreq = self.frequencies[iFreq]
            nWavelets = int(np.ceil(self.orders[iFreq]))

            self.superlets.append([])
            for iWave in range(nWavelets):
                # create morlet wavelet
                self.superlets[iFreq].append(
                    morlet(centerFreq, (iWave + 1) * baseCycles, samplingRate))

    def __del__(self):
        """
        Destructor.

        Cleans up resources when the object is deleted.
        """
        self.clear()

    def clear(self):
        """
        Clear the transform.

        Resets all internal variables to None, freeing memory.
        """
        # fields
        self.inputSize = None
        self.superlets = None
        self.poolBuffer = None
        self.convBuffer = None
        self.frequencies = None
        self.orders = None

    def longestWaveletSize(self):
        """
        Return the size of the longest wavelet.

        Returns
        -------
        int
            Size of the longest wavelet in samples.
        """
        max = 0
        for s in self.superlets:
            for w in s:
                if w.shape[0] > max:
                    max = w.shape[0]
        return max

    def validTimeRegion(self):
        """
        Compute the start and end of the valid spectrum region.

        Returns
        -------
        tuple
            A tuple containing:

            - start : int
                The start of the valid time region.
            - end : int
                The end of the valid time region.
        """
        pad = self.longestWaveletSize() // 2
        start = self.inputSize + pad
        end = self.inputSize - pad
        return start, end

    def transform(self, inputData):
        """
        Apply the transform to a buffer or list of buffers.

        Parameters
        ----------
        inputData : ndarray
            An NDarray of input data. Can be a single buffer or a list of
             buffers.

        Returns
        -------
        ndarray
            The transformed data as a time-frequency representation.

        Raises
        ------
        Exception
            If input data size doesn't match the defined input size for this
             transform.
        """

        # compute number of arrays to transform
        if len(inputData.shape) == 1:
            if inputData.shape[0] != self.inputSize:
                raise ValueError("Input data must meet the defined input size"
                                 " for this transform.")

            result = np.zeros((self.inputSize, len(self.frequencies)),
                              dtype=np.float64)
            self.transformOne(inputData, result)
            return result

        else:
            n = int(np.sum(inputData.shape[0:len(inputData.shape) - 1]))
            insize = int(inputData.shape[len(inputData.shape) - 1])

            if insize != self.inputSize:
                raise ValueError("Input data must meet the defined input size"
                                 " for this transform.")

            # reshape to data list
            datalist = np.reshape(inputData, (n, insize), 'C')
            result = np.zeros((len(self.frequencies), self.inputSize),
                              dtype=np.float64)

            for i in range(0, n):
                self.transformOne(datalist[i, :], result)

            return result / n

    def transformOne(self, inputData, accumulator):
        """
        Apply the superlet transform on a single data buffer.

        Parameters
        ----------
        inputData : ndarray
            A 1xInputSize array containing the signal to be transformed.
        accumulator : ndarray
            A spectrum to accumulate the resulting superlet transform.

        Notes
        -----
        This method modifies the accumulator array in-place.
        """
        accumulator.resize((len(self.frequencies), self.inputSize))

        for iFreq in range(len(self.frequencies)):

            # init pooling buffer
            self.poolBuffer.fill(1)

            if len(self.superlets[iFreq]) > 1:

                # superlet
                nWavelets = int(np.floor(self.orders[iFreq]))
                rfactor = 1.0 / nWavelets

                for iWave in range(nWavelets):
                    self.convBuffer = fftconvolve(inputData,
                                                  self.superlets[iFreq][iWave],
                                                  "same")
                    self.poolBuffer *= 2 * np.abs(self.convBuffer) ** 2

                if fractional(self.orders[iFreq]) != 0 and len(
                        self.superlets[iFreq]) == nWavelets + 1:
                    # apply the fractional wavelet
                    exponent = self.orders[iFreq] - nWavelets
                    rfactor = 1 / (nWavelets + exponent)

                    self.convBuffer = fftconvolve(inputData,
                                                  self.superlets[iFreq][
                                                      nWavelets], "same")
                    self.poolBuffer *= (2 * np.abs(
                        self.convBuffer) ** 2) ** exponent

                # perform geometric mean
                accumulator[iFreq, :] += self.poolBuffer ** rfactor

            else:
                # wavelet transform
                accumulator[iFreq, :] += (2 * np.abs(
                    fftconvolve(inputData, self.superlets[iFreq][0],
                                "same")) ** 2).astype(np.float64)


def cropSpectrum(spectrum, paddingSize):
    """
    Remove paddingSize samples at both ends of the spectrum.

    Parameters
    ----------
    spectrum : ndarray
        A 2D numpy array representing the time-frequency spectrum.
    paddingSize : int
        Number of samples to remove - equals to longestWaveletSize() / 2
        of the computing SuperletTransform object.

    Returns
    -------
    ndarray
        The spectrum with the padding removed.
    """
    return spectrum[:, paddingSize:(spectrum.shape[1] - paddingSize)]


# main superlet function
def superlets(data,
              fs,
              foi,
              c1,
              ord):
    """
    Perform fractional adaptive superlet transform (FASLT) on a list of trials.

    Parameters
    ----------
    data : ndarray or cupy.ndarray
        A numpy or cupy array of data. The rightmost dimension of the data is the trial
         size. The result will be the average over all the spectra.
        If a CuPy array is provided, GPU acceleration will be used automatically.
    fs : float
        The sampling rate in Hz.
    foi : array_like
        List of frequencies of interest.
    c1 : float
        Base number of cycles parameter.
    ord : tuple or list
        The order (for SLT) or order range (for FASLT), spanned across the
        frequencies of interest.

    Returns
    -------
    ndarray or cupy.ndarray
        A matrix containing the average superlet spectrum (Frequency x Time).
        Returns CuPy array if input was CuPy array, otherwise NumPy array.

    Notes
    -----
    This is the main function for computing the superlet transform.
    Uses adaptive_superlet_transform internally for optimized computation.
    Automatically detects CuPy arrays and uses GPU acceleration.
    """
    # Detect if input is CuPy array using Array API
    xp = array_namespace(data, foi)
    is_gpu = is_cupy(xp)
    
    data = xp.asarray(data)
    foi = xp.asarray(foi)
    
    # determine buffer size
    bufferSize = data.shape[-1]

    # make order parameter
    if len(ord) == 1:
        ord = (ord, ord)
    
    min_order, max_order = int(ord[0]), int(ord[1])

    # Reshape data to (n_trials, n_samples)
    if data.ndim == 1:
        data = data[None, :]
    
    n_trials = int(xp.prod(xp.asarray(data.shape[:-1])))
    data_reshaped = data.reshape(n_trials, bufferSize)
    
    # Compute superlet transform for each trial
    results = []
    # Show progress bar for GPU processing or when processing many trials
    if is_gpu or n_trials > 1:
        iterator = tqdm(data_reshaped, desc="Superlet transform", total=n_trials)
    else:
        iterator = data_reshaped
    
    for trial_data in iterator:
        result = adaptive_superlet_transform(
            signal=trial_data,
            freqs=foi,
            sampling_freq=fs,
            base_cycle=c1,
            min_order=min_order,
            max_order=max_order,
            mode="mul",
            use_gpu=is_gpu
        )
        results.append(result)
    
    # Average over trials
    if is_gpu and CUPY_AVAILABLE:
        result = cp.mean(cp.stack(results), axis=0)
    else:
        result = np.mean(np.stack(results), axis=0)
    
    return result


def superlet_tfr(inst: Signal,
                 foi: list[float],
                 c1: float,
                 ord: tuple[int, int] = (1, 1),
                 decim: int = 1,
                 n_jobs: Union[int, str] = 1) -> Signal:
    """
    Compute the superlet time-frequency representation of the input signal.

    Parameters
    ----------
    inst : Signal
        The input signal (e.g., Raw, Epochs, Evoked).
    foi : list[float]
        List of frequencies of interest.
    c1 : float
        Base number of cycles parameter.
    ord : tuple[int, int], optional
        The order (for SLT) or order range (for FASLT), spanned across the
        frequencies of interest. Default is (1, 1).
    decim : int, optional
        Decimation factor for the output. Default is 1.
    n_jobs : int or str, optional
        Number of parallel jobs. If "cuda" or "gpu", uses GPU acceleration.
        Default is 1.

    Returns
    -------
    Signal
        The time-frequency representation of the input signal.
    """
    # Check for GPU usage
    use_gpu = (n_jobs == "cuda" or n_jobs == "gpu") and CUPY_AVAILABLE
    
    if (n_jobs == "cuda" or n_jobs == "gpu") and not CUPY_AVAILABLE:
        warnings.warn(
            "CuPy not available. Falling back to CPU. Install cupy for GPU support.",
            UserWarning
        )
        use_gpu = False
    
    # check if the input is a Raw or Epochs object
    times = inst.times[::decim]
    sfreq = inst.info['sfreq']
    if isinstance(inst, (mne.io.BaseRaw | mne.Epochs)):
        data = inst.get_data()

    # check if the input is an Evoked object
    elif isinstance(inst, mne.Evoked):
        data = inst.data[np.newaxis, :]

    else:
        raise ValueError("Input must be a Raw, Epochs or Evoked object.")

    # compute superlet transform
    foi = np.asarray(foi)
    
    # make order parameter
    if len(ord) == 1:
        ord = (ord, ord)
    
    min_order, max_order = int(ord[0]), int(ord[1])
    
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install cupy for GPU support.")
        # Transfer data to GPU
        data_gpu = cp.asarray(data)
        foi_gpu = cp.asarray(foi)
        out = cp.zeros(data.shape[:-1] + (len(foi), len(times)),
                      dtype=data.dtype)
        
        # Process all trials on GPU with progress bar
        indices = list(np.ndindex(data.shape[:-1]))
        n_trials = len(indices)
        
        for idx in tqdm(indices, desc="GPU superlet transform", total=n_trials):
            trial_data = data_gpu[idx]
            tout = adaptive_superlet_transform(
                signal=trial_data,
                freqs=foi_gpu,
                sampling_freq=sfreq,
                base_cycle=c1,
                min_order=min_order,
                max_order=max_order,
                mode="mul",
                use_gpu=True
            )
            # Apply decimation
            out[idx] = tout[:, ::decim]
        
        # Transfer result back to CPU for MNE
        out = cp.asnumpy(out)
    else:
        out = np.zeros(data.shape[:-1] + (len(foi), len(times)),
                       dtype=data.dtype)

        def _apply_transform(idx):
            trial_data = data[idx]
            tout = adaptive_superlet_transform(
                signal=trial_data,
                freqs=foi,
                sampling_freq=sfreq,
                base_cycle=c1,
                min_order=min_order,
                max_order=max_order,
                mode="mul"
            )
            # Transpose to match expected shape (n_times, n_freqs) -> (n_freqs, n_times)
            # and apply decimation
            return tout[:, ::decim], idx

        # apply transform in parallel
        par = Parallel(n_jobs=n_jobs, return_as='generator_unordered', verbose=10)(
            delayed(_apply_transform)(i) for i in np.ndindex(data.shape[:-1]))
        for o, i in par:
            # apply transform
            out[i] = o

    # create TFR object and return it
    tfr = mne.time_frequency.EpochsTFRArray(inst.info, out, times, foi,
                                            events=inst.events,
                                            event_id=inst.event_id)

    return tfr


# def hyperlet_wavelet(freq, cycles, fs, t):
#     """
#     Generate a hyperlet wavelet (enhanced Morlet with hypergeometric properties).
#     Based on hyperlet transform principles for super-resolution.
#     Uses pywavelets Morlet wavelet ('morl') as the base.
#
#     Parameters
#     ----------
#     freq : float
#         Center frequency in Hz
#     cycles : float
#         Number of cycles
#     fs : float
#         Sampling frequency
#     t : np.ndarray
#         Time array
#
#     Returns
#     -------
#     wavelet : np.ndarray
#         Complex hyperlet wavelet
#     """
#     # Convert frequency to scale for pywavelets
#     # For Morlet wavelet in pywavelets, scale relates to frequency as:
#     # scale = (central_freq * sampling_rate) / frequency
#     # pywavelets Morlet has central frequency ~0.849
#     central_freq = 0.849  # Central frequency of pywavelets Morlet
#     scale = (central_freq * fs) / freq
#
#     # Adjust scale based on number of cycles
#     # More cycles = larger scale (narrower frequency band, better frequency resolution)
#     scale = scale * (cycles / 5.0)  # Normalize to standard 5 cycles
#
#     # Generate base Morlet wavelet using pywavelets
#     # pywavelets.wavefun generates the wavelet function
#     # For Morlet, we need to generate it at the specified scale
#     n_samples = len(t)
#     if n_samples == 0:
#         return np.array([], dtype=complex)
#
#     # Create time array centered at zero
#     t_centered = t - t[n_samples // 2] if n_samples > 0 else t
#
#     # # Generate Morlet wavelet using pywavelets
#     # # pywavelets doesn't have a direct function to generate Morlet at arbitrary scale,
#     # # so we'll use the wavelet function and scale it
#     # try:
#     # Use pywt to get Morlet wavelet function
#     # For continuous wavelets, we can use the wavelet function directly
#     # Generate a test signal and use cwt to get the wavelet shape
#     test_signal = np.zeros(n_samples)
#     test_signal[n_samples // 2] = 1.0  # Impulse at center
#
#     # Use CWT with single scale to get the Morlet wavelet shape
#     scales_array = np.array([scale])
#     coeffs, _ = pywt.cwt(test_signal, scales_array, 'morl', method='conv')
#     morlet_base = coeffs[0, :]
#
#     # Convert to complex (pywavelets Morlet is real, so we create analytic version)
#     # Use Hilbert transform approach or create directly
#     # For hyperlet, we enhance with hypergeometric properties
#     sigma = cycles / (2 * np.pi * freq)
#     gaussian_envelope = np.exp(-t_centered ** 2 / (2 * sigma ** 2))
#
#     # Hypergeometric enhancement factor (improves frequency resolution)
#     enhancement = 1 + 0.1 * (t_centered / sigma) ** 2 * np.exp(
#         -(t_centered / (2 * sigma)) ** 2)
#
#     # Create complex Morlet with enhancement
#     carrier = np.exp(1j * 2 * np.pi * freq * t_centered)
#     wavelet = enhancement * gaussian_envelope * carrier
#
#     # Blend with pywavelets Morlet for better properties
#     morlet_norm = np.max(np.abs(morlet_base)) if np.max(
#         np.abs(morlet_base)) > 0 else 1.0
#     wavelet = 0.7 * wavelet + 0.3 * (morlet_base / morlet_norm) * np.abs(
#         wavelet)
#     #
#     # except Exception:
#     # Fallback: create Morlet directly if pywavelets fails
#     # sigma = cycles / (2 * np.pi * freq)
#     # gaussian = np.exp(-t_centered**2 / (2 * sigma**2))
#     # enhancement = 1 + 0.1 * (t_centered / sigma)**2 * np.exp(-(t_centered / (2*sigma))**2)
#     # carrier = np.exp(1j * 2 * np.pi * freq * t_centered)
#     # wavelet = enhancement * gaussian * carrier
#
#     # Normalize
#     norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
#     if norm > 0:
#         wavelet = wavelet / norm
#
#     return wavelet
#
#
# def hyperlet_transform(signal, freqs, fs, base_cycles=3, order_min=1,
#                        order_max=10):
#     """
#     Hyperlet Transform - Enhanced superlet transform with hypergeometric wavelets.
#     Based on: https://www.sciencedirect.com/science/article/pii/S0031320325012610
#
#     The hyperlet transform extends superlets by using enhanced wavelets with
#     hypergeometric properties for improved time-frequency resolution.
#
#     Parameters
#     ----------
#     signal : np.ndarray
#         1D signal array
#     freqs : np.ndarray
#         Frequencies of interest (Hz)
#     fs : float
#         Sampling frequency
#     base_cycles : float
#         Base number of cycles
#     order_min : int
#         Minimum order
#     order_max : int
#         Maximum order
#
#     Returns
#     -------
#     result : np.ndarray
#         2D time-frequency representation (freqs x time)
#     """
#     n_samples = len(signal)
#     n_freqs = len(freqs)
#
#     # Compute adaptive orders for each frequency (similar to adaptive superlets)
#     f_min = freqs.min()
#     f_max = freqs.max()
#     if f_max == f_min:
#         orders = np.full(n_freqs, order_min, dtype=int)
#     else:
#         normalized = (freqs - f_min) / (f_max - f_min)
#         orders_float = order_min + (order_max - order_min) * normalized
#         orders = np.round(orders_float).astype(int)
#
#     # Pre-allocate output array
#     # Shape: (max_order, n_freqs, n_samples)
#     max_order = order_max
#     out = np.zeros((max_order, n_freqs, n_samples), dtype=np.complex128)
#
#     # Time array for wavelets
#     # Use longer time window for hyperlets (better frequency resolution)
#     max_wavelet_length = int(2 * fs)  # 2 seconds max
#     t_wavelet = np.arange(-max_wavelet_length // 2,
#                           max_wavelet_length // 2) / fs
#
#     # Compute hyperlet transform for each order and frequency
#     for i_order in range(max_order):
#         order = i_order + 1
#         cycles = base_cycles * order
#
#         for i_freq, freq in enumerate(freqs):
#             # Generate hyperlet wavelet
#             # Crop time array to reasonable length based on frequency
#             wavelet_length = min(max_wavelet_length,
#                                  int(6 * cycles / freq * fs))
#             if wavelet_length % 2 == 0:
#                 wavelet_length += 1
#             t_local = np.arange(-wavelet_length // 2,
#                                 wavelet_length // 2 + 1) / fs
#
#             wavelet = hyperlet_wavelet(freq, cycles, fs, t_local)
#
#             # Convolve signal with wavelet
#             conv_result = oaconvolve(signal, wavelet, mode='same')
#
#             # Compute power
#             power = 2.0 * np.abs(conv_result) ** 2
#             out[i_order, i_freq, :] = power
#
#     # Apply adaptive masking and geometric mean (similar to superlets)
#     result = np.zeros((n_freqs, n_samples), dtype=np.float64)
#     eps = 1e-12
#
#     for i_freq in range(n_freqs):
#         order = orders[i_freq]
#
#         # Apply mask: set values to 1.0 where (i_order + 1) > order
#         for i_order in range(max_order):
#             if (i_order + 1) > order:
#                 out[i_order, i_freq, :] = 1.0
#
#         if order > 0:
#             # Compute geometric mean: exp(sum(log(X + eps)) / order)
#             log_sum = np.sum(np.log(out[:order, i_freq, :] + eps), axis=0)
#             result[i_freq, :] = np.exp(log_sum / order)
#         else:
#             result[i_freq, :] = out[0, i_freq, :]
#
#     return result
