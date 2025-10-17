import numpy as np
from ieeg.calc._fast.ufuncs import mean_diff as _md, t_test as _ttest
from ieeg.calc._fast.ufuncs import meanvar as _meanvar
from ieeg.calc._fast.mixup import mixupnd as cmixup, normnd as cnorm
from ieeg.calc._fast.permgt import permgtnd as permgt
from ieeg.arrays.api import array_namespace, is_numpy, is_torch, Array, is_cupy
from scipy.stats import rankdata
from functools import partial

__all__ = ["mean_diff", "mixup", "mixup2", "permgt", "norm", "concatenate_arrays",
           "ttest", "brunnermunzel", "meanvar"]

# Cached CuPy kernels for meanvar (GPU backend)
_CUPY_MEANVAR_KERNELS = {}


def _get_cupy_meanvar_kernel(xp, dtype):
    key = ('meanvar', str(dtype))
    kern = _CUPY_MEANVAR_KERNELS.get(key)
    if kern is not None:
        return kern
    if dtype == xp.float64:
        code = r"""
        #define _CUDA_NAN_D __longlong_as_double(0x7ff8000000000000ULL)
        extern "C" __global__ void meanvar_f64(const double* __restrict__ x,
                                                const long long outer_size,
                                                const int last_dim,
                                                const double ddof,
                                                double* __restrict__ out_mean,
                                                double* __restrict__ out_var)
        {
            const long long outer_idx = blockIdx.x;
            if (outer_idx >= outer_size) return;
            const int tid = threadIdx.x;
            const int nthreads = blockDim.x;
            const double* row = x + outer_idx * (long long)last_dim;
            double sum = 0.0;
            double sumsq = 0.0;
            int count = 0;
            for (int j = tid; j < last_dim; j += nthreads) {
                double v = row[j];
                if (v == v) { // not NaN
                    sum += v;
                    sumsq += v * v;
                    count += 1;
                }
            }
            extern __shared__ unsigned char smem_raw[];
            double* s_sum = (double*)smem_raw;
            double* s_sumsq = (double*)(s_sum + nthreads);
            int* s_count = (int*)(s_sumsq + nthreads);
            s_sum[tid] = sum;
            s_sumsq[tid] = sumsq;
            s_count[tid] = count;
            __syncthreads();
            for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    s_sum[tid] += s_sum[tid + offset];
                    s_sumsq[tid] += s_sumsq[tid + offset];
                    s_count[tid] += s_count[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int n = s_count[0];
                if (n == 0) {
                    out_mean[outer_idx] = _CUDA_NAN_D;
                    out_var[outer_idx] = _CUDA_NAN_D;
                } else {
                    const double mean = s_sum[0] / (double)n;
                    const double denom = (double)n - ddof;
                    if (denom <= 0.0) {
                        out_mean[outer_idx] = mean;
                        out_var[outer_idx] = _CUDA_NAN_D;
                    } else {
                        const double var_num = s_sumsq[0] - (s_sum[0] * s_sum[0]) / (double)n;
                        out_mean[outer_idx] = mean;
                        out_var[outer_idx] = var_num / denom;
                    }
                }
            }
        }
        """
        kern = xp.RawKernel(code, 'meanvar_f64')
        _CUPY_MEANVAR_KERNELS[key] = kern
        return kern
    else:
        code = r"""
        #define _CUDA_NAN_F __int_as_float(0x7fffffff)
        extern "C" __global__ void meanvar_f32(const float* __restrict__ x,
                                                const long long outer_size,
                                                const int last_dim,
                                                const float ddof,
                                                float* __restrict__ out_mean,
                                                float* __restrict__ out_var)
        {
            const long long outer_idx = blockIdx.x;
            if (outer_idx >= outer_size) return;
            const int tid = threadIdx.x;
            const int nthreads = blockDim.x;
            const float* row = x + outer_idx * (long long)last_dim;
            float sum = 0.0f;
            float sumsq = 0.0f;
            int count = 0;
            for (int j = tid; j < last_dim; j += nthreads) {
                float v = row[j];
                if (v == v) { // not NaN
                    sum += v;
                    sumsq += v * v;
                    count += 1;
                }
            }
            extern __shared__ unsigned char smem_raw[];
            float* s_sum = (float*)smem_raw;
            float* s_sumsq = (float*)(s_sum + nthreads);
            int* s_count = (int*)(s_sumsq + nthreads);
            s_sum[tid] = sum;
            s_sumsq[tid] = sumsq;
            s_count[tid] = count;
            __syncthreads();
            for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    s_sum[tid] += s_sum[tid + offset];
                    s_sumsq[tid] += s_sumsq[tid + offset];
                    s_count[tid] += s_count[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int n = s_count[0];
                if (n == 0) {
                    out_mean[outer_idx] = _CUDA_NAN_F;
                    out_var[outer_idx] = _CUDA_NAN_F;
                } else {
                    const float mean = s_sum[0] / (float)n;
                    const float denom = (float)n - ddof;
                    if (denom <= 0.0f) {
                        out_mean[outer_idx] = mean;
                        out_var[outer_idx] = _CUDA_NAN_F;
                    } else {
                        const float var_num = s_sumsq[0] - (s_sum[0] * s_sum[0]) / (float)n;
                        out_mean[outer_idx] = mean;
                        out_var[outer_idx] = var_num / denom;
                    }
                }
            }
        }
        """
        kern = xp.RawKernel(code, 'meanvar_f32')
        _CUPY_MEANVAR_KERNELS[key] = kern
        return kern


def _get_cupy_ttest_kernel(xp, dtype):
    key = ('ttest', str(dtype))
    kern = _CUPY_MEANVAR_KERNELS.get(key)
    if kern is not None:
        return kern
    if dtype == xp.float64:
        code = r"""
        #define _CUDA_NAN_D __longlong_as_double(0x7ff8000000000000ULL)
        extern "C" __global__ void ttest_f64(const double* __restrict__ a,
                                              const double* __restrict__ b,
                                              const long long outer_size,
                                              const int last_dim,
                                              double* __restrict__ out)
        {
            const long long outer_idx = blockIdx.x;
            if (outer_idx >= outer_size) return;
            const int tid = threadIdx.x;
            const int nthreads = blockDim.x;
            const double* row_a = a + outer_idx * (long long)last_dim;
            const double* row_b = b + outer_idx * (long long)last_dim;
            double sum_a = 0.0, sumsq_a = 0.0;
            double sum_b = 0.0, sumsq_b = 0.0;
            int cnt_a = 0, cnt_b = 0;
            for (int j = tid; j < last_dim; j += nthreads) {
                double va = row_a[j];
                if (va == va) { sum_a += va; sumsq_a += va * va; cnt_a += 1; }
                double vb = row_b[j];
                if (vb == vb) { sum_b += vb; sumsq_b += vb * vb; cnt_b += 1; }
            }
            extern __shared__ unsigned char smem_raw[];
            double* s_sum_a = (double*)smem_raw;
            double* s_sumsq_a = (double*)(s_sum_a + nthreads);
            int* s_cnt_a = (int*)(s_sumsq_a + nthreads);
            double* s_sum_b = (double*)(s_cnt_a + nthreads);
            double* s_sumsq_b = (double*)(s_sum_b + nthreads);
            int* s_cnt_b = (int*)(s_sumsq_b + nthreads);
            s_sum_a[tid] = sum_a; s_sumsq_a[tid] = sumsq_a; s_cnt_a[tid] = cnt_a;
            s_sum_b[tid] = sum_b; s_sumsq_b[tid] = sumsq_b; s_cnt_b[tid] = cnt_b;
            __syncthreads();
            for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    s_sum_a[tid] += s_sum_a[tid + offset];
                    s_sumsq_a[tid] += s_sumsq_a[tid + offset];
                    s_cnt_a[tid] += s_cnt_a[tid + offset];
                    s_sum_b[tid] += s_sum_b[tid + offset];
                    s_sumsq_b[tid] += s_sumsq_b[tid + offset];
                    s_cnt_b[tid] += s_cnt_b[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int n1 = s_cnt_a[0];
                const int n2 = s_cnt_b[0];
                if (n1 == 0 || n2 == 0 || (n1 == 1 && n2 == 1)) {
                    out[outer_idx] = _CUDA_NAN_D;
                    return;
                }
                const double sum1 = s_sum_a[0], sum2 = s_sum_b[0];
                const double mean1 = sum1 / (double)n1;
                const double mean2 = sum2 / (double)n2;
                const double varnum1 = s_sumsq_a[0] - (sum1 * sum1) / (double)n1;
                const double varnum2 = s_sumsq_b[0] - (sum2 * sum2) / (double)n2;
                double denom;
                if (n1 == 1) {
                    denom = sqrt(varnum2 / ((double)(n2 - 1) * (double)n2));
                } else if (n2 == 1) {
                    denom = sqrt(varnum1 / ((double)(n1 - 1) * (double)n1));
                } else {
                    denom = sqrt(varnum1 / ((double)(n1 - 1) * (double)n1)
                                + varnum2 / ((double)(n2 - 1) * (double)n2));
                }
                if (denom == 0.0) {
                    out[outer_idx] = _CUDA_NAN_D;
                } else {
                    out[outer_idx] = (mean1 - mean2) / denom;
                }
            }
        }
        """
        kern = xp.RawKernel(code, 'ttest_f64')
        _CUPY_MEANVAR_KERNELS[key] = kern
        return kern
    else:
        code = r"""
        #define _CUDA_NAN_F __int_as_float(0x7fffffff)
        extern "C" __global__ void ttest_f32(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              const long long outer_size,
                                              const int last_dim,
                                              float* __restrict__ out)
        {
            const long long outer_idx = blockIdx.x;
            if (outer_idx >= outer_size) return;
            const int tid = threadIdx.x;
            const int nthreads = blockDim.x;
            const float* row_a = a + outer_idx * (long long)last_dim;
            const float* row_b = b + outer_idx * (long long)last_dim;
            float sum_a = 0.0f, sumsq_a = 0.0f;
            float sum_b = 0.0f, sumsq_b = 0.0f;
            int cnt_a = 0, cnt_b = 0;
            for (int j = tid; j < last_dim; j += nthreads) {
                float va = row_a[j];
                if (va == va) { sum_a += va; sumsq_a += va * va; cnt_a += 1; }
                float vb = row_b[j];
                if (vb == vb) { sum_b += vb; sumsq_b += vb * vb; cnt_b += 1; }
            }
            extern __shared__ unsigned char smem_raw[];
            float* s_sum_a = (float*)smem_raw;
            float* s_sumsq_a = (float*)(s_sum_a + nthreads);
            int* s_cnt_a = (int*)(s_sumsq_a + nthreads);
            float* s_sum_b = (float*)(s_cnt_a + nthreads);
            float* s_sumsq_b = (float*)(s_sum_b + nthreads);
            int* s_cnt_b = (int*)(s_sumsq_b + nthreads);
            s_sum_a[tid] = sum_a; s_sumsq_a[tid] = sumsq_a; s_cnt_a[tid] = cnt_a;
            s_sum_b[tid] = sum_b; s_sumsq_b[tid] = sumsq_b; s_cnt_b[tid] = cnt_b;
            __syncthreads();
            for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    s_sum_a[tid] += s_sum_a[tid + offset];
                    s_sumsq_a[tid] += s_sumsq_a[tid + offset];
                    s_cnt_a[tid] += s_cnt_a[tid + offset];
                    s_sum_b[tid] += s_sum_b[tid + offset];
                    s_sumsq_b[tid] += s_sumsq_b[tid + offset];
                    s_cnt_b[tid] += s_cnt_b[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int n1 = s_cnt_a[0];
                const int n2 = s_cnt_b[0];
                if (n1 == 0 || n2 == 0 || (n1 == 1 && n2 == 1)) {
                    out[outer_idx] = _CUDA_NAN_F;
                    return;
                }
                const float sum1 = s_sum_a[0], sum2 = s_sum_b[0];
                const float mean1 = sum1 / (float)n1;
                const float mean2 = sum2 / (float)n2;
                const float varnum1 = s_sumsq_a[0] - (sum1 * sum1) / (float)n1;
                const float varnum2 = s_sumsq_b[0] - (sum2 * sum2) / (float)n2;
                float denom;
                if (n1 == 1) {
                    denom = sqrtf(varnum2 / ((float)(n2 - 1) * (float)n2));
                } else if (n2 == 1) {
                    denom = sqrtf(varnum1 / ((float)(n1 - 1) * (float)n1));
                } else {
                    denom = sqrtf(varnum1 / ((float)(n1 - 1) * (float)n1)
                                 + varnum2 / ((float)(n2 - 1) * (float)n2));
                }
                if (denom == 0.0f) {
                    out[outer_idx] = _CUDA_NAN_F;
                } else {
                    out[outer_idx] = (mean1 - mean2) / denom;
                }
            }
        }
        """
        kern = xp.RawKernel(code, 'ttest_f32')
        _CUPY_MEANVAR_KERNELS[key] = kern
        return kern


def brunnermunzel(x: np.ndarray, y: np.ndarray, axis=None, nan_policy='omit'):
    """
    Compute the Brunner-Munzel test statistic for two independent samples.

    The Brunner-Munzel test is used to compare the stochastic dominance of two
    independent samples and does not assume equal variances. It is a
     nonparametric statistical test that operates using ranked data. This
      implementation allows handling NaN values based on the specified policy.

    Parameters
    ----------
    x : np.ndarray
        The first input array representing sample data.
    y : np.ndarray
        The second input array representing sample data.
    axis : int or None, optional
        The axis along which to compute the test statistic. If None, the arrays
        are flattened before computation. Default is None.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle NaN values in the inputs:
        - 'propagate': Returns NaN in the result if NaN is present in the input
        - 'raise': Raises an error if NaN is detected in the input.
        - 'omit': Omits NaN values during the computation.
        Default is 'omit'.

    Returns
    -------
    np.ndarray or float
        The computed Brunner-Munzel statistic, returned as a scalar if the
         input arrays are 1D and as an array otherwise.
    """

    if axis is None:
        nx, ny = x.size, y.size
        idxx = slice(0, nx)
        idxy = slice(nx, nx+ny)
        x, y = x.flat, y.flat
        concat = np.concatenate((x, y), axis=0)
    else:
        while axis < 0:
            axis += x.ndim
        nx, ny = x.shape[axis], y.shape[axis]
        idxx = tuple(slice(None) if i != axis else slice(0, nx)
                     for i in range(x.ndim))
        idxy = tuple(slice(None) if i != axis else slice(nx, nx+ny)
                     for i in range(x.ndim))
        concat = np.concatenate((x, y), axis=axis)

    where = ~np.isnan(concat)
    if nan_policy == 'omit':
        rank = partial(rankdata, nan_policy=nan_policy)
        wherex, wherey = where[idxx], where[idxy]
    else:
        rank = rankdata
        wherex = wherey = None
        if np.any(~where) and nan_policy == 'raise':
            raise ValueError("The input contains NaN.")

    kwargsx = dict(axis=axis, where=wherex, keepdims=True)
    kwargsy = dict(axis=axis, where=wherey, keepdims=True)

    rankc = rank(concat, axis=axis)
    rankcx, rankcy = rankc[idxx], rankc[idxy]
    rankcx_mean, rankcy_mean = rankcx.mean(**kwargsx), rankcy.mean(**kwargsy)
    rankx, ranky = rank(x, axis=axis), rank(y, axis=axis)
    rankx_mean, ranky_mean = rankx.mean(**kwargsx), ranky.mean(**kwargsy)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0),
                **kwargsx) / (nx - 1)
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0),
                **kwargsy) / (ny - 1)

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
    return np.squeeze(wbfn)


def ttest(group1: np.ndarray, group2: np.ndarray,
          axis: int, xp=None) -> np.ndarray:
    """Calculate the t-statistic between two groups.

    This function is the default statistic function for time_perm_cluster. It
    calculates the t-statistic between two groups along the specified axis.

    Parameters
    ----------
    group1 : array, shape (..., time)
        The first group of observations.
    group2 : array, shape (..., time)
        The second group of observations.
    axis : int or tuple of ints, optional
        The axis or axes along which to compute the t-statistic. If None,
        compute the t-statistic over all axes.

    Returns
    -------
    t : array
        The t-statistic between the two groups.

    Examples
    --------
    >>> import numpy as np
    >>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]])
    >>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    >>> ttest(group1, group2, 1)
    array([      nan, 1.2004901])
    >>> ttest(group1, group2, 0)
    array([0.        , 1.01680311, 0.        , 1.10431526, 0.        ])
    >>> import cupy as cp
    >>> group1 = cp.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]]
    ... )
    >>> group2 = cp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    >>> ttest(group1, group2, 1)
    array([      nan, 1.2004901])
    """
    if xp is None:
        xp = array_namespace(group1, group2)
    while axis < 0:
        axis += group1.ndim
    if is_numpy(xp):
        return _ttest(group1, group2, axes=[axis, axis])
    elif is_cupy(xp):
        # Use fused CuPy kernel
        moved1 = xp.moveaxis(group1, axis, -1)
        moved2 = xp.moveaxis(group2, axis, -1)
        # Ensure floating dtype for kernel
        kdtype = xp.float64 if moved1.dtype == xp.float64 or moved2.dtype == xp.float64 else xp.float32
        a_cast = moved1.astype(kdtype, copy=False)
        b_cast = moved2.astype(kdtype, copy=False)
        outer = int(xp.prod(xp.asarray(a_cast.shape[:-1]))) if a_cast.ndim > 1 else 1
        last_dim = a_cast.shape[-1]
        a_flat = a_cast.reshape(outer, last_dim)
        b_flat = b_cast.reshape(outer, last_dim)
        out = xp.empty((outer,), dtype=kdtype)
        kern = _get_cupy_ttest_kernel(xp, kdtype)
        threads = 256
        blocks = outer
        # shared memory: sum/sumsq/count for both groups
        size_per_thread = (kdtype().itemsize * 2 + xp.dtype('int32').itemsize) * 2
        shmem = threads * size_per_thread
        kern((blocks,), (threads,), (a_flat, b_flat, outer, int(last_dim), out), shared_mem=shmem)
        return out.reshape(a_cast.shape[:-1])
    else:
        raise NotImplementedError("T-test is not implemented for this array"
                                  " type.")


def concatenate_arrays(arrays: tuple[np.ndarray, ...], axis: int = 0
                       ) -> np.ndarray:
    """Concatenate arrays along a specified axis, filling in empty arrays with
    nan values.

    Parameters
    ----------
    arrays
        A list of arrays to concatenate
    axis
        The axis along which to concatenate the arrays

    Returns
    -------
    result
        The concatenated arrays

    Examples
    --------
    >>> concatenate_arrays((np.array([[1, 2, 3]]), np.array([[4, 5]])), axis=0)
    array([[ 1.,  2.,  3.],
           [ 4.,  5., nan]])
    >>> concatenate_arrays((np.array([1, 2, 3]), np.array([4, 5])), axis=0)
    array([1., 2., 3., 4., 5.])
    >>> arr1 = np.arange(60, dtype=float).reshape(10,2)
    >>> arr2 = np.arange(240, dtype=float).reshape(20,3)
    >>> concatenate_arrays((arr1, arr2), axis=0)[0]
    array([[ 0.,  1.,  2., nan],
           [ 3.,  4.,  5., nan],
           [nan, nan, nan, nan]])
    >>> concatenate_arrays((arr2[0], arr1[0]), axis=1)
    array([[ 0.,  1.,  2.,  3.,  0.,  1.,  2.],
           [ 4.,  5.,  6.,  7.,  3.,  4.,  5.],
           [ 8.,  9., 10., 11., nan, nan, nan]])
    >>> arr = concatenate_arrays((arr1[0], arr2[0]), axis=None)
    >>> arr
    array([[[ 0.,  1.,  2., nan],
            [ 3.,  4.,  5., nan],
            [nan, nan, nan, nan]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]])
    >>> concatenate_arrays((arr2[0].astype('f2'), arr1[0].astype('f2')),
    ... axis=1)
    array([[ 0.,  1.,  2.,  3.,  0.,  1.,  2.],
           [ 4.,  5.,  6.,  7.,  3.,  4.,  5.],
           [ 8.,  9., 10., 11., nan, nan, nan]], dtype=float16)
    """

    if axis is None:
        axis = 0
        arrays = [np.expand_dims(ar, axis) for ar in arrays]

    arrays = [ar.astype(float) if ar.dtype.kind in 'iu' else ar
              for ar in arrays if ar.size > 0]
    if len(arrays) == 0:
        return np.array([])

    while axis < 0:
        axis += max(a.ndim for a in arrays)

    max_shape = [max(a.shape[ax] for a in arrays) if ax != axis else
                 sum(a.shape[ax] for a in arrays)
                 for ax in range(arrays[0].ndim)]
    out = np.full(max_shape, np.nan, dtype=arrays[0].dtype)
    start = 0
    for i, ar in enumerate(arrays):
        slices = tuple(slice(start, start + ar.shape[ax]) if ax == axis else
                       slice(ar.shape[ax]) for ax in range(ar.ndim))
        out[slices] = ar
        start += ar.shape[axis]
    return out


def _mixup_np(arr: np.ndarray, obs_axis: int, alpha: float = 1.,
              rng: int = None) -> None:
    """Oversample by mixing two random non-NaN observations

    Parameters
    ----------
    arr : array
        The data to oversample.
    obs_axis : int
        The axis along which to apply func.
    alpha : float
        The alpha parameter for the beta distribution. If alpha is 0, then
        the distribution is uniform. If alpha is 1, then the distribution is
        symmetric. If alpha is greater than 1, then the distribution is
        skewed towards the first observation. If alpha is less than 1, then
        the distribution is skewed towards the second observation.

    Examples
    --------
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> _mixup_np(arr, 0, rng=42)
    >>> arr # doctest: +NORMALIZE_WHITESPACE +SKIP
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.24946679, 6.24946679]])
    >>> arr2 = np.arange(24, dtype=float).reshape(2,3)
    >>> arr2[0, 2, :] = [float("nan")] * 4
    >>> _mixup_np(arr2, 1, rng=42)
    >>> arr2 # doctest: +NORMALIZE_WHITESPACE +SKIP
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 2.33404428,  3.33404428,  4.33404428,  5.33404428]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> arr3 = np.arange(24).reshape(3,2).astype("f2")
    >>> arr3[0, :, :] = float("nan")
    >>> _mixup_np(arr3, 0, rng=42)
    >>> arr3 # doctest: +NORMALIZE_WHITESPACE +SKIP
    array([[[12.67, 13.67, 14.67, 15.67],
            [17.31, 18.31, 19.31, 20.31]],
    <BLANKLINE>
           [[ 8.  ,  9.  , 10.  , 11.  ],
            [12.  , 13.  , 14.  , 15.  ]],
    <BLANKLINE>
           [[16.  , 17.  , 18.  , 19.  ],
            [20.  , 21.  , 22.  , 23.  ]]], dtype=float16)
    """

    if obs_axis == 0:
        arr = arr.swapaxes(1, obs_axis)
    if arr.ndim > 3:
        for i in range(arr.shape[0]):
            _mixup_np(arr[i], obs_axis - 1, alpha, rng)
    elif arr.ndim == 1:
        raise ValueError("Array must have at least 2 dimensions")
    else:
        if rng is None:
            rng = np.random.randint(0, 2 ** 16 - 1)

        if arr.dtype != np.float64:
            temp = arr.astype('f8', copy=True)
            cmixup(temp, 1, alpha, rng)
            arr[...] = temp
        else:
            cmixup(arr, 1, alpha, rng)


def mixup(arr: Array, obs_axis: int, alpha: float = 1.,
          rng=None) -> None:
    """Replace rows along the observation axis that are “missing” (i.e. contain
     any NaNs) with a random convex combination of two non‐missing rows (the
      “mixup”).

    This function works for arrays of arbitrary dimension so long as the
    observation axis (obs_axis) contains the “rows” to mix up and the last axis
    holds features. In higher dimensions the axes other than obs_axis and the
    last axis are treated as independent batch indices. (Every such batch is
     assumed to have at least one non-NaN row.)

    The mixup coefficient for each missing row is drawn from a beta
     distribution with parameters (alpha, alpha) and then “flipped” if it is
      less than 0.5 (so that the coefficient is always >=0.5).

    Parameters
    ----------
    arr : np.ndarray
        Array of data. In the 2D case it should have shape (n_obs, n_features).
        For higher dimensions, the last axis is taken as features and obs_axis
        (which must not be the last axis) is the observation axis.
    obs_axis : int
        The axis along which to look for rows that contain any NaN.
    alpha : float, default=1.
        The alpha parameter for the beta distribution.
    rng : np.random.RandomState or similar, optional
        A random number generator (if None, one is created using
         np.random.RandomState()).

    Returns
    -------
    None; arr is modified in-place.

    Examples
    --------
    >>> arr = np.array([[1, 2],
    ...                 [4, 5],
    ...                 [7, 8],
    ...                 [float("nan"), float("nan")]])
    >>> mixup(arr, 0, rng=42)
    >>> arr # doctest: +SKIP
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.24946679, 6.24946679]])

    For a 3D example (here we mix along axis 1):
    >>> arr3 = np.arange(24, dtype=float).reshape(2,3,4)
    >>> arr3[0, 2, :] = [float("nan")] * 4
    >>> mixup(arr3, 1, rng=42)
    >>> arr3 # doctest: +SKIP
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 2.33404428,  3.33404428,  4.33404428,  5.33404428]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> np.random.seed(0)
    >>> group2 = np.random.rand(500, 10, 10, 100).astype("float16")
    >>> group2[::2, 0, 0, :] = np.nan
    >>> mixup(group2, 0)
    >>> group2[:10, 0, 0, :5] # doctest: +SKIP
    array([[0.3274 , 0.2805 , 0.1257 , 0.1256 , 0.3027 ],
           [0.748  , 0.1802 , 0.389  , 0.0376 , 0.01179],
           [0.6484 , 0.829  , 0.8213 , 0.2578 , 0.5327 ],
           [0.7583 , 0.5034 , 0.177  , 0.8325 , 0.5166 ],
           [0.7397 , 0.857  , 0.449  , 0.5913 , 0.714  ],
           [0.3076 , 0.062  , 0.989  , 0.719  , 0.758  ],
           [0.571  , 0.176  , 0.679  , 0.6924 , 0.636  ],
           [0.6323 , 0.07513, 0.722  , 0.4668 , 0.7417 ],
           [0.6987 , 0.3787 , 0.4668 , 0.04987, 0.915  ],
           [0.1912 , 0.05853, 0.4368 , 0.72   , 0.824  ]], dtype=float16)
    >>> import cupy as cp
    >>> group3 = cp.random.randn(100, 10, 10, 100)
    >>> group3[0::2, 0, 0, :] = float("nan")
    >>> mixup(group3, 0)
    >>> group3[0, 0, :, :5]
    array([[0.3274 , 0.2805 , 0.1257 , 0.1256 , 0.3027 ],
           [0.748  , 0.1802 , 0.389  , 0.0376 , 0.01179],
           [0.6484 , 0.829  , 0.8213 , 0.2578 , 0.5327 ],
           [0.7583 , 0.5034 , 0.177  , 0.8325 , 0.5166 ],
           [0.7397 , 0.857  , 0.449  , 0.5913 , 0.714  ],
           [0.3076 , 0.062  , 0.989  , 0.719  , 0.758  ],
           [0.571  , 0.176  , 0.679  , 0.6924 , 0.636  ],
           [0.6323 , 0.07513, 0.722  , 0.4668 , 0.7417 ],
           [0.6987 , 0.3787 , 0.4668 , 0.04987, 0.915  ],
           [0.1912 , 0.05853, 0.4368 , 0.72   , 0.824  ]], dtype=float16)
    """
    xp = array_namespace(arr)
    if is_numpy(xp):
        _mixup_np(arr, obs_axis, alpha, rng)
        return
    elif is_torch(xp):  # TODO: remove this crutch to keep data on the GPU
        temp = arr.numpy(force=True).astype(float)
        _mixup_np(temp, obs_axis, alpha, rng)
        arr.copy_(xp.from_numpy(temp))
        return

    if rng is None:
        if is_torch(xp):
            xp.random.manual_seed(xp.random.seed())
            rng = xp
            xp.beta = xp.distributions.beta.Beta(alpha, alpha)
        else:
            rng = xp.random.RandomState()

    # Bring the observation axis to the front; this is a view.
    arr_view = xp.moveaxis(arr, obs_axis, 0)

    # For ndim >= 3, assume that the last axis holds features.
    # Flatten all intermediate (batch) dimensions into one.
    n_obs = arr_view.shape[0]
    n_features = arr_view.shape[-1]
    # if is_torch(xp) and not arr_view.is_contiguous():
    #     arr_view = arr_view.contiguous()

    arr_flat = arr_view.reshape(n_obs, -1, n_features)
    # Compute a mask over the observation axis for each batch:
    mask = xp.isnan(arr_flat).any(axis=-1)
    # For each batch (i.e. each column in the flattened batch dimension) we
    # want to know the available (non-NaN) indices. We do this by sorting the
    # boolean mask along axis 0: since False sorts before True, the first few
    # indices are the non-missing ones.
    order = xp.argsort(mask, axis=0)
    # Get all indices where the observation is missing.
    missing_rows, batch_idx = xp.nonzero(mask)
    counts = xp.bincount(batch_idx, minlength=mask.shape[1]) # number of non-missing rows per batch
    if missing_rows.size:
        L = missing_rows.shape[0]
        # For each missing observation, generate a random index into the
        # available (non-missing) rows in its batch.
        idx1 = xp.astype(rng.rand(L) * counts[batch_idx], int)
        idx2 = xp.astype(rng.rand(L) * counts[batch_idx], int)
        donor1 = order[idx1, batch_idx]
        donor2 = order[idx2, batch_idx]
        lams = xp.empty(L, dtype=arr.dtype)[:, None]
        if is_torch(xp):
            lams[:] = xp.beta.sample((L,1))
        else:
            lams[:] = rng.beta(alpha, alpha, size=(L, 1))
        less = xp.flatnonzero(lams < 0.5)
        xp.subtract(1, lams[less], out=lams[less])
        # lams[:, 1] -= lams[:, 0]
        # lams.sort(axis=1)
        # lams = xp.where(lams < 0.5, 1 - lams, lams)
        # Instead of direct advanced indexing assignment, use index_put_ for
        # torch:
        if is_torch(xp):
            value = lams * arr_flat[donor1, batch_idx]
            xp.subtract(1, lams, out=lams)
            value += lams * arr_flat[donor2, batch_idx]
            arr_flat.masked_scatter_(mask[..., None], value)
        else:
            arr_flat[missing_rows, batch_idx] = lams * arr_flat[donor1, batch_idx]
            xp.subtract(1, lams, out=lams)
            arr_flat[missing_rows, batch_idx] += lams * arr_flat[donor2, batch_idx]


def mixup2(arr: Array, labels: Array, obs_axis: int, alpha: float = 1.,
           seed=None, xp=None) -> None:
    """Label-aware mixup that pairs the larger lambda with a same-class donor.

    This function mirrors the vectorized implementation of ``mixup`` but
    enforces that the larger coefficient multiplies a donor sampled from the
    same class as the target missing observation, while the smaller coefficient
    multiplies a donor sampled from any available non-NaN observation.

    Parameters
    ----------
    arr : Array
        Input data. The last axis is treated as features, and ``obs_axis`` is
        the observation axis where rows with any NaN are imputed.
    labels : Array
        1-D class labels aligned with the observation axis.
    obs_axis : int
        Axis of observations.
    alpha : float
        Beta distribution parameter for the mixing coefficient.
    rng : RandomState-like, optional
        Random number generator. If None, a new one is created.

    Returns
    -------
    None; modifies ``arr`` in-place.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5],
    ... [float("nan"), float("nan")]])
    >>> labels = np.array([1, 1, 0])
    >>> mixup2(arr[None], labels, 1)
    >>> arr
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [6.03943491, 7.03943491]])
    >>> arr3 = np.arange(24, dtype=float).reshape(2,3,4)
    >>> arr3[:, 2, :] = float("nan")
    >>> mixup2(arr3, np.array([1, 0, 1]), 1, seed=42)
    >>> arr3
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 1.99984539,  2.99984539,  3.99984539,  4.99984539]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [12.25137354, 13.25137354, 14.25137354, 15.25137354]]])
    >>> import cupy as cp
    >>> arr4 = cp.array([[1, 2],[3,4], [5, 6],
    ... [7, 8], [9, 10], [cp.nan, cp.nan]])
    >>> labels4 = cp.array([0, 0, 0, 1, 1, 1, 1])
    >>> mixup2(arr4, labels4, 0, seed=0)
    >>> arr4
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [4.54459201, 5.54459201]])
    >>> arr4 = cp.array([[1, 2, 3, 4],[3,4, 5, 6], [5, 6, 7, 8],
    ... [7, 8, 9, 10], [9, 10, 11, 12], [cp.nan, cp.nan, cp.nan, cp.nan]])
    >>> new = cp.stack(cp.stack(tuple(arr4 for _ in range(1000))) for _ in range(1000))
    >>> mixup2(new, labels4, 2, seed=0)
    >>> new[0, 0]
    array([[ 1.        ,  2.        ,  3.        ,  4.        ],
           [ 3.        ,  4.        ,  5.        ,  6.        ],
           [ 5.        ,  6.        ,  7.        ,  8.        ],
           [ 7.        ,  8.        ,  9.        , 10.        ],
           [ 9.        , 10.        , 11.        , 12.        ],
           [ 6.37764196,  7.37764196,  8.37764196,  9.37764196]])
    >>> new = cp.ascontiguousarray(cp.stack(cp.stack(tuple(arr4 for _ in range(1000))) for _ in range(1000)).swapaxes(1,2))
    >>> list(mixup2(new.copy(), labels4, 1, seed=0) for _ in range(100))
    >>> mixup2(new, labels4, 1, seed=0)
    >>> new[0, -1, 0:10]
    array([[ 1.        ,  2.        ,  3.        ,  4.        ],
           [ 3.        ,  4.        ,  5.        ,  6.        ],
           [ 5.        ,  6.        ,  7.        ,  8.        ],
           [ 7.        ,  8.        ,  9.        , 10.        ],
           [ 9.        , 10.        , 11.        , 12.        ],
           [ 6.37764196,  7.37764196,  8.37764196,  9.37764196]])
    """
    if xp is None:
        xp = array_namespace(arr, labels)

    # Torch: move to CPU, operate in NumPy, then copy back (keep data on GPU otherwise)
    if is_torch(xp):  # TODO: remove this crutch to keep data on the GPU
        temp = arr.numpy(force=True).astype(float)
        temp_labels = labels.numpy(force=True)
        mixup2(temp, temp_labels, obs_axis, alpha, seed)
        arr.copy_(xp.from_numpy(temp))
        return

    if seed is None:
        rng = xp.random.RandomState()
    elif isinstance(seed, int):
        rng = xp.random.RandomState(seed)
    else:
        rng = seed

    # Move obs axis to -2 and keep features on -1 to maintain a view-only transform
    arr_moved = xp.moveaxis(arr, [obs_axis, -1], [-2, -1])
    batch_shape = arr_moved.shape[:-2]
    obs = arr_moved.shape[-2]
    # Mask of rows (along obs) with any NaN in features
    isnan = xp.isnan(arr_moved).any(-1)  # shape: batch_shape + (obs,)
    # Disallow batches where every obs is NaN
    isn_sum = isnan.sum(-1, dtype='uintp')
    if (isn_sum == obs).any():
        raise ValueError("Cannot mixup if any rows are completely NaN")

    # Flatten batch dims for index computation only (does not touch data array)
    B = int(xp.prod(xp.asarray(batch_shape))) if batch_shape else 1
    mask = isnan.reshape(B, obs).T  # shape: (obs, B)

    # One pass to derive both non-missing and missing indices per batch
    counts_nonmissing_by_batch = (~mask).sum(axis=0)
    counts_missing_by_batch = mask.shape[0] - counts_nonmissing_by_batch

    # Restrict processing to columns with at least one missing row
    cols_proc = xp.flatnonzero(counts_missing_by_batch > 0)
    if cols_proc.size == 0:
        return
    order_proc = xp.argsort(mask[:, cols_proc], axis=0)  # False (non-missing) before True
    n_cols_proc = cols_proc.shape[0]
    counts_nonmissing_proc = counts_nonmissing_by_batch[cols_proc]

    # Build non-missing indices per processed column using broadcast_to
    max_nn = int(counts_nonmissing_proc.max())
    order_T = order_proc[:max_nn].T  # shape (n_cols_proc, max_nn)
    nn_mask = xp.arange(max_nn)[None] < counts_nonmissing_proc[:, None]
    rows_nn = order_T[nn_mask]
    batch_nn = xp.broadcast_to(cols_proc[:, None], (n_cols_proc, max_nn))[nn_mask]

    # Build missing indices per processed column using broadcast_to (tail of order)
    counts_missing_proc = counts_missing_by_batch[cols_proc]
    max_miss = int(counts_missing_proc.max())
    order_tail_T = order_proc[-max_miss:].T  # (n_cols_proc, max_miss)
    miss_sel = xp.arange(max_miss)[None] >= (max_miss - counts_missing_proc)[:, None]
    missing_rows = order_tail_T[miss_sel]
    batch_idx = xp.broadcast_to(cols_proc[:, None], (n_cols_proc, max_miss))[miss_sel]

    # If there are no missing rows anywhere, we're done
    if missing_rows.size == 0:
        return

    # Donor 2 (any class): sample uniformly from non-missing rows in each batch
    pool_sizes_any = counts_nonmissing_by_batch[batch_idx]
    if xp.any(pool_sizes_any == 0):
        raise ValueError("Not enough non-nan values to mixup")
    idx2 = (rng.rand(missing_rows.shape[0]) * pool_sizes_any).astype(xp.uintp)
    donor2 = order_proc[idx2, xp.searchsorted(cols_proc, batch_idx)]

    # Donor 1 (same class): build per-batch, per-class pools from non-missing rows
    # Map labels to compact class ids [0, K)
    unique_labels, label_ids = xp.unique(labels, return_inverse=True)
    K = int(unique_labels.shape[0])

    non_class_ids = label_ids[rows_nn]

    # Compose (batch, class) into a single index for grouping
    comp = batch_nn * K + non_class_ids
    max_len = int(mask.shape[1]) * K
    counts_comp = xp.bincount(comp, minlength=max_len)
    offsets_comp = xp.empty(counts_comp.shape[0] + 1, dtype=counts_comp.dtype)
    offsets_comp[0] = 0
    offsets_comp[1:] = xp.cumsum(counts_comp)
    order_comp = xp.argsort(comp)
    rows_nn_sorted = rows_nn[order_comp]

    # For each target (missing_rows, batch_idx) choose a donor from same-class pool
    target_class_ids = label_ids[missing_rows]
    comp_targets = batch_idx * K + target_class_ids
    pool_sizes_same = counts_comp[comp_targets]
    if xp.any(pool_sizes_same == 0):
        raise ValueError("Not enough non-nan values to mixup")
    r1 = (rng.rand(missing_rows.shape[0]) * pool_sizes_same).astype(offsets_comp.dtype)
    pos1 = offsets_comp[comp_targets] + r1
    donor1 = rows_nn_sorted[pos1]

    # Mixing coefficients: ensure larger lambda pairs with same-class donor
    lams = xp.empty((missing_rows.shape[0], 1), dtype=f'f{arr.nbytes // arr.size}')
    lams[:, 0] = rng.beta(alpha, alpha, size=(missing_rows.shape[0],))
    less = lams < 0.5
    xp.subtract(1., lams[less], out=lams[less])

    # In-place assignment back into arr_moved using advanced indexing over batch dims
    if batch_shape:
        batch_coords = xp.unravel_index(batch_idx, batch_shape)
        lhs_idx = batch_coords + (missing_rows, slice(None))
        d1_idx = batch_coords + (donor1, slice(None))
        d2_idx = batch_coords + (donor2, slice(None))
    else:
        lhs_idx = (missing_rows, slice(None))
        d1_idx = (donor1, slice(None))
        d2_idx = (donor2, slice(None))

    value = lams * arr_moved[d1_idx]
    xp.subtract(1., lams, out=lams)
    value += lams * arr_moved[d2_idx]
    arr_moved[lhs_idx] = value

def norm(arr: np.ndarray, obs_axis: int = -1) -> None:
    """Oversample by obtaining the distribution and randomly selecting

    Parameters
    ----------
    arr : array
        The data to oversample.
    obs_axis : int
        The axis along which to apply func.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([1, 2, 4, 5, 7, 8,
    ... float("nan"), float("nan")])
    >>> norm(arr)
    >>> arr
    array([1.        , 2.        , 4.        , 5.        , 7.        ,
           8.        , 8.91013086, 5.50039302])
    """
    cnorm(arr, obs_axis)


def mean_diff(group1: Array, group2: Array,
              axis: int = -1, xp=None) -> np.ndarray[float] | float:
    """Calculate the mean difference between two groups.

    This function is the default statistic function for time_perm_cluster. It
    calculates the mean difference between two groups along the specified axis.

    Parameters
    ----------
    group1 : array, shape (..., time)
        The first group of observations.
    group2 : array, shape (..., time)
        The second group of observations.
    axis : int or tuple of ints, optional
        The axis or axes along which to compute the mean difference. If None,
        compute the mean difference over all axes.

    Returns
    -------
    avg1 - avg2 : array or float
        The mean difference between the two groups.

    Examples
    --------
    >>> import numpy as np
    >>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]])
    >>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    >>> mean_diff(group1, group2, axis=1)
    array([ 0., 14.])
    >>> mean_diff(group1, group2, axis=0)
    array([ 0., 30.,  0.,  5.,  0.])
    >>> group3 = np.arange(100000, dtype=float).reshape(20000,5)
    >>> mean_diff(group3, group1, axis=0)
    array([49997., 49968., 49999., 49995., 50001.])
    """

    if xp is None:
        xp = array_namespace(group1, group2)
    if is_numpy(xp):
        return _md(group1, group2, axes=[axis, axis])
    else:
        return group1.mean(axis=axis) - group2.mean(axis=axis)


def meanvar(arr: Array, axis: int = -1, ddof: int = 1, xp=None):
    """Compute mean and variance along an axis, ignoring NaNs.

    Returns a tuple (mean, var). Uses optimized C ufunc for NumPy;
    CuPy uses a RawKernel implementation.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [4, 5, np.nan], [7, 8, 9]])
    >>> meanvar(arr, axis=0)
    (array([4., 5., 6.]), array([9., 9., 9.]))
    >>> meanvar(arr, axis=1)
    (array([2., 4.5, 8.]), array([1., 0.5, 1.]))
    >>> import cupy as cp
    >>> arr_cp = cp.array([[1, 2, 3], [4, 5, cp.nan], [7, 8, 9]])
    >>> mv = meanvar(arr_cp, axis=0)
    >>> mv[0].shape, mv[1].shape
    ((3,), (3,))
    """
    while axis < 0:
        axis += arr.ndim

    if xp is None:
        xp = array_namespace(arr)

    if is_numpy(xp):
        # NumPy backend: C gufunc
        ddof_arg = np.array(ddof, dtype=arr.dtype)
        mean, var = _meanvar(arr, ddof_arg, axes=[(axis,), (), (), ()])
        return mean, var

    elif is_cupy(xp):
        # CuPy backend: RawKernel reduction along the last axis
        moved = xp.moveaxis(arr, axis, -1)
        outer = int(xp.prod(xp.asarray(moved.shape[:-1]))) if moved.ndim > 1 else 1
        last_dim = moved.shape[-1]
        x_flat = moved.reshape(outer, last_dim)
        out_mean = xp.empty((outer,), dtype=moved.dtype)
        out_var = xp.empty((outer,), dtype=moved.dtype)
        kern = _get_cupy_meanvar_kernel(xp, moved.dtype)
        threads = 256
        blocks = outer
        shmem = threads * (moved.dtype.itemsize * 2 + xp.dtype('int32').itemsize)
        kern((blocks,), (threads,),
             (x_flat, outer, int(last_dim), moved.dtype.type(ddof), out_mean, out_var),
             shared_mem=shmem)
        mean = out_mean.reshape(moved.shape[:-1])
        var = out_var.reshape(moved.shape[:-1])
        return mean, var

    else:

        raise NotImplementedError("meanvar not implemented for this array type")


if __name__ == "__main__":
    import numpy as np
    from timeit import timeit
    from scipy import stats

    # np.random.seed(0)
    rng = np.random.default_rng()
    rvs1 = np.array([
        stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
        for _ in range(100)]) / 10000
    # group2[::2] = np.nan
    rvs3 = np.array([
        stats.norm.rvs(loc=8, scale=5, size=2000, random_state=rng)
        for _ in range(100)]) / 10000
    res1 = ttest(rvs1, rvs3, axis=1)
    res2 = stats.ttest_ind(rvs1, rvs3, axis=1, equal_var=False)
    res3 = _ttest(rvs1, rvs3, axes=[1, 1])

    n = 10000
    kwargs = dict(globals=globals(), number=n)
    time1 = timeit('ttest(rvs1, rvs3, 1)', **kwargs)
    print(f"ttest: {time1 / n:.3g} per run")
    # time2 = timeit('stats.ttest_ind(rvs1, rvs3, axis=1, equal_var=False)',
    # **kwargs)
    # print(f"scipy: {time2 / n:.3g} per run")
    time3 = timeit('_ttest(rvs1, rvs3, axes=[1,1])', **kwargs)
    print(f"_ttest: {time3 / n:.3g} per run")
