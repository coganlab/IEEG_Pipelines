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
