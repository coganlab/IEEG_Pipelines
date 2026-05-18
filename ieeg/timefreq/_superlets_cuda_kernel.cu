/*
 * _superlets_cuda_kernel.cu — CUDA kernel for the adaptive superlet
 * Morlet wavelet batch.
 *
 * Replaces the previous ``@numba.cuda.jit`` kernel in superlets.py.
 * The kernel writes directly into a complex128 output buffer (flat
 * double pairs: real, imag) so the host wrapper can avoid the
 * real-then-imag-then-combine round-trip that numba's kernel required.
 *
 * Loaded at runtime via ``cupy.RawKernel`` (which uses nvrtc, the
 * CUDA runtime compiler bundled with the CUDA toolkit).
 *
 * Grid layout: one thread per (i_order, i_freq) pair. The thread
 * walks the n_samples inner loop sequentially. Suggested launch:
 *   threads_per_block = 256
 *   blocks = (max_order * n_freqs + 255) / 256
 */

extern "C" __global__ void cxmorelet_batch_kernel(
    const double * __restrict__ freqs,
    const double * __restrict__ cycles,
    const double sampling_freq,
    double * __restrict__ wavelets_out,   /* (max_order, n_freqs, n_samples) complex128
                                             * as flat double pairs */
    const int n_samples,
    const int max_order,
    const int n_freqs)
{
    const double k_sd      = 5.0;
    const double sqrt_2pi  = 2.5066282746310002;
    const double two_pi    = 6.283185307179586;

    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)max_order * n_freqs) return;

    const int i_order = (int)(idx / n_freqs);
    const int i_freq  = (int)(idx % n_freqs);

    const double cycle_count = cycles[i_order];
    const double freq        = freqs[i_freq];

    const double bc = cycle_count / (k_sd * freq);
    const double norm = 1.0 / (bc * sqrt_2pi);
    const double inv_two_bc_sq = 1.0 / (2.0 * bc * bc);
    const double inv_n_minus_1 = (n_samples > 1) ? (1.0 / (double)(n_samples - 1)) : 0.0;
    const double two_pi_freq = two_pi * freq;

    /* Pointer to this thread's row: 2*n_samples doubles. */
    double *row = wavelets_out + ((long long)i_order * n_freqs + i_freq) * (long long)n_samples * 2LL;

    double abs_sum = 0.0;

    /* First pass: write wavelet values, accumulate the absolute-value
     * sum used for L1 normalisation. */
    for (int i_sample = 0; i_sample < n_samples; ++i_sample) {
        /* Match np.linspace(-1.0, 1.0, n_samples): endpoint-inclusive,
         * step = 2 / (n_samples - 1). For n_samples == 1 the numpy
         * convention is [-1.0], so use that as a fallback. */
        const double t_val = (n_samples > 1)
            ? (-1.0 + 2.0 * (double)i_sample * inv_n_minus_1)
            : -1.0;
        const double gauss = exp(-(t_val * t_val) * inv_two_bc_sq);
        const double angle = two_pi_freq * t_val;
        const double sr = cos(angle);
        const double si = sin(angle);
        const double w_re = norm * gauss * sr;
        const double w_im = norm * gauss * si;
        row[2 * i_sample + 0] = w_re;
        row[2 * i_sample + 1] = w_im;
        abs_sum += sqrt(w_re * w_re + w_im * w_im);
    }

    /* Second pass: L1-normalise. */
    if (abs_sum > 0.0) {
        const double inv_abs_sum = 1.0 / abs_sum;
        for (int i_sample = 0; i_sample < n_samples; ++i_sample) {
            row[2 * i_sample + 0] *= inv_abs_sum;
            row[2 * i_sample + 1] *= inv_abs_sum;
        }
    }
}
