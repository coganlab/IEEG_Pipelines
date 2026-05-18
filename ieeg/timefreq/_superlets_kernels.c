/*
 * _superlets_kernels.c — OpenMP CPU kernels for the adaptive superlet
 * transform.
 *
 * Replaces the two ``@numba.njit`` functions previously in
 * ``superlets.py``:
 *
 *   cxmorelet_batch(freqs, cycles, sampling_freq, wavelets_out)
 *     Batch computation of complex Morlet wavelets into a pre-allocated
 *     (max_order, n_freqs, n_samples) complex128 output array.
 *     Parallelized over the order axis via OpenMP.
 *
 *   apply_mask_and_geomean(out, orders, eps, result)
 *     Mask `out[i_order, i_freq, :]` to 1.0 where ``i_order + 1 >
 *     orders[i_freq]``, then write the geometric mean of the first
 *     ``orders[i_freq]`` entries to ``result[i_freq, :]``. ``out`` is
 *     mutated in place; ``result`` is pre-allocated and written into.
 *     Parallelized over the frequency axis.
 *
 * Both kernels run on float64 input/output. Complex128 buffers are
 * treated as flat ``double *`` arrays (real, imag interleaved) to avoid
 * MSVC ``_Dcomplex`` vs gcc ``double _Complex`` divergence.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#define _SUPERLETS_TWO_PI 6.283185307179586
#define _SUPERLETS_SQRT_2PI 2.5066282746310002


/* ============================================================ */
/* cxmorelet_batch                                                */
/* ============================================================ */

static PyObject *
cxmorelet_batch(PyObject *self, PyObject *args)
{
    PyArrayObject *freqs_arr, *cycles_arr, *out_arr;
    double sampling_freq;
    if (!PyArg_ParseTuple(args, "O!O!dO!",
            &PyArray_Type, &freqs_arr,
            &PyArray_Type, &cycles_arr,
            &sampling_freq,
            &PyArray_Type, &out_arr)) {
        return NULL;
    }

    if (PyArray_TYPE(freqs_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "freqs must be float64"); return NULL;
    }
    if (PyArray_TYPE(cycles_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "cycles must be float64"); return NULL;
    }
    if (PyArray_TYPE(out_arr) != NPY_COMPLEX128) {
        PyErr_SetString(PyExc_TypeError,
                        "wavelets_out must be complex128"); return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(freqs_arr) ||
        !PyArray_IS_C_CONTIGUOUS(cycles_arr) ||
        !PyArray_IS_C_CONTIGUOUS(out_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "all arrays must be C-contiguous"); return NULL;
    }
    if (PyArray_NDIM(out_arr) != 3) {
        PyErr_SetString(PyExc_ValueError,
                        "wavelets_out must be 3D"); return NULL;
    }

    const npy_intp max_order = PyArray_DIM(out_arr, 0);
    const npy_intp n_freqs   = PyArray_DIM(out_arr, 1);
    const npy_intp n_samples = PyArray_DIM(out_arr, 2);
    if (PyArray_NDIM(cycles_arr) != 1 ||
        PyArray_DIM(cycles_arr, 0) != max_order) {
        PyErr_SetString(PyExc_ValueError, "cycles shape mismatch"); return NULL;
    }
    if (PyArray_NDIM(freqs_arr) != 1 ||
        PyArray_DIM(freqs_arr, 0) != n_freqs) {
        PyErr_SetString(PyExc_ValueError, "freqs shape mismatch"); return NULL;
    }
    const npy_intp expected_n_samples = (npy_intp)(sampling_freq * 2.0);
    if (n_samples != expected_n_samples) {
        PyErr_Format(PyExc_ValueError,
            "wavelets_out.shape[2]=%lld but expected sampling_freq*2=%lld",
            (long long)n_samples, (long long)expected_n_samples);
        return NULL;
    }

    const double *freqs  = (const double *)PyArray_DATA(freqs_arr);
    const double *cycles = (const double *)PyArray_DATA(cycles_arr);
    /* Treat complex128 as flat (real, imag) doubles. */
    double * const out   = (double *)PyArray_DATA(out_arr);

    const double k_sd = 5.0;
    /* Numba's ``np.linspace(-1.0, 1.0, n_samples)`` includes endpoints;
     * step = 2 / (n_samples - 1). For n_samples == 1 numpy returns
     * [-1.0]; we match that. */
    const double inv_n_minus_1 = (n_samples > 1)
        ? (1.0 / (double)(n_samples - 1)) : 0.0;

    NPY_BEGIN_ALLOW_THREADS;

    npy_intp i_order;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i_order = 0; i_order < max_order; ++i_order) {
        const double cycle_count = cycles[i_order];
        npy_intp i_freq;
        for (i_freq = 0; i_freq < n_freqs; ++i_freq) {
            const double freq = freqs[i_freq];
            const double bc = cycle_count / (k_sd * freq);
            const double norm = 1.0 / (bc * _SUPERLETS_SQRT_2PI);
            const double inv_two_bc_sq = 1.0 / (2.0 * bc * bc);
            const double two_pi_freq = _SUPERLETS_TWO_PI * freq;

            double * const row = out
                + (i_order * n_freqs + i_freq) * n_samples * 2;

            double abs_sum = 0.0;
            for (npy_intp i_sample = 0; i_sample < n_samples; ++i_sample) {
                /* Match np.linspace(-1.0, 1.0, n_samples) exactly: that
                 * function uses `start + step * i` where step =
                 * (stop - start) / (n - 1). Numba would also follow
                 * that since it uses np.linspace internally. */
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

            if (abs_sum > 0.0) {
                const double inv_abs_sum = 1.0 / abs_sum;
                for (npy_intp i_sample = 0; i_sample < n_samples; ++i_sample) {
                    row[2 * i_sample + 0] *= inv_abs_sum;
                    row[2 * i_sample + 1] *= inv_abs_sum;
                }
            }
        }
    }

    NPY_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}


/* ============================================================ */
/* apply_mask_and_geomean                                          */
/* ============================================================ */

static PyObject *
apply_mask_and_geomean(PyObject *self, PyObject *args)
{
    PyArrayObject *out_arr, *orders_arr, *result_arr;
    double eps;
    if (!PyArg_ParseTuple(args, "O!O!dO!",
            &PyArray_Type, &out_arr,
            &PyArray_Type, &orders_arr,
            &eps,
            &PyArray_Type, &result_arr)) {
        return NULL;
    }

    if (PyArray_TYPE(out_arr) != NPY_DOUBLE ||
        PyArray_TYPE(result_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError,
                        "out and result must be float64"); return NULL;
    }
    if (PyArray_TYPE(orders_arr) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError,
                        "orders must be int64 (cast on the Python side)"); return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(out_arr) ||
        !PyArray_IS_C_CONTIGUOUS(orders_arr) ||
        !PyArray_IS_C_CONTIGUOUS(result_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "all arrays must be C-contiguous"); return NULL;
    }
    if (PyArray_NDIM(out_arr) != 3) {
        PyErr_SetString(PyExc_ValueError, "out must be 3D"); return NULL;
    }
    if (PyArray_NDIM(result_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "result must be 2D"); return NULL;
    }

    const npy_intp max_order = PyArray_DIM(out_arr, 0);
    const npy_intp n_freqs   = PyArray_DIM(out_arr, 1);
    const npy_intp n_times   = PyArray_DIM(out_arr, 2);
    if (PyArray_NDIM(orders_arr) != 1 ||
        PyArray_DIM(orders_arr, 0) != n_freqs) {
        PyErr_SetString(PyExc_ValueError, "orders shape mismatch"); return NULL;
    }
    if (PyArray_DIM(result_arr, 0) != n_freqs ||
        PyArray_DIM(result_arr, 1) != n_times) {
        PyErr_SetString(PyExc_ValueError,
                        "result shape mismatch"); return NULL;
    }

    double * const out         = (double *)PyArray_DATA(out_arr);
    const int64_t * const orders = (const int64_t *)PyArray_DATA(orders_arr);
    double * const result      = (double *)PyArray_DATA(result_arr);

    NPY_BEGIN_ALLOW_THREADS;

    npy_intp i_freq;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i_freq = 0; i_freq < n_freqs; ++i_freq) {
        const npy_intp order = (npy_intp)orders[i_freq];

        /* Mask: set out[i_order, i_freq, :] = 1.0 where i_order+1 > order. */
        for (npy_intp i_order = 0; i_order < max_order; ++i_order) {
            if ((i_order + 1) > order) {
                double *row = out + (i_order * n_freqs + i_freq) * n_times;
                for (npy_intp i_time = 0; i_time < n_times; ++i_time) {
                    row[i_time] = 1.0;
                }
            }
        }

        if (order > 0) {
            const double inv_order = 1.0 / (double)order;
            for (npy_intp i_time = 0; i_time < n_times; ++i_time) {
                double log_sum = 0.0;
                for (npy_intp i_order = 0; i_order < order; ++i_order) {
                    const double v = out[(i_order * n_freqs + i_freq) * n_times + i_time];
                    log_sum += log(v + eps);
                }
                result[i_freq * n_times + i_time] = exp(log_sum * inv_order);
            }
        } else {
            /* order == 0: copy first-order slice. */
            const double *src = out + (0 * n_freqs + i_freq) * n_times;
            double *dst = result + i_freq * n_times;
            memcpy(dst, src, (size_t)n_times * sizeof(double));
        }
    }

    NPY_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}


/* ============================================================ */
/* Module init                                                    */
/* ============================================================ */

static PyMethodDef superlets_methods[] = {
    {"cxmorelet_batch", cxmorelet_batch, METH_VARARGS,
     "cxmorelet_batch(freqs, cycles, sampling_freq, wavelets_out)\n\n"
     "Fill ``wavelets_out`` (max_order, n_freqs, n_samples complex128)\n"
     "with normalised complex Morlet wavelets. Parallelised over the\n"
     "order axis via OpenMP."},
    {"apply_mask_and_geomean", apply_mask_and_geomean, METH_VARARGS,
     "apply_mask_and_geomean(out, orders, eps, result)\n\n"
     "Mask ``out`` in place (per-frequency, order > orders[i_freq]\n"
     "becomes 1.0) and write the geometric mean of the first\n"
     "orders[i_freq] entries to ``result[i_freq, :]``. Parallelised\n"
     "over the frequency axis."},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef superlets_module = {
    PyModuleDef_HEAD_INIT,
    "_superlets_kernels",
    "OpenMP CPU kernels for the adaptive superlet transform.",
    -1, superlets_methods,
    NULL, NULL, NULL, NULL,
};


PyMODINIT_FUNC
PyInit__superlets_kernels(void)
{
    PyObject *m = PyModule_Create(&superlets_module);
    if (m == NULL) return NULL;
    import_array();
    return m;
}
