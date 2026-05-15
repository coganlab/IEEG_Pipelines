/*
 * _hilbert_kernel.c — OpenMP inner kernel for filterbank Hilbert transform.
 *
 * Replaces the OpenMP-using portion of the previous Cython
 * implementation (``hilbert.pyx``). The Python orchestration (FFT,
 * inverse FFT, abs) lives in ``hilbert.py``; this module provides:
 *
 *   build_apply_H_c64(freqs, cfs, sds, h, Xf, out, minf, maxf)
 *     For a single channel's frequency-domain signal Xf (shape (N,)
 *     complex64), build the gaussian filter bank H (shape (N, n_cfs)
 *     complex64) and compute ``Xf[:, None] * H`` (shape (N, n_cfs)
 *     complex64) into ``out``. Parallelized across the frequency axis
 *     via OpenMP.
 *
 * Both H construction and the Xf*H multiplication are fused into a
 * single parallel loop with no cross-thread dependencies — output is
 * bit-identical to a serial implementation.
 *
 * Complex memory layout: numpy guarantees ``complex64`` is two
 * contiguous floats (real, imag). We cast through ``float *`` to avoid
 * the MSVC ``_Fcomplex`` / gcc ``float _Complex`` divergence in the
 * ``npy_complex64`` typedef.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif


/* Build one row of H and apply the Xf multiplication.
 * Inputs are raw float pointers for the complex arrays (each complex
 * value occupies two contiguous floats: real, imag).
 *
 * For the gaussian half (i in [1, n_freqs)), H is gaussian(freqs[i] - cfs[j])
 * times h[i]. For the mirror half (i in [n_freqs, N)), we use the gaussian
 * computed at freqs[N - i], then multiply by h[i]. Row 0 of H is zero.
 *
 * Then we compute out[i, j] = Xf[i] * H[i, j].
 */
static PyObject *
build_apply_H_c64(PyObject *self, PyObject *args)
{
    PyArrayObject *freqs_arr, *cfs_arr, *sds_arr, *h_arr, *Xf_arr, *out_arr;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
            &PyArray_Type, &freqs_arr,
            &PyArray_Type, &cfs_arr,
            &PyArray_Type, &sds_arr,
            &PyArray_Type, &h_arr,
            &PyArray_Type, &Xf_arr,
            &PyArray_Type, &out_arr)) {
        return NULL;
    }

    /* Dtype/shape validation */
    if (PyArray_TYPE(freqs_arr) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "freqs must be float32"); return NULL;
    }
    if (PyArray_TYPE(cfs_arr) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "cfs must be float32"); return NULL;
    }
    if (PyArray_TYPE(sds_arr) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "sds must be float32"); return NULL;
    }
    if (PyArray_TYPE(h_arr) != NPY_COMPLEX64) {
        PyErr_SetString(PyExc_TypeError, "h must be complex64"); return NULL;
    }
    if (PyArray_TYPE(Xf_arr) != NPY_COMPLEX64) {
        PyErr_SetString(PyExc_TypeError, "Xf must be complex64"); return NULL;
    }
    if (PyArray_TYPE(out_arr) != NPY_COMPLEX64) {
        PyErr_SetString(PyExc_TypeError, "out must be complex64"); return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(freqs_arr) ||
        !PyArray_IS_C_CONTIGUOUS(cfs_arr) ||
        !PyArray_IS_C_CONTIGUOUS(sds_arr) ||
        !PyArray_IS_C_CONTIGUOUS(h_arr) ||
        !PyArray_IS_C_CONTIGUOUS(Xf_arr) ||
        !PyArray_IS_C_CONTIGUOUS(out_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "all input arrays must be C-contiguous");
        return NULL;
    }

    const npy_intp n_freqs = PyArray_DIM(freqs_arr, 0);
    const npy_intp n_cfs   = PyArray_DIM(cfs_arr, 0);
    if (PyArray_DIM(sds_arr, 0) != n_cfs) {
        PyErr_SetString(PyExc_ValueError, "sds shape mismatch"); return NULL;
    }
    /* h is treated as a 1-D vector of length N; caller may pass (N, 1) or (N,). */
    npy_intp N;
    if (PyArray_NDIM(h_arr) == 1) {
        N = PyArray_DIM(h_arr, 0);
    } else if (PyArray_NDIM(h_arr) == 2 && PyArray_DIM(h_arr, 1) == 1) {
        N = PyArray_DIM(h_arr, 0);
    } else {
        PyErr_SetString(PyExc_ValueError,
                        "h must have shape (N,) or (N, 1)"); return NULL;
    }
    if (PyArray_NDIM(Xf_arr) != 1 || PyArray_DIM(Xf_arr, 0) != N) {
        PyErr_SetString(PyExc_ValueError,
                        "Xf must have shape (N,) matching h"); return NULL;
    }
    if (PyArray_NDIM(out_arr) != 2 ||
        PyArray_DIM(out_arr, 0) != N || PyArray_DIM(out_arr, 1) != n_cfs) {
        PyErr_SetString(PyExc_ValueError,
                        "out must have shape (N, n_cfs)"); return NULL;
    }
    if (n_freqs > N) {
        PyErr_SetString(PyExc_ValueError, "n_freqs > N"); return NULL;
    }

    /* Cast complex buffers to float[N * n_cfs * 2] for portability. */
    const float * const freqs = (const float *)PyArray_DATA(freqs_arr);
    const float * const cfs   = (const float *)PyArray_DATA(cfs_arr);
    const float * const sds   = (const float *)PyArray_DATA(sds_arr);
    const float * const h     = (const float *)PyArray_DATA(h_arr);  /* (N, 2) */
    const float * const Xf    = (const float *)PyArray_DATA(Xf_arr); /* (N, 2) */
    float * const out         = (float *)PyArray_DATA(out_arr);      /* (N, n_cfs, 2) */

    NPY_BEGIN_ALLOW_THREADS;

    /* Row 0 of out = Xf[0] * 0 = 0. (H[0] is zero per the original. ) */
    for (npy_intp j = 0; j < n_cfs; ++j) {
        out[2 * j + 0] = 0.0f;
        out[2 * j + 1] = 0.0f;
    }

    /* Rows 1 to n_freqs - 1: gaussian * h[i], then multiply by Xf[i].
     * MSVC's OpenMP requires the loop variable to be declared outside
     * the parallel-for pragma, so we use signed npy_intp pre-declared. */
    npy_intp i;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = 1; i < n_freqs; ++i) {
        const float fi = freqs[i];
        const float hi_re = h[2 * i + 0];
        const float hi_im = h[2 * i + 1];
        const float xi_re = Xf[2 * i + 0];
        const float xi_im = Xf[2 * i + 1];
        float * const out_row = out + i * n_cfs * 2;
        npy_intp j;
        for (j = 0; j < n_cfs; ++j) {
            const float k = fi - cfs[j];
            const float r = k / sds[j];
            const float gauss = expf(-0.5f * (r * r));  /* match numpy's `(x**2)` precedence */
            /* H[i, j] = gauss * h[i]  (gauss is real) */
            const float H_re = gauss * hi_re;
            const float H_im = gauss * hi_im;
            /* out[i, j] = Xf[i] * H[i, j] */
            out_row[2 * j + 0] = xi_re * H_re - xi_im * H_im;
            out_row[2 * j + 1] = xi_re * H_im + xi_im * H_re;
        }
    }

    /* Rows n_freqs to N - 1: mirror from gaussian at index N - i, multiply
     * by h[i], then by Xf[i]. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (i = n_freqs; i < N; ++i) {
        const npy_intp mirror = N - i;          /* 1 <= mirror < n_freqs */
        const float fm = freqs[mirror];
        const float hi_re = h[2 * i + 0];
        const float hi_im = h[2 * i + 1];
        const float xi_re = Xf[2 * i + 0];
        const float xi_im = Xf[2 * i + 1];
        float * const out_row = out + i * n_cfs * 2;
        npy_intp j;
        for (j = 0; j < n_cfs; ++j) {
            const float k = fm - cfs[j];
            const float r = k / sds[j];
            const float gauss = expf(-0.5f * (r * r));  /* match numpy's `(x**2)` precedence */
            const float H_re = gauss * hi_re;
            const float H_im = gauss * hi_im;
            out_row[2 * j + 0] = xi_re * H_re - xi_im * H_im;
            out_row[2 * j + 1] = xi_re * H_im + xi_im * H_re;
        }
    }

    NPY_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}


static PyMethodDef hilbert_kernel_methods[] = {
    {"build_apply_H_c64", build_apply_H_c64, METH_VARARGS,
     "build_apply_H_c64(freqs, cfs, sds, h, Xf, out)\n\n"
     "Fused construction of the gaussian filter-bank H and its product\n"
     "with a single channel's Xf. Writes the (N, n_cfs) result into\n"
     "the pre-allocated complex64 ``out`` array. All inputs must be\n"
     "C-contiguous; freqs/cfs/sds float32, h/Xf/out complex64."},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef hilbert_kernel_module = {
    PyModuleDef_HEAD_INIT,
    "_hilbert_kernel",
    "OpenMP inner kernel for the filterbank Hilbert transform.",
    -1, hilbert_kernel_methods,
    NULL, NULL, NULL, NULL,
};


PyMODINIT_FUNC
PyInit__hilbert_kernel(void)
{
    PyObject *m = PyModule_Create(&hilbert_kernel_module);
    if (m == NULL) return NULL;
    import_array();
    return m;
}
