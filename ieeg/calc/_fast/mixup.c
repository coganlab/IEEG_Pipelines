/*
 * mixup.c — NaN-row imputation by convex combination of two non-NaN donors.
 *
 * Replaces the Cython implementation in mixup.pyx. Supports float16,
 * float32, and float64 inputs; mixing arithmetic is performed in double
 * regardless of the input precision. Uses NumPy's bitgen_t throughout
 * for both donor selection and beta sampling (the previous Cython
 * version mixed libc rand()/srand() with numpy.random.Generator.beta,
 * which was inconsistent and made the output not fully reproducible
 * from a single seed).
 *
 * Public API:
 *   mixupnd(arr, obs_axis, alpha=1.0, seed=-1)
 *     Replace each row (along obs_axis) that contains any NaN with
 *     ``lam * donor1 + (1 - lam) * donor2``, where donor1 and donor2 are
 *     distinct rows sampled uniformly from the non-NaN rows in the same
 *     batch, and ``lam ~ Beta(alpha, alpha)``. arr must be float16,
 *     float32, or float64; the function modifies arr in place.
 *
 *   normnd(arr, obs_axis=-1)
 *     Replace each NaN with a draw from N(mean, std) where mean/std are
 *     computed from the non-NaN values in the same 1-D slice along
 *     obs_axis. arr must be float16, float32, or float64; modified in
 *     place. Uses the LEGACY global numpy RNG (np.random.normal) so
 *     ``np.random.seed`` continues to drive the output stream.
 *
 * Per-dtype kernels are generated from DEFINE_* macros that mirror the
 * pattern in ``shared/meanvar_core.{c,h}``. Each kernel receives raw
 * byte pointers + strides and uses the LOAD_F* / STORE_F* macros from
 * meanvar_core.h to read/write values in the input dtype while
 * accumulating in double.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>
#include <numpy/halffloat.h>
#include <numpy/npy_math.h>
#include "meanvar_core.h"  /* LOAD_F16/F32/F64, STORE_F16/F32/F64, isnan_f64 */
#include <math.h>
#include <string.h>
#include <stdlib.h>


/* =============================================================== */
/* RNG helpers                                                       */
/* =============================================================== */

/* Build a fresh ``numpy.random.Generator(SFC64(seed))``. Caller owns
 * the returned reference. Pass ``seed < 0`` for OS-entropy seeding. */
static PyObject *
_make_generator(int seed)
{
    PyObject *random_mod = PyImport_ImportModule("numpy.random");
    if (random_mod == NULL) return NULL;

    PyObject *sfc64_cls = PyObject_GetAttrString(random_mod, "SFC64");
    if (sfc64_cls == NULL) { Py_DECREF(random_mod); return NULL; }

    PyObject *bg;
    if (seed < 0) {
        bg = PyObject_CallObject(sfc64_cls, NULL);
    } else {
        bg = PyObject_CallFunction(sfc64_cls, "i", seed);
    }
    Py_DECREF(sfc64_cls);
    if (bg == NULL) { Py_DECREF(random_mod); return NULL; }

    PyObject *gen_cls = PyObject_GetAttrString(random_mod, "Generator");
    Py_DECREF(random_mod);
    if (gen_cls == NULL) { Py_DECREF(bg); return NULL; }

    PyObject *gen = PyObject_CallOneArg(gen_cls, bg);
    Py_DECREF(gen_cls);
    Py_DECREF(bg);
    return gen;
}


/* Extract the underlying ``bitgen_t *`` from a numpy Generator. */
static bitgen_t *
_get_bitgen(PyObject *generator)
{
    PyObject *bg_obj = PyObject_GetAttrString(generator, "bit_generator");
    if (bg_obj == NULL) return NULL;
    PyObject *capsule = PyObject_GetAttrString(bg_obj, "capsule");
    Py_DECREF(bg_obj);
    if (capsule == NULL) return NULL;
    bitgen_t *bg = (bitgen_t *)PyCapsule_GetPointer(capsule, "BitGenerator");
    Py_DECREF(capsule);
    return bg;
}


/* Bounded uniform integer in ``[0, n)``. Modulo bias is negligible for
 * typical batch sizes (≪ 2^32). */
static NPY_INLINE npy_intp
_uniform_bounded(bitgen_t *bg, npy_intp n)
{
    return (npy_intp)(bg->next_uint64(bg->state) % (uint64_t)n);
}


/* Call ``generator.beta(alpha, alpha, size=(a_size, b_size))``. The
 * returned array is shape (a_size, b_size), dtype float64, C-contiguous. */
static PyArrayObject *
_draw_beta(PyObject *generator, double alpha,
           npy_intp a_size, npy_intp b_size)
{
    PyObject *size_tuple = Py_BuildValue("(nn)",
        (Py_ssize_t)a_size, (Py_ssize_t)b_size);
    if (size_tuple == NULL) return NULL;

    PyObject *kwargs = PyDict_New();
    if (kwargs == NULL) { Py_DECREF(size_tuple); return NULL; }
    if (PyDict_SetItemString(kwargs, "size", size_tuple) < 0) {
        Py_DECREF(size_tuple); Py_DECREF(kwargs); return NULL;
    }
    Py_DECREF(size_tuple);

    PyObject *args = Py_BuildValue("(dd)", alpha, alpha);
    if (args == NULL) { Py_DECREF(kwargs); return NULL; }

    PyObject *beta_method = PyObject_GetAttrString(generator, "beta");
    if (beta_method == NULL) {
        Py_DECREF(args); Py_DECREF(kwargs); return NULL;
    }
    PyObject *result = PyObject_Call(beta_method, args, kwargs);
    Py_DECREF(beta_method);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return (PyArrayObject *)result;
}


/* Call legacy ``numpy.random.normal(loc=mean, scale=std, size=size)``.
 * Uses the GLOBAL numpy RNG state (settable via ``np.random.seed``).
 * Matches the previous Cython behaviour for ``normnd`` so existing
 * doctests that seed via ``np.random.seed(0)`` keep reproducing. */
static PyArrayObject *
_draw_normal_legacy(double mean, double std, npy_intp size)
{
    PyObject *random_mod = PyImport_ImportModule("numpy.random");
    if (random_mod == NULL) return NULL;
    PyObject *normal_func = PyObject_GetAttrString(random_mod, "normal");
    Py_DECREF(random_mod);
    if (normal_func == NULL) return NULL;
    PyObject *result = PyObject_CallFunction(
        normal_func, "ddn", mean, std, (Py_ssize_t)size);
    Py_DECREF(normal_func);
    return (PyArrayObject *)result;
}


/* =============================================================== */
/* Per-dtype kernels — generated from macros                         */
/* =============================================================== */

/* Count rows (along axis 0 of an x×y slice) that contain any NaN.
 * Mixing precision: load to double, isnan check works for all floats. */
#define DEFINE_COUNT_NAN_ROWS_2D(NAME, LOAD)                                              \
static npy_intp                                                                            \
_count_nan_rows_2d_##NAME(const char *arr_data, npy_intp x, npy_intp y,                   \
                          npy_intp s0, npy_intp s1)                                       \
{                                                                                          \
    npy_intp n_nan = 0;                                                                    \
    for (npy_intp i = 0; i < x; ++i) {                                                     \
        const char *row = arr_data + i * s0;                                               \
        for (npy_intp j = 0; j < y; ++j) {                                                 \
            double v = (double)LOAD(row + j * s1);                                         \
            if (isnan_f64(v)) { n_nan++; break; }                                          \
        }                                                                                  \
    }                                                                                      \
    return n_nan;                                                                          \
}

DEFINE_COUNT_NAN_ROWS_2D(f16, LOAD_F16)
DEFINE_COUNT_NAN_ROWS_2D(f32, LOAD_F32)
DEFINE_COUNT_NAN_ROWS_2D(f64, LOAD_F64)


/* 2D mixup kernel: operate on an x×y slice with byte strides (s0, s1).
 * Identify NaN rows, pick two distinct non-NaN donors per NaN row,
 * write ``lam[i] * d1 + (1 - lam[i]) * d2`` back in place. Mixing is
 * performed in double; results are stored back through STORE which
 * handles the conversion to the input dtype. */
#define DEFINE_MIXUP_2D_KERNEL(NAME, LOAD, STORE)                                         \
static int                                                                                 \
_mixup_2d_kernel_##NAME(                                                                   \
    char *arr_data, npy_intp x, npy_intp y,                                                \
    npy_intp s0, npy_intp s1,                                                              \
    const double *lam_data, bitgen_t *bg)                                                  \
{                                                                                          \
    npy_intp *non_nan = (npy_intp *)malloc((size_t)x * sizeof(npy_intp));                  \
    npy_intp *nan_rows = (npy_intp *)malloc((size_t)x * sizeof(npy_intp));                 \
    if (non_nan == NULL || nan_rows == NULL) {                                             \
        free(non_nan); free(nan_rows);                                                     \
        PyErr_NoMemory();                                                                  \
        return -1;                                                                         \
    }                                                                                      \
    npy_intp k_non_nan = 0, k_nan = 0;                                                     \
    for (npy_intp i = 0; i < x; ++i) {                                                     \
        int has_nan = 0;                                                                   \
        char *row_p = arr_data + i * s0;                                                   \
        for (npy_intp j = 0; j < y; ++j) {                                                 \
            double v = (double)LOAD(row_p + j * s1);                                       \
            if (isnan_f64(v)) { has_nan = 1; break; }                                      \
        }                                                                                  \
        if (has_nan) nan_rows[k_nan++] = i;                                                \
        else non_nan[k_non_nan++] = i;                                                     \
    }                                                                                      \
    /* Need >= 2 distinct donors and >= 1 NaN row to do anything. */                       \
    if (k_non_nan < 2 || k_nan == 0) {                                                     \
        free(non_nan); free(nan_rows);                                                     \
        return 0;                                                                          \
    }                                                                                      \
    for (npy_intp i = 0; i < k_nan; ++i) {                                                 \
        npy_intp row = nan_rows[i];                                                        \
        npy_intp d1 = non_nan[_uniform_bounded(bg, k_non_nan)];                            \
        npy_intp d2 = non_nan[_uniform_bounded(bg, k_non_nan)];                            \
        while (d1 == d2) {                                                                 \
            d2 = non_nan[_uniform_bounded(bg, k_non_nan)];                                 \
        }                                                                                  \
        double lam = lam_data[i];                                                          \
        double cmpl = 1.0 - lam;                                                           \
        char *dst  = arr_data + row * s0;                                                  \
        char *src1 = arr_data + d1  * s0;                                                  \
        char *src2 = arr_data + d2  * s0;                                                  \
        for (npy_intp j = 0; j < y; ++j) {                                                 \
            double v1 = (double)LOAD(src1 + j * s1);                                       \
            double v2 = (double)LOAD(src2 + j * s1);                                       \
            double mixed = lam * v1 + cmpl * v2;                                           \
            STORE(dst + j * s1, mixed);                                                    \
        }                                                                                  \
    }                                                                                      \
    free(non_nan); free(nan_rows);                                                         \
    return 0;                                                                              \
}

DEFINE_MIXUP_2D_KERNEL(f16, LOAD_F16, STORE_F16)
DEFINE_MIXUP_2D_KERNEL(f32, LOAD_F32, STORE_F32)
DEFINE_MIXUP_2D_KERNEL(f64, LOAD_F64, STORE_F64)


/* 1-D normnd kernel: replace NaNs in a 1-D slice with draws from
 * Normal(mean, std) where mean/std come from the non-NaN values.
 * Samples are drawn via the legacy np.random.normal (always float64);
 * STORE converts back to the input dtype. */
#define DEFINE_NORM_1D_KERNEL(NAME, LOAD, STORE)                                          \
static int                                                                                 \
_norm_1d_kernel_##NAME(char *arr_data, npy_intp n, npy_intp stride)                       \
{                                                                                          \
    npy_intp non_nan_count = 0;                                                            \
    double sum = 0.0;                                                                      \
    for (npy_intp i = 0; i < n; ++i) {                                                     \
        double v = (double)LOAD(arr_data + i * stride);                                    \
        if (!isnan_f64(v)) { sum += v; non_nan_count++; }                                  \
    }                                                                                      \
    if (non_nan_count < 1) {                                                               \
        PyErr_SetString(PyExc_ValueError,                                                  \
                        "No test data to fit distribution");                               \
        return -1;                                                                         \
    }                                                                                      \
    double mean = sum / (double)non_nan_count;                                             \
    double var_sum = 0.0;                                                                  \
    for (npy_intp i = 0; i < n; ++i) {                                                     \
        double v = (double)LOAD(arr_data + i * stride);                                    \
        if (!isnan_f64(v)) {                                                               \
            double d = v - mean;                                                           \
            var_sum += d * d;                                                              \
        }                                                                                  \
    }                                                                                      \
    double std = sqrt(var_sum / (double)non_nan_count);                                    \
    npy_intp nan_count = n - non_nan_count;                                                \
    if (nan_count == 0) return 0;                                                          \
    PyArrayObject *samples = _draw_normal_legacy(mean, std, nan_count);                    \
    if (samples == NULL) return -1;                                                        \
    const double *sd = (const double *)PyArray_DATA(samples);                              \
    npy_intp si = 0;                                                                       \
    for (npy_intp i = 0; i < n; ++i) {                                                     \
        double v = (double)LOAD(arr_data + i * stride);                                    \
        if (isnan_f64(v)) {                                                                \
            STORE(arr_data + i * stride, sd[si]);                                          \
            si++;                                                                          \
        }                                                                                  \
    }                                                                                      \
    Py_DECREF(samples);                                                                    \
    return 0;                                                                              \
}

DEFINE_NORM_1D_KERNEL(f16, LOAD_F16, STORE_F16)
DEFINE_NORM_1D_KERNEL(f32, LOAD_F32, STORE_F32)
DEFINE_NORM_1D_KERNEL(f64, LOAD_F64, STORE_F64)


/* =============================================================== */
/* Dispatch tables                                                   */
/* =============================================================== */

typedef npy_intp (*count_nan_2d_fn)(const char *, npy_intp, npy_intp,
                                    npy_intp, npy_intp);
typedef int (*mixup_2d_fn)(char *, npy_intp, npy_intp,
                           npy_intp, npy_intp,
                           const double *, bitgen_t *);
typedef int (*norm_1d_fn)(char *, npy_intp, npy_intp);


static count_nan_2d_fn
_pick_count_nan_2d(int typenum)
{
    switch (typenum) {
        case NPY_HALF:   return _count_nan_rows_2d_f16;
        case NPY_FLOAT:  return _count_nan_rows_2d_f32;
        case NPY_DOUBLE: return _count_nan_rows_2d_f64;
        default:         return NULL;
    }
}


static mixup_2d_fn
_pick_mixup_2d(int typenum)
{
    switch (typenum) {
        case NPY_HALF:   return _mixup_2d_kernel_f16;
        case NPY_FLOAT:  return _mixup_2d_kernel_f32;
        case NPY_DOUBLE: return _mixup_2d_kernel_f64;
        default:         return NULL;
    }
}


static norm_1d_fn
_pick_norm_1d(int typenum)
{
    switch (typenum) {
        case NPY_HALF:   return _norm_1d_kernel_f16;
        case NPY_FLOAT:  return _norm_1d_kernel_f32;
        case NPY_DOUBLE: return _norm_1d_kernel_f64;
        default:         return NULL;
    }
}


/* =============================================================== */
/* Mixup dispatchers                                                 */
/* =============================================================== */

static int _mixup_recursive(PyArrayObject *arr, double alpha,
                            PyObject *gen, bitgen_t *bg);


static int
_mixup_2d_dispatch(PyArrayObject *arr, double alpha,
                   PyObject *gen, bitgen_t *bg)
{
    int typenum = PyArray_TYPE(arr);
    count_nan_2d_fn count_fn = _pick_count_nan_2d(typenum);
    mixup_2d_fn mix_fn = _pick_mixup_2d(typenum);
    if (count_fn == NULL || mix_fn == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "mixupnd: unsupported dtype (need float16/32/64)");
        return -1;
    }

    npy_intp x = PyArray_DIM(arr, 0);
    npy_intp y = PyArray_DIM(arr, 1);
    npy_intp *strides = PyArray_STRIDES(arr);
    char *base = (char *)PyArray_DATA(arr);

    npy_intp n_nan = count_fn(base, x, y, strides[0], strides[1]);
    if (n_nan == 0) return 0;

    PyArrayObject *lam;
    if (alpha > 0.0) {
        lam = _draw_beta(gen, alpha, 1, n_nan);
        if (lam == NULL) return -1;
    } else {
        npy_intp lam_dims[2] = {1, n_nan};
        lam = (PyArrayObject *)PyArray_EMPTY(2, lam_dims, NPY_DOUBLE, 0);
        if (lam == NULL) return -1;
        double *p = (double *)PyArray_DATA(lam);
        for (npy_intp k = 0; k < n_nan; ++k) p[k] = 1.0;
    }

    int rc = mix_fn(base, x, y, strides[0], strides[1],
                    (double *)PyArray_DATA(lam), bg);
    Py_DECREF(lam);
    return rc;
}


static int
_mixup_3d_dispatch(PyArrayObject *arr, double alpha,
                   PyObject *gen, bitgen_t *bg)
{
    int typenum = PyArray_TYPE(arr);
    count_nan_2d_fn count_fn = _pick_count_nan_2d(typenum);
    mixup_2d_fn mix_fn = _pick_mixup_2d(typenum);
    if (count_fn == NULL || mix_fn == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "mixupnd: unsupported dtype (need float16/32/64)");
        return -1;
    }

    npy_intp x = PyArray_DIM(arr, 0);
    npy_intp y = PyArray_DIM(arr, 1);
    npy_intp z = PyArray_DIM(arr, 2);
    npy_intp *strides = PyArray_STRIDES(arr);
    char *base = (char *)PyArray_DATA(arr);

    npy_intp *n_nan = (npy_intp *)malloc((size_t)x * sizeof(npy_intp));
    if (n_nan == NULL) { PyErr_NoMemory(); return -1; }

    npy_intp max_n = 0;
    for (npy_intp i = 0; i < x; ++i) {
        npy_intp c = count_fn(base + i * strides[0],
                              y, z, strides[1], strides[2]);
        n_nan[i] = c;
        if (c > max_n) max_n = c;
    }

    if (max_n == 0) { free(n_nan); return 0; }

    PyArrayObject *lam;
    if (alpha > 0.0) {
        lam = _draw_beta(gen, alpha, x, max_n);
        if (lam == NULL) { free(n_nan); return -1; }
    } else {
        npy_intp lam_dims[2] = {x, max_n};
        lam = (PyArrayObject *)PyArray_EMPTY(2, lam_dims, NPY_DOUBLE, 0);
        if (lam == NULL) { free(n_nan); return -1; }
        double *p = (double *)PyArray_DATA(lam);
        npy_intp total = x * max_n;
        for (npy_intp k = 0; k < total; ++k) p[k] = 1.0;
    }

    double *lam_data = (double *)PyArray_DATA(lam);
    for (npy_intp i = 0; i < x; ++i) {
        if (n_nan[i] == 0) continue;
        int rc = mix_fn(base + i * strides[0],
                        y, z, strides[1], strides[2],
                        lam_data + i * max_n, bg);
        if (rc < 0) {
            Py_DECREF(lam); free(n_nan);
            return -1;
        }
    }

    Py_DECREF(lam);
    free(n_nan);
    return 0;
}


static int
_mixup_recursive(PyArrayObject *arr, double alpha,
                 PyObject *gen, bitgen_t *bg)
{
    int ndim = PyArray_NDIM(arr);
    if (ndim == 2) return _mixup_2d_dispatch(arr, alpha, gen, bg);
    if (ndim == 3) return _mixup_3d_dispatch(arr, alpha, gen, bg);
    /* ndim > 3: recurse over the leading axis. obs_axis stays at -2
     * for the inner subarray since slicing along axis 0 doesn't
     * affect the position of the obs axis from the back. */
    npy_intp x = PyArray_DIM(arr, 0);
    for (npy_intp i = 0; i < x; ++i) {
        PyObject *idx = PyLong_FromSsize_t((Py_ssize_t)i);
        if (idx == NULL) return -1;
        PyArrayObject *sub = (PyArrayObject *)PyObject_GetItem((PyObject *)arr, idx);
        Py_DECREF(idx);
        if (sub == NULL) return -1;
        int rc = _mixup_recursive(sub, alpha, gen, bg);
        Py_DECREF(sub);
        if (rc < 0) return -1;
    }
    return 0;
}


/* =============================================================== */
/* Public: mixupnd                                                   */
/* =============================================================== */

static int
_is_supported_float_dtype(int typenum)
{
    return typenum == NPY_HALF || typenum == NPY_FLOAT || typenum == NPY_DOUBLE;
}


static PyObject *
mixupnd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *arr_obj;
    int obs_axis;
    double alpha = 1.0;
    int seed = -1;
    static char *kwlist[] = {"arr", "obs_axis", "alpha", "seed", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oi|di", kwlist,
            &arr_obj, &obs_axis, &alpha, &seed)) {
        return NULL;
    }

    if (!PyArray_Check(arr_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "mixupnd: arr must be a numpy ndarray");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)arr_obj;
    if (!_is_supported_float_dtype(PyArray_TYPE(arr))) {
        PyErr_SetString(PyExc_TypeError,
                        "mixupnd: arr must be float16, float32, or float64");
        return NULL;
    }

    int ndim = PyArray_NDIM(arr);
    if (ndim < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot apply mixup to a 1-dimensional array");
        return NULL;
    }

    int ax = obs_axis;
    if (ax < 0) ax += ndim;
    if (ax < 0 || ax >= ndim) {
        PyErr_Format(PyExc_ValueError,
                     "obs_axis %d out of range for ndim %d", obs_axis, ndim);
        return NULL;
    }

    /* Swap obs axis to -2 so the LAST axis is "features". */
    PyArrayObject *arr_in;
    if (ax == ndim - 2) {
        arr_in = arr;
        Py_INCREF(arr_in);
    } else {
        arr_in = (PyArrayObject *)PyArray_SwapAxes(arr, ax, ndim - 2);
        if (arr_in == NULL) return NULL;
    }

    PyObject *gen = _make_generator(seed);
    if (gen == NULL) { Py_DECREF(arr_in); return NULL; }
    bitgen_t *bg = _get_bitgen(gen);
    if (bg == NULL) {
        Py_DECREF(gen); Py_DECREF(arr_in);
        return NULL;
    }

    int rc = _mixup_recursive(arr_in, alpha, gen, bg);

    Py_DECREF(gen);
    Py_DECREF(arr_in);
    if (rc < 0) return NULL;
    Py_RETURN_NONE;
}


/* =============================================================== */
/* normnd                                                            */
/* =============================================================== */

static int
_norm_recursive(PyArrayObject *arr)
{
    int ndim = PyArray_NDIM(arr);
    if (ndim == 1) {
        norm_1d_fn fn = _pick_norm_1d(PyArray_TYPE(arr));
        if (fn == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "normnd: unsupported dtype (need float16/32/64)");
            return -1;
        }
        return fn((char *)PyArray_DATA(arr),
                  PyArray_DIM(arr, 0),
                  PyArray_STRIDE(arr, 0));
    }
    npy_intp x = PyArray_DIM(arr, 0);
    for (npy_intp i = 0; i < x; ++i) {
        PyObject *idx = PyLong_FromSsize_t((Py_ssize_t)i);
        if (idx == NULL) return -1;
        PyArrayObject *sub = (PyArrayObject *)PyObject_GetItem((PyObject *)arr, idx);
        Py_DECREF(idx);
        if (sub == NULL) return -1;
        int rc = _norm_recursive(sub);
        Py_DECREF(sub);
        if (rc < 0) return -1;
    }
    return 0;
}


static PyObject *
normnd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *arr_obj;
    int obs_axis = -1;
    static char *kwlist[] = {"arr", "obs_axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|i", kwlist, &arr_obj, &obs_axis)) {
        return NULL;
    }
    if (!PyArray_Check(arr_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "normnd: arr must be a numpy ndarray");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)arr_obj;
    if (!_is_supported_float_dtype(PyArray_TYPE(arr))) {
        PyErr_SetString(PyExc_TypeError,
                        "normnd: arr must be float16, float32, or float64");
        return NULL;
    }
    int ndim = PyArray_NDIM(arr);
    if (ndim < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot apply norm to a 0-dimensional array");
        return NULL;
    }

    int ax = obs_axis;
    if (ax < 0) ax += ndim;
    if (ax < 0 || ax >= ndim) {
        PyErr_Format(PyExc_ValueError,
                     "obs_axis %d out of range for ndim %d", obs_axis, ndim);
        return NULL;
    }

    PyArrayObject *arr_in;
    if (ax == ndim - 1) {
        arr_in = arr;
        Py_INCREF(arr_in);
    } else {
        arr_in = (PyArrayObject *)PyArray_SwapAxes(arr, ax, ndim - 1);
        if (arr_in == NULL) return NULL;
    }

    int rc = _norm_recursive(arr_in);
    Py_DECREF(arr_in);
    if (rc < 0) return NULL;
    Py_RETURN_NONE;
}


/* =============================================================== */
/* Module init                                                       */
/* =============================================================== */

static PyMethodDef mixup_methods[] = {
    {"mixupnd", (PyCFunction)mixupnd, METH_VARARGS | METH_KEYWORDS,
     "mixupnd(arr, obs_axis, alpha=1.0, seed=-1)\n\n"
     "Replace each row (along obs_axis) containing any NaN with\n"
     "``lam * donor1 + (1 - lam) * donor2``, where donors are distinct\n"
     "non-NaN rows in the same batch and ``lam ~ Beta(alpha, alpha)``.\n"
     "Modifies arr in place; arr must be float16/float32/float64 with\n"
     "ndim >= 2. Mixing arithmetic is performed in double regardless\n"
     "of input precision."},
    {"normnd", (PyCFunction)normnd, METH_VARARGS | METH_KEYWORDS,
     "normnd(arr, obs_axis=-1)\n\n"
     "Replace each NaN with a draw from Normal(mean, std) of the\n"
     "non-NaN values in the same 1-D slice along ``obs_axis``.\n"
     "arr must be float16/float32/float64."},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef mixup_module = {
    PyModuleDef_HEAD_INIT,
    "mixup",
    "NaN imputation via mixup or per-slice Normal. C re-implementation of "
    "the previous Cython kernel; supports float16/32/64 via the LOAD_F* / "
    "STORE_F* macros from shared/meanvar_core.",
    -1, mixup_methods,
    NULL, NULL, NULL, NULL,
};


PyMODINIT_FUNC
PyInit_mixup(void)
{
    PyObject *m = PyModule_Create(&mixup_module);
    if (m == NULL) return NULL;
    import_array();
    return m;
}
