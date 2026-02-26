#ifndef PY_ARRAY_UNIQUE_SYMBOL
#error "PY_ARRAY_UNIQUE_SYMBOL must be defined when building meanvar_core.c"
#endif
#define NO_IMPORT_ARRAY
#include "meanvar_core.h"

#define DEFINE_MEANVAR(NAME, ACC_T, LOAD, STORE, ISNAN, NAN_OUT)                        \
void meanvar_##NAME(                                                                    \
    char **args,                                                                        \
    const npy_intp *dimensions,                                                         \
    const npy_intp *steps,                                                              \
    void *extra)                                                                         \
{                                                                                       \
    char *in = args[0], *ddofp = args[1], *out_mean = args[2], *out_var = args[3];      \
    const npy_intp nloops = dimensions[0];                                              \
    const npy_intp len    = dimensions[1];                                              \
    const npy_intp step_in  = steps[0];                                                 \
    const npy_intp step_dd  = steps[1];                                                 \
    const npy_intp step_m   = steps[2];                                                 \
    const npy_intp step_v   = steps[3];                                                 \
    const npy_intp inner_in = steps[4];                                                 \
    for (npy_intp i = 0; i < nloops; ++i, in += step_in, ddofp += step_dd, out_mean += step_m, out_var += step_v) { \
        ACC_T sum = 0;                                                                   \
        ACC_T sumsq = 0;                                                                 \
        npy_intp cnt = 0;                                                                \
        for (npy_intp j = 0; j < len; ++j) {                                            \
            ACC_T v = (ACC_T)LOAD(in + j * inner_in);                                    \
            if (!ISNAN(v)) { sum += v; sumsq += v * v; ++cnt; }                          \
        }                                                                                \
        if (cnt == 0) {                                                                  \
            STORE(out_mean, NAN_OUT);                                                    \
            STORE(out_var,  NAN_OUT);                                                    \
            continue;                                                                    \
        }                                                                                \
        ACC_T mean = sum / (ACC_T)cnt;                                                   \
        ACC_T varsum = sumsq - (sum * sum) / (ACC_T)cnt;                                 \
        if (varsum < (ACC_T)0) varsum = (ACC_T)0;                                        \
        ACC_T ddof = (ACC_T)LOAD(ddofp);                                                 \
        if (ISNAN(ddof)) ddof = (ACC_T)1;                                                \
        if (ddof < (ACC_T)0) ddof = (ACC_T)0;                                            \
        ACC_T denom = (ACC_T)cnt - ddof;                                                 \
        ACC_T var = (denom > (ACC_T)0) ? (varsum / denom) : NAN_OUT;                     \
        STORE(out_mean, mean);                                                           \
        STORE(out_var,  var);                                                            \
    }                                                                                    \
    (void)extra;                                                                         \
}

DEFINE_MEANVAR(half,       float,       LOAD_F16, STORE_F16, isnan_f32, (float)NPY_NAN)
DEFINE_MEANVAR(float,      float,       LOAD_F32, STORE_F32, isnan_f32, (float)NPY_NAN)
DEFINE_MEANVAR(double,     long double, LOAD_F64, STORE_F64, isnan_f64, NPY_NAN)
DEFINE_MEANVAR(longdouble, long double, LOAD_F80, STORE_F80, isnan_f80, (long double)NPY_NAN)

static NPY_INLINE int
validate_reduce_2d(PyArrayObject *arr2d, const char *name,
                   int *typ, npy_intp *outer, npy_intp *len)
{
    if (PyArray_NDIM(arr2d) != 2) {
        PyErr_Format(PyExc_ValueError, "%s expects a 2D array", name);
        return -1;
    }
    int t = PyArray_TYPE(arr2d);
    if (!PyTypeNum_ISFLOAT(t)) {
        PyErr_Format(PyExc_TypeError, "%s only supports float dtypes", name);
        return -1;
    }
    *typ = t;
    *outer = PyArray_DIM(arr2d, 0);
    *len = PyArray_DIM(arr2d, 1);
    return 0;
}

#define DEFINE_REDUCE_MEANSTD(NAME, ACC_T, LOAD, STORE, ISNAN, NAN_OUT, SQRTFUN)       \
static NPY_INLINE void reduce_meanstd_##NAME(                                          \
    const char *in, const npy_intp nloops, const npy_intp len,                          \
    const npy_intp step_in, const npy_intp inner_in,                                   \
    char *out_mean, const npy_intp step_mean,                                          \
    char *out_std, const npy_intp step_std,                                             \
    ACC_T ddof)                                                                        \
{                                                                                      \
    if (ISNAN(ddof)) ddof = (ACC_T)1;                                                   \
    if (ddof < (ACC_T)0) ddof = (ACC_T)0;                                               \
    char *m = out_mean;                                                                \
    char *s = out_std;                                                                 \
    for (npy_intp i = 0; i < nloops; ++i, in += step_in) {                              \
        ACC_T sum = 0;                                                                 \
        ACC_T sumsq = 0;                                                               \
        npy_intp cnt = 0;                                                              \
        for (npy_intp j = 0; j < len; ++j) {                                            \
            ACC_T v = (ACC_T)LOAD(in + j * inner_in);                                  \
            if (!ISNAN(v)) { sum += v; sumsq += v * v; ++cnt; }                        \
        }                                                                              \
        if (cnt == 0) {                                                                \
            if (m) { STORE(m, NAN_OUT); m += step_mean; }                              \
            if (s) { STORE(s, NAN_OUT); s += step_std; }                               \
            continue;                                                                  \
        }                                                                              \
        ACC_T mean = sum / (ACC_T)cnt;                                                 \
        ACC_T varsum = sumsq - (sum * sum) / (ACC_T)cnt;                               \
        if (varsum < (ACC_T)0) varsum = (ACC_T)0;                                      \
        ACC_T denom = (ACC_T)cnt - ddof;                                               \
        ACC_T var = (denom > (ACC_T)0) ? (varsum / denom) : NAN_OUT;                   \
        if (m) { STORE(m, mean); m += step_mean; }                                     \
        if (s) {                                                                       \
            if (denom > (ACC_T)0) {                                                    \
                STORE(s, (ACC_T)SQRTFUN(var));                                         \
            } else {                                                                   \
                STORE(s, NAN_OUT);                                                     \
            }                                                                          \
            s += step_std;                                                             \
        }                                                                              \
    }                                                                                  \
}

DEFINE_REDUCE_MEANSTD(half,       float,       LOAD_F16, STORE_F16, isnan_f32, (float)NPY_NAN, sqrtf)
DEFINE_REDUCE_MEANSTD(float,      float,       LOAD_F32, STORE_F32, isnan_f32, (float)NPY_NAN, sqrtf)
DEFINE_REDUCE_MEANSTD(double,     long double, LOAD_F64, STORE_F64, isnan_f64, NPY_NAN,         sqrtl)
DEFINE_REDUCE_MEANSTD(longdouble, long double, LOAD_F80, STORE_F80, isnan_f80, (long double)NPY_NAN, sqrtl)

int meanvar_reduce_2d(PyArrayObject *arr2d, double ddof,
                      PyArrayObject **out_mean, PyArrayObject **out_var)
{
    int typ = 0;
    npy_intp outer = 0, len = 0;
    if (validate_reduce_2d(arr2d, "meanvar_reduce_2d", &typ, &outer, &len) < 0) {
        return -1;
    }
    npy_intp odims[1] = {outer};
    PyArrayObject *mean = (PyArrayObject *)PyArray_SimpleNew(1, odims, typ);
    PyArrayObject *var = (PyArrayObject *)PyArray_SimpleNew(1, odims, typ);
    if (!mean || !var) {
        Py_XDECREF(mean);
        Py_XDECREF(var);
        return -1;
    }

    char *args[4];
    npy_intp dims[2] = {outer, len};
    npy_intp steps[5];
    steps[0] = PyArray_STRIDE(arr2d, 0);
    steps[1] = 0;
    steps[2] = PyArray_STRIDE(mean, 0);
    steps[3] = PyArray_STRIDE(var, 0);
    steps[4] = PyArray_STRIDE(arr2d, 1);
    args[0] = PyArray_BYTES(arr2d);
    args[2] = PyArray_BYTES(mean);
    args[3] = PyArray_BYTES(var);

    switch (typ) {
        case NPY_HALF: {
            npy_half dd = npy_float_to_half((float)ddof);
            args[1] = (char *)&dd;
            meanvar_half(args, dims, steps, NULL);
            break;
        }
        case NPY_FLOAT: {
            float dd = (float)ddof;
            args[1] = (char *)&dd;
            meanvar_float(args, dims, steps, NULL);
            break;
        }
        case NPY_DOUBLE: {
            double dd = (double)ddof;
            args[1] = (char *)&dd;
            meanvar_double(args, dims, steps, NULL);
            break;
        }
        case NPY_LONGDOUBLE: {
            long double dd = (long double)ddof;
            args[1] = (char *)&dd;
            meanvar_longdouble(args, dims, steps, NULL);
            break;
        }
        default:
            Py_DECREF(mean);
            Py_DECREF(var);
            PyErr_SetString(PyExc_TypeError, "unsupported dtype");
            return -1;
    }

    *out_mean = mean;
    *out_var = var;
    return 0;
}

int std_reduce_2d(PyArrayObject *arr2d, double ddof,
                  PyArrayObject **out_std)
{
    int typ = 0;
    npy_intp outer = 0, len = 0;
    if (validate_reduce_2d(arr2d, "std_reduce_2d", &typ, &outer, &len) < 0) {
        return -1;
    }
    npy_intp odims[1] = {outer};
    PyArrayObject *std = (PyArrayObject *)PyArray_SimpleNew(1, odims, typ);
    if (!std) return -1;

    const char *in = PyArray_BYTES(arr2d);
    const npy_intp step_in = PyArray_STRIDE(arr2d, 0);
    const npy_intp inner_in = PyArray_STRIDE(arr2d, 1);
    const npy_intp step_std = PyArray_STRIDE(std, 0);
    char *out_std_ptr = PyArray_BYTES(std);

    switch (typ) {
        case NPY_HALF: {
            float dd = (float)ddof;
            reduce_meanstd_half(in, outer, len, step_in, inner_in, NULL, 0, out_std_ptr, step_std, dd);
            break;
        }
        case NPY_FLOAT: {
            float dd = (float)ddof;
            reduce_meanstd_float(in, outer, len, step_in, inner_in, NULL, 0, out_std_ptr, step_std, dd);
            break;
        }
        case NPY_DOUBLE: {
            long double dd = (long double)ddof;
            reduce_meanstd_double(in, outer, len, step_in, inner_in, NULL, 0, out_std_ptr, step_std, dd);
            break;
        }
        case NPY_LONGDOUBLE: {
            long double dd = (long double)ddof;
            reduce_meanstd_longdouble(in, outer, len, step_in, inner_in, NULL, 0, out_std_ptr, step_std, dd);
            break;
        }
        default:
            Py_DECREF(std);
            PyErr_SetString(PyExc_TypeError, "unsupported dtype");
            return -1;
    }

    *out_std = std;
    return 0;
}

int meanstd_reduce_2d(PyArrayObject *arr2d, double ddof,
                      PyArrayObject **out_mean, PyArrayObject **out_std)
{
    int typ = 0;
    npy_intp outer = 0, len = 0;
    if (validate_reduce_2d(arr2d, "meanstd_reduce_2d", &typ, &outer, &len) < 0) {
        return -1;
    }
    npy_intp odims[1] = {outer};
    PyArrayObject *mean = (PyArrayObject *)PyArray_SimpleNew(1, odims, typ);
    PyArrayObject *std = (PyArrayObject *)PyArray_SimpleNew(1, odims, typ);
    if (!mean || !std) {
        Py_XDECREF(mean);
        Py_XDECREF(std);
        return -1;
    }

    const char *in = PyArray_BYTES(arr2d);
    const npy_intp step_in = PyArray_STRIDE(arr2d, 0);
    const npy_intp inner_in = PyArray_STRIDE(arr2d, 1);
    const npy_intp step_mean = PyArray_STRIDE(mean, 0);
    const npy_intp step_std = PyArray_STRIDE(std, 0);
    char *out_mean_ptr = PyArray_BYTES(mean);
    char *out_std_ptr = PyArray_BYTES(std);

    switch (typ) {
        case NPY_HALF: {
            float dd = (float)ddof;
            reduce_meanstd_half(in, outer, len, step_in, inner_in, out_mean_ptr, step_mean, out_std_ptr, step_std, dd);
            break;
        }
        case NPY_FLOAT: {
            float dd = (float)ddof;
            reduce_meanstd_float(in, outer, len, step_in, inner_in, out_mean_ptr, step_mean, out_std_ptr, step_std, dd);
            break;
        }
        case NPY_DOUBLE: {
            long double dd = (long double)ddof;
            reduce_meanstd_double(in, outer, len, step_in, inner_in, out_mean_ptr, step_mean, out_std_ptr, step_std, dd);
            break;
        }
        case NPY_LONGDOUBLE: {
            long double dd = (long double)ddof;
            reduce_meanstd_longdouble(in, outer, len, step_in, inner_in, out_mean_ptr, step_mean, out_std_ptr, step_std, dd);
            break;
        }
        default:
            Py_DECREF(mean);
            Py_DECREF(std);
            PyErr_SetString(PyExc_TypeError, "unsupported dtype");
            return -1;
    }

    *out_mean = mean;
    *out_std = std;
    return 0;
}

/* ---------- numpy function cache ---------- */
static PyObject *np_func_cache = NULL;

PyObject *
get_numpy_func_cached(const char *name)
{
    if (!np_func_cache) {
        np_func_cache = PyDict_New();
        if (!np_func_cache) return NULL;
    }
    PyObject *key = PyUnicode_FromString(name);
    if (!key) return NULL;
    PyObject *func = PyDict_GetItemWithError(np_func_cache, key);
    if (func) {
        Py_INCREF(func);
        Py_DECREF(key);
        return func;
    }
    if (PyErr_Occurred()) { Py_DECREF(key); return NULL; }
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy) { Py_DECREF(key); return NULL; }
    func = PyObject_GetAttrString(numpy, name);
    Py_DECREF(numpy);
    if (!func) { Py_DECREF(key); return NULL; }
    if (PyDict_SetItem(np_func_cache, key, func) < 0) {
        Py_DECREF(key);
        Py_DECREF(func);
        return NULL;
    }
    Py_DECREF(key);
    return func;
}

/* ---------- nan reduction helpers ---------- */
typedef enum {
    NANREDUCE_MEANVAR = 0,
    NANREDUCE_STD = 1,
    NANREDUCE_MEANSTD = 2
} NanReduceMode;

static NPY_INLINE PyObject *
build_nan_kwargs(PyObject *axis_obj, PyObject *dtype_obj, PyObject *where_obj)
{
    PyObject *kwargs = PyDict_New();
    if (!kwargs) return NULL;
    if (axis_obj && axis_obj != Py_None) {
        if (PyDict_SetItemString(kwargs, "axis", axis_obj) < 0) {
            Py_DECREF(kwargs);
            return NULL;
        }
    }
    if (dtype_obj && dtype_obj != Py_None) {
        if (PyDict_SetItemString(kwargs, "dtype", dtype_obj) < 0) {
            Py_DECREF(kwargs);
            return NULL;
        }
    }
    if (where_obj && where_obj != Py_None) {
        if (PyDict_SetItemString(kwargs, "where", where_obj) < 0) {
            Py_DECREF(kwargs);
            return NULL;
        }
    }
    return kwargs;
}

static NPY_INLINE PyObject *
call_nanfunc(PyObject *func, PyObject *args, PyObject *kwargs,
             int use_ddof, double ddof)
{
    PyObject *kw = kwargs;
    if (use_ddof) {
        kw = PyDict_Copy(kwargs);
        if (!kw) return NULL;
        PyObject *ddof_obj = PyFloat_FromDouble(ddof);
        if (!ddof_obj) {
            Py_DECREF(kw);
            return NULL;
        }
        if (PyDict_SetItemString(kw, "ddof", ddof_obj) < 0) {
            Py_DECREF(ddof_obj);
            Py_DECREF(kw);
            return NULL;
        }
        Py_DECREF(ddof_obj);
    }
    PyObject *res = PyObject_Call(func, args, kw);
    if (use_ddof) Py_DECREF(kw);
    return res;
}

static NPY_INLINE PyObject *
reshape_reduction_out(PyArrayObject *arr, int keep_nd, npy_intp *keep_shape)
{
    if (keep_nd == 0) {
        PyObject *out = PyArray_Squeeze(arr);
        Py_DECREF(arr);
        return out;
    }
    PyArray_Dims nds; nds.ptr = keep_shape; nds.len = keep_nd;
    PyObject *out = PyArray_Newshape(arr, &nds, NPY_CORDER);
    Py_DECREF(arr);
    return out;
}

static NPY_INLINE int
normalize_axes(PyObject *axis_obj, int ndim, int **axes_out, int *naxes_out, int *axis_none)
{
    *axes_out = NULL;
    *naxes_out = 0;
    *axis_none = 0;
    if (axis_obj == NULL || axis_obj == Py_None) {
        *axis_none = 1;
        if (ndim == 0) return 0;
        int *axes = (int *)PyMem_Malloc(sizeof(int) * (size_t)ndim);
        if (!axes) { PyErr_NoMemory(); return -1; }
        for (int i = 0; i < ndim; ++i) axes[i] = i;
        *axes_out = axes;
        *naxes_out = ndim;
        return 0;
    }

    if (PyLong_Check(axis_obj)) {
        long axl = PyLong_AsLong(axis_obj);
        if (axl == -1 && PyErr_Occurred()) return -1;
        int ax = (int)axl;
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim) {
            PyErr_SetString(PyExc_ValueError, "axis out of bounds");
            return -1;
        }
        int *axes = (int *)PyMem_Malloc(sizeof(int));
        if (!axes) { PyErr_NoMemory(); return -1; }
        axes[0] = ax;
        *axes_out = axes;
        *naxes_out = 1;
        return 0;
    }

    if (!PySequence_Check(axis_obj)) {
        PyErr_SetString(PyExc_TypeError, "axis must be int or sequence of ints");
        return -1;
    }

    PyObject *seq = PySequence_Fast(axis_obj, "axis must be sequence");
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0) {
        Py_DECREF(seq);
        *naxes_out = 0;
        return 0;
    }
    int *axes = (int *)PyMem_Malloc(sizeof(int) * (size_t)n);
    if (!axes) { Py_DECREF(seq); PyErr_NoMemory(); return -1; }
    char *seen = (char *)PyMem_Calloc((size_t)ndim, sizeof(char));
    if (!seen) { Py_DECREF(seq); PyMem_Free(axes); PyErr_NoMemory(); return -1; }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, i);
        long axl = PyLong_AsLong(it);
        if (axl == -1 && PyErr_Occurred()) { PyMem_Free(seen); PyMem_Free(axes); Py_DECREF(seq); return -1; }
        int ax = (int)axl;
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim) {
            PyMem_Free(seen); PyMem_Free(axes); Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "axis out of bounds");
            return -1;
        }
        if (seen[ax]) {
            PyMem_Free(seen); PyMem_Free(axes); Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "duplicate axis");
            return -1;
        }
        seen[ax] = 1;
        axes[i] = ax;
    }
    PyMem_Free(seen);
    Py_DECREF(seq);
    *axes_out = axes;
    *naxes_out = (int)n;
    return 0;
}

static NPY_INLINE int
nanreduce_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
               PyObject *where_obj, double ddof, NanReduceMode mode,
               PyObject **mean_out, PyObject **var_out, PyObject **std_out,
               int **axes_out, int *naxes_out)
{
    PyArrayObject *arr_base = NULL;
    PyArrayObject *arr_cast = NULL;
    PyArrayObject *transposed = NULL;
    PyArrayObject *work2d = NULL;
    PyArrayObject *mean_arr = NULL;
    PyArrayObject *var_arr = NULL;
    PyArrayObject *std_arr = NULL;
    PyObject *mean = NULL;
    PyObject *var = NULL;
    PyObject *std = NULL;
    PyObject *args = NULL;
    PyObject *kwargs = NULL;
    PyObject *nanmean = NULL;
    PyObject *nanvar = NULL;
    PyObject *nanstd = NULL;
    PyArray_Descr *descr = NULL;
    npy_intp *keep_shape = NULL;
    int *axes = NULL;
    int naxes = 0;
    int axis_none = 0;
    int ndim = 0;
    int use_numpy = 0;
    int status = -1;

    if (mean_out) *mean_out = NULL;
    if (var_out) *var_out = NULL;
    if (std_out) *std_out = NULL;
    if (axes_out) *axes_out = NULL;
    if (naxes_out) *naxes_out = 0;

    arr_base = (PyArrayObject *)PyArray_View((PyArrayObject *)self_obj, NULL, &PyArray_Type);
    if (!arr_base) goto cleanup;

    ndim = PyArray_NDIM(arr_base);
    if (normalize_axes(axis_obj, ndim, &axes, &naxes, &axis_none) < 0) goto cleanup;
    (void)axis_none;

    if (where_obj && where_obj != Py_None && where_obj != Py_True) use_numpy = 1;
    if (naxes == 0) use_numpy = 1;

    if (!use_numpy) {
        if (dtype_obj && dtype_obj != Py_None) {
            if (!PyArray_DescrConverter(dtype_obj, &descr)) goto cleanup;
            if (!PyTypeNum_ISFLOAT(descr->type_num)) {
                use_numpy = 1;
                Py_DECREF(descr);
                descr = NULL;
            } else {
                arr_cast = (PyArrayObject *)PyArray_FromAny((PyObject *)arr_base, descr, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                descr = NULL;
                if (!arr_cast) goto cleanup;
            }
        } else {
            int typ = PyArray_TYPE(arr_base);
            if (PyTypeNum_ISFLOAT(typ)) {
                Py_INCREF(arr_base);
                arr_cast = arr_base;
            } else if (PyTypeNum_ISCOMPLEX(typ)) {
                use_numpy = 1;
            } else {
                descr = PyArray_DescrFromType(NPY_DOUBLE);
                if (!descr) goto cleanup;
                arr_cast = (PyArrayObject *)PyArray_FromAny((PyObject *)arr_base, descr, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                descr = NULL;
                if (!arr_cast) goto cleanup;
            }
        }
    }

    if (use_numpy) {
        args = PyTuple_New(1);
        if (!args) goto cleanup;
        Py_INCREF(arr_base);
        PyTuple_SET_ITEM(args, 0, (PyObject *)arr_base);

        kwargs = build_nan_kwargs(axis_obj, dtype_obj, where_obj);
        if (!kwargs) goto cleanup;

        if (mode == NANREDUCE_MEANVAR || mode == NANREDUCE_MEANSTD) {
            nanmean = get_numpy_func_cached("nanmean");
            if (!nanmean) goto cleanup;
            mean = call_nanfunc(nanmean, args, kwargs, 0, ddof);
            if (!mean) goto cleanup;
        }
        if (mode == NANREDUCE_MEANVAR) {
            nanvar = get_numpy_func_cached("nanvar");
            if (!nanvar) goto cleanup;
            var = call_nanfunc(nanvar, args, kwargs, 1, ddof);
            if (!var) goto cleanup;
        }
        if (mode == NANREDUCE_STD || mode == NANREDUCE_MEANSTD) {
            nanstd = get_numpy_func_cached("nanstd");
            if (!nanstd) goto cleanup;
            std = call_nanfunc(nanstd, args, kwargs, 1, ddof);
            if (!std) goto cleanup;
        }
        status = 0;
        goto cleanup;
    }

    int keep_nd = ndim - naxes;
    if (naxes == 1 && axes[0] == ndim - 1) {
        Py_INCREF(arr_cast);
        transposed = arr_cast;
    } else {
        char *drop = (char *)PyMem_Calloc((size_t)ndim, sizeof(char));
        if (!drop) { PyErr_NoMemory(); goto cleanup; }
        for (int j = 0; j < naxes; ++j) {
            int ax = axes[j];
            if (ax >= 0 && ax < ndim) drop[ax] = 1;
        }
        int *perm = (int *)PyMem_Malloc(sizeof(int) * (size_t)ndim);
        if (!perm) { PyMem_Free(drop); PyErr_NoMemory(); goto cleanup; }
        int w = 0;
        for (int i = 0; i < ndim; ++i) if (!drop[i]) perm[w++] = i;
        for (int j = 0; j < naxes; ++j) perm[w++] = axes[j];
        PyMem_Free(drop);
        npy_intp *perm_i = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)ndim);
        if (!perm_i) { PyMem_Free(perm); PyErr_NoMemory(); goto cleanup; }
        for (int i = 0; i < ndim; ++i) perm_i[i] = (npy_intp)perm[i];
        PyArray_Dims pd; pd.ptr = perm_i; pd.len = ndim;
        transposed = (PyArrayObject *)PyArray_Transpose(arr_cast, &pd);
        PyMem_Free(perm_i);
        PyMem_Free(perm);
        if (!transposed) goto cleanup;
    }

    if (keep_nd > 0) {
        keep_shape = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)keep_nd);
        if (!keep_shape) { PyErr_NoMemory(); goto cleanup; }
        for (int i = 0; i < keep_nd; ++i) keep_shape[i] = PyArray_DIM(transposed, i);
    }

    npy_intp len = 1;
    for (int i = keep_nd; i < keep_nd + naxes; ++i) len *= PyArray_DIM(transposed, i);
    npy_intp outer = 1;
    for (int i = 0; i < keep_nd; ++i) outer *= keep_shape[i];

    npy_intp dims2[2] = {outer, len};
    PyArray_Dims nds2; nds2.ptr = dims2; nds2.len = 2;
    work2d = (PyArrayObject *)PyArray_Newshape(transposed, &nds2, NPY_CORDER);
    Py_DECREF(transposed);
    transposed = NULL;
    if (!work2d) goto cleanup;

    if (mode == NANREDUCE_MEANVAR) {
        if (meanvar_reduce_2d(work2d, ddof, &mean_arr, &var_arr) < 0) goto cleanup;
    } else if (mode == NANREDUCE_STD) {
        if (std_reduce_2d(work2d, ddof, &std_arr) < 0) goto cleanup;
    } else {
        if (meanstd_reduce_2d(work2d, ddof, &mean_arr, &std_arr) < 0) goto cleanup;
    }
    Py_DECREF(work2d);
    work2d = NULL;

    if (mode == NANREDUCE_MEANVAR) {
        mean = reshape_reduction_out(mean_arr, keep_nd, keep_shape);
        mean_arr = NULL;
        var = reshape_reduction_out(var_arr, keep_nd, keep_shape);
        var_arr = NULL;
        if (!mean || !var) goto cleanup;
    } else if (mode == NANREDUCE_STD) {
        std = reshape_reduction_out(std_arr, keep_nd, keep_shape);
        std_arr = NULL;
        if (!std) goto cleanup;
    } else {
        mean = reshape_reduction_out(mean_arr, keep_nd, keep_shape);
        mean_arr = NULL;
        std = reshape_reduction_out(std_arr, keep_nd, keep_shape);
        std_arr = NULL;
        if (!mean || !std) goto cleanup;
    }

    status = 0;

cleanup:
    if (status == 0) {
        if (mean_out) *mean_out = mean;
        if (var_out) *var_out = var;
        if (std_out) *std_out = std;
        if (axes_out) {
            *axes_out = axes;
        } else {
            PyMem_Free(axes);
        }
        if (naxes_out) *naxes_out = naxes;
    } else {
        Py_XDECREF(mean);
        Py_XDECREF(var);
        Py_XDECREF(std);
        PyMem_Free(axes);
    }

    Py_XDECREF(mean_arr);
    Py_XDECREF(var_arr);
    Py_XDECREF(std_arr);
    Py_XDECREF(work2d);
    Py_XDECREF(transposed);
    Py_XDECREF(arr_base);
    Py_XDECREF(arr_cast);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_XDECREF(nanmean);
    Py_XDECREF(nanvar);
    Py_XDECREF(nanstd);
    Py_XDECREF(descr);
    PyMem_Free(keep_shape);
    return status;
}

int
nanmeanvar_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
                PyObject *where_obj, double ddof,
                PyObject **mean_out, PyObject **var_out,
                int **axes_out, int *naxes_out)
{
    return nanreduce_core(self_obj, axis_obj, dtype_obj, where_obj, ddof,
                          NANREDUCE_MEANVAR, mean_out, var_out, NULL,
                          axes_out, naxes_out);
}

int
nanstd_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
            PyObject *where_obj, double ddof,
            PyObject **std_out, int **axes_out, int *naxes_out)
{
    return nanreduce_core(self_obj, axis_obj, dtype_obj, where_obj, ddof,
                          NANREDUCE_STD, NULL, NULL, std_out,
                          axes_out, naxes_out);
}

int
nanmeanstd_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
                PyObject *where_obj, double ddof,
                PyObject **mean_out, PyObject **std_out,
                int **axes_out, int *naxes_out)
{
    return nanreduce_core(self_obj, axis_obj, dtype_obj, where_obj, ddof,
                          NANREDUCE_MEANSTD, mean_out, NULL, std_out,
                          axes_out, naxes_out);
}
