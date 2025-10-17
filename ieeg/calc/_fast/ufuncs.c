#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/halffloat.h>

#ifndef NPY_INLINE
#define NPY_INLINE inline
#endif

#define NPY_NAN_HALF (npy_half)NPY_NAN

static PyMethodDef Meandiff_Methods[] = {
    {NULL, NULL, 0, NULL}
};

/* Macro-generated sums_and_counts for all dtypes */
#define ISNAN_VAL_F32(v) ((v) != (v))
#define ISNAN_VAL_F64(v) ((v) != (v))
#define ISNAN_VAL_F80(v) ((v) != (v))

#define TO_ACC_FROM_HALF(v) ((float)npy_half_to_float((v)))
#define TO_ACC_FROM_F32(v)  (v)
#define TO_ACC_FROM_F64(v)  (v)
#define TO_ACC_FROM_F80(v)  (v)

#define DEFINE_SUMS_AND_COUNTS(NAME, IN_T, ACC_T, TO_ACC, ISNAN_VAL)                               \
static NPY_INLINE void sums_and_counts_##NAME(const char *inl, const char *ins,                    \
    const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl,                           \
    const npy_intp innersteps, ACC_T * const suml, ACC_T * const sums,                             \
    npy_intp * const countl, npy_intp * const counts)                                              \
{                                                                                                  \
    for (npy_intp j = 0; j < lenl; ++j) {                                                          \
        IN_T vall = *(IN_T *)(inl + j * innerstepl);                                               \
        if (!ISNAN_VAL(vall)) { *suml += (ACC_T)TO_ACC(vall); (*countl)++; }                       \
        if (j < lens) {                                                                            \
            IN_T vals = *(IN_T *)(ins + j * innersteps);                                           \
            if (!ISNAN_VAL(vals)) { *sums += (ACC_T)TO_ACC(vals); (*counts)++; }                   \
        }                                                                                          \
    }                                                                                              \
}

DEFINE_SUMS_AND_COUNTS(half,       npy_half,    float,       TO_ACC_FROM_HALF, npy_half_isnan)
DEFINE_SUMS_AND_COUNTS(float,      float,       float,       TO_ACC_FROM_F32,  ISNAN_VAL_F32)
DEFINE_SUMS_AND_COUNTS(double,     double,      long double, TO_ACC_FROM_F64,  ISNAN_VAL_F64)
DEFINE_SUMS_AND_COUNTS(longdouble, long double, long double, TO_ACC_FROM_F80,  ISNAN_VAL_F80)

/*
 * Macro-generated branchless dtype-specialized loops
 */
static NPY_INLINE int isnan_f32(float x) { return x != x; }
static NPY_INLINE int isnan_f64(double x) { return x != x; }
static NPY_INLINE int isnan_f80(long double x) { return x != x; }

#define LOAD_F16(p) ((float)npy_half_to_float(*(npy_half*)(p)))
#define LOAD_F32(p) (*(float*)(p))
#define LOAD_F64(p) (*(double*)(p))
#define LOAD_F80(p) (*(long double*)(p))

#define STORE_F16(p, x) (*((npy_half*)(p)) = npy_float_to_half((float)(x)))
#define STORE_F32(p, x) (*((float*)(p)) = (float)(x))
#define STORE_F64(p, x) (*((double*)(p)) = (double)(x))
#define STORE_F80(p, x) (*((long double*)(p)) = (long double)(x))

#define DEFINE_MEAN_DIFF(NAME, ACC_T, LOAD, STORE, ISNAN, NAN_OUT)                    \
static void mean_diff_##NAME(                                                          \
    char **args,                                                                       \
    const npy_intp *dimensions,                                                        \
    const npy_intp *steps,                                                             \
    void *extra)                                                                        \
{                                                                                      \
    char *in1 = args[0], *in2 = args[1], *out = args[2];                               \
                                                                                       \
    const npy_intp nloops = dimensions[0];                                             \
    const npy_intp len1 = dimensions[1];                                               \
    const npy_intp len2 = dimensions[2];                                               \
                                                                                       \
    const npy_intp step1 = steps[0];                                                   \
    const npy_intp step2 = steps[1];                                                   \
    const npy_intp step_out = steps[2];                                                \
    const npy_intp innerstep1 = steps[3];                                              \
    const npy_intp innerstep2 = steps[4];                                              \
                                                                                       \
    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {\
        ACC_T sum1 = 0, sum2 = 0;                                                      \
        npy_intp count1 = 0, count2 = 0;                                               \
        const npy_intp Lmax = (len1 > len2) ? len1 : len2;                             \
        for (npy_intp j = 0; j < Lmax; ++j) {                                          \
            if (j < len1) {                                                            \
                ACC_T v1 = (ACC_T)LOAD(in1 + j * innerstep1);                          \
                if (!ISNAN(v1)) { sum1 += v1; count1++; }                              \
            }                                                                          \
            if (j < len2) {                                                            \
                ACC_T v2 = (ACC_T)LOAD(in2 + j * innerstep2);                          \
                if (!ISNAN(v2)) { sum2 += v2; count2++; }                              \
            }                                                                          \
        }                                                                              \
        if ((count1 > 0) && (count2 > 0)) {                                            \
            ACC_T md = sum1 / (ACC_T)count1 - sum2 / (ACC_T)count2;                    \
            STORE(out, md);                                                            \
        } else {                                                                       \
            STORE(out, NAN_OUT);                                                       \
        }                                                                              \
    }                                                                                  \
}

#define DEFINE_T_TEST(NAME, ACC_T, LOAD, STORE, ISNAN, NAN_OUT, SQRTFUN)               \
static NPY_INLINE ACC_T pooled_sigma_##NAME(ACC_T s1, ACC_T s2,                        \
                                            npy_intp n1, npy_intp n2) {                \
    if (n1 == 1) return SQRTFUN(s2 / ((n2 - 1) * (ACC_T)n2));                          \
    if (n2 == 1) return SQRTFUN(s1 / ((n1 - 1) * (ACC_T)n1));                          \
    return SQRTFUN(s1 / ((n1 - 1) * (ACC_T)n1) + s2 / ((n2 - 1) * (ACC_T)n2));         \
}                                                                                      \
static void t_test_##NAME(                                                             \
    char **args,                                                                       \
    const npy_intp *dimensions,                                                        \
    const npy_intp *steps,                                                             \
    void *extra)                                                                        \
{                                                                                      \
    char *in1 = args[0], *in2 = args[1], *out = args[2];                               \
                                                                                       \
    const npy_intp nloops = dimensions[0];                                             \
    const npy_intp len1 = dimensions[1];                                               \
    const npy_intp len2 = dimensions[2];                                               \
                                                                                       \
    const npy_intp step1 = steps[0];                                                   \
    const npy_intp step2 = steps[1];                                                   \
    const npy_intp step_out = steps[2];                                                \
    const npy_intp innerstep1 = steps[3];                                              \
    const npy_intp innerstep2 = steps[4];                                              \
                                                                                       \
    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {\
        ACC_T s1 = 0, s2 = 0;                                                          \
        npy_intp n1 = 0, n2 = 0;                                                       \
        const npy_intp Lmax = (len1 > len2) ? len1 : len2;                             \
        for (npy_intp j = 0; j < Lmax; ++j) {                                          \
            if (j < len1) { ACC_T v = (ACC_T)LOAD(in1 + j * innerstep1); if (!ISNAN(v)) { s1 += v; n1++; } } \
            if (j < len2) { ACC_T v = (ACC_T)LOAD(in2 + j * innerstep2); if (!ISNAN(v)) { s2 += v; n2++; } } \
        }                                                                              \
        if ((n1 == 0) || (n2 == 0) || (n1 == 1 && n2 == 1)) {                          \
            STORE(out, NAN_OUT);                                                       \
            continue;                                                                  \
        }                                                                              \
        ACC_T m1 = s1 / (ACC_T)n1, m2 = s2 / (ACC_T)n2;                                \
        ACC_T vs1 = 0, vs2 = 0;                                                        \
        for (npy_intp j = 0; j < Lmax; ++j) {                                          \
            if (j < len1) { ACC_T v = (ACC_T)LOAD(in1 + j * innerstep1); if (!ISNAN(v)) { ACC_T d = v - m1; vs1 += d * d; } } \
            if (j < len2) { ACC_T v = (ACC_T)LOAD(in2 + j * innerstep2); if (!ISNAN(v)) { ACC_T d = v - m2; vs2 += d * d; } } \
        }                                                                              \
        ACC_T denom = pooled_sigma_##NAME(vs1, vs2, n1, n2);                           \
        if (denom != 0) {                                                              \
            STORE(out, (ACC_T)((m1 - m2) / denom));                                    \
        } else {                                                                       \
            STORE(out, NAN_OUT);                                                       \
        }                                                                              \
    }                                                                                  \
}

/* Instantiate specializations */
DEFINE_MEAN_DIFF(half,       float,       LOAD_F16, STORE_F16, isnan_f32, NPY_NAN_HALF)
DEFINE_MEAN_DIFF(float,      float,       LOAD_F32, STORE_F32, isnan_f32, NAN)
DEFINE_MEAN_DIFF(double,     long double, LOAD_F64, STORE_F64, isnan_f80, NAN)
DEFINE_MEAN_DIFF(longdouble, long double, LOAD_F80, STORE_F80, isnan_f80, NAN)

DEFINE_T_TEST(half,       float,       LOAD_F16, STORE_F16, isnan_f32, NPY_NAN_HALF, sqrtf)
DEFINE_T_TEST(float,      float,       LOAD_F32, STORE_F32, isnan_f32, NAN,          sqrtf)
DEFINE_T_TEST(double,     double,      LOAD_F64, STORE_F64, isnan_f64, NAN,          sqrt)
DEFINE_T_TEST(longdouble, long double, LOAD_F80, STORE_F80, isnan_f80, NAN,          sqrtl)

/* mean_diff_* generated via macros above */

/* Macro-generated varsums and sum_var helpers */
#define DEFINE_VARSUMS(NAME, IN_T, ACC_T, LOADVAL, ISNAN_VAL)                                      \
static NPY_INLINE void varsums_##NAME(const char *inl, const char *ins,                            \
    const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl,                           \
    const npy_intp innersteps, const ACC_T meanl, const ACC_T means,                               \
    ACC_T * const varsuml, ACC_T * const varsums)                                                   \
{                                                                                                  \
    for (npy_intp j = 0; j < lenl; ++j) {                                                          \
        IN_T vall = *(IN_T *)(inl + j * innerstepl);                                               \
        if (!ISNAN_VAL(vall)) {                                                                    \
            ACC_T v = (ACC_T)LOADVAL(vall) - meanl;                                                \
            *varsuml += v * v;                                                                     \
        }                                                                                          \
        if (j < lens) {                                                                            \
            IN_T vals = *(IN_T *)(ins + j * innersteps);                                           \
            if (!ISNAN_VAL(vals)) {                                                                \
                ACC_T v = (ACC_T)LOADVAL(vals) - means;                                            \
                *varsums += v * v;                                                                 \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
}

#define DEFINE_SUM_VAR(NAME, ACC_T, SQRTFUN)                                                       \
static NPY_INLINE ACC_T sum_var_##NAME(const ACC_T sum1, const ACC_T sum2,                         \
                                       const npy_intp n1, const npy_intp n2) {                     \
    if (n1 == 1) {                                                                                 \
        return (ACC_T)SQRTFUN(sum2 / ((n2 - 1) * (ACC_T)n2));                                      \
    } else if (n2 == 1) {                                                                          \
        return (ACC_T)SQRTFUN(sum1 / ((n1 - 1) * (ACC_T)n1));                                      \
    } else {                                                                                       \
        const ACC_T var1 = sum1 / ((n1 - 1) * (ACC_T)n1);                                          \
        const ACC_T var2 = sum2 / ((n2 - 1) * (ACC_T)n2);                                          \
        return (ACC_T)SQRTFUN(var1 + var2);                                                        \
    }                                                                                              \
}

DEFINE_VARSUMS(half,       npy_half,    float,       npy_half_to_float, npy_half_isnan)
DEFINE_VARSUMS(float,      float,       float,       (float),            ISNAN_VAL_F32)
DEFINE_VARSUMS(double,     double,      double,      (double),           ISNAN_VAL_F64)
DEFINE_VARSUMS(longdouble, long double, long double, (long double),       ISNAN_VAL_F80)

DEFINE_SUM_VAR(float,      float,       sqrtf)
DEFINE_SUM_VAR(double,     double,      sqrt)
DEFINE_SUM_VAR(longdouble, long double, sqrtl)

#undef DEFINE_MEANVAR
#define DEFINE_MEANVAR(NAME, ACC_T, LOAD, STORE, ISNAN, NAN_OUT)                        \
static void meanvar_##NAME(                                                             \
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
        npy_intp cnt = 0;                                                                \
        for (npy_intp j = 0; j < len; ++j) {                                            \
            ACC_T v = (ACC_T)LOAD(in + j * inner_in);                                    \
            if (!ISNAN(v)) { sum += v; ++cnt; }                                          \
        }                                                                                \
        if (cnt == 0) {                                                                  \
            STORE(out_mean, NAN_OUT);                                                    \
            STORE(out_var,  NAN_OUT);                                                    \
            continue;                                                                    \
        }                                                                                \
        ACC_T mean = sum / (ACC_T)cnt;                                                   \
        ACC_T varsum = 0;                                                                \
        for (npy_intp j = 0; j < len; ++j) {                                            \
            ACC_T v = (ACC_T)LOAD(in + j * inner_in);                                    \
            if (!ISNAN(v)) { ACC_T d = v - mean; varsum += d * d; }                      \
        }                                                                                \
        ACC_T ddof = (ACC_T)LOAD(ddofp);                                                 \
        if (ISNAN(ddof)) ddof = (ACC_T)1;                                                \
        if (ddof < (ACC_T)0) ddof = (ACC_T)0;                                            \
        ACC_T denom = (ACC_T)cnt - ddof;                                                 \
        ACC_T var = (denom > (ACC_T)0) ? (varsum / denom) : NAN_OUT;                     \
        STORE(out_mean, mean);                                                           \
        STORE(out_var,  var);                                                            \
    }                                                                                    \
}

DEFINE_MEANVAR(half,       float,       LOAD_F16, STORE_F16, isnan_f32, NPY_NAN_HALF)
DEFINE_MEANVAR(float,      float,       LOAD_F32, STORE_F32, isnan_f32, NAN)
DEFINE_MEANVAR(double,     long double, LOAD_F64, STORE_F64, isnan_f64, NAN)
DEFINE_MEANVAR(longdouble, long double, LOAD_F80, STORE_F80, isnan_f80, NAN)

static PyUFuncGenericFunction funcs[8] = {&mean_diff_half,
                                          &mean_diff_float,
                                          &mean_diff_double,
                                          &mean_diff_longdouble,
                                          &t_test_half,
                                          &t_test_float,
                                          &t_test_double,
                                          &t_test_longdouble};

static const char md_types[12] = {NPY_HALF, NPY_HALF, NPY_HALF,
                            NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                            NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                            NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};

static const char t_types[12] = {NPY_HALF, NPY_HALF, NPY_HALF,
                           NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                           NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                           NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};

static PyUFuncGenericFunction funcs_meanvar[4] = {&meanvar_half,
                                                  &meanvar_float,
                                                  &meanvar_double,
                                                  &meanvar_longdouble};

/* in types: input array dtype, ddof scalar dtype; out types: mean dtype, var dtype */
static const char mv_types[16] = {NPY_HALF, NPY_HALF, NPY_HALF, NPY_HALF,
                                  NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                                  NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                                  NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufuncs",
    NULL,
    -1,
    Meandiff_Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

static char *doc = "Calculate the mean difference between two groups."
    "\n\n"
    "This function is the default statistic function for time_perm_cluster. It"
    "calculates the mean difference between two groups along the specified axis."
    "\n\n"
    "Parameters"
    "----------"
    "group1 : array, shape (..., time)"
    "    The first group of observations."
    "group2 : array, shape (..., time)"
    "    The second group of observations."
    "axis : int or tuple of ints, optional"
    "    The axis or axes along which to compute the mean difference. If None,"
    "    compute the mean difference over all axes."
    "\n\n"
    "Returns"
    "-------"
    "avg1 - avg2 : array or float"
    "    The mean difference between the two groups."
    "\n\n"
    "Examples"
    "--------"
    ">>> import numpy as np"
    ">>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]])"
    ">>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])"
    ">>> mean_diff(group1, group2, axis=1)"
    "array([ 0., 14.])"
    ">>> mean_diff(group1, group2, axis=0)"
    "array([ 0., 30.,  0.,  5.,  0.])"
    ">>> group3 = np.arange(100000, dtype=float).reshape(20000, 5)"
    ">>> mean_diff(group3, group1, axis=0)"
    "array([49997., 49968., 49999., 49995., 50001.])";

PyMODINIT_FUNC PyInit_ufuncs(void) {
    PyObject *m, *ufunc1, *ufunc2, *ufunc3, *d;
    import_array();
    import_ufunc();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    ufunc1 = PyUFunc_FromFuncAndDataAndSignature(funcs, NULL, md_types, 4, 2, 1, PyUFunc_Zero, "mean_diff",
    doc, 0, "(i),(j)->()");

    ufunc2 = PyUFunc_FromFuncAndDataAndSignature(funcs + 4, NULL, t_types, 4, 2, 1, PyUFunc_ReorderableNone, "t_test",
    "", 0, "(i),(j)->()");

    ufunc3 = PyUFunc_FromFuncAndDataAndSignature(funcs_meanvar, NULL, mv_types, 4, 2, 2, PyUFunc_Zero, "meanvar",
    "", 0, "(i),()->(),()");

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "mean_diff", ufunc1);
    PyDict_SetItemString(d, "t_test", ufunc2);
    PyDict_SetItemString(d, "meanvar", ufunc3);
    Py_DECREF(ufunc1);
    Py_DECREF(ufunc2);
    Py_DECREF(ufunc3);

    return m;
}
