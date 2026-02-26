#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include "shared/meanvar_core.h"

static PyMethodDef Meandiff_Methods[] = {
    {NULL, NULL, 0, NULL}
};

/* Macro-generated sums_and_counts for all dtypes */
#define ISNAN_VAL_F32(v) isnan_f32(v)
#define ISNAN_VAL_F64(v) isnan_f64(v)
#define ISNAN_VAL_F80(v) isnan_f80(v)

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
    const npy_intp Lmax = (lenl > lens) ? lenl : lens;                                             \
    for (npy_intp j = 0; j < Lmax; ++j) {                                                          \
        if (j < lenl) {                                                                            \
            IN_T vall = *(IN_T *)(inl + j * innerstepl);                                           \
            if (!ISNAN_VAL(vall)) { *suml += (ACC_T)TO_ACC(vall); (*countl)++; }                   \
        }                                                                                          \
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

/* Macro-generated varsums and sum_var helpers */
#define DEFINE_VARSUMS(NAME, IN_T, ACC_T, LOADVAL, ISNAN_VAL)                                      \
static NPY_INLINE void varsums_##NAME(const char *inl, const char *ins,                            \
    const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl,                           \
    const npy_intp innersteps, const ACC_T meanl, const ACC_T means,                               \
    ACC_T * const varsuml, ACC_T * const varsums)                                                   \
{                                                                                                  \
    const npy_intp Lmax = (lenl > lens) ? lenl : lens;                                             \
    for (npy_intp j = 0; j < Lmax; ++j) {                                                          \
        if (j < lenl) {                                                                            \
            IN_T vall = *(IN_T *)(inl + j * innerstepl);                                           \
            if (!ISNAN_VAL(vall)) {                                                                \
                ACC_T v = (ACC_T)LOADVAL(vall) - meanl;                                            \
                *varsuml += v * v;                                                                 \
            }                                                                                      \
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
DEFINE_VARSUMS(double,     double,      long double, (long double),       ISNAN_VAL_F64)
DEFINE_VARSUMS(longdouble, long double, long double, (long double),       ISNAN_VAL_F80)

DEFINE_SUM_VAR(half,       float,       sqrtf)
DEFINE_SUM_VAR(float,      float,       sqrtf)
DEFINE_SUM_VAR(double,     long double, sqrtl)
DEFINE_SUM_VAR(longdouble, long double, sqrtl)

/* Macro-generated branchless dtype-specialized loops */

#define DEFINE_MEAN_DIFF(NAME, ACC_T, STORE, NAN_OUT)                                  \
static void mean_diff_##NAME(                                                         \
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
        sums_and_counts_##NAME(in1, in2, len1, len2, innerstep1, innerstep2,            \
                               &sum1, &sum2, &count1, &count2);                        \
        if ((count1 > 0) && (count2 > 0)) {                                            \
            ACC_T md = sum1 / (ACC_T)count1 - sum2 / (ACC_T)count2;                    \
            STORE(out, md);                                                            \
        } else {                                                                       \
            STORE(out, NAN_OUT);                                                       \
        }                                                                              \
    }                                                                                  \
    (void)extra;                                                                       \
}

#define DEFINE_T_TEST(NAME, ACC_T, STORE, NAN_OUT)                                     \
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
        sums_and_counts_##NAME(in1, in2, len1, len2, innerstep1, innerstep2,            \
                               &s1, &s2, &n1, &n2);                                    \
        if ((n1 == 0) || (n2 == 0) || (n1 == 1 && n2 == 1)) {                          \
            STORE(out, NAN_OUT);                                                       \
            continue;                                                                  \
        }                                                                              \
        ACC_T m1 = s1 / (ACC_T)n1, m2 = s2 / (ACC_T)n2;                                \
        ACC_T vs1 = 0, vs2 = 0;                                                        \
        varsums_##NAME(in1, in2, len1, len2, innerstep1, innerstep2, m1, m2,            \
                       &vs1, &vs2);                                                    \
        ACC_T denom = sum_var_##NAME(vs1, vs2, n1, n2);                                \
        if (denom != 0) {                                                              \
            STORE(out, (ACC_T)((m1 - m2) / denom));                                    \
        } else {                                                                       \
            STORE(out, NAN_OUT);                                                       \
        }                                                                              \
    }                                                                                  \
    (void)extra;                                                                       \
}

/* Instantiate specializations */
DEFINE_MEAN_DIFF(half,       float,       STORE_F16, (float)NPY_NAN)
DEFINE_MEAN_DIFF(float,      float,       STORE_F32, (float)NPY_NAN)
DEFINE_MEAN_DIFF(double,     long double, STORE_F64, NPY_NAN)
DEFINE_MEAN_DIFF(longdouble, long double, STORE_F80, (long double)NPY_NAN)

DEFINE_T_TEST(half,       float,       STORE_F16, (float)NPY_NAN)
DEFINE_T_TEST(float,      float,       STORE_F32, (float)NPY_NAN)
DEFINE_T_TEST(double,     long double, STORE_F64, NPY_NAN)
DEFINE_T_TEST(longdouble, long double, STORE_F80, (long double)NPY_NAN)

/* mean_diff_* generated via macros above */

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
