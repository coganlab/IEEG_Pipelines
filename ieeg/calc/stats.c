#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <math.h>

static PyMethodDef Meandiff_Methods[] = {
    {NULL, NULL, 0, NULL}
};

double avg_non_nan(char *input, npy_intp len, npy_intp innerstep) {
    double sum = 0.0;
    npy_intp count = 0;

    for (npy_intp j = 0; j < len; ++j) {
        double val = *(double *)(input + j*innerstep);
        if (!isnan(val)) {
            sum += val;
            count++;
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

static void mean_diff(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    npy_intp nloops = dimensions[0];  // Number of outer loops
    npy_intp len1 = dimensions[1];    // Core dimension i
    npy_intp len2 = dimensions[2];    // Core dimension j

    npy_intp step1 = steps[0];        // Outer loop step size for the first input
    npy_intp step2 = steps[1];        // Outer loop step size for the second input
    npy_intp step_out = steps[2];     // Outer loop step size for the output
    npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    for (npy_intp i = 0; i < nloops;
         i++, in1 += step1, in2 += step2, out += step_out) {

        // core calculation
        double mean1 = avg_non_nan(in1, len1, innerstep1);
        double mean2 = avg_non_nan(in2, len2, innerstep2);

        *((double *)out) = mean1 - mean2;
    }
}

static void _perm_gt(char **args, const npy_intp *dimensions, const npy_intp *steps, void *extra) {

    char *in1 = args[0], *in2 = args[1], *out = args[2];

    npy_intp nloops = dimensions[0];  // Number of outer loops
    npy_intp len = dimensions[1];    // Core dimension m

    npy_intp step1 = steps[0];        // Outer loop step size for the first input
    npy_intp step2 = steps[1];        // Outer loop step size for the second input
    npy_intp step_out = steps[2];     // Outer loop step size for the output
    npy_intp innerstep = steps[3];   // Step size of elements within dimension m

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {

        // core calculation
        npy_intp count = 0;
        double val = *((double *)in1);
        for (npy_intp j = 0; j < len; j++) {
            double compare = *((double *)(in2 + j*innerstep));
            if (val > compare) {
                count++;
            }
        }

        *((double *)out) = (double)count / (double)len;
    }
}

PyUFuncGenericFunction funcs[2] = {&mean_diff, &_perm_gt};

static char types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cstats",
    NULL,
    -1,
    Meandiff_Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_cstats(void) {
    PyObject *m, *ufunc1, *ufunc2, *d;
    import_array();
    import_ufunc();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    ufunc1 = PyUFunc_FromFuncAndDataAndSignature(funcs, NULL, types, 2, 2, 1, PyUFunc_None, "mean_diff",
    "Calculate the mean difference of two numpy arrays.", 0, "(i),(j)->()");

    ufunc2 = PyUFunc_FromFuncAndDataAndSignature(funcs + 1, NULL, types + 3, 1, 2, 1, PyUFunc_None, "_perm_gt",
    "Calculate the proportion of elements in compare that are less than vals.", 0, "(),(m)->()");

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "mean_diff", ufunc1);
    PyDict_SetItemString(d, "_perm_gt", ufunc2);
    Py_DECREF(ufunc1);
    Py_DECREF(ufunc2);

    return m;
}
