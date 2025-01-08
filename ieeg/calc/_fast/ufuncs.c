#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

static PyMethodDef Meandiff_Methods[] = {
    {NULL, NULL, 0, NULL}
};

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

    if (len1 > len2) {
        for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
            double sum1 = 0.0, sum2 = 0.0;
            npy_intp count1 = 0, count2 = 0;

            for (npy_intp j = 0; j < len1; ++j) {
                double val1 = *(double *)(in1 + j * innerstep1);
                if (val1 == val1) {
                    sum1 += val1;
                    count1++;
                }
                if (j < len2) {
                    double val2 = *(double *)(in2 + j * innerstep2);
                    if (val2 == val2) {
                        sum2 += val2;
                        count2++;
                    }
                }
            }

            double mean1 = (count1 > 0) ? sum1 / count1 : 0.0;
            double mean2 = (count2 > 0) ? sum2 / count2 : 0.0;

            // Calculate the difference
            *((double *)out) = mean1 - mean2;
        }
    } else if (len1 < len2) {
        for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
            double sum1 = 0.0, sum2 = 0.0;
            npy_intp count1 = 0, count2 = 0;

            for (npy_intp j = 0; j < len2; ++j) {
                if (j < len1) {
                    double val1 = *(double *)(in1 + j * innerstep1);
                    if (val1 == val1) {
                        sum1 += val1;
                        count1++;
                    }
                }
                double val2 = *(double *)(in2 + j * innerstep2);
                if (val2 == val2) {
                    sum2 += val2;
                    count2++;
                }
            }

            double mean1 = (count1 > 0) ? sum1 / count1 : 0.0;
            double mean2 = (count2 > 0) ? sum2 / count2 : 0.0;

            // Calculate the difference
            *((double *)out) = mean1 - mean2;
        }
    } else {
        for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
            double sum1 = 0.0, sum2 = 0.0;
            npy_intp count1 = 0, count2 = 0;

            for (npy_intp j = 0; j < len2; ++j) {
                double val1 = *(double *)(in1 + j * innerstep1);
                if (val1 == val1) {
                    sum1 += val1;
                    count1++;
                }
                double val2 = *(double *)(in2 + j * innerstep2);
                if (val2 == val2) {
                    sum2 += val2;
                    count2++;
                }
            }

            double mean1 = (count1 > 0) ? sum1 / count1 : 0.0;
            double mean2 = (count2 > 0) ? sum2 / count2 : 0.0;

            // Calculate the difference
            *((double *)out) = mean1 - mean2;
        }
    }
}

PyUFuncGenericFunction funcs[1] = {&mean_diff};

static char md_types[3] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

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

char *doc = "Calculate the mean difference between two groups."
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
    PyObject *m, *ufunc1, *d;
    import_array();
    import_ufunc();
    import_umath();


    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    ufunc1 = PyUFunc_FromFuncAndDataAndSignature(funcs, NULL, md_types, 1, 2, 1, PyUFunc_None, "mean_diff",
    doc, 0, "(i),(j)->()");

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "mean_diff", ufunc1);
    Py_DECREF(ufunc1);

    return m;
}
