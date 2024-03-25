#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

static PyMethodDef Meandiff_Methods[] = {
    {NULL, NULL, 0, NULL}
};

double avg_non_nan(char *input, npy_intp len, npy_intp innerstep) {
    double sum = 0.0;
    npy_intp count = 0;

    for (npy_intp j = 0; j < len; ++j) {
        double val = *(double *)(input + j * innerstep);
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

// Function to shuffle an array
void shuffle(char *array, npy_intp len, npy_intp innerstep)
{
    if (len > 1)
    {
        npy_intp i;
        for (i = 0; i < len - 1; i++)
        {
          npy_intp j = i + rand() / (RAND_MAX / (len - i) + 1);
          double t = *(double *)(array + j * innerstep);
          *(array + j * innerstep) = *(array + i * innerstep);
          *((double *)(array + i * innerstep)) = t;
        }
    }
}


// Function to perform the permutation test
static void perm_test(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *in3 = args[2], *out = args[3];

    npy_intp nloops = dimensions[0];  // Number of outer loops
    npy_intp len1 = dimensions[1];    // Core dimension i
    npy_intp len2 = dimensions[2];    // Core dimension j
    unsigned int nperm = *((unsigned int *)in3);   // Number of permutations

    npy_intp step1 = steps[0];        // Outer loop step size for the first input
    npy_intp step2 = steps[1];        // Outer loop step size for the second input
    npy_intp step_out = steps[3];     // Outer loop step size for the output
    npy_intp innerstep1 = steps[4];   // Step size of elements within the first input
    npy_intp innerstep2 = steps[5];   // Step size of elements within the second input


    // Allocate memory for s based on the size of double
    char* s = malloc((len1 + len2) * sizeof(double));
    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        // core calculation
        double mean1 = avg_non_nan(in1, len1, innerstep1);
        double mean2 = avg_non_nan(in2, len2, innerstep2);
        double diff = mean1 - mean2;
        memcpy(s, in1, len1);
        memcpy(s + len1, in2, len2);

        int count = 0;
        for (int j = 0; j < nperm; j++) {
            shuffle(s, len1 + len2, innerstep1);
            mean1 = avg_non_nan(s, len1, innerstep1);
            mean2 = avg_non_nan(s + len1, len2, innerstep2);
            double perms = mean1 - mean2;
            if (perms > diff) {
                count++;
            }
        }
        *((double *)out) = (double)count / (double)nperm;
    }
    free(s);
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
        int count = 0;
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

// Robert Jenkins' 96 bit Mix Function
unsigned long seeder(void)
{
    unsigned long a = getpid(), b = clock(), c = time(NULL);
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

PyUFuncGenericFunction funcs[3] = {&mean_diff, &_perm_gt, &perm_test};

static char types[10] = {
NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_INT, NPY_DOUBLE
};

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
    PyObject *m, *ufunc1, *ufunc2, *ufunc3, *d;
    import_array();
    import_ufunc();
    import_umath();

    srand(seeder());

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    ufunc1 = PyUFunc_FromFuncAndDataAndSignature(funcs, NULL, types, 2, 2, 1, PyUFunc_None, "mean_diff",
    "Calculate the mean difference of two numpy arrays.", 0, "(i),(j)->()");

    ufunc2 = PyUFunc_FromFuncAndDataAndSignature(funcs + 1, NULL, types + 3, 1, 2, 1, PyUFunc_None, "_perm_gt",
    "Calculate the proportion of elements in compare that are less than vals.", 0, "(),(m)->()");

    ufunc3 = PyUFunc_FromFuncAndDataAndSignature(funcs + 2, NULL, types + 6, 2, 3, 1, PyUFunc_None, "perm_test",
    "Calculate the proportion of permutations that are greater than the observed difference.", 0, "(i),(j),()->()");
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "mean_diff", ufunc1);
    PyDict_SetItemString(d, "_perm_gt", ufunc2);
    PyDict_SetItemString(d, "perm_test", ufunc3);
    Py_DECREF(ufunc1);
    Py_DECREF(ufunc2);
    Py_DECREF(ufunc3);

    return m;
}
