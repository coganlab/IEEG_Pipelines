#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void get_array_coords(PyArrayObject* array, npy_intp index, npy_intp* coords) {
    for (int i = PyArray_NDIM(array) - 1; i >= 0; i--) {
        coords[i] = index / PyArray_STRIDE(array, i);
        index %= PyArray_STRIDE(array, i);
    }
}

// Function to create a new array with the maximum shape of the input arrays, filled with NaN values
static PyArrayObject* create_max_shape_array(PyArrayObject** arrays, int n_arrays, int axis) {
    npy_intp* max_shape = malloc(PyArray_NDIM(arrays[0]) * sizeof(npy_intp));
    memcpy(max_shape, PyArray_DIMS(arrays[0]), PyArray_NDIM(arrays[0]) * sizeof(npy_intp));

    for (int i = 1; i < n_arrays; i++) {
        for (int j = 0; j < PyArray_NDIM(arrays[i]); j++) {
            if (j != axis && PyArray_DIMS(arrays[i])[j] > max_shape[j]) {
                max_shape[j] = PyArray_DIMS(arrays[i])[j];
            }
        }
    }

    PyArrayObject* max_shape_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(arrays[0]), max_shape, NPY_DOUBLE);
    double nan_value = NAN;
    PyArray_FillWithScalar(max_shape_array, PyArray_Scalar(&nan_value, PyArray_DescrFromType(NPY_DOUBLE), NULL));

    free(max_shape);

    return max_shape_array;
}

// Function to copy the contents of the input arrays into the new array, leaving NaN values where the input arrays do not have elements
static void copy_arrays_to_max_shape_array(PyArrayObject* max_shape_array, PyArrayObject** arrays, int n_arrays, int axis) {
    for (int i = 0; i < n_arrays; i++) {
        npy_intp* indices = calloc(PyArray_NDIM(arrays[i]), sizeof(npy_intp));
        for (npy_intp j = 0; j < PyArray_SIZE(arrays[i]); j++) {
            get_array_coords(arrays[i], j, indices);
            *(double*)PyArray_GetPtr(max_shape_array, indices) = *(double*)PyArray_GetPtr(arrays[i], indices);
        }
        free(indices);
    }
}

// Python wrapper function
static PyObject* py_concatenate_and_pad(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    int axis;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "Oi", &list_obj, &axis)) {
        return NULL;
    }

    // Convert Python list of arrays to C array of PyArrayObjects
    Py_ssize_t n_arrays = PyList_Size(list_obj);
    PyArrayObject** arrays = malloc(n_arrays * sizeof(PyArrayObject*));
    for (Py_ssize_t i = 0; i < n_arrays; i++) {
        PyObject* array_obj = PyList_GetItem(list_obj, i);
        arrays[i] = (PyArrayObject*)PyArray_FROM_OTF(array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (arrays[i] == NULL) {
            return NULL;
        }
    }

    // Call C functions
    PyArrayObject* max_shape_array = create_max_shape_array(arrays, n_arrays, axis);
    copy_arrays_to_max_shape_array(max_shape_array, arrays, n_arrays, axis);

    // Convert result back to Python object
    PyObject* result = PyArray_Return(max_shape_array);

    // Clean up
    for (Py_ssize_t i = 0; i < n_arrays; i++) {
        Py_DECREF(arrays[i]);
    }
    free(arrays);

    return result;
}

static PyMethodDef Meandiff_Methods[] = {
    {"concatenate_and_pad", py_concatenate_and_pad, METH_VARARGS, "Concatenate arrays and pad with NaN values."},
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

double rand_normal(double mean, double stddev)
{
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    double x = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    return mean + stddev * x;
}

static void norm_fill(char **args, const npy_intp *dimensions, const npy_intp *steps, void *extra) {

    char *in1 = args[0], *out = args[1];

    npy_intp nloops = dimensions[0];  // Number of outer loops
    npy_intp len = dimensions[1];    // Core dimension m

    npy_intp step1 = steps[0];        // Outer loop step size for the first input
    npy_intp step_out = steps[1];     // Outer loop step size for the output
    npy_intp innerstep = steps[2];   // Step size of elements within dimension m

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, out += step_out) {

        // core calculation

        double sum = 0.0;
        double sum_of_squares = 0.0;
        npy_intp count = 0;

        for (npy_intp j = 0; j < len; ++j) {
            double val = *(double *)(in1 + j * innerstep);
            if (!isnan(val)) {
                sum += val;
                sum_of_squares += val * val;
                count++;
            }
        }

        double avg = (count > 0) ? sum / count : 0.0;
        double variance = (count > 1) ? (sum_of_squares - ((sum * sum) / count)) / (count - 1) : 0.0;
        double std = sqrt(variance);

        // fill in the missing values
        for (npy_intp j = 0; j < len; ++j) {
            double val = *(double *)(in1 + j * innerstep);
            if (isnan(val)) {
                *(double *)(out + j * innerstep) = rand_normal(avg, std);
            } else {
                *(double *)(out + j * innerstep) = val;
            }
        }
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

PyUFuncGenericFunction funcs[3] = {&mean_diff, &perm_test, &norm_fill};

static char md_types[3] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static char pt_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_INT, NPY_DOUBLE};

static char nf_types[2] = {NPY_DOUBLE, NPY_DOUBLE};

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

PyMODINIT_FUNC PyInit_ufuncs(void) {
    PyObject *m, *ufunc1, *ufunc2, *ufunc3, *d;
    import_array();
    import_ufunc();
    import_umath();

    srand(seeder());

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    ufunc1 = PyUFunc_FromFuncAndDataAndSignature(funcs, NULL, md_types, 1, 2, 1, PyUFunc_None, "mean_diff",
    "Calculate the mean difference of two numpy arrays.", 0, "(i),(j)->()");

    ufunc2 = PyUFunc_FromFuncAndDataAndSignature(funcs + 1, NULL, pt_types, 1, 3, 1, PyUFunc_None, "perm_test",
    "Calculate the proportion of permutations that are greater than the observed difference.", 0, "(i),(j),()->()");

    ufunc3 = PyUFunc_FromFuncAndDataAndSignature(funcs + 2, NULL, nf_types, 1, 1, 1, PyUFunc_None, "norm_fill",
    "Fill in missing values with random values from a normal distribution.", 0, "(i)->(i)");

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "mean_diff", ufunc1);
    PyDict_SetItemString(d, "perm_test", ufunc2);
    PyDict_SetItemString(d, "norm_fill", ufunc3);
    Py_DECREF(ufunc1);
    Py_DECREF(ufunc2);
    Py_DECREF(ufunc3);

    return m;
}
