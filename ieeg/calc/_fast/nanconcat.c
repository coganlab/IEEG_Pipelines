#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to create a new array with the maximum shape of the input arrays, filled with NaN values
static PyArrayObject* create_max_shape_array(PyArrayObject** arrays, int n_arrays, int axis) {
    npy_intp* max_shape = malloc(PyArray_NDIM(arrays[0]) * sizeof(npy_intp));
    memcpy(max_shape, PyArray_DIMS(arrays[0]), PyArray_NDIM(arrays[0]) * sizeof(npy_intp));

    for (int i = 1; i < n_arrays; i++) {
        for (int j = 0; j < PyArray_NDIM(arrays[i]); j++) {
            if (j != axis && PyArray_DIMS(arrays[i])[j] > max_shape[j]) {
                max_shape[j] = PyArray_DIMS(arrays[i])[j];
            } else if (j == axis) {
                max_shape[j] += PyArray_DIMS(arrays[i])[j];
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
static int copy_arrays_to_max_shape_array(PyArrayObject* max_shape_array, PyArrayObject** arrays, int n_arrays, int axis) {
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    npy_intp *multi_index = malloc(PyArray_NDIM(arrays[0]) * sizeof(npy_intp));
    npy_intp *multi_index_out = malloc(PyArray_NDIM(arrays[0]) * sizeof(npy_intp));
    npy_intp offset = 0;

    for (int i = 0; i < n_arrays; i++) {
        iter = NpyIter_New(
            arrays[i], NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX,
            NPY_KEEPORDER, NPY_NO_CASTING, NULL);

        if (iter == NULL) {
            return -1;
        }
        if (NpyIter_GetIterSize(iter) != 0) {
            iternext = NpyIter_GetIterNext(iter, NULL);
            if (iternext == NULL) {
                NpyIter_Deallocate(iter);
                return -1;
            }
            NpyIter_GetMultiIndexFunc *get_multi_index =
            NpyIter_GetGetMultiIndex(iter, NULL);
            if (get_multi_index == NULL) {
                NpyIter_Deallocate(iter);
                return -1;
            }
            do {
                get_multi_index(iter, multi_index);
                for (int j = 0; j < PyArray_NDIM(arrays[i]); j++) {
                    multi_index_out[j] = multi_index[j];
                }
                multi_index_out[axis] += offset;

                fprintf(stderr, "multi_index is [%" NPY_INTP_FMT ", %" NPY_INTP_FMT "]\n",
                       multi_index_out[0], multi_index_out[1]);
                // copy the value from the input array to the output array
                memcpy(
                    (char*)PyArray_DATA(max_shape_array) + PyArray_STRIDES(max_shape_array)[0] * multi_index_out[0] +
                    PyArray_STRIDES(max_shape_array)[1] * multi_index_out[1],
                    (char*)PyArray_DATA(arrays[i]) + PyArray_STRIDES(arrays[i])[0] * multi_index[0] +
                    PyArray_STRIDES(arrays[i])[1] * multi_index[1],
                    sizeof(double)
                );
            } while (iternext(iter));
        }
        if (!NpyIter_Deallocate(iter)) {
            return -1;
        }
        offset += PyArray_DIMS(arrays[i])[axis];
//        NpyIter_Deallocate(iter);
    }
    free(multi_index);
    free(multi_index_out);
    return 0;
}

int PrintMultiIndex(PyArrayObject *arr) {
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    npy_intp *multi_index = malloc(PyArray_NDIM(arr) * sizeof(npy_intp));

    iter = NpyIter_New(
        arr, NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        return -1;
    }
    if (NpyIter_GetNDim(iter) != 2) {
        NpyIter_Deallocate(iter);
        PyErr_SetString(PyExc_ValueError, "Array must be 2-D");
        return -1;
    }
    if (NpyIter_GetIterSize(iter) != 0) {
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }
        NpyIter_GetMultiIndexFunc *get_multi_index =
            NpyIter_GetGetMultiIndex(iter, NULL);
        if (get_multi_index == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        do {
            get_multi_index(iter, multi_index);
            printf("multi_index is [%" NPY_INTP_FMT ", %" NPY_INTP_FMT "]\n",
                   multi_index[0], multi_index[1]);
        } while (iternext(iter));
    }
    if (!NpyIter_Deallocate(iter)) {
        return -1;
    }
    return 0;
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
    fprintf(stderr, "here\n");
//    PrintMultiIndex(max_shape_array);
//    PrintMultiIndex(arrays[0]);
    int check = copy_arrays_to_max_shape_array(max_shape_array, arrays, n_arrays, axis);
    if (check == -1) {
        return NULL;
    }

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

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "nanconcat",
    NULL,
    -1,
    Meandiff_Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_nanconcat(void) {
    PyObject *m, *d;
    import_array();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    d = PyModule_GetDict(m);

    return m;
}