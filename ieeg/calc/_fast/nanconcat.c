#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

// Function to create a new array with the maximum shape of the input arrays, filled with NaN values
static inline PyArrayObject* create_max_shape_array(PyArrayObject** arrays, int n_arrays, int axis) {
    npy_intp* max_shape = malloc(PyArray_NDIM(arrays[0]) * sizeof(npy_intp));
    npy_intp ndim = PyArray_NDIM(arrays[0]), dtype = PyArray_TYPE(arrays[0]);
    memcpy(max_shape, PyArray_DIMS(arrays[0]), ndim * sizeof(npy_intp));
    npy_intp *shape = malloc(ndim * sizeof(npy_intp));

    for (int i = 1; i < n_arrays; i++) {
        shape = PyArray_DIMS(arrays[i]);
        for (int j = 0; j < ndim; j++) {
            if (j != axis && shape[j] > max_shape[j]) {
                max_shape[j] = shape[j];
            } else if (j == axis) {
                max_shape[j] += shape[j];
            }
        }
    }

    PyArrayObject* max_shape_array = (PyArrayObject*)PyArray_SimpleNew(ndim, max_shape, dtype);

    // make an array of npy_nan's
    double nan = NPY_NAN;
    PyArray_FillWithScalar(max_shape_array, PyArray_Scalar(&nan, PyArray_DescrFromType(NPY_DOUBLE), NULL));

    free(max_shape);

    return max_shape_array;
}

// Function to copy the contents of the input arrays into the new array, leaving NaN values where the input arrays do not have elements
static inline void copy_arrays_to_max_shape_array(PyArrayObject* max_shape_array, PyArrayObject** arrays, int n_arrays, int axis) {
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    npy_intp *multi_index = malloc(PyArray_NDIM(arrays[0]) * sizeof(npy_intp));
    npy_intp offset = 0;
    npy_intp innerstride, outerstride;
    npy_intp *out_strides = PyArray_STRIDES(max_shape_array);
    char *outptrarray = (char*)PyArray_DATA(max_shape_array);
    npy_intp itemsize = PyArray_ITEMSIZE(max_shape_array);
    npy_intp ndim = PyArray_NDIM(max_shape_array);

    NPY_BEGIN_THREADS_DEF;

    for (int i = 0; i < n_arrays; i++) {
        iter = NpyIter_New(
            arrays[i], NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX,
            NPY_KEEPORDER, NPY_NO_CASTING, NULL);

        if (NpyIter_GetIterSize(iter) == 0) {
            if (iter != NULL){
                NpyIter_Deallocate(iter);
            }
            continue;
        }
        iternext = NpyIter_GetIterNext(iter, NULL);
        NpyIter_GetMultiIndexFunc *get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
        char *dataptrarray = NpyIter_GetDataPtrArray(iter)[0];

        NPY_BEGIN_ALLOW_THREADS;
        do {
            get_multi_index(iter, multi_index);
            outerstride = 0;
            innerstride = 0;
            for (int j = 0; j < ndim; j++) {
                innerstride += multi_index[j] * NpyIter_GetAxisStrideArray(iter, j)[0];
                if (j != axis) {
                    outerstride += multi_index[j] * out_strides[j];
                } else {
                    outerstride += (multi_index[j] + offset) * out_strides[j];
                }
            }
            // copy the value from the input array to the output array
            memcpy(outptrarray + outerstride, dataptrarray + innerstride, itemsize);

        } while (iternext(iter));
        NPY_END_ALLOW_THREADS;

        NpyIter_Deallocate(iter);
        offset += PyArray_DIMS(arrays[i])[axis];
    }
    free(multi_index);
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
        arrays[i] = (PyArrayObject*)PyArray_FROM_OF(array_obj, NPY_ARRAY_IN_ARRAY);
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