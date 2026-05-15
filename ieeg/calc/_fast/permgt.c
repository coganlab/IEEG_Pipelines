/*
 * permgt.c — per-element rank proportions along an axis.
 *
 * For each 1-D slice along ``axis``, compute argsort and then write
 *   out[..., argsorted[..., i], ...] = i / (n - 1)
 * so that each element of the output holds its rank position normalized
 * to [0, 1] within its slice. Used for permutation-greater-than statistics
 * in ``ieeg.calc.stats``.
 *
 * Replaces the previous Cython implementation in permgt.pyx.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <string.h>


/* Increment a multi-index over `ndim` dims of size `dims[k]`.
 * Returns 0 when the iteration wraps past the end. */
static inline int
_advance_multi(npy_intp *multi, const npy_intp *dims, int ndim)
{
    int k = ndim - 1;
    while (k >= 0) {
        ++multi[k];
        if (multi[k] < dims[k]) {
            return 1;
        }
        multi[k] = 0;
        --k;
    }
    return 0;
}


static PyObject *
permgtnd(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *diff_obj;
    int axis = 0;
    static char *kwlist[] = {"diff", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|i", kwlist, &diff_obj, &axis)) {
        return NULL;
    }

    /* Promote input to float64; argsort works on any dtype but downstream
     * statistical code expects float64 and that's what the Cython version
     * declared via ``DTYPE = np.float64``. */
    PyArrayObject *diff = (PyArrayObject *)PyArray_FROMANY(
        diff_obj, NPY_DOUBLE, 1, 0, NPY_ARRAY_DEFAULT);
    if (diff == NULL) {
        return NULL;
    }

    int ndim = PyArray_NDIM(diff);
    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot apply perm_gt to a 0-dimensional array");
        Py_DECREF(diff);
        return NULL;
    }

    /* Normalize negative axis. */
    int ax = axis;
    if (ax < 0) ax += ndim;
    if (ax < 0 || ax >= ndim) {
        PyErr_Format(PyExc_ValueError,
                     "axis %d out of range for ndim %d", axis, ndim);
        Py_DECREF(diff);
        return NULL;
    }

    /* argsort along the chosen axis. Returns array of NPY_INTP with same
     * shape as input. */
    PyArrayObject *idx = (PyArrayObject *)PyArray_ArgSort(
        diff, ax, NPY_QUICKSORT);
    if (idx == NULL) {
        Py_DECREF(diff);
        return NULL;
    }

    /* Output: same shape as input, dtype float64. */
    npy_intp *dims = PyArray_DIMS(diff);
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(
        ndim, dims, NPY_DOUBLE, 0);
    if (out == NULL) {
        Py_DECREF(idx);
        Py_DECREF(diff);
        return NULL;
    }

    npy_intp n = dims[ax];

    /* Degenerate single-element axis: rank/(n-1) = 0/0. Match what the
     * Cython version would produce: ``proportions[sorted[0]] = 0 / 0 = NaN``.
     * Fill output with NaN to be consistent. */
    if (n == 1) {
        double nan = NPY_NAN;
        PyObject *scalar = PyArray_Scalar(
            &nan, PyArray_DescrFromType(NPY_DOUBLE), NULL);
        if (scalar == NULL) {
            Py_DECREF(out);
            Py_DECREF(idx);
            Py_DECREF(diff);
            return NULL;
        }
        int fill_rc = PyArray_FillWithScalar(out, scalar);
        Py_DECREF(scalar);
        Py_DECREF(idx);
        Py_DECREF(diff);
        if (fill_rc < 0) {
            Py_DECREF(out);
            return NULL;
        }
        return (PyObject *)out;
    }

    double inv_m = 1.0 / (double)(n - 1);

    /* Build the outer multi-index over all axes except `ax`. */
    npy_intp outer_dims[NPY_MAXDIMS];
    npy_intp idx_outer_strides[NPY_MAXDIMS];
    npy_intp out_outer_strides[NPY_MAXDIMS];
    int outer_ndim = 0;
    npy_intp *idx_strides = PyArray_STRIDES(idx);
    npy_intp *out_strides = PyArray_STRIDES(out);
    for (int k = 0; k < ndim; ++k) {
        if (k != ax) {
            outer_dims[outer_ndim] = dims[k];
            idx_outer_strides[outer_ndim] = idx_strides[k];
            out_outer_strides[outer_ndim] = out_strides[k];
            ++outer_ndim;
        }
    }
    npy_intp idx_axis_stride = idx_strides[ax];
    npy_intp out_axis_stride = out_strides[ax];

    char *idx_base = (char *)PyArray_DATA(idx);
    char *out_base = (char *)PyArray_DATA(out);

    /* Iterate over outer multi-index; for each outer position, walk the
     * 1-D slice along `ax` and scatter ranks into `out`. */
    npy_intp multi[NPY_MAXDIMS] = {0};

    if (outer_ndim == 0) {
        /* ndim == 1 case: no outer multi-index, just the inner loop once. */
        for (npy_intp i = 0; i < n; ++i) {
            npy_intp si;
            memcpy(&si, idx_base + i * idx_axis_stride, sizeof(npy_intp));
            double v = (double)i * inv_m;
            memcpy(out_base + si * out_axis_stride, &v, sizeof(double));
        }
    } else {
        do {
            npy_intp idx_off = 0;
            npy_intp out_off = 0;
            for (int k = 0; k < outer_ndim; ++k) {
                idx_off += multi[k] * idx_outer_strides[k];
                out_off += multi[k] * out_outer_strides[k];
            }
            char *idx_p = idx_base + idx_off;
            char *out_p = out_base + out_off;
            for (npy_intp i = 0; i < n; ++i) {
                npy_intp si;
                memcpy(&si, idx_p + i * idx_axis_stride, sizeof(npy_intp));
                double v = (double)i * inv_m;
                memcpy(out_p + si * out_axis_stride, &v, sizeof(double));
            }
        } while (_advance_multi(multi, outer_dims, outer_ndim));
    }

    Py_DECREF(idx);
    Py_DECREF(diff);
    return (PyObject *)out;
}


static PyMethodDef permgt_methods[] = {
    {"permgtnd", (PyCFunction)permgtnd, METH_VARARGS | METH_KEYWORDS,
     "permgtnd(diff, axis=0)\n\n"
     "Return per-element rank proportions along ``axis`` (each element\n"
     "is replaced by its rank position within its 1-D slice, normalized\n"
     "to ``[0, 1]``). Output dtype is float64."},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef permgt_module = {
    PyModuleDef_HEAD_INIT,
    "permgt",
    "Per-element rank proportions along an axis. C re-implementation of the "
    "previous Cython kernel; same API.",
    -1,
    permgt_methods,
    NULL, NULL, NULL, NULL,
};


PyMODINIT_FUNC
PyInit_permgt(void)
{
    PyObject *m = PyModule_Create(&permgt_module);
    if (m == NULL) {
        return NULL;
    }
    import_array();
    return m;
}
