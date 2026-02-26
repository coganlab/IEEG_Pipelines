/* (moved down after includes/typedefs) */
/* ndarray subtype: LabeledArray with C-level labels storage (stride-safe) */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include "labels.h"
#include <numpy/npy_math.h>
#include <numpy/halffloat.h>
#include "../calc/_fast/shared/meanvar_core.h"
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NPY_INLINE
#define NPY_INLINE inline
#endif

static NPY_INLINE void
write_intp_1d(char *base, npy_intp stride, npy_intp i, npy_intp v)
{
    memcpy(base + i * stride, &v, sizeof(npy_intp));
}



static PyTypeObject LabeledArray_Type; /* forward */


/* Forward declare struct tag so self-referential pointers work */

/* explicit ndarray sub-type instance layout */
typedef struct {
    PyArrayObject base;
    LabelsBlock *labels_block;
    PyObject *labels_cache; /* cached tuple-of-ndarray(unicode, 1d) per axis */
    char *delimiter; /* default join for label combinations */
} LabeledArrayObject;

/* Convert args for __array_function__ fallback: replace LabeledArray with base ndarray, recurse into lists/tuples */
static NPY_INLINE PyObject *
to_base_arg(PyObject *arg)
{
    if (PyObject_TypeCheck(arg, &LabeledArray_Type)) {
        return PyArray_View((PyArrayObject *)arg, NULL, &PyArray_Type);
    }
    if (PyTuple_Check(arg)) {
        Py_ssize_t n = PyTuple_GET_SIZE(arg);
        PyObject *out = PyTuple_New(n);
        if (!out) return NULL;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *it = PyTuple_GET_ITEM(arg, i);
            PyObject *conv = to_base_arg(it);
            if (!conv) { Py_DECREF(out); return NULL; }
            PyTuple_SET_ITEM(out, i, conv); /* steals */
        }
        return out;
    }
    if (PyList_Check(arg)) {
        Py_ssize_t n = PyList_GET_SIZE(arg);
        PyObject *out = PyList_New(n);
        if (!out) return NULL;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *it = PyList_GET_ITEM(arg, i);
            PyObject *conv = to_base_arg(it);
            if (!conv) { Py_DECREF(out); return NULL; }
            PyList_SET_ITEM(out, i, conv); /* steals */
        }
        return out;
    }
    Py_INCREF(arg);
    return arg;
}

/* Convert kwargs: return a new dict with 'out' and 'where' converted to ndarray views if they are LabeledArray; returns NULL if no kwargs */
static NPY_INLINE PyObject *
convert_kwargs_out_where(PyObject *kwargs)
{
    if (!kwargs || kwargs == Py_None || !PyDict_Check(kwargs)) return NULL;
    PyObject *kw2 = PyDict_Copy(kwargs);
    if (!kw2) return NULL;
    /* out */
    PyObject *out_kw = PyDict_GetItemString(kw2, "out"); /* borrowed */
    if (out_kw && out_kw != Py_None) {
        if (PyTuple_Check(out_kw)) {
            Py_ssize_t nout = PyTuple_GET_SIZE(out_kw);
            PyObject *out_conv = PyTuple_New(nout);
            if (!out_conv) { Py_DECREF(kw2); return NULL; }
            for (Py_ssize_t i = 0; i < nout; ++i) {
                PyObject *oi = PyTuple_GET_ITEM(out_kw, i);
                PyObject *seto = oi;
                if (oi && oi != Py_None && PyObject_TypeCheck(oi, &LabeledArray_Type)) {
                    seto = PyArray_View((PyArrayObject *)oi, NULL, &PyArray_Type);
                    if (!seto) { Py_DECREF(out_conv); Py_DECREF(kw2); return NULL; }
                } else {
                    Py_INCREF(seto ? seto : Py_None);
                    if (!seto) seto = Py_None;
                }
                PyTuple_SET_ITEM(out_conv, i, seto);
            }
            if (PyDict_SetItemString(kw2, "out", out_conv) < 0) { Py_DECREF(out_conv); Py_DECREF(kw2); return NULL; }
            Py_DECREF(out_conv);
        } else if (PyObject_TypeCheck(out_kw, &LabeledArray_Type)) {
            PyObject *out_conv = PyArray_View((PyArrayObject *)out_kw, NULL, &PyArray_Type);
            if (!out_conv) { Py_DECREF(kw2); return NULL; }
            if (PyDict_SetItemString(kw2, "out", out_conv) < 0) { Py_DECREF(out_conv); Py_DECREF(kw2); return NULL; }
            Py_DECREF(out_conv);
        }
    }
    /* where */
    PyObject *where_kw = PyDict_GetItemString(kw2, "where"); /* borrowed */
    if (where_kw && PyObject_TypeCheck(where_kw, &LabeledArray_Type)) {
        PyObject *where_conv = PyArray_View((PyArrayObject *)where_kw, NULL, &PyArray_Type);
        if (!where_conv) { Py_DECREF(kw2); return NULL; }
        if (PyDict_SetItemString(kw2, "where", where_conv) < 0) { Py_DECREF(where_conv); Py_DECREF(kw2); return NULL; }
        Py_DECREF(where_conv);
    }
    return kw2;
}

static NPY_INLINE void
labels_cache_clear(LabeledArrayObject *obj)
{
    if (obj) {
        Py_XDECREF(obj->labels_cache);
        obj->labels_cache = NULL;
    }
}

/* Attach labels to a LabeledArray view and clear cache */
static NPY_INLINE void
labelsblock_attach_to_view(PyObject *view, LabelsBlock *out_lb)
{
    LabeledArrayObject *vobj = (LabeledArrayObject *)view;
    LabelsBlock *old = vobj->labels_block;
    vobj->labels_block = out_lb;
    labelsblock_decref(old);
    labels_cache_clear(vobj);
}


static NPY_INLINE PyObject *
build_keepdims_labels(LabeledArrayObject *self, const int *axes, int naxes)
{
    LabelsBlock *lb = self->labels_block;
    if (!lb) {
        lb = labelsblock_new_default((PyArrayObject *)self);
        if (!lb) return NULL;
        self->labels_block = lb;
    }
    int nd = lb->ndim;
    char *reduced = (char *)PyMem_Calloc((size_t)nd, sizeof(char));
    if (!reduced) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < naxes; ++i) {
        int ax = axes[i];
        if (ax >= 0 && ax < nd) reduced[ax] = 1;
    }
    PyObject *orig = labelsblock_to_py_tuple(lb);
    if (!orig) { PyMem_Free(reduced); return NULL; }
    PyObject *out = PyTuple_New(nd);
    if (!out) { Py_DECREF(orig); PyMem_Free(reduced); return NULL; }
    const char *joiner = self->delimiter ? self->delimiter : "-";
    for (int ax = 0; ax < nd; ++ax) {
        if (reduced[ax]) {
            PyObject *joined = join_axis_labels(lb, ax, joiner);
            if (!joined) { Py_DECREF(orig); Py_DECREF(out); PyMem_Free(reduced); return NULL; }
            PyObject *t = PyTuple_New(1);
            if (!t) { Py_DECREF(joined); Py_DECREF(orig); Py_DECREF(out); PyMem_Free(reduced); return NULL; }
            PyTuple_SET_ITEM(t, 0, joined);
            PyTuple_SET_ITEM(out, ax, t);
        } else {
            PyObject *item = PyTuple_GET_ITEM(orig, ax);
            Py_INCREF(item);
            PyTuple_SET_ITEM(out, ax, item);
        }
    }
    Py_DECREF(orig);
    PyMem_Free(reduced);
    return out;
}



/* Wrap an ndarray result to LabeledArray and attach labels via ck, returning scalar if 0d. Does not touch ck's refcount. */
/* ---------- labels slot helpers (explicit struct) ---------- */
static NPY_INLINE LabelsBlock **
labels_slot(PyObject *self)
{
    return &((LabeledArrayObject *)self)->labels_block;
}


/* ---------- getters/setters ---------- */
static PyObject *
LabeledArray_get_labels(PyObject *self, void *closure)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    if (obj->labels_cache) {
        Py_INCREF(obj->labels_cache);
        return obj->labels_cache;
    }
    LabelsBlock *lb = obj->labels_block;
    if (!lb) {
        lb = labelsblock_new_default((PyArrayObject *)self);
        if (!lb) return NULL;
        obj->labels_block = lb;
    }
    /* Build list-of-lists of labels per axis */
    PyObject *out = PyList_New(lb->ndim);
    if (!out) return NULL;
    for (int ax = 0; ax < lb->ndim; ++ax) {
        int n = lb->axis_len[ax];
        PyObject *lst = PyList_New(n);
        if (!lst) { Py_DECREF(out); return NULL; }
        for (int i = 0; i < n; ++i) {
            PyObject *s = PyUnicode_FromString(lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "");
            if (!s) { Py_DECREF(lst); Py_DECREF(out); return NULL; }
            PyList_SET_ITEM(lst, i, s); /* steals ref */
        }
        PyList_SET_ITEM(out, ax, lst); /* steals ref */
    }
    obj->labels_cache = out; /* cache holds a ref */
    Py_INCREF(obj->labels_cache);
    return out;
}

static int
LabeledArray_set_labels(PyObject *self, PyObject *value, void *closure)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    LabelsBlock *nb = labelsblock_from_py(value, (PyArrayObject *)self);
    if (!nb) return -1;
    LabelsBlock *old = obj->labels_block;
    obj->labels_block = nb;
    labelsblock_decref(old);
    labels_cache_clear(obj);
    return 0;
}

/* ---------- __new__/finalize/dealloc ---------- */
static PyObject *
LabeledArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *input = NULL; PyObject *labels_in = NULL; PyObject *dtype_obj = NULL; const char *delim_in = NULL;
    static char *kwlist[] = {"input_array", "labels", "dtype", "delimiter", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOs:LabeledArray", kwlist, &input, &labels_in, &dtype_obj, &delim_in)) return NULL;

    PyArray_Descr *descr = NULL;
    if (dtype_obj && dtype_obj != Py_None) {
        if (!PyArray_DescrConverter(dtype_obj, &descr)) {
            PyErr_SetString(PyExc_TypeError, "invalid dtype");
            return NULL;
        }
    }

    PyArrayObject *arr = (PyArrayObject *)PyArray_FromAny(input, descr, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    if (!arr) {
        if (descr) Py_DECREF(descr);
        return NULL;
    }

    PyObject *view = PyArray_View(arr, NULL, type);
    if (!view) { Py_DECREF(arr); return NULL; }
    if (!PyObject_TypeCheck(view, &LabeledArray_Type)) {
        Py_DECREF(arr);
        Py_DECREF(view);
        PyErr_SetString(PyExc_RuntimeError, "failed to construct LabeledArray view");
        return NULL;
    }
    Py_DECREF(arr);

    *labels_slot(view) = labelsblock_from_py(labels_in, (PyArrayObject *)view);
    if (!*labels_slot(view)) { Py_DECREF(view); return NULL; }
    ((LabeledArrayObject *)view)->labels_cache = NULL;
    ((LabeledArrayObject *)view)->delimiter = NULL;
    if (delim_in && *delim_in) {
        size_t L = strlen(delim_in) + 1;
        ((LabeledArrayObject *)view)->delimiter = (char *)PyDataMem_NEW(L);
        if (!((LabeledArrayObject *)view)->delimiter) { Py_DECREF(view); return NULL; }
        memcpy(((LabeledArrayObject *)view)->delimiter, delim_in, L);
    }
    return view;
}

static void
LabeledArray_dealloc(PyObject *self)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    labelsblock_decref(obj->labels_block);
    obj->labels_block = NULL;
    Py_XDECREF(obj->labels_cache);
    obj->labels_cache = NULL;
    if (obj->delimiter) { free(obj->delimiter); obj->delimiter = NULL; }
    PyArray_Type.tp_dealloc(self);
}

/* ---------- label-based mapping ---------- */
static NPY_INLINE PyObject *
expand_and_convert_key_c(LabeledArrayObject *self, PyObject *key)
{
    int ndim = PyArray_NDIM((PyArrayObject *)self);
    LabelsBlock *lb = self->labels_block;
    if (!lb) { lb = labelsblock_new_default((PyArrayObject *)self); if (!lb) return NULL; self->labels_block = lb; }

    PyObject *items = PyTuple_Check(key) ? (Py_INCREF(key), key) : (Py_INCREF(key), PyTuple_Pack(1, key));
    if (!items) return NULL;
    Py_ssize_t n_items = PyTuple_GET_SIZE(items);

    /* First pass: count tokens */
    int consumed = 0; /* non-ellipsis, non-None */
    int c_none = 0;
    int c_ellipsis = 0;
    for (Py_ssize_t i = 0; i < n_items; ++i) {
        PyObject *k = PyTuple_GET_ITEM(items, i);
        if (k == Py_None) { c_none++; continue; }
        if (k == Py_Ellipsis) { c_ellipsis++; continue; }
        consumed++;
    }
    int need_slices = ndim - consumed;
    if (need_slices < 0) { Py_DECREF(items); PyErr_SetString(PyExc_IndexError, "too many indices"); return NULL; }
    /* Total output length: ndim + c_none (+ extra slices for additional ellipses) */
    Py_ssize_t out_len = (Py_ssize_t)ndim + (Py_ssize_t)c_none + (c_ellipsis > 0 ? (c_ellipsis - 1) : 0);
    PyObject *out = PyTuple_New(out_len);
    if (!out) { Py_DECREF(items); return NULL; }

    int axis = 0; /* how many data axes consumed */
    Py_ssize_t w = 0; /* write index into out */
    int ellipsis_done = 0;
    for (Py_ssize_t i = 0; i < n_items; ++i) {
        PyObject *k = PyTuple_GET_ITEM(items, i);
        if (k == Py_Ellipsis) {
            int n = ellipsis_done ? 1 : need_slices;
            for (int j = 0; j < n; ++j) {
                PyObject *sl = PySlice_New(NULL, NULL, NULL);
                if (!sl) { Py_DECREF(out); Py_DECREF(items); return NULL; }
                PyTuple_SET_ITEM(out, w++, sl); /* steals */
                axis++;
            }
            ellipsis_done = 1;
            continue;
        }
        if (k == Py_None) {
            Py_INCREF(Py_None);
            PyTuple_SET_ITEM(out, w++, Py_None);
            continue;
        }
        if (axis >= ndim) { Py_DECREF(out); Py_DECREF(items); PyErr_SetString(PyExc_IndexError, "too many indices"); return NULL; }

        PyObject *ck = NULL;
        if (PyUnicode_Check(k) || PyList_Check(k) || PyTuple_Check(k) || PyArray_Check(k)) {
            if (PyUnicode_Check(k)) {
                Py_ssize_t pos; int rc = labelsblock_find(lb, axis, k, &pos);
                if (rc != 0) { Py_DECREF(out); Py_DECREF(items); PyErr_SetString(PyExc_IndexError, "label not found"); return NULL; }
                ck = PyLong_FromSsize_t(pos);
                if (!ck) { Py_DECREF(out); Py_DECREF(items); return NULL; }
            } else if (PyArray_Check(k)) {
                PyArrayObject *arrk = (PyArrayObject *)k;
                int t = PyArray_TYPE(arrk);
                if (PyTypeNum_ISINTEGER(t) || t == NPY_BOOL) {
                    Py_INCREF(k);
                    ck = k;
                } else if (t == NPY_UNICODE || t == NPY_STRING || t == NPY_OBJECT) {
                    PyArrayObject *ind = map_label_index_array_to_int(lb, axis, arrk);
                    if (!ind) { Py_DECREF(out); Py_DECREF(items); PyErr_SetString(PyExc_IndexError, "label not found in array index"); return NULL; }
                    ck = (PyObject *)ind;
                } else {
                    Py_INCREF(k);
                    ck = k;
                }
            } else {
                PyObject *seq = PySequence_Fast(k, "expected sequence");
                if (!seq) { Py_DECREF(out); Py_DECREF(items); return NULL; }
                Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                ck = PyList_New(n);
                if (!ck) { Py_DECREF(seq); Py_DECREF(out); Py_DECREF(items); return NULL; }
                for (Py_ssize_t j = 0; j < n; ++j) {
                    PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
                    if (PyUnicode_Check(it)) {
                        Py_ssize_t pos; int rc = labelsblock_find(lb, axis, it, &pos);
                        if (rc != 0) { Py_DECREF(ck); Py_DECREF(seq); Py_DECREF(out); Py_DECREF(items); PyErr_SetString(PyExc_IndexError, "label not found in sequence"); return NULL; }
                        PyObject *idx = PyLong_FromSsize_t(pos);
                        if (!idx) { Py_DECREF(ck); Py_DECREF(seq); Py_DECREF(out); Py_DECREF(items); return NULL; }
                        PyList_SET_ITEM(ck, j, idx);
                    } else {
                        Py_INCREF(it);
                        PyList_SET_ITEM(ck, j, it);
                    }
                }
                Py_DECREF(seq);
            }
        } else {
            Py_INCREF(k);
            ck = k;
        }
        PyTuple_SET_ITEM(out, w++, ck); /* steals */
        axis++;
    }
    /* Trailing slices if any */
    while (axis < ndim) {
        PyObject *sl = PySlice_New(NULL, NULL, NULL);
        if (!sl) { Py_DECREF(out); Py_DECREF(items); return NULL; }
        PyTuple_SET_ITEM(out, w++, sl);
        axis++;
    }
    Py_DECREF(items);
    return out;
}

static PyObject *
LabeledArray_subscript(PyObject *self, PyObject *key)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    PyObject *ck = expand_and_convert_key_c(obj, key);
    if (!ck) return NULL;
    PyObject *res = PyArray_Type.tp_as_mapping->mp_subscript(self, ck);
    if (res && PyArray_Check(res)) {
        PyObject *target = res;
        if (!PyObject_TypeCheck(res, (PyTypeObject *)Py_TYPE(self))) {
            PyObject *viewres = PyArray_View((PyArrayObject *)res, NULL, (PyTypeObject *)Py_TYPE(self));
            Py_DECREF(res);
            if (!viewres) { Py_DECREF(ck); return NULL; }
            target = viewres;
        }
        LabelsBlock *parent_lb = obj->labels_block;
        if (parent_lb) {
            LabelsBlock *child = labelsblock_slice(parent_lb, ck, (PyArrayObject *)target);
            if (child) {
                LabeledArrayObject *tobj = (LabeledArrayObject *)target;
                LabelsBlock *old = tobj->labels_block;
                tobj->labels_block = child;
                labelsblock_decref(old);
                labels_cache_clear(tobj);
            }
        }
        if (PyArray_NDIM((PyArrayObject *)target) == 0) {
            PyObject *scalar = PyObject_CallMethod(target, "item", NULL);
            Py_DECREF(target);
            Py_DECREF(ck);
            return scalar;
        }
        Py_DECREF(ck);
        return target;
    }
    Py_DECREF(ck);
    return res;
}

static int
LabeledArray_ass_subscript(PyObject *self, PyObject *key, PyObject *value)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    PyObject *ck = expand_and_convert_key_c(obj, key);
    if (!ck) return -1;
    int rc = PyArray_Type.tp_as_mapping->mp_ass_subscript(self, ck, value);
    Py_DECREF(ck);
    return rc;
}

/* ---------- str printing ---------- */
static PyObject *
LabeledArray_str(PyObject *self_obj)
{
    PyObject *arr_str = PyArray_Type.tp_str(self_obj);
    if (!arr_str) return NULL;
    LabelsBlock *lb = *labels_slot(self_obj);
    if (!lb) { lb = labelsblock_new_default((PyArrayObject *)self_obj); if (!lb){ Py_DECREF(arr_str); return NULL;} *labels_slot(self_obj)=lb; }
    PyObject *labs = labelsblock_to_py_tuple(lb);
    if (!labs) { Py_DECREF(arr_str); return NULL; }
    PyObject *labs_str = PyObject_Str(labs); Py_DECREF(labs);
    if (!labs_str) { Py_DECREF(arr_str); return NULL; }
    PyObject *out = PyUnicode_FromFormat("%U\n%U", arr_str, labs_str);
    Py_DECREF(arr_str); Py_DECREF(labs_str); return out;
}

/* ---------- type table ---------- */

static PyGetSetDef LabeledArray_getset[] = {
    {"labels", (getter)LabeledArray_get_labels, (setter)LabeledArray_set_labels, "per-axis labels", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyObject *
array_finalize(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyObject *parent = NULL;
    if (args && PyTuple_Check(args) && PyTuple_GET_SIZE(args) >= 1) {
        parent = PyTuple_GET_ITEM(args, 0);
    }
    LabeledArrayObject *self = (LabeledArrayObject *)self_obj;
    self->labels_block = NULL;
    self->labels_cache = NULL;
    self->delimiter = NULL;
    if (parent && PyObject_TypeCheck(parent, &LabeledArray_Type)) {
        LabeledArrayObject *par  = (LabeledArrayObject *)parent;
        LabelsBlock *plb = par->labels_block;
        if (plb) plb->refcount++;
        self->labels_block = plb;
        if (par->delimiter) {
            size_t L = strlen(par->delimiter) + 1;
            self->delimiter = (char *)PyDataMem_NEW(L);
            if (self->delimiter) memcpy(self->delimiter, par->delimiter, L);
        }
    }
    Py_RETURN_NONE;
}

static PyObject *
array_wrap(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyObject *out_arr = NULL;
    PyObject *context = NULL;
    PyObject *return_scalar = NULL;
    if (!PyArg_ParseTuple(args, "O|OO:__array_wrap__", &out_arr, &context, &return_scalar)) {
        return NULL;
    }
    if (!PyArray_Check(out_arr)) { Py_INCREF(out_arr); return out_arr; }

    if (PyArray_NDIM((PyArrayObject *)out_arr) == 0) {
        return PyObject_CallMethod(out_arr, "item", NULL);
    }

    PyObject *view = PyArray_View((PyArrayObject *)out_arr, NULL, (PyTypeObject *)Py_TYPE(self_obj));
    if (!view) return NULL;

    LabeledArrayObject *self = (LabeledArrayObject *)self_obj;
    LabeledArrayObject *vobj = (LabeledArrayObject *)view;

    LabelsBlock *plb = self->labels_block;
    if (plb) plb->refcount++;
    LabelsBlock *old = vobj->labels_block;
    vobj->labels_block = plb;
    labelsblock_decref(old);
    labels_cache_clear(vobj);

    int parent_nd = plb ? plb->ndim : 0;
    int out_nd = PyArray_NDIM((PyArrayObject *)view);
    /* Prefer deterministic label mapping using reduction axes from context (3rd element) */
    int used_context = 0;
    if (plb && out_nd < parent_nd && context && PyTuple_Check(context) && PyTuple_GET_SIZE(context) >= 3) {
        PyObject *axes_obj = PyTuple_GET_ITEM(context, 2); /* borrowed */
        if (axes_obj && axes_obj != Py_None) {
            int *drop = (int *)calloc((size_t)parent_nd, sizeof(int));
            if (drop) {
                if (PyLong_Check(axes_obj)) {
                    long axl = PyLong_AsLong(axes_obj);
                    if (!(axl == -1 && PyErr_Occurred())) {
                        int ax = (int)axl; if (ax < 0) ax += parent_nd;
                        if (ax >= 0 && ax < parent_nd) drop[ax] = 1;
                    } else {
                        PyErr_Clear();
                    }
                } else if (PySequence_Check(axes_obj)) {
                    PyObject *seq = PySequence_Fast(axes_obj, "axes must be sequence");
                    if (seq) {
                        Py_ssize_t na = PySequence_Fast_GET_SIZE(seq);
                        for (Py_ssize_t i = 0; i < na; ++i) {
                            long axl = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
                            if (axl == -1 && PyErr_Occurred()) { PyErr_Clear(); continue; }
                            int ax = (int)axl; if (ax < 0) ax += parent_nd;
                            if (ax >= 0 && ax < parent_nd) drop[ax] = 1;
                        }
                        Py_DECREF(seq);
                    }
                }
                int kept_count = 0; for (int i = 0; i < parent_nd; ++i) if (!drop[i]) kept_count++;
                if (kept_count == out_nd) {
                    LabelsBlock *child = labelsblock_alloc(out_nd);
                    if (child) {
                        int dst = 0;
                        for (int src = 0; src < parent_nd; ++src) {
                            if (drop[src]) continue;
                            int n = plb->axis_len[src];
                            if (labelsblock_set_axis_copy(child, dst, plb->axis_labels[src], n) < 0) { labelsblock_decref(child); child = NULL; break; }
                            dst++;
                        }
                        if (child) {
                            LabelsBlock *old2 = vobj->labels_block;
                            vobj->labels_block = child;
                            labelsblock_decref(old2);
                            labels_cache_clear(vobj);
                            used_context = 1;
                        }
                    }
                }
                free(drop);
            }
        }
    }
    if (used_context) {
        return view;
    }
    if (plb && out_nd < parent_nd) {
        const npy_intp *oshape = PyArray_SHAPE((PyArrayObject *)view);
        int *kept_src = (int *)calloc((size_t)out_nd, sizeof(int));
        if (!kept_src) { Py_DECREF(view); return NULL; }
        int dst = 0;
        for (int src = 0; src < parent_nd && dst < out_nd; ++src) {
            int src_n = plb->axis_len[src];
            if (src_n == (int)oshape[dst]) { kept_src[dst++] = src; }
        }
        if (dst == out_nd) {
            LabelsBlock *child = labelsblock_alloc(out_nd);
            if (!child) { free(kept_src); Py_DECREF(view); return NULL; }
            for (int i = 0; i < out_nd; ++i) {
                int src = kept_src[i];
                int n = (int)oshape[i];
                if (labelsblock_set_axis_copy(child, i, plb->axis_labels[src], n) < 0) { free(kept_src); labelsblock_decref(child); Py_DECREF(view); return NULL; }
            }
            LabelsBlock *old2 = vobj->labels_block;
            vobj->labels_block = child;
            labelsblock_decref(old2);
            labels_cache_clear(vobj);
        }
        free(kept_src);
    }

    return view;
}

/* ---------- __array_function__ override (handle np.concatenate) ---------- */
static PyObject *
array_function(PyObject *self_obj, PyObject *args)
{
    /* Signature: __array_function__(self, func, types, fargs, fkwargs) */
    PyObject *func = NULL, *types = NULL, *fargs = NULL, *fkwargs = NULL;
    if (!PyArg_ParseTuple(args, "OOOO:__array_function__", &func, &types, &fargs, &fkwargs)) {
        return NULL;
    }

    /* Identify numpy.concatenate, numpy.squeeze, numpy.stack, numpy.expand_dims, numpy.take, numpy.transpose, numpy.swapaxes, numpy.take_along_axis */
    static PyObject *np_concatenate = NULL;
    static PyObject *np_squeeze = NULL;
    static PyObject *np_stack = NULL;
    static PyObject *np_expand_dims = NULL;
    static PyObject *np_take = NULL;
    static PyObject *np_transpose = NULL;
    static PyObject *np_swapaxes = NULL;
    static PyObject *np_taa = NULL;
    static PyObject *np_nanmean = NULL;
    static PyObject *np_nanstd = NULL;
    if (np_concatenate == NULL || np_squeeze == NULL || np_stack == NULL || np_expand_dims == NULL || np_take == NULL || np_transpose == NULL || np_swapaxes == NULL || np_taa == NULL || np_nanmean == NULL || np_nanstd == NULL) {
        PyObject *numpy = PyImport_ImportModule("numpy");
        if (!numpy) return NULL;
        if (np_concatenate == NULL) np_concatenate = PyObject_GetAttrString(numpy, "concatenate");
        if (np_squeeze == NULL) np_squeeze = PyObject_GetAttrString(numpy, "squeeze");
        if (np_stack == NULL) np_stack = PyObject_GetAttrString(numpy, "stack");
        if (np_expand_dims == NULL) np_expand_dims = PyObject_GetAttrString(numpy, "expand_dims");
        if (np_take == NULL) np_take = PyObject_GetAttrString(numpy, "take");
        if (np_transpose == NULL) np_transpose = PyObject_GetAttrString(numpy, "transpose");
        if (np_swapaxes == NULL) np_swapaxes = PyObject_GetAttrString(numpy, "swapaxes");
        if (np_taa == NULL) np_taa = PyObject_GetAttrString(numpy, "take_along_axis");
        if (np_nanmean == NULL) np_nanmean = PyObject_GetAttrString(numpy, "nanmean");
        if (np_nanstd == NULL) np_nanstd = PyObject_GetAttrString(numpy, "nanstd");
        Py_DECREF(numpy);
        if (!np_concatenate || !np_squeeze || !np_stack || !np_expand_dims || !np_take || !np_transpose || !np_swapaxes || !np_taa || !np_nanmean || !np_nanstd) return NULL;
    }
    int is_concatenate = (func == np_concatenate);
    int is_squeeze = (func == np_squeeze);
    int is_stack = (func == np_stack);
    int is_expand = (func == np_expand_dims);
    int is_take = (func == np_take);
    int is_transpose = (func == np_transpose);
    int is_swapaxes = (func == np_swapaxes);
    int is_taa = (func == np_taa);
    int is_nanmean = (func == np_nanmean);
    int is_nanstd = (func == np_nanstd);
    if (!is_concatenate && !is_squeeze && !is_stack && !is_expand && !is_take && !is_transpose && !is_swapaxes && !is_taa && !is_nanmean && !is_nanstd) {
        /* Fallback: call NumPy func on base ndarrays to avoid recursion */
        if (!PyTuple_Check(fargs)) { Py_RETURN_NOTIMPLEMENTED; }
        Py_ssize_t nfa = PyTuple_GET_SIZE(fargs);
        PyObject *new_fargs = PyTuple_New(nfa);
        if (!new_fargs) return NULL;
        for (Py_ssize_t i = 0; i < nfa; ++i) {
            PyObject *argi = PyTuple_GET_ITEM(fargs, i);
            PyObject *to_set = to_base_arg(argi);
            if (!to_set) { Py_DECREF(new_fargs); return NULL; }
            PyTuple_SET_ITEM(new_fargs, i, to_set);
        }
        PyObject *new_fkwargs = convert_kwargs_out_where(fkwargs);
        PyObject *res = PyObject_Call(func, new_fargs, new_fkwargs ? new_fkwargs : fkwargs);
        if (new_fkwargs) Py_DECREF(new_fkwargs);
        Py_DECREF(new_fargs);
        if (!res) return NULL;
        if (PyArray_Check(res)) {
            /* Pass axis info (if present) via context's 3rd element */
            PyObject *ctx_axes = Py_None;
            if (fkwargs && PyDict_Check(fkwargs)) {
                PyObject *axis_obj = PyDict_GetItemString(fkwargs, "axis"); /* borrowed */
                if (axis_obj && axis_obj != Py_None) {
                    if (PyLong_Check(axis_obj)) {
                        long axl = PyLong_AsLong(axis_obj);
                        if (!(axl == -1 && PyErr_Occurred())) {
                            PyObject *t = PyTuple_New(1);
                            if (t) { PyTuple_SET_ITEM(t, 0, PyLong_FromLong(axl)); ctx_axes = t; }
                        } else {
                            PyErr_Clear();
                        }
                    } else if (PySequence_Check(axis_obj)) {
                        PyObject *t = PySequence_Tuple(axis_obj);
                        if (t) ctx_axes = t;
                    }
                }
            }
            PyObject *context = Py_BuildValue("(OOO)", Py_None, Py_None, ctx_axes);
            if (ctx_axes != Py_None) Py_DECREF(ctx_axes);
            PyObject *wrapped = PyObject_CallMethod(self_obj, "__array_wrap__", "OO", res, context ? context : Py_None);
            if (context) Py_DECREF(context);
            Py_DECREF(res);
            return wrapped;
        }
        return res;
    }

    /* Ensure all types are subclasses of LabeledArray */
    PyObject *types_seq = PySequence_Fast(types, "types must be a sequence");
    if (!types_seq) return NULL;
    Py_ssize_t ntypes = PySequence_Fast_GET_SIZE(types_seq);
    for (Py_ssize_t i = 0; i < ntypes; ++i) {
        PyObject *t = PySequence_Fast_GET_ITEM(types_seq, i);
        if (!PyType_Check(t) || !PyType_IsSubtype((PyTypeObject *)t, &LabeledArray_Type)) {
            Py_DECREF(types_seq);
            Py_RETURN_NOTIMPLEMENTED;
        }
    }
    Py_DECREF(types_seq);

    if (!PyTuple_Check(fargs) || PyTuple_GET_SIZE(fargs) < 1) { Py_RETURN_NOTIMPLEMENTED; }

    if (is_nanmean || is_nanstd) {
        PyObject *obj0 = PyTuple_GET_ITEM(fargs, 0);
        if (!PyObject_TypeCheck(obj0, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
        Py_ssize_t nfa = PyTuple_GET_SIZE(fargs);
        PyObject *args_tail = PyTuple_New(nfa - 1);
        if (!args_tail) return NULL;
        for (Py_ssize_t i = 1; i < nfa; ++i) {
            PyObject *it = PyTuple_GET_ITEM(fargs, i);
            Py_INCREF(it);
            PyTuple_SET_ITEM(args_tail, i - 1, it);
        }
        PyObject *meth = PyObject_GetAttrString(obj0, is_nanmean ? "nanmean" : "nanstd");
        if (!meth) { Py_DECREF(args_tail); return NULL; }
        PyObject *res = PyObject_Call(meth, args_tail, fkwargs);
        Py_DECREF(meth);
        Py_DECREF(args_tail);
        return res;
    }

    /* Common locals used in some branches */
    PyObject *arrays = NULL; Py_ssize_t narr = 0;
    if (is_concatenate || is_stack) {
        PyObject *arrays_obj = PyTuple_GET_ITEM(fargs, 0);
        arrays = PySequence_Fast(arrays_obj, "first arg must be a sequence of arrays");
        if (!arrays) return NULL;
        narr = PySequence_Fast_GET_SIZE(arrays);
        if (narr == 0) { Py_DECREF(arrays); Py_RETURN_NOTIMPLEMENTED; }
    }

    /* Determine axis */
    int axis = 0;
    if (is_concatenate || is_stack || is_expand || is_take || is_taa) {
        if (fkwargs && fkwargs != Py_None && PyDict_Check(fkwargs)) {
            PyObject *axis_obj = PyDict_GetItemString(fkwargs, "axis"); /* borrowed */
            if (axis_obj && axis_obj != Py_None) {
                long axl = PyLong_AsLong(axis_obj);
                if (axl == -1 && PyErr_Occurred()) { if (arrays) Py_DECREF(arrays); return NULL; }
                axis = (int)axl;
            }
        }
        if ((is_take || is_taa) && PyTuple_Check(fargs) && PyTuple_GET_SIZE(fargs) >= 3) {
            /* Positional axis for take: (a, indices, axis) */
            PyObject *axis_obj2 = PyTuple_GET_ITEM(fargs, 2);
            if (axis_obj2 != Py_None) {
                long axl = PyLong_AsLong(axis_obj2);
                if (axl == -1 && PyErr_Occurred()) { if (arrays) Py_DECREF(arrays); return NULL; }
                axis = (int)axl;
            }
        }
    }

    /* Determine base array and ndim depending on op */
    PyObject *first = NULL; int nd = 0;
    if (is_concatenate || is_stack) {
        first = PySequence_Fast_GET_ITEM(arrays, 0);
        if (!PyArray_Check(first)) { Py_DECREF(arrays); Py_RETURN_NOTIMPLEMENTED; }
        nd = PyArray_NDIM((PyArrayObject *)first);
    } else {
        first = PyTuple_GET_ITEM(fargs, 0);
        if (!PyArray_Check(first)) { Py_RETURN_NOTIMPLEMENTED; }
        nd = PyArray_NDIM((PyArrayObject *)first);
    }
    /* Handlers for transpose/swapaxes/take_along_axis */
    if (is_transpose) {
        PyObject *obj0 = PyTuple_GET_ITEM(fargs, 0);
        if (!PyObject_TypeCheck(obj0, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
        PyObject *axes = NULL;
        if (PyTuple_Check(fargs) && PyTuple_GET_SIZE(fargs) >= 2)
            axes = PyTuple_GET_ITEM(fargs, 1);
        PyObject *res_base;
        if (axes && axes != Py_None) {
            /* build dims */
            int nd0 = PyArray_NDIM((PyArrayObject *)obj0);
            npy_intp *perm = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)nd0);
            if (!perm) return NULL;
            PyObject *seq = PySequence_Fast(axes, "axes must be sequence");
            if (!seq) { PyMem_Free(perm); return NULL; }
            for (int i = 0; i < nd0; ++i) perm[i] = (npy_intp)PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
            Py_DECREF(seq);
            PyArray_Dims pd; pd.ptr = perm; pd.len = nd0;
            res_base = PyArray_Transpose((PyArrayObject *)obj0, &pd);
            PyMem_Free(perm);
            if (!res_base) return NULL;
        } else {
            res_base = PyArray_Transpose((PyArrayObject *)obj0, NULL);
            if (!res_base) return NULL;
        }
        PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(res_base);
        if (!view) return NULL;
        LabeledArrayObject *lai = (LabeledArrayObject *)obj0;
        LabelsBlock *plb = lai->labels_block;
        if (!plb) { plb = labelsblock_new_default((PyArrayObject *)obj0); if (!plb) { Py_DECREF(view); return NULL; } lai->labels_block = plb; }
        int nd_out = PyArray_NDIM((PyArrayObject *)view);
        LabelsBlock *out = labelsblock_alloc(nd_out);
        if (!out) { Py_DECREF(view); return NULL; }
        /* compute permutation */
        int *perm_i = (int *)malloc(sizeof(int) * (size_t)nd_out);
        if (!perm_i) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
        if (axes && axes != Py_None) {
            PyObject *seq = PySequence_Fast(axes, "axes must be sequence");
            if (!seq) { free(perm_i); labelsblock_decref(out); Py_DECREF(view); return NULL; }
            for (int i = 0; i < nd_out; ++i) perm_i[i] = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
            Py_DECREF(seq);
        } else {
            for (int i = 0; i < nd_out; ++i) perm_i[i] = nd_out - 1 - i;
        }
        for (int ax = 0; ax < nd_out; ++ax) {
            int src = perm_i[ax];
            out->axis_len[ax] = plb->axis_len[src];
            out->axis_labels[ax] = plb->axis_labels[src];
            out->axis_borrowed[ax] = 1;
            if (axhash_build(&out->axis_hash[ax], (const char **)out->axis_labels[ax], out->axis_len[ax]) < 0) { free(perm_i); labelsblock_decref(out); Py_DECREF(view); return NULL; }
        }
        free(perm_i);
        out->borrowed_owner = plb; plb->refcount++;
        labelsblock_attach_to_view(view, out);
        return view;
    }

    if (is_swapaxes) {
        PyObject *obj0 = PyTuple_GET_ITEM(fargs, 0);
        if (!PyObject_TypeCheck(obj0, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
        long a = PyLong_AsLong(PyTuple_GET_ITEM(fargs, 1));
        long b = PyLong_AsLong(PyTuple_GET_ITEM(fargs, 2));
        if ((a == -1 || b == -1) && PyErr_Occurred()) return NULL;
        PyObject *res_base = PyArray_SwapAxes((PyArrayObject *)obj0, (int)a, (int)b);
        if (!res_base) return NULL;
        PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(res_base);
        if (!view) return NULL;
        LabeledArrayObject *lai = (LabeledArrayObject *)obj0;
        LabelsBlock *plb = lai->labels_block;
        if (!plb) { plb = labelsblock_new_default((PyArrayObject *)obj0); if (!plb) { Py_DECREF(view); return NULL; } lai->labels_block = plb; }
        int nd_out = PyArray_NDIM((PyArrayObject *)view);
        LabelsBlock *out = labelsblock_alloc(nd_out);
        if (!out) { Py_DECREF(view); return NULL; }
        for (int ax = 0; ax < nd_out; ++ax) {
            int src = ax;
            if (ax == (int)a) src = (int)b; else if (ax == (int)b) src = (int)a;
            out->axis_len[ax] = plb->axis_len[src];
            out->axis_labels[ax] = plb->axis_labels[src];
            out->axis_borrowed[ax] = 1;
            if (axhash_build(&out->axis_hash[ax], (const char **)out->axis_labels[ax], out->axis_len[ax]) < 0) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
        }
        out->borrowed_owner = plb; plb->refcount++;
        labelsblock_attach_to_view(view, out);
        return view;
    }

    if (is_taa) {
        if (!PyTuple_Check(fargs) || PyTuple_GET_SIZE(fargs) < 3) { Py_RETURN_NOTIMPLEMENTED; }
        PyObject *obj0 = PyTuple_GET_ITEM(fargs, 0);
        PyObject *indices = PyTuple_GET_ITEM(fargs, 1);
        if (!PyObject_TypeCheck(obj0, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
        PyObject *res_base = PyObject_Call(func, fargs, fkwargs);
        if (!res_base) return NULL;
        if (!PyArray_Check(res_base)) return res_base;
        PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(res_base);
        if (!view) return NULL;
        int nd0 = PyArray_NDIM((PyArrayObject *)obj0);
        if (axis < 0) axis += nd0;
        PyObject *ck = PyTuple_New(nd0);
        if (!ck) { Py_DECREF(view); return NULL; }
        for (int ax = 0; ax < nd0; ++ax) {
            if (ax == axis) { Py_INCREF(indices); PyTuple_SET_ITEM(ck, ax, indices); }
            else { PyObject *sl = PySlice_New(NULL, NULL, NULL); if (!sl) { Py_DECREF(ck); Py_DECREF(view); return NULL; } PyTuple_SET_ITEM(ck, ax, sl); }
        }
        LabeledArrayObject *lai = (LabeledArrayObject *)obj0;
        LabelsBlock *plb = lai->labels_block;
        if (plb) {
            LabelsBlock *child = labelsblock_slice(plb, ck, (PyArrayObject *)view);
            if (child) {
                LabeledArrayObject *tobj = (LabeledArrayObject *)view;
                LabelsBlock *old = tobj->labels_block;
                tobj->labels_block = child;
                labelsblock_decref(old);
                labels_cache_clear(tobj);
            }
        }
        Py_DECREF(ck);
        return view;
    }
    if (is_take) {
        if (!PyTuple_Check(fargs) || PyTuple_GET_SIZE(fargs) < 2) { Py_RETURN_NOTIMPLEMENTED; }
        PyObject *obj = PyTuple_GET_ITEM(fargs, 0);
        PyObject *indices = PyTuple_GET_ITEM(fargs, 1);
        if (!PyObject_TypeCheck(obj, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
        int nd = PyArray_NDIM((PyArrayObject *)obj);
        if (axis < 0) axis += nd;
        if (axis < 0 || axis >= nd) { PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL; }

        /* Ensure labels block exists */
        LabeledArrayObject *lai = (LabeledArrayObject *)obj;
        LabelsBlock *lb = lai->labels_block;
        if (!lb) { lb = labelsblock_new_default((PyArrayObject *)obj); if (!lb) return NULL; lai->labels_block = lb; }

        /* Build separate tuples: ck_data used for subscripting (ints only), ck_lbl used for labels construction */
        PyObject *ck_data = PyTuple_New(nd);
        if (!ck_data) return NULL;
        PyObject *ck_lbl = PyTuple_New(nd);
        if (!ck_lbl) { Py_DECREF(ck_data); return NULL; }
        for (int ax = 0; ax < nd; ++ax) {
            if (ax == axis) {
                /* Fill later with sel */
                Py_INCREF(Py_None);
                PyTuple_SET_ITEM(ck_data, ax, Py_None);
                Py_INCREF(Py_None);
                PyTuple_SET_ITEM(ck_lbl, ax, Py_None);
            } else {
                PyObject *sl = PySlice_New(NULL, NULL, NULL);
                if (!sl) { Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                PyTuple_SET_ITEM(ck_data, ax, sl);
                Py_INCREF(sl);
                PyTuple_SET_ITEM(ck_lbl, ax, sl);
            }
        }

        /* Prepare axis index for data and labels */
        PyObject *sel_data = NULL;
        PyObject *sel_lbl = NULL;

        if (PyArray_Check(indices)) {
            PyArrayObject *arrk = (PyArrayObject *)indices;
            int t = PyArray_TYPE(arrk);
            if (PyTypeNum_ISINTEGER(t)) {
                sel_data = (PyObject *)arrk; Py_INCREF(sel_data);
                sel_lbl = (PyObject *)arrk; Py_INCREF(sel_lbl);
            } else if (t == NPY_UNICODE || t == NPY_STRING || t == NPY_OBJECT) {
                /* Map labels to intp for data; keep original for labels (supports ND indices) */
                PyArrayObject *ind = map_label_index_array_to_int(lb, axis, arrk);
                if (!ind) {
                    if (!PyErr_Occurred()) PyErr_SetString(PyExc_IndexError, "label not found in axis for take");
                    Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL;
                }
                sel_data = (PyObject *)ind; /* new ref */
                sel_lbl = (PyObject *)arrk; Py_INCREF(sel_lbl);
            } else {
                PyErr_SetString(PyExc_TypeError, "unsupported indices dtype for take");
                Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL;
            }
        } else {
            /* Scalar or Python sequence (list/tuple). Treat unicode scalars as labels, not sequences of chars. */
            if (PyUnicode_Check(indices) || PyBytes_Check(indices)) {
                Py_ssize_t pos = -1;
                if (labelsblock_find(lb, axis, indices, &pos) != 0) {
                    Py_DECREF(ck_data); Py_DECREF(ck_lbl);
                    PyErr_SetString(PyExc_IndexError, "label not found in axis for take");
                    return NULL;
                }
                PyObject *idx = PyLong_FromSsize_t(pos);
                if (!idx) { Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                sel_data = idx; /* new ref */
                sel_lbl = idx; Py_INCREF(sel_lbl);
            } else {
                PyObject *idx_int = PyNumber_Index(indices);
                if (idx_int) {
                    sel_data = idx_int; /* new ref */
                    sel_lbl = idx_int; Py_INCREF(sel_lbl);
                } else {
                    PyErr_Clear();
                    PyObject *seq = PySequence_Fast(indices, "expected sequence for indices");
                    if (!seq) { Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                    int nested = 0;
                    for (Py_ssize_t i = 0; i < n; ++i) {
                        PyObject *it = PySequence_Fast_GET_ITEM(seq, i);
                        if (PyArray_Check(it) || PyList_Check(it) || PyTuple_Check(it)) { nested = 1; break; }
                    }
                    if (nested) {
                        Py_DECREF(seq);
                        PyArrayObject *arrk = (PyArrayObject *)PyArray_FromAny(indices, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                        if (!arrk) { Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                        int t = PyArray_TYPE(arrk);
                        if (PyTypeNum_ISINTEGER(t)) {
                            sel_data = (PyObject *)arrk;
                            sel_lbl = (PyObject *)arrk; Py_INCREF(sel_lbl);
                        } else if (t == NPY_UNICODE || t == NPY_STRING || t == NPY_OBJECT) {
                            PyArrayObject *ind = map_label_index_array_to_int(lb, axis, arrk);
                            if (!ind) {
                                if (!PyErr_Occurred()) PyErr_SetString(PyExc_IndexError, "label not found in axis for take");
                                Py_DECREF(arrk); Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL;
                            }
                            sel_data = (PyObject *)ind;
                            sel_lbl = (PyObject *)arrk; /* keep arrk ref for labels */
                        } else {
                            Py_DECREF(arrk); Py_DECREF(ck_data); Py_DECREF(ck_lbl);
                            PyErr_SetString(PyExc_TypeError, "unsupported indices dtype for take");
                            return NULL;
                        }
                    } else {
                    PyObject *lst = PyList_New(n);
                    if (!lst) { Py_DECREF(seq); Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                    for (Py_ssize_t i = 0; i < n; ++i) {
                        PyObject *it = PySequence_Fast_GET_ITEM(seq, i);
                        PyObject *maybe_int = PyNumber_Index(it);
                        PyObject *to_set = NULL;
                        if (maybe_int) {
                            to_set = maybe_int; /* new ref */
                        } else {
                            PyErr_Clear();
                            if (PyUnicode_Check(it) || PyBytes_Check(it)) {
                                Py_ssize_t pos = -1;
                                if (labelsblock_find(lb, axis, it, &pos) != 0) {
                                    Py_DECREF(seq); Py_DECREF(lst); Py_DECREF(ck_data); Py_DECREF(ck_lbl);
                                    PyErr_SetString(PyExc_IndexError, "label not found in axis for take");
                                    return NULL;
                                }
                                to_set = PyLong_FromSsize_t(pos);
                                if (!to_set) { Py_DECREF(seq); Py_DECREF(lst); Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                            } else {
                                Py_DECREF(seq); Py_DECREF(lst); Py_DECREF(ck_data); Py_DECREF(ck_lbl);
                                PyErr_SetString(PyExc_TypeError, "non-integer index in indices");
                                return NULL;
                            }
                        }
                        PyList_SET_ITEM(lst, i, to_set); /* steals */
                    }
                    Py_DECREF(seq);
                    sel_data = lst;
                    sel_lbl = lst; Py_INCREF(sel_lbl);
                    }
                }
            }
        }

        /* Put selections */
        PyObject *tmpd = PyTuple_GET_ITEM(ck_data, axis);
        Py_DECREF(tmpd);
        PyTuple_SET_ITEM(ck_data, axis, sel_data);
        PyObject *tmpl = PyTuple_GET_ITEM(ck_lbl, axis);
        Py_DECREF(tmpl);
        PyTuple_SET_ITEM(ck_lbl, axis, sel_lbl);

        /* Do data selection */
        PyObject *res = PyArray_Type.tp_as_mapping->mp_subscript(obj, ck_data);
        Py_DECREF(ck_data);
        if (!res) { Py_DECREF(ck_lbl); return NULL; }

        /* Wrap and attach labels according to ck_lbl */
        if (PyArray_Check(res)) {
            PyObject *target = res;
            if (!PyObject_TypeCheck(res, (PyTypeObject *)Py_TYPE(self_obj))) {
                PyObject *viewres = PyArray_View((PyArrayObject *)res, NULL, (PyTypeObject *)Py_TYPE(self_obj));
                Py_DECREF(res);
                if (!viewres) { Py_DECREF(ck_lbl); return NULL; }
                target = viewres;
            }
            LabeledArrayObject *objla = (LabeledArrayObject *)self_obj;
            LabelsBlock *parent_lb = objla->labels_block;
            if (parent_lb) {
                LabelsBlock *child = labelsblock_slice(parent_lb, ck_lbl, (PyArrayObject *)target);
                if (child) {
                    LabeledArrayObject *tobj = (LabeledArrayObject *)target;
                    LabelsBlock *old = tobj->labels_block;
                    tobj->labels_block = child;
                    labelsblock_decref(old);
                    labels_cache_clear(tobj);
                }
            }
            Py_DECREF(ck_lbl);
            return target;
        }
        Py_DECREF(ck_lbl);
        return res;
    }

    if (is_concatenate) {
        if (axis < 0) axis += nd;
        if (axis < 0 || axis >= nd) { Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL; }
    } else if (is_stack || is_expand) {
        int nd_new = nd + 1;
        if (axis < 0) axis += nd_new;
        if (axis < 0 || axis > nd) { if (arrays) Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL; }
    }

    /* Ensure inputs are LabeledArray and compatible (for concatenate/stack) */
    LabeledArrayObject *first_la = NULL; LabelsBlock *first_lb = NULL;
    if (is_concatenate || is_stack) {
        if (PyObject_TypeCheck(first, &LabeledArray_Type)) {
            first_la = (LabeledArrayObject *)first;
        } else { Py_DECREF(arrays); Py_RETURN_NOTIMPLEMENTED; }
        first_lb = first_la->labels_block;
        if (!first_lb) { first_lb = labelsblock_new_default((PyArrayObject *)first); if (!first_lb) { Py_DECREF(arrays); return NULL; } first_la->labels_block = first_lb; }
        for (Py_ssize_t i = 1; i < narr; ++i) {
            PyObject *obji = PySequence_Fast_GET_ITEM(arrays, i);
            if (!PyObject_TypeCheck(obji, &LabeledArray_Type)) { Py_DECREF(arrays); Py_RETURN_NOTIMPLEMENTED; }
            if (PyArray_NDIM((PyArrayObject *)obji) != nd) { Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "all inputs must have same ndim"); return NULL; }
            LabeledArrayObject *lai = (LabeledArrayObject *)obji;
            LabelsBlock *lbi = lai->labels_block;
            if (!lbi) { lbi = labelsblock_new_default((PyArrayObject *)obji); if (!lbi) { Py_DECREF(arrays); return NULL; } lai->labels_block = lbi; }
            for (int ax = 0; ax < nd; ++ax) {
                if (is_concatenate && ax == axis) continue;
                if (first_lb->axis_len[ax] != lbi->axis_len[ax]) { Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "labels mismatch across axes"); return NULL; }
                int n = first_lb->axis_len[ax];
                for (int j = 0; j < n; ++j) {
                    const char *a = first_lb->axis_labels[ax][j];
                    const char *b = lbi->axis_labels[ax][j];
                    if ((a && b && strcmp(a, b) != 0) || (a == NULL && b != NULL) || (a != NULL && b == NULL)) {
                        Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "labels differ across axes"); return NULL;
                    }
                }
            }
        }
    }

    if (is_concatenate) {
        /* Concatenate data via C-API */
        PyObject *seq0 = PyTuple_GET_ITEM(fargs, 0);
        PyObject *res_base = PyArray_Concatenate(seq0, axis);
        if (!res_base) { Py_DECREF(arrays); return NULL; }

        PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(res_base);
        if (!view) { Py_DECREF(arrays); return NULL; }

        /* Build concatenated labels block */
        LabelsBlock *out_lb = labelsblock_alloc(nd);
        if (!out_lb) { Py_DECREF(view); Py_DECREF(arrays); return NULL; }

        for (int ax = 0; ax < nd; ++ax) {
            if (ax == axis) {
                int total = 0;
                for (Py_ssize_t i = 0; i < narr; ++i) {
                    LabeledArrayObject *lai = (LabeledArrayObject *)PySequence_Fast_GET_ITEM(arrays, i);
                    total += lai->labels_block->axis_len[ax];
                }
                out_lb->axis_len[ax] = total;
                out_lb->axis_labels[ax] = (char **)calloc(total, sizeof(char *));
                if (!out_lb->axis_labels[ax]) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
                int pos = 0;
                for (Py_ssize_t i = 0; i < narr; ++i) {
                    LabeledArrayObject *lai = (LabeledArrayObject *)PySequence_Fast_GET_ITEM(arrays, i);
                    LabelsBlock *lbi = lai->labels_block;
                    int n = lbi->axis_len[ax];
                    for (int j = 0; j < n; ++j, ++pos) {
                        const char *s = lbi->axis_labels[ax][j];
                        size_t L = strlen(s ? s : "") + 1;
                        out_lb->axis_labels[ax][pos] = (char *)PyDataMem_NEW(L);
                        if (!out_lb->axis_labels[ax][pos]) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
                        memcpy(out_lb->axis_labels[ax][pos], s ? s : "", L);
                    }
                }
                if (axhash_build(&out_lb->axis_hash[ax], (const char **)out_lb->axis_labels[ax], total) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
            } else {
                int n = first_lb->axis_len[ax];
                if (labelsblock_set_axis_copy(out_lb, ax, first_lb->axis_labels[ax], n) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
            }
        }

        labelsblock_attach_to_view(view, out_lb);

        Py_DECREF(arrays);
        return view;
    }

    if (is_stack) {
        /* Implement stack via C-API: expand dims for each array, then concatenate on that axis */
        int nd_new = nd + 1;
        PyObject *stack_inputs = PyTuple_New(narr);
        if (!stack_inputs) { Py_DECREF(arrays); return NULL; }
        for (Py_ssize_t i = 0; i < narr; ++i) {
            PyObject *obji = PySequence_Fast_GET_ITEM(arrays, i); /* borrowed */
            if (!PyArray_Check(obji)) { Py_DECREF(stack_inputs); Py_DECREF(arrays); Py_RETURN_NOTIMPLEMENTED; }
            const npy_intp *orig_dims = PyArray_DIMS((PyArrayObject *)obji);
            npy_intp *new_dims = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)nd_new);
            if (!new_dims) { Py_DECREF(stack_inputs); Py_DECREF(arrays); return NULL; }
            int p2 = 0;
            for (int ax = 0; ax < nd_new; ++ax) {
                if (ax == axis) new_dims[ax] = 1;
                else new_dims[ax] = orig_dims[p2++];
            }
            PyArray_Dims nds2; nds2.ptr = new_dims; nds2.len = nd_new;
            PyObject *reshaped = PyArray_Newshape((PyArrayObject *)obji, &nds2, NPY_CORDER);
            PyMem_Free(new_dims);
            if (!reshaped) { Py_DECREF(stack_inputs); Py_DECREF(arrays); return NULL; }
            PyTuple_SET_ITEM(stack_inputs, i, reshaped); /* steals */
        }
        PyObject *res_base = PyArray_Concatenate(stack_inputs, axis);
        Py_DECREF(stack_inputs);
        if (!res_base) { Py_DECREF(arrays); return NULL; }

        PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(res_base);
        if (!view) { Py_DECREF(arrays); return NULL; }

        LabelsBlock *out_lb = labelsblock_alloc(nd_new);
        if (!out_lb) { Py_DECREF(view); Py_DECREF(arrays); return NULL; }

        for (int ax = 0, src = 0; ax < nd_new; ++ax) {
            if (ax == axis) {
                int nnew = (int)narr;
                if (labelsblock_set_axis_numeric(out_lb, ax, nnew) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
            } else {
                int n = first_lb->axis_len[src];
                if (labelsblock_set_axis_copy(out_lb, ax, first_lb->axis_labels[src], n) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
                src++;
            }
        }

        labelsblock_attach_to_view(view, out_lb);

        Py_DECREF(arrays);
        return view;
    }

    /* Handle squeeze */
    /* fargs: (arr,) */
    PyObject *obj = PyTuple_GET_ITEM(fargs, 0);
    if (is_squeeze) {
        if (!PyObject_TypeCheck(obj, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }

    int orig_nd = PyArray_NDIM((PyArrayObject *)obj);
    const npy_intp *orig_dims = PyArray_DIMS((PyArrayObject *)obj);

    /* Build drop mask */
    int *drop = (int *)calloc((size_t)orig_nd, sizeof(int));
    if (!drop) { Py_DECREF(arrays); return NULL; }
    int any_drop = 0;
    if (fkwargs && fkwargs != Py_None && PyDict_Check(fkwargs)) {
        PyObject *axis_obj = PyDict_GetItemString(fkwargs, "axis");
        if (axis_obj && axis_obj != Py_None) {
            if (PyLong_Check(axis_obj)) {
                long axl = PyLong_AsLong(axis_obj);
                if (axl == -1 && PyErr_Occurred()) { free(drop); Py_DECREF(arrays); return NULL; }
                int ax = (int)axl; if (ax < 0) ax += orig_nd;
                if (ax < 0 || ax >= orig_nd) { free(drop); Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL; }
                if (orig_dims[ax] != 1) { free(drop); Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "cannot squeeze axis with size != 1"); return NULL; }
                drop[ax] = 1; any_drop = 1;
            } else {
                PyObject *seq = PySequence_Fast(axis_obj, "axis must be int or sequence of int");
                if (!seq) { free(drop); Py_DECREF(arrays); return NULL; }
                Py_ssize_t na = PySequence_Fast_GET_SIZE(seq);
                for (Py_ssize_t i = 0; i < na; ++i) {
                    PyObject *it = PySequence_Fast_GET_ITEM(seq, i);
                    long axl = PyLong_AsLong(it);
                    if (axl == -1 && PyErr_Occurred()) { Py_DECREF(seq); free(drop); Py_DECREF(arrays); return NULL; }
                    int ax = (int)axl; if (ax < 0) ax += orig_nd;
                    if (ax < 0 || ax >= orig_nd) { Py_DECREF(seq); free(drop); Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL; }
                    if (orig_dims[ax] != 1) { Py_DECREF(seq); free(drop); Py_DECREF(arrays); PyErr_SetString(PyExc_ValueError, "cannot squeeze axis with size != 1"); return NULL; }
                    if (!drop[ax]) { drop[ax] = 1; any_drop = 1; }
                }
                Py_DECREF(seq);
            }
        }
    }
    if (!any_drop) {
        for (int ax = 0; ax < orig_nd; ++ax) if (orig_dims[ax] == 1) { drop[ax] = 1; any_drop = 1; }
    }
    if (!any_drop) { /* nothing to do, return view of self */ free(drop); Py_INCREF(obj); return obj; }

    int new_nd = 0; for (int ax = 0; ax < orig_nd; ++ax) if (!drop[ax]) new_nd++;
    npy_intp *new_dims = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)new_nd);
    if (!new_dims) { free(drop); Py_DECREF(arrays); return NULL; }
    int p = 0; for (int ax = 0; ax < orig_nd; ++ax) if (!drop[ax]) new_dims[p++] = orig_dims[ax];

    PyArray_Dims nds; nds.ptr = new_dims; nds.len = new_nd;
    PyObject *res_base2 = PyArray_Newshape((PyArrayObject *)obj, &nds, NPY_CORDER);
    PyMem_Free(new_dims);
    if (!res_base2) { free(drop); return NULL; }

    if (PyArray_NDIM((PyArrayObject *)res_base2) == 0) {
        PyObject *scalar = PyObject_CallMethod(res_base2, "item", NULL);
        Py_DECREF(res_base2); free(drop); return scalar;
    }

    PyObject *view2 = PyArray_View((PyArrayObject *)res_base2, NULL, (PyTypeObject *)Py_TYPE(self_obj));
    Py_DECREF(res_base2);
    if (!view2) { free(drop); return NULL; }

    /* Build squeezed labels */
    LabeledArrayObject *lai0 = (LabeledArrayObject *)obj;
    LabelsBlock *plb = lai0->labels_block;
    if (!plb) { plb = labelsblock_new_default((PyArrayObject *)obj); if (!plb) { Py_DECREF(view2); free(drop); return NULL; } lai0->labels_block = plb; }
        LabelsBlock *out_lb2 = labelsblock_alloc(new_nd);
    if (!out_lb2) { Py_DECREF(view2); free(drop); return NULL; }
    int dst = 0;
    for (int ax = 0; ax < orig_nd; ++ax) {
        if (drop[ax]) continue;
        int n = plb->axis_len[ax];
            if (labelsblock_set_axis_copy(out_lb2, dst, plb->axis_labels[ax], n) < 0) { labelsblock_decref(out_lb2); Py_DECREF(view2); free(drop); return NULL; }
        dst++;
    }

        labelsblock_attach_to_view(view2, out_lb2);

    free(drop);
    return view2;
    }

    /* Handle expand_dims */
    if (is_expand) {
        PyObject *obj0 = PyTuple_GET_ITEM(fargs, 0);
        if (!PyObject_TypeCheck(obj0, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
        int nd_new = nd + 1;
        npy_intp *new_dims = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)nd_new);
        if (!new_dims) return NULL;
        const npy_intp *orig_dims2 = PyArray_DIMS((PyArrayObject *)obj0);
        int p2 = 0;
        for (int ax = 0; ax < nd_new; ++ax) {
            if (ax == axis) new_dims[ax] = 1;
            else new_dims[ax] = orig_dims2[p2++];
        }
        PyArray_Dims nds2; nds2.ptr = new_dims; nds2.len = nd_new;
        PyObject *res_base3 = PyArray_Newshape((PyArrayObject *)obj0, &nds2, NPY_CORDER);
        PyMem_Free(new_dims);
        if (!res_base3) return NULL;
        PyObject *view3 = PyArray_View((PyArrayObject *)res_base3, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(res_base3);
        if (!view3) return NULL;

        LabeledArrayObject *lai = (LabeledArrayObject *)obj0;
        LabelsBlock *plb = lai->labels_block;
        if (!plb) { plb = labelsblock_new_default((PyArrayObject *)obj0); if (!plb) { Py_DECREF(view3); return NULL; } lai->labels_block = plb; }

        LabelsBlock *out_lb3 = labelsblock_alloc(nd_new);
        if (!out_lb3) { Py_DECREF(view3); return NULL; }
        for (int ax = 0, src = 0; ax < nd_new; ++ax) {
            if (ax == axis) {
                if (labelsblock_set_axis_constant(out_lb3, ax, "1") < 0) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
            } else {
                int n = plb->axis_len[src];
                if (labelsblock_set_axis_copy(out_lb3, ax, plb->axis_labels[src], n) < 0) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
                src++;
            }
        }

        labelsblock_attach_to_view(view3, out_lb3);
        return view3;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

/* ---------- __array_ufunc__ (generic ufunc handler; delegates and wraps) ---------- */
static PyObject *
array_ufunc(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    /* args = (ufunc, method, *inputs) */
    if (!PyTuple_Check(args) || PyTuple_GET_SIZE(args) < 2) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PyObject *ufunc = PyTuple_GET_ITEM(args, 0);
    PyObject *method = PyTuple_GET_ITEM(args, 1);

    Py_ssize_t nin = PyTuple_GET_SIZE(args) - 2;
    PyObject *inputs = PyTuple_New(nin);
    if (!inputs) return NULL;
    for (Py_ssize_t i = 0; i < nin; ++i) {
        PyObject *it = PyTuple_GET_ITEM(args, (Py_ssize_t)2 + i);
        Py_INCREF(it);
        PyTuple_SET_ITEM(inputs, i, it);
    }

    /* Call getattr(ufunc, method)(*inputs, **kwargs) */
    PyObject *meth = PyObject_GetAttr(ufunc, method);
    if (!meth) { Py_DECREF(inputs); Py_RETURN_NOTIMPLEMENTED; }
    /* Convert LabeledArray inputs to base ndarray views to avoid recursive dispatch */
    PyObject *inputs2 = PyTuple_New(nin);
    if (!inputs2) { Py_DECREF(meth); Py_DECREF(inputs); return NULL; }
    for (Py_ssize_t i = 0; i < nin; ++i) {
        PyObject *it = PyTuple_GET_ITEM(inputs, i); /* borrowed */
        PyObject *to_set = it;
        if (PyObject_TypeCheck(it, &LabeledArray_Type)) {
            to_set = PyArray_View((PyArrayObject *)it, NULL, &PyArray_Type);
            if (!to_set) { Py_DECREF(inputs2); Py_DECREF(meth); Py_DECREF(inputs); return NULL; }
        } else {
            Py_INCREF(to_set);
        }
        PyTuple_SET_ITEM(inputs2, i, to_set); /* steals */
    }

    /* Copy kwargs and convert 'out' and 'where' to base ndarray views if provided */
    PyObject *kwargs2 = NULL;
    kwargs2 = convert_kwargs_out_where(kwargs);

    PyObject *res = PyObject_Call(meth, inputs2, kwargs2 ? kwargs2 : kwargs);
    Py_DECREF(meth);
    Py_DECREF(inputs2);
    Py_DECREF(inputs);
    if (kwargs2) Py_DECREF(kwargs2);
    if (!res) return NULL;

    /* If 'out' is provided and points to LabeledArray(s), return original out (in-place op) */
    if (kwargs && PyDict_Check(kwargs)) {
        PyObject *out_kw = PyDict_GetItemString(kwargs, "out"); /* borrowed */
        int return_direct = 0;
        PyObject *ret_obj = NULL;
        if (out_kw && out_kw != Py_None) {
            if (PyTuple_Check(out_kw)) {
                Py_ssize_t nout = PyTuple_GET_SIZE(out_kw);
                for (Py_ssize_t i = 0; i < nout; ++i) {
                    PyObject *oi = PyTuple_GET_ITEM(out_kw, i);
                    if (oi && oi != Py_None && PyObject_TypeCheck(oi, &LabeledArray_Type)) { return_direct = 1; if (!ret_obj) ret_obj = oi; }
                }
            } else if (PyObject_TypeCheck(out_kw, &LabeledArray_Type)) {
                return_direct = 1; ret_obj = out_kw;
            }
        }
        /* Also guard the self-in-place case */
        if (res == self_obj && !ret_obj) { ret_obj = self_obj; return_direct = 1; }
        if (return_direct) {
            if (!ret_obj) ret_obj = res;
            Py_INCREF(ret_obj);
            Py_DECREF(res);
            return ret_obj;
        }
    }

    /* Wrap using __array_wrap__ to preserve labels and scalar behavior */
    PyObject *context = Py_BuildValue("(OOO)", ufunc, method, Py_None);
    if (!context) { Py_DECREF(res); return NULL; }
    PyObject *wrapped = PyObject_CallMethod(self_obj, "__array_wrap__", "OO", res, context);
    Py_DECREF(context);
    Py_DECREF(res);
    if (!wrapped) return NULL;
    return wrapped;
}

static PyObject *
LabeledArray_find(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"label", "axis", NULL};
    PyObject *label_obj = NULL;
    int axis = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi:find", kwlist, &label_obj, &axis)) {
        return NULL;
    }

    if (!PyUnicode_Check(label_obj)) {
        PyErr_SetString(PyExc_TypeError, "label must be a string");
        return NULL;
    }

    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    LabelsBlock *lb = obj->labels_block;
    if (!lb) {
        lb = labelsblock_new_default((PyArrayObject *)self);
        if (!lb) return NULL;
        obj->labels_block = lb;
    }

    if (axis < 0 || axis >= lb->ndim) {
        PyErr_SetString(PyExc_ValueError, "axis out of range");
        return NULL;
    }

    Py_ssize_t pos = -1;
    int rc = labelsblock_find(lb, axis, label_obj, &pos);
    if (rc != 0) {
        PyErr_SetString(PyExc_ValueError, "label not found in axis");
        return NULL;
    }

    return PyLong_FromSsize_t(pos);
}

/* ---------- take(indices, axis=0) with label propagation ---------- */
static PyObject *
LabeledArray_take(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyObject *indices = NULL; int axis = 0;
    static char *kwlist[] = {"indices", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:take", kwlist, &indices, &axis)) return NULL;
    if (!PyObject_TypeCheck(self_obj, &LabeledArray_Type)) { Py_RETURN_NOTIMPLEMENTED; }
    int nd = PyArray_NDIM((PyArrayObject *)self_obj);
    if (axis < 0) axis += nd;
    if (axis < 0 || axis >= nd) { PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL; }

    /* Data path via numpy.take (cached) */
    static PyObject *np_take = NULL;
    if (!np_take) {
        PyObject *numpy = PyImport_ImportModule("numpy");
        if (!numpy) return NULL;
        np_take = PyObject_GetAttrString(numpy, "take");
        Py_DECREF(numpy);
        if (!np_take) return NULL;
    }
    PyObject *res_base = PyObject_CallFunction(np_take, "OOi", self_obj, indices, axis);
    if (!res_base) return NULL;
    if (!PyArray_Check(res_base)) return res_base;
    PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
    Py_DECREF(res_base);
    if (!view) return NULL;

    /* Build ck for labels using original indices to preserve order */
    LabeledArrayObject *lai = (LabeledArrayObject *)self_obj;
    LabelsBlock *plb = lai->labels_block;
    if (!plb) { plb = labelsblock_new_default((PyArrayObject *)self_obj); if (!plb) { Py_DECREF(view); return NULL; } lai->labels_block = plb; }
    PyObject *ck = PyTuple_New(nd);
    if (!ck) { Py_DECREF(view); return NULL; }
    for (int ax = 0; ax < nd; ++ax) {
        if (ax == axis) {
            if (PyArray_Check(indices)) {
                PyArrayObject *arrk = (PyArrayObject *)PyArray_FromAny(indices, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                if (!arrk) { Py_DECREF(ck); Py_DECREF(view); return NULL; }
                int t = PyArray_TYPE(arrk);
                PyArrayObject *ind = NULL;
                int mapped = 0;
                if (PyTypeNum_ISINTEGER(t)) {
                    ind = (PyArrayObject *)PyArray_FromAny((PyObject *)arrk, PyArray_DescrFromType(NPY_INTP), 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                } else if (t == NPY_UNICODE || t == NPY_STRING || t == NPY_OBJECT) {
                    mapped = 1;
                    ind = map_label_index_array_to_int(plb, ax, arrk);
                }
                Py_DECREF(arrk);
                if (!ind) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetString(mapped ? PyExc_IndexError : PyExc_TypeError,
                                        mapped ? "label not found in axis for take" : "unsupported indices for take");
                    }
                    Py_DECREF(ck); Py_DECREF(view); return NULL;
                }
                PyTuple_SET_ITEM(ck, ax, (PyObject *)ind);
            } else if (PyList_Check(indices) || PyTuple_Check(indices)) {
                PyObject *seq = PySequence_Fast(indices, "expected sequence");
                if (!seq) { Py_DECREF(ck); Py_DECREF(view); return NULL; }
                Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                int nested = 0;
                for (Py_ssize_t j = 0; j < n; ++j) {
                    PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
                    if (PyArray_Check(it) || PyList_Check(it) || PyTuple_Check(it)) { nested = 1; break; }
                }
                if (nested) {
                    Py_DECREF(seq);
                    PyArrayObject *arrk = (PyArrayObject *)PyArray_FromAny(indices, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                    if (!arrk) { Py_DECREF(ck); Py_DECREF(view); return NULL; }
                    int t = PyArray_TYPE(arrk);
                    PyArrayObject *ind = NULL;
                    int mapped = 0;
                    if (PyTypeNum_ISINTEGER(t)) {
                        ind = (PyArrayObject *)PyArray_FromAny((PyObject *)arrk, PyArray_DescrFromType(NPY_INTP), 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                    } else if (t == NPY_UNICODE || t == NPY_STRING || t == NPY_OBJECT) {
                        mapped = 1;
                        ind = map_label_index_array_to_int(plb, ax, arrk);
                    }
                    Py_DECREF(arrk);
                    if (!ind) {
                        if (!PyErr_Occurred()) {
                            PyErr_SetString(mapped ? PyExc_IndexError : PyExc_TypeError,
                                            mapped ? "label not found in axis for take" : "unsupported indices for take");
                        }
                        Py_DECREF(ck); Py_DECREF(view); return NULL;
                    }
                    PyTuple_SET_ITEM(ck, ax, (PyObject *)ind);
                } else {
                    PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(1, (npy_intp *)&n, NPY_INTP);
                    if (!ind) { Py_DECREF(seq); Py_DECREF(ck); Py_DECREF(view); return NULL; }
                    char *ibase = PyArray_BYTES(ind);
                    npy_intp istride = PyArray_STRIDES(ind)[0];
                    for (Py_ssize_t j = 0; j < n; ++j) {
                        PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
                        PyObject *maybe_int = PyNumber_Index(it);
                        npy_intp val;
                        if (maybe_int) {
                            long v = PyLong_AsLong(maybe_int); Py_DECREF(maybe_int);
                            if (v == -1 && PyErr_Occurred()) { Py_DECREF(seq); Py_DECREF(ind); Py_DECREF(ck); Py_DECREF(view); return NULL; }
                            val = (npy_intp)v;
                        } else {
                            PyErr_Clear(); Py_ssize_t pos = -1;
                            if (labelsblock_find(plb, ax, it, &pos) != 0) { Py_DECREF(seq); Py_DECREF(ind); Py_DECREF(ck); Py_DECREF(view); PyErr_SetString(PyExc_IndexError, "label not found in axis for take"); return NULL; }
                            val = (npy_intp)pos;
                        }
                        write_intp_1d(ibase, istride, (npy_intp)j, val);
                    }
                    Py_DECREF(seq);
                    PyTuple_SET_ITEM(ck, ax, (PyObject *)ind);
                }
            } else {
                Py_INCREF(indices);
                PyTuple_SET_ITEM(ck, ax, indices);
            }
        } else {
            PyObject *sl = PySlice_New(NULL, NULL, NULL);
            if (!sl) { Py_DECREF(ck); Py_DECREF(view); return NULL; }
            PyTuple_SET_ITEM(ck, ax, sl);
        }
    }
    LabelsBlock *child = labelsblock_slice(plb, ck, (PyArrayObject *)view);
    Py_DECREF(ck);
    if (child) {
        LabeledArrayObject *tobj = (LabeledArrayObject *)view;
        LabelsBlock *old = tobj->labels_block;
        tobj->labels_block = child;
        labelsblock_decref(old);
        labels_cache_clear(tobj);
    }
    return view;
}

/* ---------- combine(levels=(i,j), delim=None) ---------- */
static PyObject *
LabeledArray_combine(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"levels", "delim", NULL};
    PyObject *levels = NULL; const char *delim = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s:combine", kwlist, &levels, &delim)) {
        return NULL;
    }
    if (!PyTuple_Check(levels) || PyTuple_GET_SIZE(levels) != 2) {
        PyErr_SetString(PyExc_ValueError, "levels must be a tuple of two ints");
        return NULL;
    }
    long a0 = PyLong_AsLong(PyTuple_GET_ITEM(levels, 0));
    long a1 = PyLong_AsLong(PyTuple_GET_ITEM(levels, 1));
    if ((a0 == -1 || a1 == -1) && PyErr_Occurred()) return NULL;

    LabeledArrayObject *self = (LabeledArrayObject *)self_obj;
    PyArrayObject *arr = (PyArrayObject *)self_obj;
    int nd = PyArray_NDIM(arr);
    if (a0 < 0) a0 += nd;
    if (a1 < 0) a1 += nd;
    if (a0 < 0 || a1 < 0 || a0 >= nd || a1 >= nd || a1 <= a0) {
        PyErr_SetString(PyExc_ValueError, "invalid levels");
        return NULL;
    }

    /* const npy_intp *dims = PyArray_DIMS(arr); */

    /* Rework using C-API: move axes and reshape to merge, avoiding Python loops */
    PyArrayObject *arr0 = (PyArrayObject *)self_obj;
    /* Bring a0 adjacent to a1 and record merge position */
    npy_intp *perm = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)nd);
    if (!perm) return NULL;
    for (int i = 0; i < nd; ++i) perm[i] = (npy_intp)i;
    int merge_pos = 0;
    if (a0 < a1) {
        for (int p = a0; p < a1 - 1; ++p) { npy_intp t = perm[p]; perm[p] = perm[p+1]; perm[p+1] = t; }
        merge_pos = a1 - 1;
    } else {
        for (int p = a0; p > a1; --p) { npy_intp t = perm[p]; perm[p] = perm[p-1]; perm[p-1] = t; }
        merge_pos = a1;
    }
    PyArray_Dims pd; pd.ptr = perm; pd.len = nd;
    PyObject *moved = PyArray_Transpose(arr0, &pd);
    PyMem_Free(perm);
    if (!moved) return NULL;

    /* Now merge axes (a1-1, a1) into one: compute new dims */
    int nd_new = nd - 1;
    const npy_intp *dims0 = PyArray_DIMS((PyArrayObject *)moved);
    npy_intp *new_dims = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)nd_new);
    if (!new_dims) { Py_DECREF(moved); return NULL; }
    int w = 0;
        for (int ax = 0; ax < nd; ++ax) {
        if (ax == merge_pos) {
            npy_intp merged = dims0[ax] * dims0[ax + 1];
            new_dims[w++] = merged;
            ax++; /* skip next */
        } else {
            new_dims[w++] = dims0[ax];
        }
    }
    PyArray_Dims nds; nds.ptr = new_dims; nds.len = nd_new;
    PyObject *reshaped = PyArray_Newshape((PyArrayObject *)moved, &nds, NPY_CORDER);
    Py_DECREF(moved);
    PyMem_Free(new_dims);
    if (!reshaped) return NULL;
    PyObject *view = PyArray_View((PyArrayObject *)reshaped, NULL, (PyTypeObject *)Py_TYPE(self_obj));
    Py_DECREF(reshaped);
    if (!view) return NULL;

    /* Build labels: drop axis a0; join labels from a0 and a1 into new a1 labels */
    LabelsBlock *plb = self->labels_block;
    if (!plb) { plb = labelsblock_new_default((PyArrayObject *)self_obj); if (!plb) { Py_DECREF(view); return NULL; } self->labels_block = plb; }
    int new_nd = nd - 1;
    LabelsBlock *out = labelsblock_alloc(new_nd);
    if (!out) { Py_DECREF(view); return NULL; }

    const char *joiner = delim ? delim : (self->delimiter ? self->delimiter : "-");
    size_t joiner_len = strlen(joiner);
    int dst = 0;
    for (int ax = 0; ax < nd; ++ax) {
        if (ax == a0) continue; /* drop */
        if (ax == a1) {
            int n0 = plb->axis_len[a0];
            int n1 = plb->axis_len[a1];
            size_t n01 = (size_t)n0 * (size_t)n1;
            out->axis_len[dst] = (int)n01;
            out->axis_labels[dst] = (char **)calloc(n01, sizeof(char *));
            if (!out->axis_labels[dst]) { labelsblock_decref(out); Py_DECREF(view); return NULL; }

            size_t *len0 = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)n0);
            size_t *len1 = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)n1);
            if (!len0 || !len1) { PyMem_Free(len0); PyMem_Free(len1); labelsblock_decref(out); Py_DECREF(view); return NULL; }
            size_t sum0 = 0, sum1 = 0;
            for (int i = 0; i < n0; ++i) {
                const char *s = plb->axis_labels[a0][i];
                len0[i] = strlen(s ? s : "");
                sum0 += len0[i];
            }
            for (int j = 0; j < n1; ++j) {
                const char *s = plb->axis_labels[a1][j];
                len1[j] = strlen(s ? s : "");
                sum1 += len1[j];
            }

            size_t total = sum0 * (size_t)n1 + sum1 * (size_t)n0 + (joiner_len + 1) * n01;
            char *slab = (char *)PyDataMem_NEW(total);
            if (!slab) { PyMem_Free(len0); PyMem_Free(len1); labelsblock_decref(out); Py_DECREF(view); return NULL; }
            out->axis_slab[dst] = slab;

            int single_delim = (joiner_len == 1);
            char dch = single_delim ? joiner[0] : '\0';
            char *cursor = slab;
            NPY_BEGIN_ALLOW_THREADS;
            for (int i = 0; i < n0; ++i) {
                const char *s0 = plb->axis_labels[a0][i];
                size_t L0 = len0[i];
                for (int j = 0; j < n1; ++j) {
                    const char *s1 = plb->axis_labels[a1][j];
                    size_t L1 = len1[j];
                    size_t Ltot = L0 + joiner_len + L1 + 1;
                    char *dstp = cursor;
                    cursor += Ltot;
                    if (L0) memcpy(dstp, s0 ? s0 : "", L0);
                    if (single_delim) { dstp[L0] = dch; }
                    else if (joiner_len) { memcpy(dstp + L0, joiner, joiner_len); }
                    if (L1) memcpy(dstp + L0 + joiner_len, s1 ? s1 : "", L1);
                    dstp[L0 + joiner_len + L1] = '\0';
                    int pos = i * n1 + j;
                    out->axis_labels[dst][pos] = dstp;
                }
            }
            NPY_END_ALLOW_THREADS;
            PyMem_Free(len0);
            PyMem_Free(len1);
            if (axhash_build(&out->axis_hash[dst], (const char **)out->axis_labels[dst], (int)n01) < 0) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
        } else {
            int n = plb->axis_len[ax];
            if (labelsblock_set_axis_copy(out, dst, plb->axis_labels[ax], n) < 0) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
        }
        dst++;
    }

    LabeledArrayObject *v = (LabeledArrayObject *)view;
    LabelsBlock *old = v->labels_block; v->labels_block = out; labelsblock_decref(old); labels_cache_clear(v);
    if (v->delimiter) { free(v->delimiter); v->delimiter = NULL; }
    if (self->delimiter) {
        size_t L = strlen(self->delimiter) + 1; v->delimiter = (char *)PyDataMem_NEW(L);
        if (v->delimiter) memcpy(v->delimiter, self->delimiter, L);
    }
    return view;
}


/* ---------- reshape (fast labels for combine-only) ---------- */
static int
reshape_parse_args(PyObject *args, PyObject *kwargs, PyArray_Dims *out_dims, NPY_ORDER *out_order)
{
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 1) {
        PyErr_SetString(PyExc_TypeError, "reshape() takes at least 1 argument");
        return -1;
    }

    PyObject *order_obj = NULL;
    if (kwargs && PyDict_Check(kwargs)) {
        order_obj = PyDict_GetItemString(kwargs, "order");
    }

    Py_ssize_t shape_nargs = nargs;
    if (!order_obj && nargs >= 2) {
        PyObject *last = PyTuple_GET_ITEM(args, nargs - 1);
        if (PyUnicode_Check(last) || PyBytes_Check(last)) {
            order_obj = last;
            shape_nargs = nargs - 1;
        }
    }

    char order_char = 'C';
    if (order_obj && order_obj != Py_None) {
        const char *os = NULL;
        if (PyUnicode_Check(order_obj)) os = PyUnicode_AsUTF8(order_obj);
        else if (PyBytes_Check(order_obj)) os = PyBytes_AsString(order_obj);
        if (!os || os[0] == '\0') {
            PyErr_SetString(PyExc_TypeError, "order must be 'C' or 'F'");
            return -1;
        }
        order_char = os[0];
    }

    if (order_char == 'C' || order_char == 'c') *out_order = NPY_CORDER;
    else if (order_char == 'F' || order_char == 'f') *out_order = NPY_FORTRANORDER;
    else {
        PyErr_SetString(PyExc_NotImplementedError, "reshape order not supported");
        return -1;
    }

    PyObject *shape_obj = NULL;
    if (shape_nargs == 1) {
        shape_obj = PyTuple_GET_ITEM(args, 0);
        Py_INCREF(shape_obj);
    } else {
        shape_obj = PyTuple_GetSlice(args, 0, shape_nargs);
    }
    if (!shape_obj) return -1;
    int ok = PyArray_IntpConverter(shape_obj, out_dims);
    Py_DECREF(shape_obj);
    if (!ok) return -1;
    return 0;
}

static int
reshape_resolve_dims(PyArray_Dims *dims, npy_intp total)
{
    int neg = -1;
    npy_intp known = 1;
    for (int i = 0; i < dims->len; ++i) {
        npy_intp d = dims->ptr[i];
        if (d == -1) {
            if (neg >= 0) {
                PyErr_SetString(PyExc_ValueError, "only one -1 is allowed in reshape");
                return -1;
            }
            neg = i;
        } else if (d < 0) {
            PyErr_SetString(PyExc_ValueError, "negative dimensions are not allowed");
            return -1;
        } else {
            if (d != 0 && known > NPY_MAX_INTP / d) {
                PyErr_SetString(PyExc_OverflowError, "reshape dims overflow");
                return -1;
            }
            known *= d;
        }
    }
    if (neg >= 0) {
        if (known == 0 || total % known != 0) {
            PyErr_SetString(PyExc_ValueError, "cannot reshape array of size into requested shape");
            return -1;
        }
        dims->ptr[neg] = total / known;
    } else if (known != total) {
        PyErr_SetString(PyExc_ValueError, "cannot reshape array of size into requested shape");
        return -1;
    }
    return 0;
}



static PyObject *
reshape_general(PyObject *self_obj, PyArrayObject *arr, const npy_intp *newdims, int nd_new, NPY_ORDER order)
{
    int nd = PyArray_NDIM(arr);
    npy_intp total = PyArray_SIZE(arr);
    if (total == 0) {
        PyErr_SetString(PyExc_NotImplementedError, "reshape labels (fast) unsupported for empty arrays");
        return NULL;
    }

    LabeledArrayObject *self = (LabeledArrayObject *)self_obj;
    LabelsBlock *lb = self->labels_block;
    if (!lb) {
        lb = labelsblock_new_default(arr);
        if (!lb) return NULL;
        self->labels_block = lb;
    }

    const char *delim = (self->delimiter != NULL) ? self->delimiter : "-";
    size_t delim_len = strlen(delim);

    const npy_intp *dims = PyArray_DIMS(arr);
    int order_c = (order == NPY_CORDER);

    size_t **lab_lens = NULL;
    npy_intp *idx_old = NULL;
    char *labels_slab = NULL;
    char **labels_flat = NULL;
    IntersectState **states = NULL;
    npy_intp *idx_new = NULL;
    char **need_unique = NULL;
    StrSet ***uniq_sets = NULL;
    LabelsBlock *out = NULL;
    PyObject *view = NULL;
    PyArrayObject *reshaped = NULL;

    lab_lens = (size_t **)PyMem_Malloc(sizeof(size_t *) * (size_t)nd);
    if (!lab_lens) { PyErr_NoMemory(); goto fail; }
    for (int ax = 0; ax < nd; ++ax) lab_lens[ax] = NULL;
    for (int ax = 0; ax < nd; ++ax) {
        int n = lb->axis_len[ax];
        lab_lens[ax] = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)n);
        if (!lab_lens[ax]) { PyErr_NoMemory(); goto fail; }
        for (int i = 0; i < n; ++i) {
            const char *s = lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "";
            lab_lens[ax][i] = strlen(s);
        }
    }

    size_t total_bytes = 0;
    idx_old = (npy_intp *)PyMem_Calloc((size_t)nd, sizeof(npy_intp));
    if (!idx_old) { PyErr_NoMemory(); goto fail; }
    for (npy_intp flat = 0; flat < total; ++flat) {
        size_t len = 0; int parts = 0;
        for (int ax = 0; ax < nd; ++ax) {
            size_t L = lab_lens[ax][(int)idx_old[ax]];
            if (L) { len += L; parts++; }
        }
        if (parts > 1 && delim_len) len += delim_len * (size_t)(parts - 1);
        if (SIZE_MAX - total_bytes < len + 1) { PyErr_NoMemory(); goto fail; }
        total_bytes += len + 1;
        idx_increment_order(idx_old, dims, nd, order_c);
    }

    labels_slab = (char *)PyDataMem_NEW(total_bytes ? total_bytes : 1);
    if (!labels_slab) { PyErr_NoMemory(); goto fail; }
    labels_flat = (char **)PyMem_Malloc(sizeof(char *) * (size_t)total);
    if (!labels_flat) { PyErr_NoMemory(); goto fail; }

    memset(idx_old, 0, sizeof(npy_intp) * (size_t)nd);
    char *cursor = labels_slab;
    for (npy_intp flat = 0; flat < total; ++flat) {
        char *dst = cursor;
        int parts_written = 0;
        for (int ax = 0; ax < nd; ++ax) {
            const char *s = lb->axis_labels[ax][(int)idx_old[ax]] ? lb->axis_labels[ax][(int)idx_old[ax]] : "";
            size_t L = lab_lens[ax][(int)idx_old[ax]];
            if (L == 0) continue;
            if (parts_written > 0 && delim_len) { memcpy(dst, delim, delim_len); dst += delim_len; }
            memcpy(dst, s, L); dst += L; parts_written++;
        }
        *dst = '\0';
        labels_flat[flat] = cursor;
        cursor = dst + 1;
        idx_increment_order(idx_old, dims, nd, order_c);
    }

    states = (IntersectState **)PyMem_Calloc((size_t)nd_new, sizeof(IntersectState *));
    if (!states) { PyErr_NoMemory(); goto fail; }
    for (int j = 0; j < nd_new; ++j) {
        states[j] = (IntersectState *)PyMem_Calloc((size_t)newdims[j], sizeof(IntersectState));
        if (!states[j]) { PyErr_NoMemory(); goto fail; }
    }

    idx_new = (npy_intp *)PyMem_Calloc((size_t)nd_new, sizeof(npy_intp));
    if (!idx_new) { PyErr_NoMemory(); goto fail; }

    Token stack_tokens[32];
    for (npy_intp flat = 0; flat < total; ++flat) {
        const char *lab = labels_flat[flat];
        Token *tokens = NULL; int ntok = 0;
        if (split_tokens(lab, delim, delim_len, &tokens, &ntok, stack_tokens, 32) < 0) { PyErr_NoMemory(); goto fail; }
        for (int j = 0; j < nd_new; ++j) {
            IntersectState *st = &states[j][(int)idx_new[j]];
            if (!st->initialized) {
                st->set = strset_from_tokens(tokens, ntok);
                if (!st->set) { if (tokens != stack_tokens) PyMem_Free(tokens); goto fail; }
                st->initialized = 1;
            } else {
                StrSet *ns = strset_intersect_tokens(st->set, tokens, ntok);
                if (!ns) { if (tokens != stack_tokens) PyMem_Free(tokens); goto fail; }
                strset_free(st->set); PyMem_Free(st->set);
                st->set = ns;
            }
        }
        if (tokens != stack_tokens) PyMem_Free(tokens);
        idx_increment_order(idx_new, newdims, nd_new, order_c);
    }

    need_unique = (char **)PyMem_Calloc((size_t)nd_new, sizeof(char *));
    uniq_sets = (StrSet ***)PyMem_Calloc((size_t)nd_new, sizeof(StrSet **));
    if (!need_unique || !uniq_sets) { PyErr_NoMemory(); goto fail; }
    int any_need = 0;
    for (int j = 0; j < nd_new; ++j) {
        need_unique[j] = (char *)PyMem_Calloc((size_t)newdims[j], sizeof(char));
        uniq_sets[j] = (StrSet **)PyMem_Calloc((size_t)newdims[j], sizeof(StrSet *));
        if (!need_unique[j] || !uniq_sets[j]) { PyErr_NoMemory(); goto fail; }
        for (npy_intp i = 0; i < newdims[j]; ++i) {
            IntersectState *st = &states[j][(int)i];
            if (!st->set || st->set->used == 0) {
                need_unique[j][(int)i] = 1;
                any_need = 1;
            }
        }
    }

    if (any_need) {
        memset(idx_new, 0, sizeof(npy_intp) * (size_t)nd_new);
        for (npy_intp flat = 0; flat < total; ++flat) {
            const char *lab = labels_flat[flat];
            size_t L = strlen(lab);
            for (int j = 0; j < nd_new; ++j) {
                if (!need_unique[j][(int)idx_new[j]]) continue;
                if (!uniq_sets[j][(int)idx_new[j]]) {
                    uniq_sets[j][(int)idx_new[j]] = strset_new(8, 0);
                    if (!uniq_sets[j][(int)idx_new[j]]) { PyErr_NoMemory(); goto fail; }
                }
                if (strset_add(uniq_sets[j][(int)idx_new[j]], lab, L) < 0) { PyErr_NoMemory(); goto fail; }
            }
            idx_increment_order(idx_new, newdims, nd_new, order_c);
        }
    }

    out = labelsblock_alloc(nd_new);
    if (!out) { PyErr_NoMemory(); goto fail; }

    for (int j = 0; j < nd_new; ++j) {
        int n = (int)newdims[j];
        out->axis_len[j] = n;
        out->axis_labels[j] = (char **)calloc((size_t)n, sizeof(char *));
        if (!out->axis_labels[j]) { PyErr_NoMemory(); goto fail; }

        size_t *join_lens = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)n);
        if (!join_lens) { PyErr_NoMemory(); goto fail; }
        size_t total_axis = 0;
        for (int i = 0; i < n; ++i) {
            StrSet *set = (states[j][i].set && states[j][i].set->used > 0) ? states[j][i].set : uniq_sets[j][i];
            size_t L = strset_join_len(set, delim_len);
            join_lens[i] = L;
            if (SIZE_MAX - total_axis < L + 1) { PyMem_Free(join_lens); PyErr_NoMemory(); goto fail; }
            total_axis += L + 1;
        }

        char *slab = (char *)PyDataMem_NEW(total_axis ? total_axis : 1);
        if (!slab) { PyMem_Free(join_lens); PyErr_NoMemory(); goto fail; }
        out->axis_slab[j] = slab;
        char *cur = slab;

        for (int i = 0; i < n; ++i) {
            StrSet *set = (states[j][i].set && states[j][i].set->used > 0) ? states[j][i].set : uniq_sets[j][i];
            if (!set || set->used == 0) {
                *cur = '\0';
                out->axis_labels[j][i] = cur;
                cur += 1;
                continue;
            }
            char *stack_keys[64];
            int nkeys = 0;
            char **keys = strset_collect_keys(set, &nkeys, stack_keys, 64);
            if (!keys) { PyMem_Free(join_lens); PyErr_NoMemory(); goto fail; }
            qsort(keys, (size_t)nkeys, sizeof(char *), cmp_cstr);

            char *dst = cur;
            for (int k = 0; k < nkeys; ++k) {
                if (k > 0 && delim_len) { memcpy(dst, delim, delim_len); dst += delim_len; }
                size_t L = strlen(keys[k]);
                memcpy(dst, keys[k], L); dst += L;
            }
            *dst = '\0';
            out->axis_labels[j][i] = cur;
            cur = dst + 1;
            if (keys != stack_keys) PyMem_Free(keys);
        }

        PyMem_Free(join_lens);
        if (make_array_unique_c(out->axis_labels[j], n, delim, &out->axis_slab[j]) < 0) { PyErr_NoMemory(); goto fail; }
        if (axhash_build(&out->axis_hash[j], (const char **)out->axis_labels[j], n) < 0) { goto fail; }
        if (labelsblock_finalize_axis_from_c(out, j) < 0) { goto fail; }
    }

    PyArray_Dims nds; nds.ptr = (npy_intp *)newdims; nds.len = nd_new;
    reshaped = (PyArrayObject *)PyArray_Newshape(arr, &nds, order);
    if (!reshaped) goto fail;
    view = PyArray_View(reshaped, NULL, (PyTypeObject *)Py_TYPE(self_obj));
    Py_DECREF(reshaped); reshaped = NULL;
    if (!view) goto fail;

    LabeledArrayObject *v = (LabeledArrayObject *)view;
    LabelsBlock *old = v->labels_block;
    v->labels_block = out;
    labelsblock_decref(old);
    labels_cache_clear(v);
    out = NULL;

fail:
    if (labels_slab) PyDataMem_FREE(labels_slab);
    if (labels_flat) PyMem_Free(labels_flat);
    if (lab_lens) {
        for (int ax = 0; ax < nd; ++ax) PyMem_Free(lab_lens[ax]);
        PyMem_Free(lab_lens);
    }
    PyMem_Free(idx_old);
    if (states) {
        for (int j = 0; j < nd_new; ++j) {
            if (!states[j]) continue;
            for (npy_intp i = 0; i < newdims[j]; ++i) {
                if (states[j][i].set) { strset_free(states[j][i].set); PyMem_Free(states[j][i].set); }
            }
            PyMem_Free(states[j]);
        }
        PyMem_Free(states);
    }
    if (need_unique) {
        for (int j = 0; j < nd_new; ++j) PyMem_Free(need_unique[j]);
        PyMem_Free(need_unique);
    }
    if (uniq_sets) {
        for (int j = 0; j < nd_new; ++j) {
            if (!uniq_sets[j]) continue;
            for (npy_intp i = 0; i < newdims[j]; ++i) {
                if (uniq_sets[j][i]) { strset_free(uniq_sets[j][i]); PyMem_Free(uniq_sets[j][i]); }
            }
            PyMem_Free(uniq_sets[j]);
        }
        PyMem_Free(uniq_sets);
    }
    PyMem_Free(idx_new);
    if (out) labelsblock_decref(out);
    if (!view) return NULL;
    return view;
}

static PyObject *
LabeledArray_reshape(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *arr = (PyArrayObject *)self_obj;
    PyArray_Dims newdims;
    NPY_ORDER order = NPY_CORDER;
    if (reshape_parse_args(args, kwargs, &newdims, &order) < 0) return NULL;

    npy_intp total = PyArray_SIZE(arr);
    if (reshape_resolve_dims(&newdims, total) < 0) { PyDimMem_FREE(newdims.ptr); return NULL; }

    int nd = PyArray_NDIM(arr);
    int nd_new = (int)newdims.len;
    if (nd_new == 0) {
        PyDimMem_FREE(newdims.ptr);
        PyErr_SetString(PyExc_NotImplementedError, "reshape labels (fast) does not support scalar shapes");
        return NULL;
    }
    if (order != NPY_CORDER && order != NPY_FORTRANORDER) {
        PyDimMem_FREE(newdims.ptr);
        PyErr_SetString(PyExc_NotImplementedError, "reshape order not supported");
        return NULL;
    }

    int use_combine = 0;
    int *group_start = NULL;
    int *group_end = NULL;
    if (order == NPY_CORDER && nd_new <= nd) {
        group_start = (int *)PyMem_Malloc(sizeof(int) * (size_t)nd_new);
        group_end = (int *)PyMem_Malloc(sizeof(int) * (size_t)nd_new);
        if (!group_start || !group_end) {
            PyMem_Free(group_start); PyMem_Free(group_end);
            PyDimMem_FREE(newdims.ptr);
            PyErr_NoMemory();
            return NULL;
        }

        int src = 0;
        int unsupported = 0;
        for (int j = 0; j < nd_new; ++j) {
            npy_intp need = newdims.ptr[j];
            npy_intp prod = 1;
            int start = src;
            while (src < nd && prod < need) {
                npy_intp d = PyArray_DIM(arr, src);
                if (d == 0) { prod = 0; break; }
                if (prod > NPY_MAX_INTP / d) { unsupported = 1; break; }
                prod *= d;
                src++;
            }
            if (prod != need) { unsupported = 1; break; }
            group_start[j] = start;
            group_end[j] = src - 1;
            if (group_end[j] < group_start[j]) { unsupported = 1; break; }
        }
        if (src != nd) unsupported = 1;
        if (!unsupported) use_combine = 1;
    }

    if (use_combine) {
        PyObject *view = NULL;
        int keep_nd = nd_new;
        npy_intp *keep_shape = NULL;
        PyArrayObject *transposed = NULL;

        if (nd_new == nd) {
            Py_INCREF(arr);
            transposed = arr;
        } else {
            char *drop = (char *)PyMem_Calloc((size_t)nd, sizeof(char));
            if (!drop) { PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); PyErr_NoMemory(); return NULL; }
            for (int j = 0; j < nd_new; ++j) {
                for (int ax = group_start[j]; ax <= group_end[j]; ++ax) drop[ax] = 1;
            }
            int *perm = (int *)PyMem_Malloc(sizeof(int) * (size_t)nd);
            if (!perm) { PyMem_Free(drop); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); PyErr_NoMemory(); return NULL; }
            int w = 0;
            for (int i = 0; i < nd; ++i) if (drop[i]) perm[w++] = i;
            PyMem_Free(drop);
            npy_intp *perm_i = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)nd);
            if (!perm_i) { PyMem_Free(perm); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); PyErr_NoMemory(); return NULL; }
            for (int i = 0; i < nd; ++i) perm_i[i] = (npy_intp)perm[i];
            PyArray_Dims pd; pd.ptr = perm_i; pd.len = nd;
            transposed = (PyArrayObject *)PyArray_Transpose(arr, &pd);
            PyMem_Free(perm_i); PyMem_Free(perm);
            if (!transposed) { PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }
        }

        if (keep_nd > 0) {
            keep_shape = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)keep_nd);
            if (!keep_shape) { Py_DECREF(transposed); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); PyErr_NoMemory(); return NULL; }
            for (int i = 0; i < keep_nd; ++i) keep_shape[i] = PyArray_DIM(transposed, i);
        }

        PyArray_Dims nds; nds.ptr = (npy_intp *)newdims.ptr; nds.len = nd_new;
        PyObject *reshaped = PyArray_Newshape(transposed, &nds, NPY_CORDER);
        Py_DECREF(transposed);
        if (!reshaped) { PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }
        view = PyArray_View((PyArrayObject *)reshaped, NULL, (PyTypeObject *)Py_TYPE(self_obj));
        Py_DECREF(reshaped);
        if (!view) { PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }

        LabeledArrayObject *self = (LabeledArrayObject *)self_obj;
        LabelsBlock *plb = self->labels_block;
        if (!plb) { plb = labelsblock_new_default((PyArrayObject *)self_obj); if (!plb) { Py_DECREF(view); PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; } self->labels_block = plb; }
        LabelsBlock *out = labelsblock_alloc(nd_new);
        if (!out) { Py_DECREF(view); PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }

        const char *joiner = (self->delimiter != NULL) ? self->delimiter : "-";
        for (int j = 0; j < nd_new; ++j) {
            int s = group_start[j];
            int e = group_end[j];
            if (s == e) {
                int n = plb->axis_len[s];
                if (labelsblock_set_axis_copy(out, j, plb->axis_labels[s], n) < 0) { labelsblock_decref(out); Py_DECREF(view); PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }
            } else {
                char **labels = NULL; char *slab = NULL; int n = 0;
                if (build_combined_labels_range(plb, s, e, joiner, &labels, &slab, &n) < 0) { labelsblock_decref(out); Py_DECREF(view); PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }
                out->axis_len[j] = n;
                out->axis_labels[j] = labels;
                out->axis_slab[j] = slab;
                if (axhash_build(&out->axis_hash[j], (const char **)out->axis_labels[j], n) < 0) { labelsblock_decref(out); Py_DECREF(view); PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }
                if (labelsblock_finalize_axis_from_c(out, j) < 0) { labelsblock_decref(out); Py_DECREF(view); PyMem_Free(keep_shape); PyMem_Free(group_start); PyMem_Free(group_end); PyDimMem_FREE(newdims.ptr); return NULL; }
            }
        }

        LabeledArrayObject *v = (LabeledArrayObject *)view;
        LabelsBlock *old = v->labels_block;
        v->labels_block = out;
        labelsblock_decref(old);
        labels_cache_clear(v);

        PyMem_Free(keep_shape);
        PyMem_Free(group_start); PyMem_Free(group_end);
        PyDimMem_FREE(newdims.ptr);
        return view;
    }

    PyMem_Free(group_start); PyMem_Free(group_end);
    PyObject *res = reshape_general(self_obj, arr, (const npy_intp *)newdims.ptr, nd_new, order);
    PyDimMem_FREE(newdims.ptr);
    return res;
}

/* ---------- dropna (C-accelerated) ---------- */
static PyObject *
LabeledArray_dropna(PyObject *self_obj, PyObject *Py_UNUSED(ignored))
{
    if (!PyObject_TypeCheck(self_obj, &LabeledArray_Type)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PyArrayObject *arr = (PyArrayObject *)self_obj;
    int nd = PyArray_NDIM(arr);
    if (nd == 0) { Py_INCREF(self_obj); return self_obj; }

    int typ = PyArray_TYPE(arr);
    if (!PyTypeNum_ISFLOAT(typ) && !PyTypeNum_ISCOMPLEX(typ)) {
        /* no NaN concept -> return self */
        Py_INCREF(self_obj); return self_obj;
    }

    const npy_intp *dims = PyArray_DIMS(arr);
    const npy_intp *strides = PyArray_STRIDES(arr);

    /* Allocate per-axis keep flags */
    npy_bool **keep = (npy_bool **)calloc((size_t)nd, sizeof(npy_bool *));
    if (!keep) return NULL;
    int alloc_ok = 1;
    for (int ax = 0; ax < nd; ++ax) {
        keep[ax] = (npy_bool *)calloc((size_t)dims[ax], sizeof(npy_bool));
        if (!keep[ax]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int ax = 0; ax < nd; ++ax) free(keep[ax]);
        free(keep);
        return NULL;
    }

    char *base = (char *)PyArray_BYTES(arr);

    /* Helper to test non-NaN across supported dtypes */
    /* Select nan tester per dtype once */
    enum { NT_FLOAT16, NT_FLOAT32, NT_FLOAT64, NT_CFLOAT, NT_CDOUBLE, NT_OTHER } nt = NT_OTHER;
    switch (typ) {
        case NPY_HALF:    nt = NT_FLOAT16;  break;
        case NPY_FLOAT:   nt = NT_FLOAT32;  break;
        case NPY_DOUBLE:  nt = NT_FLOAT64;  break;
        case NPY_CFLOAT:  nt = NT_CFLOAT;   break;
        case NPY_CDOUBLE: nt = NT_CDOUBLE;  break;
        default:          nt = NT_OTHER;    break;
    }

    /* Release GIL around data scan */
    NPY_BEGIN_ALLOW_THREADS;

    if (nd == 1) {
        char *ptr = base;
        for (npy_intp i0 = 0; i0 < dims[0]; ++i0, ptr += strides[0]) {
            int not_nan = 1;
            switch (nt) {
                case NT_FLOAT16:  { npy_half h = *(npy_half *)ptr; double v = npy_half_to_double(h); not_nan = !npy_isnan(v); } break;
                case NT_FLOAT32:  not_nan = !npy_isnan((double)(*(float *)ptr)); break;
                case NT_FLOAT64:  not_nan = !npy_isnan(*(double *)ptr); break;
                case NT_CFLOAT:   { float *p = (float *)ptr;  not_nan = !(npy_isnan((double)p[0]) || npy_isnan((double)p[1])); } break;
                case NT_CDOUBLE:  { double *p = (double *)ptr; not_nan = !(npy_isnan(p[0]) || npy_isnan(p[1])); } break;
                default:          not_nan = 1; break;
            }
            if (not_nan) keep[0][i0] = 1;
        }
    } else {
        /* Parallelize across outermost axis when available */
        #pragma omp parallel
        {
            npy_bool **keep_loc = NULL;
            #ifdef _OPENMP
            keep_loc = (npy_bool **)calloc((size_t)nd, sizeof(npy_bool *));
            int ok = keep_loc != NULL;
            if (ok) {
                for (int ax = 0; ax < nd; ++ax) {
                    keep_loc[ax] = (npy_bool *)calloc((size_t)dims[ax], sizeof(npy_bool));
                    if (!keep_loc[ax]) { ok = 0; break; }
                }
            }
            #endif

            int n0 = (int)dims[0];
            int i0;
            #pragma omp for schedule(static)
            for (i0 = 0; i0 < n0; ++i0) {
                char *ptr0 = base + i0 * strides[0];
                npy_intp idx[NPY_MAXDIMS];
                for (int a = 0; a < nd; ++a) idx[a] = 0;
                idx[0] = i0;
                char *ptr = ptr0;
                while (1) {
                    int not_nan = 1;
                    switch (nt) {
                        case NT_FLOAT16:  { npy_half h = *(npy_half *)ptr; double v = npy_half_to_double(h); not_nan = !npy_isnan(v); } break;
                        case NT_FLOAT32:  { float v = *(float *)ptr;  not_nan = !npy_isnan((double)v); } break;
                        case NT_FLOAT64:  { double v = *(double *)ptr; not_nan = !npy_isnan(v); } break;
                        case NT_CFLOAT:   { float *p = (float *)ptr;  not_nan = !(npy_isnan((double)p[0]) || npy_isnan((double)p[1])); } break;
                        case NT_CDOUBLE:  { double *p = (double *)ptr; not_nan = !(npy_isnan(p[0]) || npy_isnan(p[1])); } break;
                        default:          not_nan = 1; break;
                    }
                    if (not_nan) {
                        #ifdef _OPENMP
                        for (int ax = 0; ax < nd; ++ax) keep_loc[ax][idx[ax]] = 1;
                        #else
                        for (int ax = 0; ax < nd; ++ax) keep[ax][idx[ax]] = 1;
                        #endif
                    }

                    /* advance odometer along last axis */
                    int ax = nd - 1;
                    ptr += strides[ax];
                    idx[ax]++;
                    while (ax > 0 && idx[ax] >= dims[ax]) {
                        /* reset this axis */
                        idx[ax] = 0;
                        ptr -= strides[ax] * dims[ax];
                        ax--;
                        /* advance next axis */
                        ptr += strides[ax];
                        idx[ax]++;
                    }
                    if (ax == 0 && idx[0] != i0) break;
                }
            }

            #ifdef _OPENMP
            /* reduce local keep into global */
            #pragma omp critical
            {
                for (int ax = 0; ax < nd; ++ax) {
                    for (npy_intp i = 0; i < dims[ax]; ++i) {
                        if (keep_loc[ax][i]) keep[ax][i] = 1;
                    }
                }
            }
            if (keep_loc) {
                for (int ax = 0; ax < nd; ++ax) free(keep_loc[ax]);
                free(keep_loc);
            }
            #endif
        }
    }

    NPY_END_ALLOW_THREADS;

    /* Build ck from keep arrays */
    PyObject *ck = PyTuple_New(nd);
    if (!ck) { for (int ax = 0; ax < nd; ++ax) free(keep[ax]); free(keep); return NULL; }
    int any_filter = 0;
    for (int ax = 0; ax < nd; ++ax) {
        npy_intp n = dims[ax];
        npy_intp count = 0;
        for (npy_intp i = 0; i < n; ++i) if (keep[ax][i]) count++;
        if (count == n) {
            PyObject *sl = PySlice_New(NULL, NULL, NULL);
            if (!sl) { Py_DECREF(ck); for (int a = 0; a < nd; ++a) free(keep[a]); free(keep); return NULL; }
            PyTuple_SET_ITEM(ck, ax, sl);
        } else {
            any_filter = 1;
            PyObject *lst = PyList_New((Py_ssize_t)count);
            if (!lst) { Py_DECREF(ck); for (int a = 0; a < nd; ++a) free(keep[a]); free(keep); return NULL; }
            npy_intp pos = 0;
            for (npy_intp i = 0; i < n; ++i) if (keep[ax][i]) {
                PyObject *idx = PyLong_FromSsize_t((Py_ssize_t)i);
                if (!idx) { Py_DECREF(lst); Py_DECREF(ck); for (int a = 0; a < nd; ++a) free(keep[a]); free(keep); return NULL; }
                PyList_SET_ITEM(lst, (Py_ssize_t)pos++, idx);
            }
            PyTuple_SET_ITEM(ck, ax, lst);
        }
    }
    for (int ax = 0; ax < nd; ++ax) free(keep[ax]);
    free(keep);

    if (!any_filter) { Py_DECREF(ck); Py_INCREF(self_obj); return self_obj; }

    /* Perform selection using base mapping, then attach sliced labels */
    PyObject *res = PyArray_Type.tp_as_mapping->mp_subscript(self_obj, ck);
    if (!res) { Py_DECREF(ck); return NULL; }
    if (PyArray_Check(res)) {
        PyObject *target = res;
        if (!PyObject_TypeCheck(res, (PyTypeObject *)Py_TYPE(self_obj))) {
            PyObject *viewres = PyArray_View((PyArrayObject *)res, NULL, (PyTypeObject *)Py_TYPE(self_obj));
            Py_DECREF(res);
            if (!viewres) { Py_DECREF(ck); return NULL; }
            target = viewres;
        }
        LabeledArrayObject *obj = (LabeledArrayObject *)self_obj;
        LabelsBlock *parent_lb = obj->labels_block;
        if (parent_lb) {
            LabelsBlock *child = labelsblock_slice(parent_lb, ck, (PyArrayObject *)target);
            if (child) {
                LabeledArrayObject *tobj = (LabeledArrayObject *)target;
                LabelsBlock *old = tobj->labels_block;
                tobj->labels_block = child;
                labelsblock_decref(old);
                labels_cache_clear(tobj);
            }
        }
        Py_DECREF(ck);
        if (PyArray_NDIM((PyArrayObject *)target) == 0) {
            PyObject *scalar = PyObject_CallMethod(target, "item", NULL);
            Py_DECREF(target);
            return scalar;
        }
        return target;
    }
    Py_DECREF(ck);
    return res;
}

static int
copy_to_out(PyObject *out_obj, PyObject *src)
{
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_FromAny(out_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    if (!out_arr) return -1;
    PyObject *src_arr_obj = NULL;
    if (PyObject_TypeCheck(src, &LabeledArray_Type)) {
        src_arr_obj = PyArray_View((PyArrayObject *)src, NULL, &PyArray_Type);
    } else if (PyArray_Check(src)) {
        Py_INCREF(src);
        src_arr_obj = src;
    } else {
        src_arr_obj = PyArray_FromAny(src, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
    }
    if (!src_arr_obj) { Py_DECREF(out_arr); return -1; }
    if (PyArray_CopyInto(out_arr, (PyArrayObject *)src_arr_obj) < 0) {
        Py_DECREF(out_arr); Py_DECREF(src_arr_obj); return -1;
    }
    Py_DECREF(out_arr);
    Py_DECREF(src_arr_obj);
    return 0;
}

static PyObject *
wrap_reduction_result(PyObject *self_obj, PyObject *result, const int *axes, int naxes, int keepdims)
{
    if (!PyArray_Check(result)) { Py_INCREF(result); return result; }

    if (!keepdims) {
        PyObject *axes_tuple = PyTuple_New(naxes);
        if (!axes_tuple) return NULL;
        for (int i = 0; i < naxes; ++i) {
            PyTuple_SET_ITEM(axes_tuple, i, PyLong_FromLong(axes[i]));
        }
        PyObject *context = Py_BuildValue("(OOO)", Py_None, Py_None, axes_tuple);
        Py_DECREF(axes_tuple);
        if (!context) return NULL;
        PyObject *wrapped = PyObject_CallMethod(self_obj, "__array_wrap__", "OO", result, context);
        Py_DECREF(context);
        return wrapped;
    }

    int orig_nd = PyArray_NDIM((PyArrayObject *)self_obj);
    if (orig_nd == 0) {
        PyObject *wrapped = PyObject_CallMethod(self_obj, "__array_wrap__", "OO", result, Py_None);
        return wrapped;
    }

    char *reduced = (char *)PyMem_Calloc((size_t)orig_nd, sizeof(char));
    if (!reduced) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < naxes; ++i) {
        int ax = axes[i];
        if (ax >= 0 && ax < orig_nd) reduced[ax] = 1;
    }
    npy_intp *new_dims = (npy_intp *)PyMem_Malloc(sizeof(npy_intp) * (size_t)orig_nd);
    if (!new_dims) { PyMem_Free(reduced); PyErr_NoMemory(); return NULL; }
    PyArrayObject *res_arr = (PyArrayObject *)result;
    int j = 0;
    for (int i = 0; i < orig_nd; ++i) {
        if (reduced[i]) new_dims[i] = 1;
        else new_dims[i] = PyArray_DIM(res_arr, j++);
    }
    PyArray_Dims nds; nds.ptr = new_dims; nds.len = orig_nd;
    PyObject *expanded = PyArray_Newshape(res_arr, &nds, NPY_CORDER);
    PyMem_Free(new_dims);
    PyMem_Free(reduced);
    if (!expanded) return NULL;
    PyObject *wrapped = PyObject_CallMethod(self_obj, "__array_wrap__", "OO", expanded, Py_None);
    Py_DECREF(expanded);
    if (!wrapped) return NULL;
    if (PyObject_TypeCheck(wrapped, &LabeledArray_Type)) {
        PyObject *labels = build_keepdims_labels((LabeledArrayObject *)self_obj, axes, naxes);
        if (!labels) { Py_DECREF(wrapped); return NULL; }
        if (PyObject_SetAttrString(wrapped, "labels", labels) < 0) {
            Py_DECREF(labels);
            Py_DECREF(wrapped);
            return NULL;
        }
        Py_DECREF(labels);
    }
    return wrapped;
}

static PyObject *
LabeledArray_nanmean(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyObject *axis_obj = Py_None, *dtype_obj = Py_None, *out_obj = Py_None, *where_obj = Py_None;
    int keepdims = 0;
    static char *kwlist[] = {"axis", "dtype", "out", "keepdims", "where", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOpO:nanmean", kwlist,
                                     &axis_obj, &dtype_obj, &out_obj, &keepdims, &where_obj)) {
        return NULL;
    }
    PyObject *mean = NULL, *var = NULL;
    int *axes = NULL, naxes = 0;
    if (nanmeanvar_core(self_obj, axis_obj, dtype_obj, where_obj, 0.0, &mean, &var, &axes, &naxes) < 0) return NULL;
    Py_DECREF(var);
    PyObject *wrapped = wrap_reduction_result(self_obj, mean, axes, naxes, keepdims);
    Py_DECREF(mean);
    PyMem_Free(axes);
    if (!wrapped) return NULL;
    if (out_obj && out_obj != Py_None) {
        if (copy_to_out(out_obj, wrapped) < 0) { Py_DECREF(wrapped); return NULL; }
        Py_DECREF(wrapped);
        Py_INCREF(out_obj);
        return out_obj;
    }
    return wrapped;
}

static PyObject *
LabeledArray_nanstd(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyObject *axis_obj = Py_None, *dtype_obj = Py_None, *out_obj = Py_None, *where_obj = Py_None, *ddof_obj = NULL;
    int keepdims = 0;
    static char *kwlist[] = {"axis", "dtype", "out", "ddof", "keepdims", "where", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOpO:nanstd", kwlist,
                                     &axis_obj, &dtype_obj, &out_obj, &ddof_obj, &keepdims, &where_obj)) {
        return NULL;
    }
    double ddof = 0.0;
    if (ddof_obj && ddof_obj != Py_None) {
        ddof = PyFloat_AsDouble(ddof_obj);
        if (ddof == -1.0 && PyErr_Occurred()) return NULL;
    }
    PyObject *std = NULL;
    int *axes = NULL, naxes = 0;
    if (nanstd_core(self_obj, axis_obj, dtype_obj, where_obj, ddof, &std, &axes, &naxes) < 0) return NULL;

    PyObject *wrapped = wrap_reduction_result(self_obj, std, axes, naxes, keepdims);
    Py_DECREF(std);
    PyMem_Free(axes);
    if (!wrapped) return NULL;
    if (out_obj && out_obj != Py_None) {
        if (copy_to_out(out_obj, wrapped) < 0) { Py_DECREF(wrapped); return NULL; }
        Py_DECREF(wrapped);
        Py_INCREF(out_obj);
        return out_obj;
    }
    return wrapped;
}

static PyObject *
LabeledArray_nanmean_std(PyObject *self_obj, PyObject *args, PyObject *kwargs)
{
    PyObject *axis_obj = Py_None, *dtype_obj = Py_None, *where_obj = Py_None, *ddof_obj = NULL;
    int keepdims = 0;
    static char *kwlist[] = {"axis", "dtype", "ddof", "keepdims", "where", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOpO:nanmean_std", kwlist,
                                     &axis_obj, &dtype_obj, &ddof_obj, &keepdims, &where_obj)) {
        return NULL;
    }
    double ddof = 0.0;
    if (ddof_obj && ddof_obj != Py_None) {
        ddof = PyFloat_AsDouble(ddof_obj);
        if (ddof == -1.0 && PyErr_Occurred()) return NULL;
    }
    PyObject *mean = NULL, *std = NULL;
    int *axes = NULL, naxes = 0;
    if (nanmeanstd_core(self_obj, axis_obj, dtype_obj, where_obj, ddof, &mean, &std, &axes, &naxes) < 0) return NULL;

    PyObject *mean_wrapped = wrap_reduction_result(self_obj, mean, axes, naxes, keepdims);
    Py_DECREF(mean);
    PyObject *std_wrapped = wrap_reduction_result(self_obj, std, axes, naxes, keepdims);
    Py_DECREF(std);
    PyMem_Free(axes);
    if (!mean_wrapped || !std_wrapped) {
        Py_XDECREF(mean_wrapped);
        Py_XDECREF(std_wrapped);
        return NULL;
    }
    PyObject *out = PyTuple_New(2);
    if (!out) { Py_DECREF(mean_wrapped); Py_DECREF(std_wrapped); return NULL; }
    PyTuple_SET_ITEM(out, 0, mean_wrapped);
    PyTuple_SET_ITEM(out, 1, std_wrapped);
    return out;
}


static PyMethodDef LabeledArray_methods[] = {
    {"__array_finalize__", (PyCFunction)array_finalize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"__array_wrap__", (PyCFunction)array_wrap, METH_VARARGS | METH_KEYWORDS, NULL},
    {"__array_function__", (PyCFunction)array_function, METH_VARARGS, NULL},
    {"__array_ufunc__", (PyCFunction)array_ufunc, METH_VARARGS | METH_KEYWORDS, NULL},
    {"take", (PyCFunction)LabeledArray_take, METH_VARARGS | METH_KEYWORDS, NULL},
    {"combine", (PyCFunction)LabeledArray_combine, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape", (PyCFunction)LabeledArray_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
    {"find", (PyCFunction)LabeledArray_find, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dropna", (PyCFunction)LabeledArray_dropna, METH_NOARGS, NULL},
    {"nanmean", (PyCFunction)LabeledArray_nanmean, METH_VARARGS | METH_KEYWORDS, NULL},
    {"nanstd", (PyCFunction)LabeledArray_nanstd, METH_VARARGS | METH_KEYWORDS, NULL},
    {"nanmean_std", (PyCFunction)LabeledArray_nanmean_std, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyMappingMethods LabeledArray_mapping = {
    0,
    LabeledArray_subscript,
    LabeledArray_ass_subscript
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "labeledarray",
    "ndarray subtype with C-level labels and label-based indexing.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_labeledarray(void)
{
    PyObject *m;
    
    import_array();

    if (sizeof(PyArrayObject) < PyArray_Type.tp_basicsize) {
        PyErr_SetString(PyExc_ImportError,
           "Binary incompatibility with NumPy, must recompile/update X.");
        return NULL;
    }
    
    LabeledArray_Type.tp_name = "ieeg.arrays.labeledarray.LabeledArray";
    LabeledArray_Type.tp_basicsize = (int)sizeof(LabeledArrayObject);
    LabeledArray_Type.tp_itemsize = 0;
    LabeledArray_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    LabeledArray_Type.tp_new = LabeledArray_new;
    LabeledArray_Type.tp_base = &PyArray_Type;
    LabeledArray_Type.tp_methods = LabeledArray_methods;
    LabeledArray_Type.tp_getset = LabeledArray_getset;
    LabeledArray_Type.tp_as_mapping = &LabeledArray_mapping;
    LabeledArray_Type.tp_dealloc = LabeledArray_dealloc;
    LabeledArray_Type.tp_str = LabeledArray_str;

    if (PyType_Ready(&LabeledArray_Type) < 0) return NULL;

    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    Py_INCREF(&LabeledArray_Type);
    if (PyModule_AddObject(m, "LabeledArray", (PyObject *)&LabeledArray_Type) < 0) {
        Py_DECREF(&LabeledArray_Type);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}
