/* (moved down after includes/typedefs) */
/* ndarray subtype: LabeledArray with C-level labels storage (stride-safe) */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include <numpy/halffloat.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NPY_INLINE
#define NPY_INLINE inline
#endif


static PyTypeObject LabeledArray_Type; /* forward */
/* Forward declare struct tag so self-referential pointers work */
typedef struct LabelsBlock LabelsBlock;

typedef struct {
    size_t cap;         /* power-of-two capacity */
    size_t used;        /* number of entries */
    uint64_t *hashes;   /* hashes[slot] */
    const char **keys;  /* pointers into axis_labels strings */
    int *vals;          /* index per key */
} AxisHash;

typedef struct LabelsBlock {
    int ndim;
    int *axis_len;       /* length per axis */
    char ***axis_labels; /* axis_labels[axis][index] -> C string */
    AxisHash *axis_hash; /* per-axis C hash */
    unsigned char *axis_borrowed; /* 1 if this axis' labels are borrowed */
    LabelsBlock *borrowed_owner; /* keep owner alive if borrowing */
    PyArrayObject **axis_arr; /* per-axis 1D unicode ndarray of labels */
    PyObject **axis_index;    /* per-axis dict: unicode -> int */
    char **axis_slab;    /* per-axis contiguous short-string slab (<=32B labels) */
    int refcount;        /* manual refcount */
} LabelsBlock;

/* explicit ndarray sub-type instance layout */
typedef struct {
    PyArrayObject base;
    LabelsBlock *labels_block;
    PyObject *labels_cache; /* cached tuple-of-ndarray(unicode, 1d) per axis */
    char *delimiter; /* default join for label combinations */
} LabeledArrayObject;

/* ---------- small inline utils ---------- */
static NPY_INLINE uint64_t
fnv1a64(const char *data)
{
    /* FNV-1a 64-bit */
    const uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)data; *p; ++p) {
        hash ^= (uint64_t)(*p);
        hash *= FNV_PRIME;
    }
    return hash ? hash : 1469598103934665603ULL; /* avoid zero */
}

static NPY_INLINE size_t
next_pow2(size_t n)
{
    size_t p = 1; while (p < n) p <<= 1; return p;
}

static NPY_INLINE int
axhash_find(const AxisHash *ah, const char *key, Py_ssize_t *out)
{
    if (!ah || ah->cap == 0) return 1;
    uint64_t h = fnv1a64(key ? key : "");
    size_t m = ah->cap - 1;
    size_t pos = (size_t)h & m;
    while (1) {
        const char *k = ah->keys[pos];
        if (k == NULL) return 1; /* empty slot => not found */
        if (ah->hashes[pos] == h && strcmp(k, key) == 0) {
            *out = (Py_ssize_t)ah->vals[pos];
            return 0;
        }
        pos = (pos + 1) & m;
    }
}

/* ---------- small C helpers for speed/readability ---------- */
static NPY_INLINE int
labelsblock_find(const LabelsBlock *lb, int axis, PyObject *value, Py_ssize_t *out)
{
    if (axis < 0 || axis >= lb->ndim) return -1;
    if (lb->axis_index && lb->axis_index[axis]) {
        PyObject *key = NULL;
        if (PyUnicode_Check(value)) { key = value; Py_INCREF(key); }
        else if (PyBytes_Check(value)) { const char *c = PyBytes_AsString(value); if (!c) return -1; key = PyUnicode_FromString(c); if (!key) return -1; }
        else { key = PyObject_Str(value); if (!key) return -1; }
        PyObject *pos_obj = PyDict_GetItemWithError(lb->axis_index[axis], key);
        Py_DECREF(key);
        if (pos_obj) { *out = PyLong_AsSsize_t(pos_obj); if (*out == -1 && PyErr_Occurred()) return -1; return 0; }
        if (PyErr_Occurred()) return -1;
        return 1;
    }
    if (PyUnicode_Check(value)) {
        Py_ssize_t sz; const char *c = PyUnicode_AsUTF8AndSize(value, &sz);
        if (!c) return -1;
        return axhash_find(&lb->axis_hash[axis], c, out);
    }
    /* fallback scan: build a C string once, then strcmp */
    const char *needle = NULL; PyObject *tmp_str = NULL;
    if (PyBytes_Check(value)) {
        needle = PyBytes_AsString(value);
        if (!needle) return -1;
    } else {
        tmp_str = PyObject_Str(value);
        if (!tmp_str) return -1;
        needle = PyUnicode_AsUTF8(tmp_str);
        if (!needle) { Py_DECREF(tmp_str); return -1; }
    }
    int n = lb->axis_len[axis];
    for (int i = 0; i < n; ++i) {
        if (strcmp(lb->axis_labels[axis][i], needle) == 0) {
            if (tmp_str) Py_DECREF(tmp_str);
            *out = i; return 0;
        }
    }
    if (tmp_str) Py_DECREF(tmp_str);
    return 1;
}

static NPY_INLINE char *
cstr_dup(const char *s)
{
    const char *src = s ? s : "";
    size_t L = strlen(src) + 1;
    char *dst = (char *)PyDataMem_NEW(L);
    if (!dst) return NULL;
    memcpy(dst, src, L);
    return dst;
}

static NPY_INLINE int
axis_labels_copy(char **dst, char **src, int n)
{
    for (int i = 0; i < n; ++i) {
        dst[i] = cstr_dup(src[i]);
        if (!dst[i]) return -1;
    }
    return 0;
}

/* Build per-axis numpy unicode array and dict index from C labels */
static NPY_INLINE int
labelsblock_finalize_axis_from_c(LabelsBlock *lb, int ax)
{
    int n = lb->axis_len[ax];
    if (n < 0) return -1;
    PyObject *lst = PyList_New(n);
    if (!lst) return -1;
    for (int i = 0; i < n; ++i) {
        PyObject *s = PyUnicode_FromString(lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "");
        if (!s) { Py_DECREF(lst); return -1; }
        PyList_SET_ITEM(lst, i, s);
    }
    PyObject *arr = PyArray_FromAny(lst, NULL, 1, 1, NPY_ARRAY_ENSUREARRAY, NULL);
    Py_DECREF(lst);
    if (!arr) return -1;
    PyArray_CLEARFLAGS((PyArrayObject *)arr, NPY_ARRAY_WRITEABLE);
    PyObject *d = PyDict_New();
    if (!d) { Py_DECREF(arr); return -1; }
    for (int i = 0; i < n; ++i) {
        PyObject *key = PyUnicode_FromString(lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "");
        if (!key) { Py_DECREF(d); Py_DECREF(arr); return -1; }
        PyObject *val = PyLong_FromLong(i);
        if (!val) { Py_DECREF(key); Py_DECREF(d); Py_DECREF(arr); return -1; }
        if (PyDict_SetItem(d, key, val) < 0) { Py_DECREF(val); Py_DECREF(key); Py_DECREF(d); Py_DECREF(arr); return -1; }
        Py_DECREF(val);
        Py_DECREF(key);
    }
    Py_XDECREF(lb->axis_arr ? (PyObject *)lb->axis_arr[ax] : NULL);
    Py_XDECREF(lb->axis_index ? lb->axis_index[ax] : NULL);
    if (lb->axis_arr) lb->axis_arr[ax] = (PyArrayObject *)arr; else Py_DECREF(arr);
    if (lb->axis_index) lb->axis_index[ax] = d; else Py_DECREF(d);
    return 0;
}


static NPY_INLINE char **
build_numeric_labels(int n)
{
    char **labels = (char **)calloc((size_t)n, sizeof(char *));
    if (!labels) return NULL;
    for (int j = 0; j < n; ++j) {
        char buf[32];
        int m = snprintf(buf, sizeof(buf), "%d", j);
        if (m < 0 || m >= (int)sizeof(buf)) { /* cleanup */
            for (int k = 0; k < j; ++k) free(labels[k]);
            free(labels);
            return NULL;
        }
        labels[j] = cstr_dup(buf);
        if (!labels[j]) {
            for (int k = 0; k < j; ++k) free(labels[k]);
            free(labels);
            return NULL;
        }
    }
    return labels;
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

/* ---------- AxisHash management ---------- */
static NPY_INLINE void
axhash_free(AxisHash *ah)
{
    if (!ah) return;
    free(ah->hashes);
    free(ah->keys);
    free(ah->vals);
}

/* ---------- LabelsBlock management ---------- */
static NPY_INLINE void
labelsblock_decref(LabelsBlock *lb)
{
    if (!lb) return;
    if (--lb->refcount == 0) {
        if (lb->axis_labels) {
            for (int ax = 0; ax < lb->ndim; ++ax) {
                if (lb->axis_labels[ax]) {
                    int borrowed = lb->axis_borrowed ? lb->axis_borrowed[ax] : 0;
                    if (!borrowed) {
                        if (lb->axis_slab && lb->axis_slab[ax]) {
                            PyDataMem_FREE(lb->axis_slab[ax]);
                        } else {
                        for (int i = 0; i < lb->axis_len[ax]; ++i) {
                            if (lb->axis_labels[ax][i]) PyDataMem_FREE(lb->axis_labels[ax][i]);
                        }
                        }
                        free(lb->axis_labels[ax]);
                    }
                }
            }
            free(lb->axis_labels);
        }
        if (lb->axis_hash) {
            for (int ax = 0; ax < lb->ndim; ++ax) axhash_free(&lb->axis_hash[ax]);
            free(lb->axis_hash);
        }
        if (lb->axis_len) free(lb->axis_len);
        if (lb->axis_borrowed) free(lb->axis_borrowed);
        if (lb->axis_arr) {
            for (int ax = 0; ax < lb->ndim; ++ax) Py_XDECREF(lb->axis_arr[ax]);
            free(lb->axis_arr);
        }
        if (lb->axis_index) {
            for (int ax = 0; ax < lb->ndim; ++ax) Py_XDECREF(lb->axis_index[ax]);
            free(lb->axis_index);
        }
        if (lb->axis_slab) free(lb->axis_slab);
        if (lb->borrowed_owner) {
            labelsblock_decref(lb->borrowed_owner);
        }
        free(lb);
    }
}

/* Allocate LabelsBlock with arrays for given nd; refcount set to 1 */
static NPY_INLINE LabelsBlock *
labelsblock_alloc(int nd)
{
    LabelsBlock *lb = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!lb) return NULL;
    lb->ndim = nd;
    lb->refcount = 1;
    lb->axis_len = (int *)calloc(nd, sizeof(int));
    lb->axis_labels = (char ***)calloc(nd, sizeof(char **));
    lb->axis_hash = (AxisHash *)calloc(nd, sizeof(AxisHash));
    lb->axis_borrowed = (unsigned char *)calloc(nd, sizeof(unsigned char));
    lb->borrowed_owner = NULL;
    lb->axis_arr = (PyArrayObject **)calloc(nd, sizeof(PyArrayObject *));
    lb->axis_index = (PyObject **)calloc(nd, sizeof(PyObject *));
    lb->axis_slab = (char **)calloc(nd, sizeof(char *));
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash || !lb->axis_borrowed || !lb->axis_arr || !lb->axis_index || !lb->axis_slab) { labelsblock_decref(lb); return NULL; }
    return lb;
}

static NPY_INLINE int
axhash_build(AxisHash *ah, const char **keys, int n)
{
    ah->cap = next_pow2((size_t)(n * 2 + 1));
    ah->used = 0;
    ah->hashes = (uint64_t *)calloc(ah->cap, sizeof(uint64_t));
    ah->keys   = (const char **)calloc(ah->cap, sizeof(const char *));
    ah->vals   = (int *)calloc(ah->cap, sizeof(int));
    if (!ah->hashes || !ah->keys || !ah->vals) return -1;
    for (int i = 0; i < n; ++i) {
        const char *k = keys[i] ? keys[i] : "";
        uint64_t h = fnv1a64(k);
        size_t m = ah->cap - 1;
        size_t pos = (size_t)h & m;
        while (ah->keys[pos] != NULL) {
            pos = (pos + 1) & m;
        }
        ah->hashes[pos] = h;
        ah->keys[pos]   = k;
        ah->vals[pos]   = i;
        ah->used++;
    }
    return 0;
}

/* Set axis from source labels: alloc, copy, and build hash */
static NPY_INLINE int
labelsblock_set_axis_copy(LabelsBlock *lb, int ax, char **src_labels, int n)
{
    lb->axis_len[ax] = n;
    lb->axis_labels[ax] = (char **)calloc((size_t)n, sizeof(char *));
    if (!lb->axis_labels[ax]) return -1;
    /* short-string arena for labels <=32B */
    size_t total = 0; int use_slab = 1;
    for (int i = 0; i < n; ++i) {
        size_t L = strlen(src_labels[i] ? src_labels[i] : "") + 1;
        if (L > 32) { use_slab = 0; break; }
        total += L;
    }
    char *slab = NULL; char *cursor = NULL;
    if (use_slab && n > 0) {
        slab = (char *)PyDataMem_NEW(total);
        if (!slab) { free(lb->axis_labels[ax]); lb->axis_labels[ax] = NULL; return -1; }
        if (!lb->axis_slab) { PyDataMem_FREE(slab); free(lb->axis_labels[ax]); lb->axis_labels[ax] = NULL; return -1; }
        if (!lb->axis_slab[ax]) lb->axis_slab[ax] = slab; else { PyDataMem_FREE(slab); use_slab = 0; }
        cursor = lb->axis_slab[ax];
    }
    for (int i = 0; i < n; ++i) {
        const char *s = src_labels[i] ? src_labels[i] : "";
        size_t L = strlen(s) + 1;
        if (use_slab) {
            memcpy(cursor, s, L);
            lb->axis_labels[ax][i] = cursor;
            cursor += L;
        } else {
            lb->axis_labels[ax][i] = (char *)PyDataMem_NEW(L);
            if (!lb->axis_labels[ax][i]) { return -1; }
            memcpy(lb->axis_labels[ax][i], s, L);
        }
    }
    if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], n) < 0) return -1;
    if (labelsblock_finalize_axis_from_c(lb, ax) < 0) return -1;
            return 0;
        }

/* Set axis to numeric labels 0..n-1 with hash */
static NPY_INLINE int
labelsblock_set_axis_numeric(LabelsBlock *lb, int ax, int n)
{
    lb->axis_len[ax] = n;
    char **labels = build_numeric_labels(n);
    if (!labels) return -1;
    lb->axis_labels[ax] = labels;
    if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], n) < 0) return -1;
    if (labelsblock_finalize_axis_from_c(lb, ax) < 0) return -1;
    return 0;
}

/* Set axis to a single constant string label */
static NPY_INLINE int
labelsblock_set_axis_constant(LabelsBlock *lb, int ax, const char *s)
{
    lb->axis_len[ax] = 1;
    lb->axis_labels[ax] = (char **)calloc(1, sizeof(char *));
    if (!lb->axis_labels[ax]) return -1;
    size_t L = strlen(s ? s : "") + 1;
    lb->axis_labels[ax][0] = (char *)PyDataMem_NEW(L);
    if (!lb->axis_labels[ax][0]) return -1;
    memcpy(lb->axis_labels[ax][0], s ? s : "", L);
    if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], 1) < 0) return -1;
    if (labelsblock_finalize_axis_from_c(lb, ax) < 0) return -1;
    return 0;
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


static NPY_INLINE int
fill_labels_from_int_indices(char ***dst_labels_ptr, int out_len, char **src_labels, PyArrayObject *ind)
{
    char **dst = (char **)calloc((size_t)out_len, sizeof(char *));
    if (!dst) return -1;
    for (int j = 0; j < out_len; ++j) {
        npy_intp jj = (npy_intp)j;
        npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, &jj));
        if (idxv < 0) { for (int t=0;t<j;++t) free(dst[t]); free(dst); return -1; }
        const char *s = src_labels[(int)idxv];
        size_t L = strlen(s ? s : "") + 1;
        dst[j] = (char *)PyDataMem_NEW(L);
        if (!dst[j]) { for (int t=0;t<j;++t) free(dst[t]); free(dst); return -1; }
        memcpy(dst[j], s ? s : "", L);
    }
    *dst_labels_ptr = dst;
    return 0;
}

/* Build labels from a 1D boolean mask along a source axis by converting to integer indices */
static NPY_INLINE int
build_labels_from_bool_mask(const LabelsBlock *lb, int src_axis, PyArrayObject *mask, int out_len, char ***out_labels_ptr)
{
    int ind_nd = PyArray_NDIM(mask);
    if (ind_nd != 1) { PyErr_SetString(PyExc_TypeError, "boolean index must be 1D for a single axis"); return -1; }
    const npy_intp *adims = PyArray_DIMS(mask);
    npy_intp n = adims[0];
    /* First pass: count true */
    npy_intp count = 0;
    for (npy_intp i = 0; i < n; ++i) {
        npy_bool v = *(npy_bool *)PyArray_GetPtr(mask, &i);
        if (v) count++;
    }
    if ((int)count != out_len) {
        /* shape mismatch safeguard */
        PyErr_SetString(PyExc_ValueError, "boolean mask true count does not match result shape");
        return -1;
    }
    /* Build integer indices */
    PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(1, &count, NPY_INTP);
    if (!ind) return -1;
    npy_intp pos = 0;
    for (npy_intp i = 0; i < n; ++i) {
        npy_bool v = *(npy_bool *)PyArray_GetPtr(mask, &i);
        if (v) {
            *((npy_intp *)PyArray_GetPtr(ind, &pos)) = i;
            pos++;
        }
    }
    int rc = fill_labels_from_int_indices(out_labels_ptr, out_len, lb->axis_labels[src_axis], ind);
    Py_DECREF(ind);
    return rc;
}

/* Build labels from a Python list/tuple of integers by converting to an intp ndarray and reusing int-path */
static NPY_INLINE int
build_labels_from_py_sequence(const LabelsBlock *lb, int src_axis, PyObject *seq_obj, int out_len, char ***out_labels_ptr)
{
    PyObject *seq = PySequence_Fast(seq_obj, "expected sequence");
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if ((int)n != out_len) { Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "sequence length mismatch for result shape"); return -1; }
    PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(1, (npy_intp *)&n, NPY_INTP);
    if (!ind) { Py_DECREF(seq); return -1; }
    for (Py_ssize_t j = 0; j < n; ++j) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
        PyObject *maybe_int = PyNumber_Index(it);
        npy_intp val;
        if (maybe_int) {
            long idx = PyLong_AsLong(maybe_int);
            Py_DECREF(maybe_int);
            if (idx == -1 && PyErr_Occurred()) { Py_DECREF(ind); Py_DECREF(seq); return -1; }
            if (idx < 0 || idx >= lb->axis_len[src_axis]) { Py_DECREF(ind); Py_DECREF(seq); PyErr_SetString(PyExc_IndexError, "index out of range"); return -1; }
            val = (npy_intp)idx;
        } else if (PyUnicode_Check(it) || PyBytes_Check(it)) {
            Py_ssize_t pos = -1;
            if (labelsblock_find(lb, src_axis, it, &pos) != 0) { Py_DECREF(ind); Py_DECREF(seq); PyErr_SetString(PyExc_IndexError, "label not found in axis"); return -1; }
            val = (npy_intp)pos;
        } else {
            Py_DECREF(ind); Py_DECREF(seq); PyErr_SetString(PyExc_TypeError, "sequence contains non-int/non-string"); return -1; }
        *((npy_intp *)PyArray_GetPtr(ind, &j)) = val;
    }
    Py_DECREF(seq);
    int rc = fill_labels_from_int_indices(out_labels_ptr, out_len, lb->axis_labels[src_axis], ind);
    Py_DECREF(ind);
    return rc;
}

static NPY_INLINE int
build_labels_from_int_sequence(const LabelsBlock *lb, int src_axis, PyObject *seq_obj, int out_len, char ***out_labels_ptr)
{
    PyObject *seq = PySequence_Fast(seq_obj, "expected sequence");
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if ((int)n != out_len) { Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "sequence length mismatch for result shape"); return -1; }
    PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(1, (npy_intp *)&n, NPY_INTP);
    if (!ind) { Py_DECREF(seq); return -1; }
    for (Py_ssize_t j = 0; j < n; ++j) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
        long idx = PyLong_AsLong(it);
        if (idx == -1 && PyErr_Occurred()) { Py_DECREF(ind); Py_DECREF(seq); return -1; }
        if (idx < 0 || idx >= lb->axis_len[src_axis]) { Py_DECREF(ind); Py_DECREF(seq); PyErr_SetString(PyExc_IndexError, "index out of range"); return -1; }
        npy_intp jj = (npy_intp)j;
        *((npy_intp *)PyArray_GetPtr(ind, &jj)) = (npy_intp)idx;
    }
    Py_DECREF(seq);
    int rc = fill_labels_from_int_indices(out_labels_ptr, out_len, lb->axis_labels[src_axis], ind);
    Py_DECREF(ind);
    return rc;
}

/* Build row and column joined labels for 2D integer index array along a single source axis */
static NPY_INLINE int
build_rowcol_labels_from_2d_indices(const LabelsBlock *lb, int src_axis, PyArrayObject *ind, int R, int C, char ***out_rows_ptr, char ***out_cols_ptr)
{
    char **rows = (char **)calloc((size_t)R, sizeof(char *));
    if (!rows) return -1;
    for (int r = 0; r < R; ++r) {
        size_t tot = 1;
        for (int c = 0; c < C; ++c) {
            npy_intp ij[2] = {r, c};
            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, ij));
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            tot += strlen(s) + (c ? 1 : 0);
        }
        char *buf = (char *)PyDataMem_NEW(tot);
        if (!buf) { for (int t = 0; t < r; ++t) free(rows[t]); free(rows); return -1; }
        size_t posw = 0;
        for (int c = 0; c < C; ++c) {
            npy_intp ij[2] = {r, c};
            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, ij));
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            if (c) buf[posw++] = '-';
            size_t Ls = strlen(s);
            memcpy(buf + posw, s, Ls);
            posw += Ls;
        }
        buf[posw] = '\0';
        rows[r] = buf;
    }

    char **cols = (char **)calloc((size_t)C, sizeof(char *));
    if (!cols) { for (int t = 0; t < R; ++t) free(rows[t]); free(rows); return -1; }
    for (int c = 0; c < C; ++c) {
        size_t tot = 1;
        for (int r = 0; r < R; ++r) {
            npy_intp ij[2] = {r, c};
            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, ij));
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            tot += strlen(s) + (r ? 1 : 0);
        }
        char *buf = (char *)PyDataMem_NEW(tot);
        if (!buf) { for (int t = 0; t < c; ++t) free(cols[t]); for (int t = 0; t < R; ++t) free(rows[t]); free(rows); free(cols); return -1; }
        size_t posw = 0;
        for (int r = 0; r < R; ++r) {
            npy_intp ij[2] = {r, c};
            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, ij));
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            if (r) buf[posw++] = '-';
            size_t Ls = strlen(s);
            memcpy(buf + posw, s, Ls);
            posw += Ls;
        }
        buf[posw] = '\0';
        cols[c] = buf;
    }
    *out_rows_ptr = rows;
    *out_cols_ptr = cols;
    return 0;
}

/* Build labels for one axis p of a k-D integer index array by joining across other dims */
static NPY_INLINE int
build_kd_labels_for_axis(const LabelsBlock *lb, int src_axis, PyArrayObject *ind, int knd, int p, const npy_intp *adims, char ***out_axis_labels_ptr)
{
    int len_p = (int)adims[p];
    char **axis_labels = (char **)calloc((size_t)len_p, sizeof(char *));
    if (!axis_labels) return -1;
    for (int vp = 0; vp < len_p; ++vp) {
        /* First pass: total length */
        npy_intp idxk[NPY_MAXDIMS];
        for (int q = 0; q < knd; ++q) idxk[q] = 0;
        idxk[p] = vp;
        size_t tot = 1; int first = 1;
        while (1) {
            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, idxk));
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            tot += strlen(s) + (first ? 0 : 1);
            first = 0;
            int qq = knd - 1;
            while (qq >= 0) {
                if (qq == p) { qq--; continue; }
                idxk[qq]++;
                if (idxk[qq] < adims[qq]) break;
                idxk[qq] = 0; qq--;
            }
            if (qq < 0) break;
        }
        char *buf = (char *)PyDataMem_NEW(tot);
        if (!buf) { for (int t = 0; t < vp; ++t) free(axis_labels[t]); free(axis_labels); return -1; }
        size_t posw = 0; first = 1;
        for (int q = 0; q < knd; ++q) idxk[q] = 0;
        idxk[p] = vp;
        while (1) {
            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, idxk));
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            if (!first) buf[posw++] = '-';
            size_t Ls = strlen(s);
            memcpy(buf + posw, s, Ls); posw += Ls; first = 0;
            int qq = knd - 1;
            while (qq >= 0) {
                if (qq == p) { qq--; continue; }
                idxk[qq]++;
                if (idxk[qq] < adims[qq]) break;
                idxk[qq] = 0; qq--;
            }
            if (qq < 0) break;
        }
        buf[posw] = '\0';
        axis_labels[vp] = buf;
    }
    *out_axis_labels_ptr = axis_labels;
    return 0;
}

static NPY_INLINE LabelsBlock *
labelsblock_new_default(PyArrayObject *arr)
{
    int ndim = PyArray_NDIM(arr);
    npy_intp *shape = PyArray_SHAPE(arr);
    LabelsBlock *lb = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!lb) return NULL;
    lb->ndim = ndim;
    lb->refcount = 1;
    lb->axis_len = (int *)calloc(ndim, sizeof(int));
    lb->axis_labels = (char ***)calloc(ndim, sizeof(char **));
    lb->axis_hash = (AxisHash *)calloc(ndim, sizeof(AxisHash));
    lb->axis_borrowed = (unsigned char *)calloc(ndim, sizeof(unsigned char));
    lb->borrowed_owner = NULL;
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash || !lb->axis_borrowed) { labelsblock_decref(lb); return NULL; }
    for (int ax = 0; ax < ndim; ++ax) {
        int n = (int)shape[ax];
        lb->axis_len[ax] = n;
        lb->axis_labels[ax] = (char **)calloc(n, sizeof(char *));
        if (!lb->axis_labels[ax]) { labelsblock_decref(lb); return NULL; }
        for (int i = 0; i < n; ++i) {
            char buf[32];
            int m = snprintf(buf, sizeof(buf), "%d", i);
            if (m < 0 || m >= (int)sizeof(buf)) { labelsblock_decref(lb); return NULL; }
            lb->axis_labels[ax][i] = cstr_dup(buf);
            if (!lb->axis_labels[ax][i]) { labelsblock_decref(lb); return NULL; }
        }
        if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], n) < 0) { labelsblock_decref(lb); return NULL; }
    }
    return lb;
}

static NPY_INLINE LabelsBlock *
labelsblock_from_py(PyObject *labels_in, PyArrayObject *arr)
{
    if (labels_in == NULL || labels_in == Py_None) {
        return labelsblock_new_default(arr);
    }
    int ndim = PyArray_NDIM(arr);
    if (!PyTuple_Check(labels_in)) {
        labels_in = PySequence_Tuple(labels_in);
        if (!labels_in) return NULL;
    } else {
        Py_INCREF(labels_in);
    }
    if ((int)PyTuple_GET_SIZE(labels_in) != ndim) {
        Py_DECREF(labels_in);
        PyErr_SetString(PyExc_ValueError, "labels must match ndim");
        return NULL;
    }
    LabelsBlock *lb = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!lb) { Py_DECREF(labels_in); return NULL; }
    lb->ndim = ndim;
    lb->refcount = 1;
    lb->axis_len = (int *)calloc(ndim, sizeof(int));
    lb->axis_labels = (char ***)calloc(ndim, sizeof(char **));
    lb->axis_hash = (AxisHash *)calloc(ndim, sizeof(AxisHash));
    lb->axis_borrowed = (unsigned char *)calloc(ndim, sizeof(unsigned char));
    lb->borrowed_owner = NULL;
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash || !lb->axis_borrowed) { Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
    for (int ax = 0; ax < ndim; ++ax) {
        PyObject *axis_seq = PySequence_Tuple(PyTuple_GET_ITEM(labels_in, ax));
        if (!axis_seq) { Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
        Py_ssize_t n = PyTuple_GET_SIZE(axis_seq);
        if ((npy_intp)n != PyArray_SHAPE(arr)[ax]) {
            Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb);
            PyErr_SetString(PyExc_ValueError, "labels axis length mismatch");
            return NULL;
        }
        lb->axis_len[ax] = (int)n;
        lb->axis_labels[ax] = (char **)calloc((size_t)n, sizeof(char *));
        if (!lb->axis_labels[ax]) { Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *it = PyTuple_GET_ITEM(axis_seq, i);
            PyObject *s = PyObject_Str(it);
            if (!s) { Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            const char *c = PyUnicode_AsUTF8(s);
            if (!c) { Py_DECREF(s); Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            size_t L = strlen(c) + 1;
            lb->axis_labels[ax][(int)i] = (char *)PyDataMem_NEW(L);
            if (!lb->axis_labels[ax][(int)i]) { Py_DECREF(s); Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            memcpy(lb->axis_labels[ax][(int)i], c, L);
            Py_DECREF(s);
        }
        Py_DECREF(axis_seq);
        if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], (int)n) < 0) { Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
    }
    Py_DECREF(labels_in);
    return lb;
}

static NPY_INLINE PyObject *
labelsblock_to_py_tuple(const LabelsBlock *lb)
{
    PyObject *out = PyTuple_New(lb->ndim);
    if (!out) return NULL;
    for (int ax = 0; ax < lb->ndim; ++ax) {
        if (lb->axis_arr && lb->axis_arr[ax]) { Py_INCREF(lb->axis_arr[ax]); PyTuple_SET_ITEM(out, ax, (PyObject *)lb->axis_arr[ax]); }
        else {
        int n = lb->axis_len[ax];
        PyObject *axis = PyTuple_New(n);
        if (!axis) { Py_DECREF(out); return NULL; }
        for (int i = 0; i < n; ++i) {
            PyObject *s = PyUnicode_FromString(lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "");
            if (!s) { Py_DECREF(axis); Py_DECREF(out); return NULL; }
            PyTuple_SET_ITEM(axis, i, s);
        }
        PyTuple_SET_ITEM(out, ax, axis);
        }
    }
    return out;
}


/* Wrap an ndarray result to LabeledArray and attach labels via ck, returning scalar if 0d. Does not touch ck's refcount. */
/* ---------- index/label helpers ---------- */
static NPY_INLINE PyArrayObject *
map_label_index_array_to_int(const LabelsBlock *lb, int axis, PyArrayObject *arrk)
{
    int ind_nd = PyArray_NDIM(arrk);
    const npy_intp *adims = PyArray_DIMS(arrk);
    PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(ind_nd, adims, NPY_INTP);
    if (!ind) return NULL;
    if (ind_nd == 1) {
        npy_intp n = adims[0];
        for (npy_intp i = 0; i < n; ++i) {
            PyObject *it = PyArray_GETITEM(arrk, PyArray_GetPtr(arrk, &i));
            if (!it) { Py_DECREF(ind); return NULL; }
            Py_ssize_t pos = -1;
            if (labelsblock_find(lb, axis, it, &pos) != 0) { Py_DECREF(it); Py_DECREF(ind); return NULL; }
            Py_DECREF(it);
            *((npy_intp *)PyArray_GetPtr(ind, &i)) = pos;
        }
    } else {
        /* general k-D */
        npy_intp total = 1;
        for (int d = 0; d < ind_nd; ++d) total *= adims[d];
        /* iterate linear index */
        npy_intp idxk[NPY_MAXDIMS];
        for (int d = 0; d < ind_nd; ++d) idxk[d] = 0;
        while (1) {
            PyObject *it = PyArray_GETITEM(arrk, PyArray_GetPtr(arrk, idxk));
            if (!it) { Py_DECREF(ind); return NULL; }
            Py_ssize_t pos = -1;
            if (labelsblock_find(lb, axis, it, &pos) != 0) { Py_DECREF(it); Py_DECREF(ind); return NULL; }
            Py_DECREF(it);
            *((npy_intp *)PyArray_GetPtr(ind, idxk)) = pos;
            int q = ind_nd - 1;
            while (q >= 0) {
                idxk[q]++;
                if (idxk[q] < adims[q]) break;
                idxk[q] = 0; q--;
            }
            if (q < 0) break;
        }
    }
    return ind;
}

/* Build sliced labels block for a view result, given parent labels and converted key */
static NPY_INLINE LabelsBlock *
labelsblock_slice(const LabelsBlock *lb, PyObject *ck, PyArrayObject *result)
{
    int res_ndim = PyArray_NDIM(result);
    const npy_intp *res_shape = PyArray_SHAPE(result);
    LabelsBlock *out = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!out) return NULL;
    out->ndim = res_ndim;
    out->refcount = 1;
    out->axis_len = (int *)calloc(res_ndim, sizeof(int));
    out->axis_labels = (char ***)calloc(res_ndim, sizeof(char **));
    out->axis_hash = (AxisHash *)calloc(res_ndim, sizeof(AxisHash));
    if (!out->axis_len || !out->axis_labels || !out->axis_hash) { labelsblock_decref(out); return NULL; }

    int src_axis = 0;
    int dst_axis = 0;
    Py_ssize_t nkeys = PyTuple_Check(ck) ? PyTuple_GET_SIZE(ck) : 0;

    for (Py_ssize_t i = 0; i < nkeys; ++i) {
        PyObject *k = PyTuple_GET_ITEM(ck, i);
        if (k == Py_None) {
            /* newaxis: size from result shape */
            if (dst_axis >= res_ndim) { labelsblock_decref(out); return NULL; }
            int n = (int)res_shape[dst_axis];
            out->axis_len[dst_axis] = n;
            out->axis_labels[dst_axis] = (char **)calloc(n, sizeof(char *));
            if (!out->axis_labels[dst_axis]) { labelsblock_decref(out); return NULL; }
            for (int j = 0; j < n; ++j) {
                const char *s = "1";
                size_t L = strlen(s) + 1;
                out->axis_labels[dst_axis][j] = (char *)PyDataMem_NEW(L);
                if (!out->axis_labels[dst_axis][j]) { labelsblock_decref(out); return NULL; }
                memcpy(out->axis_labels[dst_axis][j], s, L);
            }
            /* build hash for this axis */
            if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], n) < 0) { labelsblock_decref(out); return NULL; }
            dst_axis += 1;
            continue;
        }
        if (PyLong_Check(k)) {
            /* integer index: drop this axis */
            src_axis += 1;
            continue;
        }
        /* selection on source axis */
        if (dst_axis >= res_ndim || src_axis >= lb->ndim) { labelsblock_decref(out); return NULL; }
        int out_len = (int)res_shape[dst_axis];
        out->axis_len[dst_axis] = out_len;
        out->axis_labels[dst_axis] = (char **)calloc(out_len, sizeof(char *));
        if (!out->axis_labels[dst_axis]) { labelsblock_decref(out); return NULL; }

        if (PySlice_Check(k)) {
            Py_ssize_t slen = lb->axis_len[src_axis];
            npy_intp start, stop, step, length;
            if (PySlice_GetIndicesEx(k, slen, &start, &stop, &step, &length) < 0) { labelsblock_decref(out); return NULL; }
            int pos = 0;
            for (npy_intp idx = start; (step > 0) ? (idx < stop) : (idx > stop); idx += step) {
                const char *s = lb->axis_labels[src_axis][(int)idx];
                size_t L = strlen(s) + 1;
                out->axis_labels[dst_axis][pos] = (char *)PyDataMem_NEW(L);
                if (!out->axis_labels[dst_axis][pos]) { labelsblock_decref(out); return NULL; }
                memcpy(out->axis_labels[dst_axis][pos], s, L);
                if (++pos >= out_len) break;
            }
        } else if (PyArray_Check(k)) {
            /* Integer or Unicode ndarray indices. Support 1D and 2D.
               1D: replace source axis with length n
               2D: insert two axes with lengths (R, C), labels are joined strings per row/col. */
            PyArrayObject *arrk = (PyArrayObject *)PyArray_FromAny(k, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if (!arrk) { labelsblock_decref(out); return NULL; }
            int kind = PyArray_TYPE(arrk);
            PyArrayObject *ind = NULL;
            /* Handle np.ix_-style multiple advanced indices: consecutive integer ndarrays
               each with ndim == m and exactly one non-singleton dimension at its own position. */
            if (PyTypeNum_ISINTEGER(kind)) {
                Py_ssize_t jscan = i;
                int m = 0;
                int nd_common = -1;
                int ix_style = 1;
                PyArrayObject *inds_arr[NPY_MAXDIMS];
                for (int t = 0; t < NPY_MAXDIMS; ++t) inds_arr[t] = NULL;
                while (jscan < nkeys) {
                    PyObject *kj = PyTuple_GET_ITEM(ck, jscan);
                    if (!PyArray_Check(kj)) break;
                    PyArrayObject *aj = (PyArrayObject *)PyArray_FromAny(kj, PyArray_DescrFromType(NPY_INTP), 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                    if (!aj) { ix_style = 0; break; }
                    int tj = PyArray_TYPE(aj);
                    if (!PyTypeNum_ISINTEGER(tj)) { Py_DECREF(aj); break; }
                    if (nd_common == -1) nd_common = PyArray_NDIM(aj);
                    if (PyArray_NDIM(aj) != nd_common) { Py_DECREF(aj); ix_style = 0; break; }
                    if (m >= NPY_MAXDIMS) { Py_DECREF(aj); ix_style = 0; break; }
                    inds_arr[m++] = aj;
                    jscan++;
                }
                if (!(m >= 2 && nd_common == m)) {
                    ix_style = 0;
                }
                if (ix_style) {
                    for (int p = 0; p < m && ix_style; ++p) {
                        const npy_intp *dims = PyArray_DIMS(inds_arr[p]);
                        for (int q = 0; q < m; ++q) {
                            npy_intp d = dims[q];
                            if (q == p) {
                                if (d <= 0) { ix_style = 0; break; }
                            } else {
                                if (d != 1) { ix_style = 0; break; }
                            }
                        }
                    }
                }
                if (ix_style) {
                    /* Free pre-allocation for current dst axis; we'll allocate per-axis below */
                    if (out->axis_labels[dst_axis]) { free(out->axis_labels[dst_axis]); out->axis_labels[dst_axis] = NULL; }
                    /* Build labels for each broadcasted index axis */
                    for (int p = 0; p < m; ++p) {
                        if (dst_axis + p >= res_ndim || src_axis + p >= lb->ndim) { for (int t = 0; t < m; ++t) Py_XDECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                        int Lp = (int)PyArray_DIMS(inds_arr[p])[p];
                        out->axis_len[dst_axis + p] = Lp;
                        out->axis_labels[dst_axis + p] = (char **)calloc((size_t)Lp, sizeof(char *));
                        if (!out->axis_labels[dst_axis + p]) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                        for (int v = 0; v < Lp && (dst_axis + p) < res_ndim && v < (int)res_shape[dst_axis + p]; ++v) {
                            npy_intp idxix[NPY_MAXDIMS];
                            for (int q = 0; q < m; ++q) idxix[q] = 0;
                            idxix[p] = v;
                            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(inds_arr[p], idxix));
                            int axlen = (src_axis + p < lb->ndim) ? lb->axis_len[src_axis + p] : 0;
                            if (axlen <= 0) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                            if (idxv < 0) idxv = 0;
                            if (idxv >= axlen) idxv = (npy_intp)(axlen - 1);
                            const char *s = lb->axis_labels[src_axis + p][(int)idxv];
                            size_t Ls = strlen(s) + 1;
                            out->axis_labels[dst_axis + p][v] = (char *)malloc(Ls);
                            if (!out->axis_labels[dst_axis + p][v]) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                            memcpy(out->axis_labels[dst_axis + p][v], s, Ls);
                        }
                        if (axhash_build(&out->axis_hash[dst_axis + p], (const char **)out->axis_labels[dst_axis + p], Lp) < 0) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                    }
                    for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]);
                    Py_DECREF(arrk);
                    src_axis += m;
                    dst_axis += m;
                    i += (m - 1);
                    continue;
                } else {
                    for (int t = 0; t < m; ++t) Py_XDECREF(inds_arr[t]);
                }
            }
            int arr_case = 0;
            if (kind == NPY_BOOL) arr_case = 1;
            else if (PyTypeNum_ISINTEGER(kind)) arr_case = 2;
            else if (kind == NPY_UNICODE || kind == NPY_STRING || kind == NPY_OBJECT) arr_case = 3;
            else arr_case = 4;

            switch (arr_case) {
                case 1: /* boolean mask */
                    if (build_labels_from_bool_mask(lb, src_axis, arrk, out_len, &out->axis_labels[dst_axis]) < 0) { Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                    if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], out_len) < 0) { Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                Py_DECREF(arrk);
                src_axis += 1;
                dst_axis += 1;
                continue;
                case 2: /* integer indices */
                ind = (PyArrayObject *)PyArray_FromAny((PyObject *)arrk, PyArray_DescrFromType(NPY_INTP), 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                    break;
                case 3: /* unicode/bytes/object labels */
                    ind = map_label_index_array_to_int(lb, src_axis, arrk);
                    if (!ind) { Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "label not found in axis"); return NULL; }
                    break;
                default:
                Py_DECREF(arrk);
                labelsblock_decref(out);
                PyErr_SetString(PyExc_TypeError, "unsupported index array dtype");
                return NULL;
            }
            Py_DECREF(arrk);

            int ind_nd = PyArray_NDIM(ind);
            const npy_intp *adims = PyArray_DIMS(ind);
            switch (ind_nd) {
                case 1: {
                int n = (int)adims[0];
                out->axis_len[dst_axis] = n;
                    if (fill_labels_from_int_indices(&out->axis_labels[dst_axis], (n < out_len ? n : out_len), lb->axis_labels[src_axis], ind) < 0) { Py_DECREF(ind); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], n) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                Py_DECREF(ind);
                    break;
                }
                case 2: {
                int R = (int)adims[0], C = (int)adims[1];
                    if (dst_axis >= res_ndim || dst_axis + 1 >= res_ndim) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                out->axis_len[dst_axis] = R;
                out->axis_len[dst_axis + 1] = C;
                out->axis_labels[dst_axis + 1] = (char **)calloc(C, sizeof(char *));
                if (!out->axis_labels[dst_axis + 1]) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    char **rows = NULL, **cols = NULL;
                    if (build_rowcol_labels_from_2d_indices(lb, src_axis, ind, R, C, &rows, &cols) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    out->axis_labels[dst_axis] = rows;
                    out->axis_labels[dst_axis + 1] = cols;
                    if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], R) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                if (axhash_build(&out->axis_hash[dst_axis + 1], (const char **)out->axis_labels[dst_axis + 1], C) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                Py_DECREF(ind);
                src_axis += 1;
                dst_axis += 2;
                continue;
                }
                default: {
                int knd = ind_nd;
                for (int p = 0; p < knd; ++p) {
                    if (dst_axis + p >= res_ndim) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    out->axis_len[dst_axis + p] = (int)adims[p];
                    if (p == 0) {
                        /* out->axis_labels[dst_axis] already allocated to res_shape[dst_axis] */
                    } else {
                        out->axis_labels[dst_axis + p] = (char **)calloc((size_t)adims[p], sizeof(char *));
                        if (!out->axis_labels[dst_axis + p]) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    }
                }
                for (int p = 0; p < knd; ++p) {
                        if (build_kd_labels_for_axis(lb, src_axis, ind, knd, p, adims, &out->axis_labels[dst_axis + p]) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    if (axhash_build(&out->axis_hash[dst_axis + p], (const char **)out->axis_labels[dst_axis + p], (int)adims[p]) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                }
                Py_DECREF(ind);
                src_axis += 1;
                dst_axis += knd;
                continue;
            }
            }
        } else if (PyList_Check(k) || PyTuple_Check(k)) {
            if (build_labels_from_py_sequence(lb, src_axis, k, out_len, &out->axis_labels[dst_axis]) < 0) { labelsblock_decref(out); return NULL; }
            /* build hash below */
        } else {
            /* fallback: copy entire axis truncated to out_len */
            int n = lb->axis_len[src_axis];
            for (int j = 0; j < n && j < out_len; ++j) {
                const char *s = lb->axis_labels[src_axis][j];
                size_t L = strlen(s) + 1;
                out->axis_labels[dst_axis][j] = (char *)PyDataMem_NEW(L);
                if (!out->axis_labels[dst_axis][j]) { labelsblock_decref(out); return NULL; }
                memcpy(out->axis_labels[dst_axis][j], s, L);
            }
        }
        /* build hash for this axis */
        if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], out_len) < 0) { labelsblock_decref(out); return NULL; }
        src_axis += 1;
        dst_axis += 1;
    }
    /* fill remaining axes as full copies if any */
    while (dst_axis < res_ndim && src_axis < lb->ndim) {
        int out_len2 = (int)res_shape[dst_axis];
        out->axis_len[dst_axis] = out_len2;
        out->axis_labels[dst_axis] = (char **)calloc(out_len2, sizeof(char *));
        if (!out->axis_labels[dst_axis]) { labelsblock_decref(out); return NULL; }
        int n = lb->axis_len[src_axis];
        for (int j = 0; j < n && j < out_len2; ++j) {
            const char *s = lb->axis_labels[src_axis][j];
            size_t L = strlen(s) + 1;
            out->axis_labels[dst_axis][j] = (char *)PyDataMem_NEW(L);
            if (!out->axis_labels[dst_axis][j]) { labelsblock_decref(out); return NULL; }
            memcpy(out->axis_labels[dst_axis][j], s, L);
        }
        if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], out_len2) < 0) { labelsblock_decref(out); return NULL; }
        src_axis += 1;
        dst_axis += 1;
    }
    return out;
}


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
    /* Build tuple of read-only 1D unicode ndarrays per axis */
    PyObject *out = PyTuple_New(lb->ndim);
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
        PyObject *arr = PyArray_FromAny(lst, NULL, 1, 1, NPY_ARRAY_ENSUREARRAY, NULL);
        Py_DECREF(lst);
        if (!arr) { Py_DECREF(out); return NULL; }
        PyArray_CLEARFLAGS((PyArrayObject *)arr, NPY_ARRAY_WRITEABLE);
        PyTuple_SET_ITEM(out, ax, arr); /* steals ref */
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
        if (PyUnicode_Check(k) || PyList_Check(k) || PyTuple_Check(k)) {
            if (PyUnicode_Check(k)) {
                Py_ssize_t pos; int rc = labelsblock_find(lb, axis, k, &pos);
                if (rc != 0) { Py_DECREF(out); Py_DECREF(items); PyErr_SetString(PyExc_IndexError, "label not found"); return NULL; }
                ck = PyLong_FromSsize_t(pos);
                if (!ck) { Py_DECREF(out); Py_DECREF(items); return NULL; }
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
    if (np_concatenate == NULL || np_squeeze == NULL || np_stack == NULL || np_expand_dims == NULL || np_take == NULL || np_transpose == NULL || np_swapaxes == NULL || np_taa == NULL) {
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
        Py_DECREF(numpy);
        if (!np_concatenate || !np_squeeze || !np_stack || !np_expand_dims || !np_take || !np_transpose || !np_swapaxes || !np_taa) return NULL;
    }
    int is_concatenate = (func == np_concatenate);
    int is_squeeze = (func == np_squeeze);
    int is_stack = (func == np_stack);
    int is_expand = (func == np_expand_dims);
    int is_take = (func == np_take);
    int is_transpose = (func == np_transpose);
    int is_swapaxes = (func == np_swapaxes);
    int is_taa = (func == np_taa);
    if (!is_concatenate && !is_squeeze && !is_stack && !is_expand && !is_take && !is_transpose && !is_swapaxes && !is_taa) {
        /* Fallback: call NumPy func on base ndarrays to avoid recursion */
        if (!PyTuple_Check(fargs)) { Py_RETURN_NOTIMPLEMENTED; }
        Py_ssize_t nfa = PyTuple_GET_SIZE(fargs);
        PyObject *new_fargs = PyTuple_New(nfa);
        if (!new_fargs) return NULL;
        for (Py_ssize_t i = 0; i < nfa; ++i) {
            PyObject *argi = PyTuple_GET_ITEM(fargs, i);
            PyObject *to_set = argi;
            if (PyObject_TypeCheck(argi, &LabeledArray_Type)) {
                to_set = PyArray_View((PyArrayObject *)argi, NULL, &PyArray_Type);
                if (!to_set) { Py_DECREF(new_fargs); return NULL; }
            } else {
                Py_INCREF(to_set);
            }
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
                /* Map strings to intp for data; keep original for labels */
                int ind_nd = PyArray_NDIM(arrk);
                const npy_intp *adims = PyArray_DIMS(arrk);
                PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(ind_nd, adims, NPY_INTP);
                if (!ind) { Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                if (ind_nd == 1) {
                    npy_intp n = adims[0];
                    for (npy_intp i = 0; i < n; ++i) {
                        PyObject *it = PyArray_GETITEM(arrk, PyArray_GetPtr(arrk, &i));
                        if (!it) { Py_DECREF(ind); Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                        Py_ssize_t pos = -1;
                        if (labelsblock_find(lb, axis, it, &pos) != 0) { Py_DECREF(it); Py_DECREF(ind); Py_DECREF(ck_data); Py_DECREF(ck_lbl); PyErr_SetString(PyExc_IndexError, "label not found in axis for take"); return NULL; }
                        Py_DECREF(it);
                        *((npy_intp *)PyArray_GetPtr(ind, &i)) = pos;
                    }
                } else {
                    npy_intp R = adims[0], C = (ind_nd > 1 ? adims[1] : 1);
                    for (npy_intp r = 0; r < R; ++r) {
                        for (npy_intp c = 0; c < C; ++c) {
                            npy_intp ij[2] = {r, c};
                            PyObject *it = PyArray_GETITEM(arrk, PyArray_GetPtr(arrk, ij));
                            if (!it) { Py_DECREF(ind); Py_DECREF(ck_data); Py_DECREF(ck_lbl); return NULL; }
                            Py_ssize_t pos = -1;
                            if (labelsblock_find(lb, axis, it, &pos) != 0) { Py_DECREF(it); Py_DECREF(ind); Py_DECREF(ck_data); Py_DECREF(ck_lbl); PyErr_SetString(PyExc_IndexError, "label not found in axis for take"); return NULL; }
                            Py_DECREF(it);
                            *((npy_intp *)PyArray_GetPtr(ind, ij)) = pos;
                        }
                    }
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

    /* Data path via numpy.take */
    PyObject *numpy = PyImport_ImportModule("numpy"); if (!numpy) return NULL;
    PyObject *np_take = PyObject_GetAttrString(numpy, "take"); Py_DECREF(numpy);
    if (!np_take) return NULL;
    PyObject *res_base = PyObject_CallFunction(np_take, "OOi", self_obj, indices, axis);
    Py_DECREF(np_take);
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
                if (PyTypeNum_ISINTEGER(t)) {
                    ind = (PyArrayObject *)PyArray_FromAny((PyObject *)arrk, PyArray_DescrFromType(NPY_INTP), 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
                } else if (t == NPY_UNICODE || t == NPY_STRING || t == NPY_OBJECT) {
                    ind = map_label_index_array_to_int(plb, ax, arrk);
                }
                Py_DECREF(arrk);
                if (!ind) { Py_DECREF(ck); Py_DECREF(view); PyErr_SetString(PyExc_TypeError, "unsupported indices for take"); return NULL; }
                PyTuple_SET_ITEM(ck, ax, (PyObject *)ind);
            } else if (PyList_Check(indices) || PyTuple_Check(indices)) {
                PyObject *seq = PySequence_Fast(indices, "expected sequence");
                if (!seq) { Py_DECREF(ck); Py_DECREF(view); return NULL; }
                Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(1, (npy_intp *)&n, NPY_INTP);
                if (!ind) { Py_DECREF(seq); Py_DECREF(ck); Py_DECREF(view); return NULL; }
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
                    *((npy_intp *)PyArray_GetPtr(ind, &j)) = val;
                }
                Py_DECREF(seq);
                PyTuple_SET_ITEM(ck, ax, (PyObject *)ind);
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
    LabelsBlock *out = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!out) { Py_DECREF(view); return NULL; }
    out->ndim = new_nd; out->refcount = 1;
    out->axis_len = (int *)calloc(new_nd, sizeof(int));
    out->axis_labels = (char ***)calloc(new_nd, sizeof(char **));
    out->axis_hash = (AxisHash *)calloc(new_nd, sizeof(AxisHash));
    if (!out->axis_len || !out->axis_labels || !out->axis_hash) { labelsblock_decref(out); Py_DECREF(view); return NULL; }

    const char *joiner = delim ? delim : (self->delimiter ? self->delimiter : "-");
    int dst = 0;
    for (int ax = 0; ax < nd; ++ax) {
        if (ax == a0) continue; /* drop */
        if (ax == a1) {
            int n0 = plb->axis_len[a0];
            int n1 = plb->axis_len[a1];
            out->axis_len[dst] = n0 * n1;
            out->axis_labels[dst] = (char **)calloc(n0 * n1, sizeof(char *));
            if (!out->axis_labels[dst]) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
            size_t joiner_len = strlen(joiner);
            size_t *len0 = (size_t *)malloc(sizeof(size_t) * (size_t)n0);
            size_t *len1 = (size_t *)malloc(sizeof(size_t) * (size_t)n1);
            if (!len0 || !len1) { free(len0); free(len1); labelsblock_decref(out); Py_DECREF(view); return NULL; }
            size_t sum_len1 = 0; /* retained for potential future heuristics */
                for (int j = 0; j < n1; ++j) {
                const char *s = plb->axis_labels[a1][j];
                size_t L = strlen(s ? s : "");
                len1[j] = L; sum_len1 += L;
            }
            for (int i = 0; i < n0; ++i) {
                const char *s = plb->axis_labels[a0][i];
                len0[i] = strlen(s ? s : "");
            }
            int single_delim = (joiner_len == 1);
            char dch = single_delim ? joiner[0] : '\0';
            /* Fill labels; release GIL during heavy memcpy/alloc */
            int build_ok = 1;
            NPY_BEGIN_ALLOW_THREADS;
            for (int i = 0; i < n0 && build_ok; ++i) {
                    const char *s0 = plb->axis_labels[a0][i];
                size_t L0 = len0[i];
                for (int j = 0; j < n1; ++j) {
                    const char *s1 = plb->axis_labels[a1][j];
                    size_t L1 = len1[j];
                    size_t Ltot = L0 + joiner_len + L1 + 1;
                    char *dstp = (char *)malloc(Ltot);
                    if (!dstp) { build_ok = 0; break; }
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
            if (!build_ok) {
                /* cleanup any partially allocated labels */
                for (int p = 0; p < n0 * n1; ++p) {
                    if (out->axis_labels[dst][p]) free(out->axis_labels[dst][p]);
                }
                free(out->axis_labels[dst]); out->axis_labels[dst] = NULL;
                labelsblock_decref(out);
                Py_DECREF(view);
                free(len0);
                free(len1);
                return NULL;
            }
            free(len0);
            free(len1);
            if (axhash_build(&out->axis_hash[dst], (const char **)out->axis_labels[dst], n0 * n1) < 0) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
        } else {
            int n = plb->axis_len[ax];
            out->axis_len[dst] = n;
            out->axis_labels[dst] = (char **)calloc(n, sizeof(char *));
            if (!out->axis_labels[dst]) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
            for (int j = 0; j < n; ++j) {
                const char *s = plb->axis_labels[ax][j]; size_t L = strlen(s ? s : "") + 1;
                out->axis_labels[dst][j] = (char *)PyDataMem_NEW(L);
                if (!out->axis_labels[dst][j]) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
                memcpy(out->axis_labels[dst][j], s ? s : "", L);
            }
            if (axhash_build(&out->axis_hash[dst], (const char **)out->axis_labels[dst], n) < 0) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
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


static PyMethodDef LabeledArray_methods[] = {
    {"__array_finalize__", (PyCFunction)array_finalize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"__array_wrap__", (PyCFunction)array_wrap, METH_VARARGS | METH_KEYWORDS, NULL},
    {"__array_function__", (PyCFunction)array_function, METH_VARARGS, NULL},
    {"__array_ufunc__", (PyCFunction)array_ufunc, METH_VARARGS | METH_KEYWORDS, NULL},
    {"take", (PyCFunction)LabeledArray_take, METH_VARARGS | METH_KEYWORDS, NULL},
    {"combine", (PyCFunction)LabeledArray_combine, METH_VARARGS | METH_KEYWORDS, NULL},
    {"find", (PyCFunction)LabeledArray_find, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dropna", (PyCFunction)LabeledArray_dropna, METH_NOARGS, NULL},
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
