/* ndarray subtype: LabeledArray with C-level labels storage (stride-safe) */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <stdint.h>
#include <string.h>

#ifndef NPY_INLINE
#define NPY_INLINE static inline
#endif

static PyTypeObject LabeledArray_Type; /* forward */

typedef struct {
    size_t cap;         /* power-of-two capacity */
    size_t used;        /* number of entries */
    uint64_t *hashes;   /* hashes[slot] */
    const char **keys;  /* pointers into axis_labels strings */
    int *vals;          /* index per key */
} AxisHash;

typedef struct {
    int ndim;
    int *axis_len;       /* length per axis */
    char ***axis_labels; /* axis_labels[axis][index] -> C string */
    AxisHash *axis_hash; /* per-axis C hash */
    int refcount;        /* manual refcount */
} LabelsBlock;

/* explicit ndarray sub-type instance layout */
typedef struct {
    PyArrayObject base;
    LabelsBlock *labels_block;
} LabeledArrayObject;

/* ---------- small inline utils ---------- */
NPY_INLINE uint64_t
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

NPY_INLINE size_t
next_pow2(size_t n)
{
    size_t p = 1; while (p < n) p <<= 1; return p;
}

/* ---------- AxisHash management ---------- */
NPY_INLINE void
axhash_free(AxisHash *ah)
{
    if (!ah) return;
    free(ah->hashes);
    free(ah->keys);
    free(ah->vals);
}

NPY_INLINE int
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

NPY_INLINE int
axhash_find(const AxisHash *ah, const char *key, Py_ssize_t *out)
{
    if (!ah || ah->cap == 0) return 1;
    uint64_t h = fnv1a64(key ? key : "");
    size_t m = ah->cap - 1;
    size_t pos = (size_t)h & m;
    const uint64_t *hashes = ah->hashes;
    const char *const *keys = ah->keys;
    const int *vals = ah->vals;
    while (1) {
        const char *k = keys[pos];
        if (k == NULL) return 1; /* empty slot => not found */
        if (hashes[pos] == h && strcmp(k, key) == 0) {
            *out = (Py_ssize_t)vals[pos];
            return 0;
        }
        pos = (pos + 1) & m;
    }
}

/* ---------- LabelsBlock management ---------- */
NPY_INLINE void
labelsblock_decref(LabelsBlock *lb)
{
    if (!lb) return;
    if (--lb->refcount == 0) {
        if (lb->axis_labels) {
            for (int ax = 0; ax < lb->ndim; ++ax) {
                if (lb->axis_labels[ax]) {
                    for (int i = 0; i < lb->axis_len[ax]; ++i) {
                        if (lb->axis_labels[ax][i]) free(lb->axis_labels[ax][i]);
                    }
                    free(lb->axis_labels[ax]);
                }
            }
            free(lb->axis_labels);
        }
        if (lb->axis_hash) {
            for (int ax = 0; ax < lb->ndim; ++ax) axhash_free(&lb->axis_hash[ax]);
            free(lb->axis_hash);
        }
        if (lb->axis_len) free(lb->axis_len);
        free(lb);
    }
}

NPY_INLINE LabelsBlock *
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
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash) { labelsblock_decref(lb); return NULL; }
    for (int ax = 0; ax < ndim; ++ax) {
        int n = (int)shape[ax];
        lb->axis_len[ax] = n;
        lb->axis_labels[ax] = (char **)calloc(n, sizeof(char *));
        if (!lb->axis_labels[ax]) { labelsblock_decref(lb); return NULL; }
        for (int i = 0; i < n; ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%d", i);
            size_t L = strlen(buf) + 1;
            lb->axis_labels[ax][i] = (char *)malloc(L);
            if (!lb->axis_labels[ax][i]) { labelsblock_decref(lb); return NULL; }
            memcpy(lb->axis_labels[ax][i], buf, L);
        }
        if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], n) < 0) { labelsblock_decref(lb); return NULL; }
    }
    return lb;
}

NPY_INLINE LabelsBlock *
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
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash) { Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
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
            const char *c = NULL;
            PyObject *s_tmp = NULL;
            if (PyUnicode_Check(it)) {
                c = PyUnicode_AsUTF8(it);
                if (!c) { Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            } else {
                s_tmp = PyObject_Str(it);
                if (!s_tmp) { Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
                c = PyUnicode_AsUTF8(s_tmp);
                if (!c) { Py_DECREF(s_tmp); Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            }
            size_t L = strlen(c) + 1;
            lb->axis_labels[ax][(int)i] = (char *)malloc(L);
            if (!lb->axis_labels[ax][(int)i]) { if (s_tmp) Py_DECREF(s_tmp); Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            memcpy(lb->axis_labels[ax][(int)i], c, L);
            if (s_tmp) Py_DECREF(s_tmp);
        }
        Py_DECREF(axis_seq);
        if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], (int)n) < 0) { Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
    }
    Py_DECREF(labels_in);
    return lb;
}

NPY_INLINE PyObject *
labelsblock_to_py_tuple(const LabelsBlock *lb)
{
    PyObject *out = PyTuple_New(lb->ndim);
    if (!out) return NULL;
    for (int ax = 0; ax < lb->ndim; ++ax) {
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
    return out;
}

NPY_INLINE int
labelsblock_find(const LabelsBlock *lb, int axis, PyObject *value, Py_ssize_t *out)
{
    if (axis < 0 || axis >= lb->ndim) return -1;
    if (PyUnicode_Check(value)) {
        Py_ssize_t sz; const char *c = PyUnicode_AsUTF8AndSize(value, &sz);
        if (!c) return -1;
        return axhash_find(&lb->axis_hash[axis], c, out);
    }
    /* fallback scan for non-unicode keys: convert value once */
    PyObject *tmp = PyObject_Str(value);
    if (!tmp) return -1;
    const char *vc = PyUnicode_AsUTF8(tmp);
    if (!vc) { Py_DECREF(tmp); return -1; }
    int n = lb->axis_len[axis];
    for (int i = 0; i < n; ++i) {
        if (strcmp(vc, lb->axis_labels[axis][i]) == 0) {
            *out = i;
            Py_DECREF(tmp);
            return 0;
        }
    }
    Py_DECREF(tmp);
    return 1;
}

/* Build sliced labels block for a view result, given parent labels and converted key */
NPY_INLINE LabelsBlock *
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
                out->axis_labels[dst_axis][j] = (char *)malloc(L);
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
                out->axis_labels[dst_axis][pos] = (char *)malloc(L);
                if (!out->axis_labels[dst_axis][pos]) { labelsblock_decref(out); return NULL; }
                memcpy(out->axis_labels[dst_axis][pos], s, L);
                if (++pos >= out_len) break;
            }
        } else if (PyList_Check(k) || PyTuple_Check(k)) {
            PyObject *seq = PySequence_Fast(k, "expected sequence");
            if (!seq) { labelsblock_decref(out); return NULL; }
            Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
            for (Py_ssize_t j = 0; j < n && j < out_len; ++j) {
                PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
                int idx = (int)PyLong_AsLong(it);
                if (idx < 0 || idx >= lb->axis_len[src_axis]) { Py_DECREF(seq); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                const char *s = lb->axis_labels[src_axis][idx];
                size_t L = strlen(s) + 1;
                out->axis_labels[dst_axis][(int)j] = (char *)malloc(L);
                if (!out->axis_labels[dst_axis][(int)j]) { Py_DECREF(seq); labelsblock_decref(out); return NULL; }
                memcpy(out->axis_labels[dst_axis][(int)j], s, L);
            }
            Py_DECREF(seq);
        } else {
            /* fallback: copy entire axis truncated to out_len */
            int n = lb->axis_len[src_axis];
            for (int j = 0; j < n && j < out_len; ++j) {
                const char *s = lb->axis_labels[src_axis][j];
                size_t L = strlen(s) + 1;
                out->axis_labels[dst_axis][j] = (char *)malloc(L);
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
            out->axis_labels[dst_axis][j] = (char *)malloc(L);
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
NPY_INLINE LabelsBlock **
labels_slot(PyObject *self)
{
    return &((LabeledArrayObject *)self)->labels_block;
}

/* ---------- getters/setters ---------- */
static PyObject *
LabeledArray_get_labels(PyObject *self, void *closure)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    LabelsBlock *lb = obj->labels_block;
    if (!lb) {
        lb = labelsblock_new_default((PyArrayObject *)self);
        if (!lb) return NULL;
        obj->labels_block = lb;
    }
    return labelsblock_to_py_tuple(lb);
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
    return 0;
}

/* ---------- __new__/finalize/dealloc ---------- */
static PyObject *
LabeledArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *input = NULL; PyObject *labels_in = NULL; PyObject *dtype_obj = NULL;
    static char *kwlist[] = {"input_array", "labels", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO:LabeledArray", kwlist, &input, &labels_in, &dtype_obj)) return NULL;

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
    return view;
}

static void
LabeledArray_dealloc(PyObject *self)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    labelsblock_decref(obj->labels_block);
    obj->labels_block = NULL;
    PyArray_Type.tp_dealloc(self);
}

/* ---------- label-based mapping ---------- */
NPY_INLINE PyObject *
expand_and_convert_key_c(LabeledArrayObject *self, PyObject *key)
{
    int ndim = PyArray_NDIM((PyArrayObject *)self);
    LabelsBlock *lb = self->labels_block;
    if (!lb) { lb = labelsblock_new_default((PyArrayObject *)self); if (!lb) return NULL; self->labels_block = lb; }

    PyObject *items = PyTuple_Check(key) ? (Py_INCREF(key), key) : (Py_INCREF(key), PyTuple_Pack(1, key));
    if (!items) return NULL;
    Py_ssize_t n_items = PyTuple_GET_SIZE(items);
    int consumed = 0; for (Py_ssize_t i=0;i<n_items;++i){PyObject *k=PyTuple_GET_ITEM(items,i); if (k==Py_Ellipsis||k==Py_None) continue; consumed++;}
    int need_slices = ndim - consumed;
    PyObject *elist = PyList_New(0); if (!elist){Py_DECREF(items); return NULL;}
    int axis=0, ellipsis_done=0;
    for (Py_ssize_t i=0;i<n_items;++i){
        PyObject *k=PyTuple_GET_ITEM(items,i);
        if (k==Py_Ellipsis){int n=need_slices; if (ellipsis_done) n=1; for(int j=0;j<n;++j){PyObject *sl=PySlice_New(NULL,NULL,NULL); if(!sl){Py_DECREF(elist); Py_DECREF(items); return NULL;} PyList_Append(elist,sl); Py_DECREF(sl); axis++;} ellipsis_done=1; continue;}
        if (k==Py_None){PyList_Append(elist,Py_None); continue;}
        if (axis>=ndim){Py_DECREF(elist); Py_DECREF(items); PyErr_SetString(PyExc_IndexError,"too many indices"); return NULL;}
        PyObject *ck=NULL;
        if (PyUnicode_Check(k)||PyList_Check(k)||PyTuple_Check(k)){
            if (PyUnicode_Check(k)){
                Py_ssize_t pos; int rc=labelsblock_find(lb,axis,k,&pos); if(rc!=0){Py_DECREF(elist); Py_DECREF(items); PyErr_SetString(PyExc_IndexError,"label not found"); return NULL;} ck=PyLong_FromSsize_t(pos);
            } else {
                PyObject *seq=PySequence_Fast(k,"expected sequence"); if(!seq){Py_DECREF(elist); Py_DECREF(items); return NULL;} Py_ssize_t n=PySequence_Fast_GET_SIZE(seq); ck=PyList_New(n); if(!ck){Py_DECREF(seq); Py_DECREF(elist); Py_DECREF(items); return NULL;} for(Py_ssize_t j=0;j<n;++j){PyObject *it=PySequence_Fast_GET_ITEM(seq,j); if(PyUnicode_Check(it)){Py_ssize_t pos; int rc=labelsblock_find(lb,axis,it,&pos); if(rc!=0){Py_DECREF(ck); Py_DECREF(seq); Py_DECREF(elist); Py_DECREF(items); PyErr_SetString(PyExc_IndexError,"label not found in sequence"); return NULL;} PyObject *idx=PyLong_FromSsize_t(pos); if(!idx){Py_DECREF(ck); Py_DECREF(seq); Py_DECREF(elist); Py_DECREF(items); return NULL;} PyList_SET_ITEM(ck,j,idx);} else {Py_INCREF(it); PyList_SET_ITEM(ck,j,it);} } Py_DECREF(seq);
            }
        } else { Py_INCREF(k); ck=k; }
        PyList_Append(elist, ck); Py_DECREF(ck); axis++; }
    while(axis<ndim){PyObject *sl=PySlice_New(NULL,NULL,NULL); if(!sl){Py_DECREF(elist); Py_DECREF(items); return NULL;} PyList_Append(elist,sl); Py_DECREF(sl); axis++;}
    Py_DECREF(items);
    PyObject *out=PyList_AsTuple(elist); Py_DECREF(elist); return out;
}

static PyObject *
LabeledArray_subscript(PyObject *self, PyObject *key)
{
    LabeledArrayObject *obj = (LabeledArrayObject *)self;
    /* hot path: single integer index (iteration common case) */
    if (PyLong_Check(key)) {
        PyObject *res = PyArray_Type.tp_as_mapping->mp_subscript(self, key);
        if (!res) return NULL;
        if (PyArray_Check(res)) {
            PyObject *target = res;
            if (!PyObject_TypeCheck(res, (PyTypeObject *)Py_TYPE(self))) {
                PyObject *viewres = PyArray_View((PyArrayObject *)res, NULL, (PyTypeObject *)Py_TYPE(self));
                Py_DECREF(res);
                if (!viewres) return NULL;
                target = viewres;
            }
            LabelsBlock *parent_lb = obj->labels_block;
            if (parent_lb) {
                PyObject *ck = PyTuple_Pack(1, key);
                if (!ck) { Py_DECREF(target); return NULL; }
                LabelsBlock *child = labelsblock_slice(parent_lb, ck, (PyArrayObject *)target);
                Py_DECREF(ck);
                if (child) {
                    LabeledArrayObject *tobj = (LabeledArrayObject *)target;
                    LabelsBlock *old = tobj->labels_block;
                    tobj->labels_block = child;
                    labelsblock_decref(old);
                }
            }
            if (PyArray_NDIM((PyArrayObject *)target) == 0) {
                PyObject *scalar = PyObject_CallMethod(target, "item", NULL);
                Py_DECREF(target);
                return scalar;
            }
            return target;
        }
        return res;
    }
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
    if (parent && PyObject_TypeCheck(parent, &LabeledArray_Type)) {
        LabeledArrayObject *par  = (LabeledArrayObject *)parent;
        LabelsBlock *plb = par->labels_block;
        if (plb) plb->refcount++;
        self->labels_block = plb;
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
        if (return_scalar == NULL || PyObject_IsTrue(return_scalar)) {
            return PyObject_CallMethod(out_arr, "item", NULL);
        }
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

    int parent_nd = plb ? plb->ndim : 0;
    int out_nd = PyArray_NDIM((PyArrayObject *)view);
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
            LabelsBlock *child = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
            if (!child) { free(kept_src); Py_DECREF(view); return NULL; }
            child->ndim = out_nd;
            child->refcount = 1;
            child->axis_len = (int *)calloc(out_nd, sizeof(int));
            child->axis_labels = (char ***)calloc(out_nd, sizeof(char **));
            child->axis_hash = (AxisHash *)calloc(out_nd, sizeof(AxisHash));
            if (!child->axis_len || !child->axis_labels || !child->axis_hash) { free(kept_src); labelsblock_decref(child); Py_DECREF(view); return NULL; }
            for (int i = 0; i < out_nd; ++i) {
                int src = kept_src[i];
                int n = (int)oshape[i];
                child->axis_len[i] = n;
                child->axis_labels[i] = (char **)calloc(n, sizeof(char *));
                if (!child->axis_labels[i]) { free(kept_src); labelsblock_decref(child); Py_DECREF(view); return NULL; }
                int src_n = plb->axis_len[src];
                for (int j = 0; j < n && j < src_n; ++j) {
                    const char *s = plb->axis_labels[src][j];
                    size_t L = strlen(s) + 1;
                    child->axis_labels[i][j] = (char *)malloc(L);
                    if (!child->axis_labels[i][j]) { free(kept_src); labelsblock_decref(child); Py_DECREF(view); return NULL; }
                    memcpy(child->axis_labels[i][j], s, L);
                }
                if (axhash_build(&child->axis_hash[i], (const char **)child->axis_labels[i], n) < 0) { free(kept_src); labelsblock_decref(child); Py_DECREF(view); return NULL; }
            }
            LabelsBlock *old2 = vobj->labels_block;
            vobj->labels_block = child;
            labelsblock_decref(old2);
        }
        free(kept_src);
    }

    return view;
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


static PyMethodDef LabeledArray_methods[] = {
    {"__array_finalize__", (PyCFunction)array_finalize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"__array_wrap__", (PyCFunction)array_wrap, METH_VARARGS | METH_KEYWORDS, NULL},
    {"find", (PyCFunction)LabeledArray_find, METH_VARARGS | METH_KEYWORDS, NULL},
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
