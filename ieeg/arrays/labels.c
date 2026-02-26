#define PY_SSIZE_T_CLEAN
#define NO_IMPORT_ARRAY
#include "labels.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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

static NPY_INLINE npy_intp
read_intp_1d(const char *base, npy_intp stride, npy_intp i)
{
    npy_intp v;
    memcpy(&v, base + i * stride, sizeof(npy_intp));
    return v;
}

static NPY_INLINE void
write_intp_1d(char *base, npy_intp stride, npy_intp i, npy_intp v)
{
    memcpy(base + i * stride, &v, sizeof(npy_intp));
}

static NPY_INLINE npy_bool
read_bool_1d(const char *base, npy_intp stride, npy_intp i)
{
    npy_bool v;
    memcpy(&v, base + i * stride, sizeof(npy_bool));
    return v;
}

static NPY_INLINE PyObject *
unicode_from_raw(const char *ptr, npy_intp itemsize)
{
    npy_intp n = itemsize / 4;
    const Py_UCS4 *u = (const Py_UCS4 *)ptr;
    while (n > 0 && u[n - 1] == 0) n--;
    return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, u, n);
}

static NPY_INLINE PyObject *
bytes_from_raw(const char *ptr, npy_intp itemsize)
{
    npy_intp n = itemsize;
    while (n > 0 && ptr[n - 1] == '\0') n--;
    return PyBytes_FromStringAndSize(ptr, n);
}

static NPY_INLINE npy_intp
read_intp_nd(const char *base, const npy_intp *strides, const npy_intp *idx, int nd)
{
    npy_intp off = 0;
    for (int d = 0; d < nd; ++d) {
        off += idx[d] * strides[d];
    }
    npy_intp v;
    memcpy(&v, base + off, sizeof(npy_intp));
    return v;
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
int labelsblock_find(const LabelsBlock *lb, int axis, PyObject *value, Py_ssize_t *out)
{
    if (axis < 0 || axis >= lb->ndim) return -1;
    if (PyUnicode_Check(value)) {
        Py_ssize_t sz; const char *c = PyUnicode_AsUTF8AndSize(value, &sz);
        if (!c) return -1;
        /* Fast-path via C hash when no embedded NULs. */
        if (memchr(c, '\0', (size_t)sz) == NULL) {
            int rc = axhash_find(&lb->axis_hash[axis], c, out);
            if (rc == 0) return 0;
            if (rc == 1 && !(lb->axis_index && lb->axis_index[axis])) return 1;
        }
        if (lb->axis_index && lb->axis_index[axis]) {
            PyObject *pos_obj = PyDict_GetItemWithError(lb->axis_index[axis], value);
            if (pos_obj) { *out = PyLong_AsSsize_t(pos_obj); if (*out == -1 && PyErr_Occurred()) return -1; return 0; }
            if (PyErr_Occurred()) return -1;
        }
        return 1;
    }
    if (lb->axis_index && lb->axis_index[axis]) {
        PyObject *key = NULL;
        if (PyBytes_Check(value)) { const char *c = PyBytes_AsString(value); if (!c) return -1; key = PyUnicode_FromString(c); if (!key) return -1; }
        else { key = PyObject_Str(value); if (!key) return -1; }
        PyObject *pos_obj = PyDict_GetItemWithError(lb->axis_index[axis], key);
        Py_DECREF(key);
        if (pos_obj) { *out = PyLong_AsSsize_t(pos_obj); if (*out == -1 && PyErr_Occurred()) return -1; return 0; }
        if (PyErr_Occurred()) return -1;
        return 1;
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
int labelsblock_finalize_axis_from_c(LabelsBlock *lb, int ax)
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
            for (int k = 0; k < j; ++k) PyDataMem_FREE(labels[k]);
            free(labels);
            return NULL;
        }
        labels[j] = cstr_dup(buf);
        if (!labels[j]) {
            for (int k = 0; k < j; ++k) PyDataMem_FREE(labels[k]);
            free(labels);
            return NULL;
        }
    }
    return labels;
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
void labelsblock_decref(LabelsBlock *lb)
{
    if (!lb) return;
    if (--lb->refcount == 0) {
        if (lb->axis_labels) {
            for (int ax = 0; ax < lb->ndim; ++ax) {
                if (lb->axis_labels[ax]) {
                    int labels_borrowed = lb->axis_borrowed ? lb->axis_borrowed[ax] : 0;
                    int strings_borrowed = labels_borrowed || (lb->axis_str_borrowed ? lb->axis_str_borrowed[ax] : 0);
                    if (!strings_borrowed) {
                        if (lb->axis_slab && lb->axis_slab[ax]) {
                            PyDataMem_FREE(lb->axis_slab[ax]);
                        } else {
                        for (int i = 0; i < lb->axis_len[ax]; ++i) {
                            if (lb->axis_labels[ax][i]) PyDataMem_FREE(lb->axis_labels[ax][i]);
                        }
                        }
                    }
                    if (!labels_borrowed) {
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
        if (lb->axis_str_borrowed) free(lb->axis_str_borrowed);
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
LabelsBlock *labelsblock_alloc(int nd)
{
    LabelsBlock *lb = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!lb) return NULL;
    lb->ndim = nd;
    lb->refcount = 1;
    lb->axis_len = (int *)calloc(nd, sizeof(int));
    lb->axis_labels = (char ***)calloc(nd, sizeof(char **));
    lb->axis_hash = (AxisHash *)calloc(nd, sizeof(AxisHash));
    lb->axis_borrowed = (unsigned char *)calloc(nd, sizeof(unsigned char));
    lb->axis_str_borrowed = (unsigned char *)calloc(nd, sizeof(unsigned char));
    lb->borrowed_owner = NULL;
    lb->axis_arr = (PyArrayObject **)calloc(nd, sizeof(PyArrayObject *));
    lb->axis_index = (PyObject **)calloc(nd, sizeof(PyObject *));
    lb->axis_slab = (char **)calloc(nd, sizeof(char *));
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash || !lb->axis_borrowed || !lb->axis_str_borrowed || !lb->axis_arr || !lb->axis_index || !lb->axis_slab) { labelsblock_decref(lb); return NULL; }
    return lb;
}

/* Default labels: numeric strings per axis */
LabelsBlock *labelsblock_new_default(PyArrayObject *arr)
{
    int nd = PyArray_NDIM(arr);
    LabelsBlock *lb = labelsblock_alloc(nd);
    if (!lb) return NULL;
    for (int ax = 0; ax < nd; ++ax) {
        int n = (int)PyArray_SHAPE(arr)[ax];
        lb->axis_len[ax] = n;
        lb->axis_labels[ax] = build_numeric_labels(n);
        if (!lb->axis_labels[ax]) { labelsblock_decref(lb); return NULL; }
        if (axhash_build(&lb->axis_hash[ax], (const char **)lb->axis_labels[ax], n) < 0) { labelsblock_decref(lb); return NULL; }
    }
    return lb;
}

int
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
int labelsblock_set_axis_copy(LabelsBlock *lb, int ax, char **src_labels, int n)
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
int labelsblock_set_axis_numeric(LabelsBlock *lb, int ax, int n)
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
int labelsblock_set_axis_constant(LabelsBlock *lb, int ax, const char *s)
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


static NPY_INLINE int
fill_labels_from_int_indices(char ***dst_labels_ptr, int out_len, char **src_labels, int src_len, PyArrayObject *ind)
{
    char **dst = *dst_labels_ptr;
    int own_dst = 0;
    if (!dst) {
        dst = (char **)calloc((size_t)out_len, sizeof(char *));
        if (!dst) return -1;
        own_dst = 1;
    }
    const char *base = PyArray_BYTES(ind);
    npy_intp stride = PyArray_STRIDES(ind)[0];
    const npy_intp *idata = PyArray_IS_C_CONTIGUOUS(ind) ? (const npy_intp *)PyArray_DATA(ind) : NULL;
    for (int j = 0; j < out_len; ++j) {
        npy_intp idxv = idata ? idata[j] : read_intp_1d(base, stride, (npy_intp)j);
        if (idxv < 0) idxv += src_len;
        if (idxv < 0 || idxv >= src_len) {
            for (int t = 0; t < j; ++t) { if (dst[t]) { PyDataMem_FREE(dst[t]); dst[t] = NULL; } }
            if (own_dst) { free(dst); *dst_labels_ptr = NULL; }
            return -1;
        }
        const char *s = src_labels[(int)idxv];
        size_t L = strlen(s ? s : "") + 1;
        dst[j] = (char *)PyDataMem_NEW(L);
        if (!dst[j]) {
            for (int t = 0; t < j; ++t) { if (dst[t]) { PyDataMem_FREE(dst[t]); dst[t] = NULL; } }
            if (own_dst) { free(dst); *dst_labels_ptr = NULL; }
            return -1;
        }
        memcpy(dst[j], s ? s : "", L);
    }
    *dst_labels_ptr = dst;
    return 0;
}
static NPY_INLINE int
fill_labels_from_int_indices_borrowed(char ***dst_labels_ptr, int out_len, char **src_labels, int src_len, PyArrayObject *ind)
{
    char **dst = *dst_labels_ptr;
    int own_dst = 0;
    if (!dst) {
        dst = (char **)calloc((size_t)out_len, sizeof(char *));
        if (!dst) return -1;
        own_dst = 1;
    }
    const char *base = PyArray_BYTES(ind);
    npy_intp stride = PyArray_STRIDES(ind)[0];
    const npy_intp *idata = PyArray_IS_C_CONTIGUOUS(ind) ? (const npy_intp *)PyArray_DATA(ind) : NULL;
    for (int j = 0; j < out_len; ++j) {
        npy_intp idxv = idata ? idata[j] : read_intp_1d(base, stride, (npy_intp)j);
        if (idxv < 0) idxv += src_len;
        if (idxv < 0 || idxv >= src_len) {
            if (own_dst) { free(dst); *dst_labels_ptr = NULL; }
            return -1;
        }
        dst[j] = src_labels[(int)idxv];
    }
    *dst_labels_ptr = dst;
    return 0;
}
/* Build labels from a Python list/tuple of integers by converting to an intp ndarray and reusing int-path */
/* Build labels from a 1D boolean mask along a source axis by converting to integer indices */
static NPY_INLINE int
build_labels_from_bool_mask(const LabelsBlock *lb, int src_axis, PyArrayObject *mask, int out_len, char ***out_labels_ptr)
{
    int ind_nd = PyArray_NDIM(mask);
    if (ind_nd != 1) { PyErr_SetString(PyExc_TypeError, "boolean index must be 1D for a single axis"); return -1; }
    const npy_intp *adims = PyArray_DIMS(mask);
    npy_intp n = adims[0];
    if ((int)n != lb->axis_len[src_axis]) { PyErr_SetString(PyExc_ValueError, "boolean mask length does not match axis"); return -1; }
    char **dst = *out_labels_ptr;
    int own_dst = 0;
    if (!dst) {
        dst = (char **)calloc((size_t)out_len, sizeof(char *));
        if (!dst) return -1;
        own_dst = 1;
    }
    const char *base = PyArray_BYTES(mask);
    npy_intp stride = PyArray_STRIDES(mask)[0];
    const npy_bool *mdata = PyArray_IS_C_CONTIGUOUS(mask) ? (const npy_bool *)PyArray_DATA(mask) : NULL;
    int pos = 0;
    for (npy_intp i = 0; i < n; ++i) {
        npy_bool v = mdata ? mdata[i] : read_bool_1d(base, stride, i);
        if (!v) continue;
        if (pos >= out_len) {
            for (int t = 0; t < pos; ++t) { if (dst[t]) { PyDataMem_FREE(dst[t]); dst[t] = NULL; } }
            if (own_dst) { free(dst); *out_labels_ptr = NULL; }
            PyErr_SetString(PyExc_ValueError, "boolean mask true count does not match result shape");
            return -1;
        }
        dst[pos] = lb->axis_labels[src_axis][(int)i];
        pos++;
    }
    if (pos != out_len) {
        if (own_dst) { free(dst); *out_labels_ptr = NULL; }
        PyErr_SetString(PyExc_ValueError, "boolean mask true count does not match result shape");
        return -1;
    }
    *out_labels_ptr = dst;
    return 0;
}
static NPY_INLINE int
build_labels_from_py_sequence(const LabelsBlock *lb, int src_axis, PyObject *seq_obj, int out_len, char ***out_labels_ptr)
{
    PyObject *seq = PySequence_Fast(seq_obj, "expected sequence");
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if ((int)n != out_len) { Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "sequence length mismatch for result shape"); return -1; }
    char **dst = *out_labels_ptr;
    int own_dst = 0;
    if (!dst) {
        dst = (char **)calloc((size_t)out_len, sizeof(char *));
        if (!dst) { Py_DECREF(seq); return -1; }
        own_dst = 1;
    }
    for (Py_ssize_t j = 0; j < n; ++j) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
        const char *s = NULL;
        PyObject *maybe_int = PyNumber_Index(it);
        if (maybe_int) {
            long idx = PyLong_AsLong(maybe_int);
            Py_DECREF(maybe_int);
            if (idx == -1 && PyErr_Occurred()) goto fail;
            if (idx < 0) idx += lb->axis_len[src_axis];
            if (idx < 0 || idx >= lb->axis_len[src_axis]) { PyErr_SetString(PyExc_IndexError, "index out of range"); goto fail; }
            s = lb->axis_labels[src_axis][(int)idx];
        } else {
            PyErr_Clear();
            if (PyUnicode_Check(it) || PyBytes_Check(it)) {
            Py_ssize_t pos = -1;
            if (labelsblock_find(lb, src_axis, it, &pos) != 0) { PyErr_SetString(PyExc_IndexError, "label not found in axis"); goto fail; }
            s = lb->axis_labels[src_axis][(int)pos];
            } else {
                PyErr_SetString(PyExc_TypeError, "sequence contains non-int/non-string");
                goto fail;
            }
        }
        dst[j] = (char *)s;
    }
    Py_DECREF(seq);
    *out_labels_ptr = dst;
    return 0;
fail:
    Py_DECREF(seq);
    if (own_dst && dst) { free(dst); *out_labels_ptr = NULL; }
    return -1;
}
static NPY_INLINE int
build_labels_from_int_sequence(const LabelsBlock *lb, int src_axis, PyObject *seq_obj, int out_len, char ***out_labels_ptr)
{
    PyObject *seq = PySequence_Fast(seq_obj, "expected sequence");
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if ((int)n != out_len) { Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "sequence length mismatch for result shape"); return -1; }
    char **dst = *out_labels_ptr;
    int own_dst = 0;
    if (!dst) {
        dst = (char **)calloc((size_t)out_len, sizeof(char *));
        if (!dst) { Py_DECREF(seq); return -1; }
        own_dst = 1;
    }
    for (Py_ssize_t j = 0; j < n; ++j) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, j);
        long idx = PyLong_AsLong(it);
        if (idx == -1 && PyErr_Occurred()) goto fail;
        if (idx < 0) idx += lb->axis_len[src_axis];
        if (idx < 0 || idx >= lb->axis_len[src_axis]) { PyErr_SetString(PyExc_IndexError, "index out of range"); goto fail; }
        dst[j] = lb->axis_labels[src_axis][(int)idx];
    }
    Py_DECREF(seq);
    *out_labels_ptr = dst;
    return 0;
fail:
    Py_DECREF(seq);
    if (own_dst && dst) { free(dst); *out_labels_ptr = NULL; }
    return -1;
}
/* Build row and column joined labels for 2D integer index array along a single source axis */
static NPY_INLINE int
build_rowcol_labels_from_2d_indices(const LabelsBlock *lb, int src_axis, PyArrayObject *ind, int R, int C,
                                    char ***out_rows_ptr, char ***out_cols_ptr, char **out_rows_slab, char **out_cols_slab)
{
    int ax_len = lb->axis_len[src_axis];
    const char *base = PyArray_BYTES(ind);
    const npy_intp *strides = PyArray_STRIDES(ind);
    int use_c = PyArray_IS_C_CONTIGUOUS(ind);
    int use_f = PyArray_IS_F_CONTIGUOUS(ind);
    const npy_intp *idata = (use_c || use_f) ? (const npy_intp *)PyArray_DATA(ind) : NULL;

    size_t *row_lens = NULL;
    size_t *col_lens = NULL;
    char **rows = NULL;
    char **cols = NULL;
    char *row_slab = NULL;
    char *col_slab = NULL;

    if (R > 0) {
        row_lens = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)R);
        if (!row_lens) return -1;
        rows = (char **)calloc((size_t)R, sizeof(char *));
        if (!rows) { PyMem_Free(row_lens); return -1; }
    }
    if (C > 0) {
        col_lens = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)C);
        if (!col_lens) { PyMem_Free(row_lens); free(rows); return -1; }
        cols = (char **)calloc((size_t)C, sizeof(char *));
        if (!cols) { PyMem_Free(col_lens); PyMem_Free(row_lens); free(rows); return -1; }
    }

    /* Row lengths and slab */
    size_t total_rows = 0;
    for (int r = 0; r < R; ++r) {
        size_t tot = 1;
        for (int c = 0; c < C; ++c) {
            npy_intp idxv;
            if (use_c) idxv = idata[(npy_intp)r * (npy_intp)C + c];
            else if (use_f) idxv = idata[(npy_intp)c * (npy_intp)R + r];
            else idxv = *(const npy_intp *)(base + (npy_intp)r * strides[0] + (npy_intp)c * strides[1]);
            if (idxv < 0) idxv += ax_len;
            if (idxv < 0 || idxv >= ax_len) { PyErr_SetString(PyExc_IndexError, "index out of range"); goto fail; }
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            tot += strlen(s) + (c ? 1 : 0);
        }
        if (row_lens) row_lens[r] = tot;
        total_rows += tot;
    }
    if (total_rows > 0) {
        row_slab = (char *)PyDataMem_NEW(total_rows);
        if (!row_slab) goto fail;
    }
    if (rows) {
        char *cursor = row_slab;
        for (int r = 0; r < R; ++r) {
            char *buf = cursor;
            cursor += row_lens[r];
            size_t posw = 0;
            for (int c = 0; c < C; ++c) {
                npy_intp idxv;
                if (use_c) idxv = idata[(npy_intp)r * (npy_intp)C + c];
                else if (use_f) idxv = idata[(npy_intp)c * (npy_intp)R + r];
                else idxv = *(const npy_intp *)(base + (npy_intp)r * strides[0] + (npy_intp)c * strides[1]);
                if (idxv < 0) idxv += ax_len;
                if (idxv < 0 || idxv >= ax_len) { PyErr_SetString(PyExc_IndexError, "index out of range"); goto fail; }
                const char *s = lb->axis_labels[src_axis][(int)idxv];
                if (c) buf[posw++] = '-';
                size_t Ls = strlen(s);
                memcpy(buf + posw, s, Ls);
                posw += Ls;
            }
            buf[posw] = '\0';
            rows[r] = buf;
        }
    }

    /* Column lengths and slab */
    size_t total_cols = 0;
    for (int c = 0; c < C; ++c) {
        size_t tot = 1;
        for (int r = 0; r < R; ++r) {
            npy_intp idxv;
            if (use_c) idxv = idata[(npy_intp)r * (npy_intp)C + c];
            else if (use_f) idxv = idata[(npy_intp)c * (npy_intp)R + r];
            else idxv = *(const npy_intp *)(base + (npy_intp)r * strides[0] + (npy_intp)c * strides[1]);
            if (idxv < 0) idxv += ax_len;
            if (idxv < 0 || idxv >= ax_len) { PyErr_SetString(PyExc_IndexError, "index out of range"); goto fail; }
            const char *s = lb->axis_labels[src_axis][(int)idxv];
            tot += strlen(s) + (r ? 1 : 0);
        }
        if (col_lens) col_lens[c] = tot;
        total_cols += tot;
    }
    if (total_cols > 0) {
        col_slab = (char *)PyDataMem_NEW(total_cols);
        if (!col_slab) goto fail;
    }
    if (cols) {
        char *cursor = col_slab;
        for (int c = 0; c < C; ++c) {
            char *buf = cursor;
            cursor += col_lens[c];
            size_t posw = 0;
            for (int r = 0; r < R; ++r) {
                npy_intp idxv;
                if (use_c) idxv = idata[(npy_intp)r * (npy_intp)C + c];
                else if (use_f) idxv = idata[(npy_intp)c * (npy_intp)R + r];
                else idxv = *(const npy_intp *)(base + (npy_intp)r * strides[0] + (npy_intp)c * strides[1]);
                if (idxv < 0) idxv += ax_len;
                if (idxv < 0 || idxv >= ax_len) { PyErr_SetString(PyExc_IndexError, "index out of range"); goto fail; }
                const char *s = lb->axis_labels[src_axis][(int)idxv];
                if (r) buf[posw++] = '-';
                size_t Ls = strlen(s);
                memcpy(buf + posw, s, Ls);
                posw += Ls;
            }
            buf[posw] = '\0';
            cols[c] = buf;
        }
    }

    *out_rows_ptr = rows;
    *out_cols_ptr = cols;
    if (out_rows_slab) *out_rows_slab = row_slab;
    if (out_cols_slab) *out_cols_slab = col_slab;
    PyMem_Free(col_lens);
    PyMem_Free(row_lens);
    return 0;
fail:
    if (row_slab) PyDataMem_FREE(row_slab);
    if (col_slab) PyDataMem_FREE(col_slab);
    if (rows) free(rows);
    if (cols) free(cols);
    PyMem_Free(col_lens);
    PyMem_Free(row_lens);
    return -1;
}
/* Build labels for one axis p of a k-D integer index array by joining across other dims */
static NPY_INLINE int
build_kd_labels_for_axis(const LabelsBlock *lb, int src_axis, PyArrayObject *ind, int knd, int p, const npy_intp *adims, char ***out_axis_labels_ptr, char **out_slab_ptr)
{
    int len_p = (int)adims[p];
    int ax_len = lb->axis_len[src_axis];
    char **axis_labels = *out_axis_labels_ptr;
    int own_axis = 0;
    if (!axis_labels) {
        axis_labels = (char **)calloc((size_t)len_p, sizeof(char *));
        if (!axis_labels) return -1;
        own_axis = 1;
    }
    const char *base = PyArray_BYTES(ind);
    const npy_intp *strides = PyArray_STRIDES(ind);

    size_t *lens = (size_t *)PyMem_Malloc(sizeof(size_t) * (size_t)len_p);
    if (!lens) { if (own_axis) free(axis_labels); return -1; }
    size_t total = 0;

    for (int vp = 0; vp < len_p; ++vp) {
        /* First pass: total length */
        npy_intp idxk[NPY_MAXDIMS];
        for (int q = 0; q < knd; ++q) idxk[q] = 0;
        idxk[p] = vp;
        size_t tot = 1; int first = 1;
        while (1) {
            npy_intp idxv = read_intp_nd(base, strides, idxk, knd);
            if (idxv < 0) idxv += ax_len;
            if (idxv < 0 || idxv >= ax_len) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                PyMem_Free(lens);
                if (own_axis) { free(axis_labels); }
                return -1;
            }
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
        lens[vp] = tot;
        total += tot;
    }

    char *slab = (char *)PyDataMem_NEW(total);
    if (!slab) { PyMem_Free(lens); if (own_axis) free(axis_labels); return -1; }
    if (out_slab_ptr) *out_slab_ptr = slab;

    char *cursor = slab;
    for (int vp = 0; vp < len_p; ++vp) {
        char *buf = cursor;
        cursor += lens[vp];
        size_t posw = 0; int first = 1;
        npy_intp idxk[NPY_MAXDIMS];
        for (int q = 0; q < knd; ++q) idxk[q] = 0;
        idxk[p] = vp;
        while (1) {
            npy_intp idxv = read_intp_nd(base, strides, idxk, knd);
            if (idxv < 0) idxv += ax_len;
            if (idxv < 0 || idxv >= ax_len) {
                PyMem_Free(lens);
                if (own_axis) { free(axis_labels); }
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return -1;
            }
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
    PyMem_Free(lens);
    *out_axis_labels_ptr = axis_labels;
    return 0;
}
LabelsBlock *labelsblock_from_py(PyObject *labels_in, PyArrayObject *arr)
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
    lb->axis_str_borrowed = (unsigned char *)calloc(ndim, sizeof(unsigned char));
    lb->borrowed_owner = NULL;
    if (!lb->axis_len || !lb->axis_labels || !lb->axis_hash || !lb->axis_borrowed || !lb->axis_str_borrowed) { Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
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

PyObject *labelsblock_to_py_tuple(const LabelsBlock *lb)
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


PyObject *join_axis_labels(const LabelsBlock *lb, int ax, const char *joiner)
{
    int n = lb->axis_len[ax];
    if (n <= 0) return PyUnicode_FromString("");
    size_t joiner_len = strlen(joiner);
    size_t total = 1;
    for (int i = 0; i < n; ++i) {
        const char *s = lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "";
        total += strlen(s);
        if (i > 0) total += joiner_len;
    }
    char *buf = (char *)PyMem_Malloc(total);
    if (!buf) { PyErr_NoMemory(); return NULL; }
    char *p = buf;
    for (int i = 0; i < n; ++i) {
        const char *s = lb->axis_labels[ax][i] ? lb->axis_labels[ax][i] : "";
        size_t L = strlen(s);
        if (i > 0 && joiner_len) { memcpy(p, joiner, joiner_len); p += joiner_len; }
        if (L) { memcpy(p, s, L); p += L; }
    }
    *p = '\0';
    PyObject *out = PyUnicode_FromString(buf);
    PyMem_Free(buf);
    return out;
}

/* ---------- index/label helpers ---------- */
PyArrayObject *map_label_index_array_to_int(const LabelsBlock *lb, int axis, PyArrayObject *arrk)
{
    int ind_nd = PyArray_NDIM(arrk);
    const npy_intp *adims = PyArray_DIMS(arrk);
    PyArrayObject *ind = (PyArrayObject *)PyArray_SimpleNew(ind_nd, adims, NPY_INTP);
    if (!ind) return NULL;
    int kind = PyArray_TYPE(arrk);
    npy_intp itemsize = PyArray_ITEMSIZE(arrk);
    PyArrayIterObject *it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arrk);
    PyArrayIterObject *itout = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ind);
    if (!it || !itout) {
        Py_XDECREF(it);
        Py_XDECREF(itout);
        Py_DECREF(ind);
        return NULL;
    }
    while (it->index < it->size) {
        PyObject *item = NULL;
        char *ptr = it->dataptr;
        if (kind == NPY_OBJECT) {
            item = *(PyObject **)ptr;
            if (!item) { PyErr_SetString(PyExc_ValueError, "object index array contains NULL"); goto fail; }
        } else if (kind == NPY_UNICODE) {
            item = unicode_from_raw(ptr, itemsize);
            if (!item) goto fail;
        } else { /* NPY_STRING */
            item = bytes_from_raw(ptr, itemsize);
            if (!item) goto fail;
        }
        Py_ssize_t pos = -1;
        int rc = labelsblock_find(lb, axis, item, &pos);
        if (kind != NPY_OBJECT) { Py_DECREF(item); item = NULL; }
        if (rc != 0) goto not_found;
        npy_intp posi = (npy_intp)pos;
        memcpy(itout->dataptr, &posi, sizeof(npy_intp));
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(itout);
    }
    Py_DECREF(it);
    Py_DECREF(itout);
    return ind;
not_found:
    Py_DECREF(it);
    Py_DECREF(itout);
    Py_DECREF(ind);
    return NULL;
fail:
    Py_DECREF(it);
    Py_DECREF(itout);
    Py_DECREF(ind);
    return NULL;
}

/* Build sliced labels block for a view result, given parent labels and converted key */
LabelsBlock *labelsblock_slice(const LabelsBlock *lb, PyObject *ck, PyArrayObject *result)
{
    int res_ndim = PyArray_NDIM(result);
    const npy_intp *res_shape = PyArray_SHAPE(result);
    LabelsBlock *out = labelsblock_alloc(res_ndim);
    if (!out) return NULL;

    int src_axis = 0;
    int dst_axis = 0;
    Py_ssize_t nkeys = PyTuple_Check(ck) ? PyTuple_GET_SIZE(ck) : 0;
    int borrowed_any = 0;

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
        if (PySlice_Check(k)) {
            Py_ssize_t slen = lb->axis_len[src_axis];
            npy_intp start, stop, step, length;
            if (PySlice_GetIndicesEx(k, slen, &start, &stop, &step, &length) < 0) { labelsblock_decref(out); return NULL; }
            int out_len = (int)res_shape[dst_axis];
            out->axis_len[dst_axis] = out_len;
            if (step == 1 && start == 0 && (int)length == lb->axis_len[src_axis]) {
                out->axis_len[dst_axis] = lb->axis_len[src_axis];
                out->axis_labels[dst_axis] = lb->axis_labels[src_axis];
                out->axis_borrowed[dst_axis] = 1;
                borrowed_any = 1;
                if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], out->axis_len[dst_axis]) < 0) { labelsblock_decref(out); return NULL; }
                src_axis += 1;
                dst_axis += 1;
                continue;
            }
            if (!out->axis_labels[dst_axis]) {
                out->axis_labels[dst_axis] = (char **)calloc(out_len, sizeof(char *));
                if (!out->axis_labels[dst_axis]) { labelsblock_decref(out); return NULL; }
            }
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
                        const char *ibase = PyArray_BYTES(inds_arr[p]);
                        const npy_intp *istrides = PyArray_STRIDES(inds_arr[p]);
                        out->axis_len[dst_axis + p] = Lp;
                        out->axis_labels[dst_axis + p] = (char **)calloc((size_t)Lp, sizeof(char *));
                        if (!out->axis_labels[dst_axis + p]) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                        for (int v = 0; v < Lp && (dst_axis + p) < res_ndim && v < (int)res_shape[dst_axis + p]; ++v) {
                            npy_intp idxix[NPY_MAXDIMS];
                            for (int q = 0; q < m; ++q) idxix[q] = 0;
                            idxix[p] = v;
                            npy_intp idxv = read_intp_nd(ibase, istrides, idxix, m);
                            int axlen = (src_axis + p < lb->ndim) ? lb->axis_len[src_axis + p] : 0;
                            if (axlen <= 0) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                            if (idxv < 0) idxv += axlen;
                            if (idxv < 0 || idxv >= axlen) { for (int t = 0; t < m; ++t) Py_DECREF(inds_arr[t]); Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                            const char *s = lb->axis_labels[src_axis + p][(int)idxv];
                            size_t Ls = strlen(s) + 1;
                            out->axis_labels[dst_axis + p][v] = (char *)PyDataMem_NEW(Ls);
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
            int out_len = (int)res_shape[dst_axis];
            out->axis_len[dst_axis] = out_len;
            int arr_case = 0;
            if (kind == NPY_BOOL) arr_case = 1;
            else if (PyTypeNum_ISINTEGER(kind)) arr_case = 2;
            else if (kind == NPY_UNICODE || kind == NPY_STRING || kind == NPY_OBJECT) arr_case = 3;
            else arr_case = 4;

            switch (arr_case) {
                case 1: /* boolean mask */
                    if (build_labels_from_bool_mask(lb, src_axis, arrk, out_len, &out->axis_labels[dst_axis]) < 0) { Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                    out->axis_str_borrowed[dst_axis] = 1;
                    borrowed_any = 1;
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
                    if (fill_labels_from_int_indices_borrowed(&out->axis_labels[dst_axis], (n < out_len ? n : out_len), lb->axis_labels[src_axis], lb->axis_len[src_axis], ind) < 0) { Py_DECREF(ind); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                out->axis_str_borrowed[dst_axis] = 1;
                borrowed_any = 1;
                Py_DECREF(ind);
                    break;
                }
                case 2: {
                int R = (int)adims[0], C = (int)adims[1];
                    if (dst_axis >= res_ndim || dst_axis + 1 >= res_ndim) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                out->axis_len[dst_axis] = R;
                out->axis_len[dst_axis + 1] = C;
                char **rows = NULL, **cols = NULL;
                    if (build_rowcol_labels_from_2d_indices(lb, src_axis, ind, R, C, &rows, &cols,
                                                           &out->axis_slab[dst_axis], &out->axis_slab[dst_axis + 1]) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
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
                }
                for (int p = 0; p < knd; ++p) {
                        if (build_kd_labels_for_axis(lb, src_axis, ind, knd, p, adims, &out->axis_labels[dst_axis + p], &out->axis_slab[dst_axis + p]) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    if (axhash_build(&out->axis_hash[dst_axis + p], (const char **)out->axis_labels[dst_axis + p], (int)adims[p]) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                }
                Py_DECREF(ind);
                src_axis += 1;
                dst_axis += knd;
                continue;
            }
            }
        } else if (PyList_Check(k) || PyTuple_Check(k)) {
            /* Get the actual sequence length */
            Py_ssize_t seq_len = PySequence_Size(k);
            if (seq_len < 0) { labelsblock_decref(out); return NULL; }
            int actual_out_len = (int)seq_len;
            /* If sequence length doesn't match out_len, reallocate */
            int out_len = (int)res_shape[dst_axis];
            if (actual_out_len != out_len) {
                out->axis_len[dst_axis] = actual_out_len;
                out_len = actual_out_len;
            } else {
                out->axis_len[dst_axis] = out_len;
            }
            if (build_labels_from_py_sequence(lb, src_axis, k, out_len, &out->axis_labels[dst_axis]) < 0) { labelsblock_decref(out); return NULL; }
            out->axis_str_borrowed[dst_axis] = 1;
            borrowed_any = 1;
            /* build hash below */
        } else {
            /* fallback: copy entire axis truncated to out_len */
            int out_len = (int)res_shape[dst_axis];
            out->axis_len[dst_axis] = out_len;
            if (!out->axis_labels[dst_axis]) {
                out->axis_labels[dst_axis] = (char **)calloc(out_len, sizeof(char *));
                if (!out->axis_labels[dst_axis]) { labelsblock_decref(out); return NULL; }
            }
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
        {
            int out_len = out->axis_len[dst_axis];
            if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], out_len) < 0) { labelsblock_decref(out); return NULL; }
        }
        src_axis += 1;
        dst_axis += 1;
    }
    /* fill remaining axes as full copies if any */
    while (dst_axis < res_ndim && src_axis < lb->ndim) {
        int out_len2 = (int)res_shape[dst_axis];
        out->axis_len[dst_axis] = out_len2;
        out->axis_labels[dst_axis] = lb->axis_labels[src_axis];
        out->axis_borrowed[dst_axis] = 1;
        borrowed_any = 1;
        if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], out_len2) < 0) { labelsblock_decref(out); return NULL; }
        src_axis += 1;
        dst_axis += 1;
    }
    if (borrowed_any) { out->borrowed_owner = (LabelsBlock *)lb; out->borrowed_owner->refcount++; }
    return out;
}



static NPY_INLINE uint64_t
fnv1a64_len(const char *data, size_t len)
{
    const uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint64_t)(unsigned char)data[i];
        hash *= FNV_PRIME;
    }
    return hash ? hash : 1469598103934665603ULL;
}

static int
strset_rehash(StrSet *set, size_t newcap)
{
    StrSet tmp = {0};
    tmp.cap = next_pow2(newcap);
    tmp.used = 0;
    tmp.owns_keys = set->owns_keys;
    tmp.keys = (char **)calloc(tmp.cap, sizeof(char *));
    tmp.lens = (size_t *)calloc(tmp.cap, sizeof(size_t));
    tmp.hashes = (uint64_t *)calloc(tmp.cap, sizeof(uint64_t));
    if (!tmp.keys || !tmp.lens || !tmp.hashes) {
        free(tmp.keys); free(tmp.lens); free(tmp.hashes);
        return -1;
    }
    for (size_t i = 0; i < set->cap; ++i) {
        if (!set->keys[i]) continue;
        size_t m = tmp.cap - 1;
        size_t pos = (size_t)set->hashes[i] & m;
        while (tmp.keys[pos] != NULL) pos = (pos + 1) & m;
        tmp.keys[pos] = set->keys[i];
        tmp.lens[pos] = set->lens[i];
        tmp.hashes[pos] = set->hashes[i];
        tmp.used++;
    }
    free(set->keys); free(set->lens); free(set->hashes);
    *set = tmp;
    return 0;
}

static int
strset_init(StrSet *set, size_t cap_hint, int owns_keys)
{
    size_t cap = next_pow2(cap_hint > 0 ? cap_hint : 1);
    set->cap = cap;
    set->used = 0;
    set->owns_keys = owns_keys;
    set->keys = (char **)calloc(cap, sizeof(char *));
    set->lens = (size_t *)calloc(cap, sizeof(size_t));
    set->hashes = (uint64_t *)calloc(cap, sizeof(uint64_t));
    if (!set->keys || !set->lens || !set->hashes) {
        free(set->keys); free(set->lens); free(set->hashes);
        set->keys = NULL; set->lens = NULL; set->hashes = NULL;
        return -1;
    }
    return 0;
}

void strset_free(StrSet *set)
{
    if (!set) return;
    if (set->owns_keys && set->keys) {
        for (size_t i = 0; i < set->cap; ++i) {
            if (set->keys[i]) PyDataMem_FREE(set->keys[i]);
        }
    }
    free(set->keys); free(set->lens); free(set->hashes);
    set->keys = NULL; set->lens = NULL; set->hashes = NULL;
    set->cap = 0; set->used = 0;
}

static int
strset_contains(const StrSet *set, const char *s, size_t len, uint64_t h)
{
    if (!set || set->cap == 0) return 0;
    size_t m = set->cap - 1;
    size_t pos = (size_t)h & m;
    while (set->keys[pos] != NULL) {
        if (set->hashes[pos] == h && set->lens[pos] == len && memcmp(set->keys[pos], s, len) == 0) return 1;
        pos = (pos + 1) & m;
    }
    return 0;
}

int strset_add(StrSet *set, const char *s, size_t len)
{
    if (set->used * 2 >= set->cap) {
        if (strset_rehash(set, set->cap * 2) < 0) return -1;
    }
    uint64_t h = fnv1a64_len(s, len);
    size_t m = set->cap - 1;
    size_t pos = (size_t)h & m;
    while (set->keys[pos] != NULL) {
        if (set->hashes[pos] == h && set->lens[pos] == len && memcmp(set->keys[pos], s, len) == 0) return 1;
        pos = (pos + 1) & m;
    }
    char *store = NULL;
    if (set->owns_keys) {
        store = (char *)PyDataMem_NEW(len + 1);
        if (!store) return -1;
        memcpy(store, s, len);
        store[len] = '\0';
    } else {
        store = (char *)s;
    }
    set->keys[pos] = store;
    set->lens[pos] = len;
    set->hashes[pos] = h;
    set->used++;
    return 0;
}

StrSet *strset_new(size_t cap_hint, int owns_keys)
{
    StrSet *set = (StrSet *)PyMem_Malloc(sizeof(StrSet));
    if (!set) return NULL;
    if (strset_init(set, cap_hint, owns_keys) < 0) { PyMem_Free(set); return NULL; }
    return set;
}

StrSet *strset_from_tokens(Token *tokens, int ntok)
{
    StrSet *set = strset_new((size_t)ntok * 2 + 1, 1);
    if (!set) return NULL;
    for (int i = 0; i < ntok; ++i) {
        if (strset_add(set, tokens[i].ptr, tokens[i].len) < 0) { strset_free(set); PyMem_Free(set); return NULL; }
    }
    return set;
}

StrSet *strset_intersect_tokens(const StrSet *set, Token *tokens, int ntok)
{
    StrSet *out = strset_new((size_t)ntok * 2 + 1, 1);
    if (!out) return NULL;
    for (int i = 0; i < ntok; ++i) {
        uint64_t h = fnv1a64_len(tokens[i].ptr, tokens[i].len);
        if (strset_contains(set, tokens[i].ptr, tokens[i].len, h)) {
            if (strset_add(out, tokens[i].ptr, tokens[i].len) < 0) { strset_free(out); PyMem_Free(out); return NULL; }
        }
    }
    return out;
}

size_t strset_join_len(const StrSet *set, size_t delim_len)
{
    if (!set || set->used == 0) return 0;
    size_t sum = 0;
    for (size_t i = 0; i < set->cap; ++i) {
        if (set->keys[i]) sum += set->lens[i];
    }
    if (delim_len && set->used > 1) sum += delim_len * (set->used - 1);
    return sum;
}

char **strset_collect_keys(const StrSet *set, int *out_n, char **stack_buf, int stack_cap)
{
    int n = set ? (int)set->used : 0;
    *out_n = n;
    if (n == 0) return NULL;
    char **arr = (n <= stack_cap) ? stack_buf : (char **)PyMem_Malloc(sizeof(char *) * (size_t)n);
    if (!arr) return NULL;
    int w = 0;
    for (size_t i = 0; i < set->cap; ++i) {
        if (set->keys[i]) arr[w++] = set->keys[i];
    }
    return arr;
}

int
cmp_cstr(const void *a, const void *b)
{
    const char *sa = *(const char * const *)a;
    const char *sb = *(const char * const *)b;
    return strcmp(sa, sb);
}

int split_tokens(const char *s, const char *delim, size_t delim_len,
             Token **out_tokens, int *out_n,
             Token *stack_buf, int stack_cap)
{
    if (delim_len == 0) {
        *out_tokens = stack_buf;
        stack_buf[0].ptr = s;
        stack_buf[0].len = strlen(s);
        *out_n = 1;
        return 0;
    }
    int count = 0;
    const char *p = s;
    const char *q = NULL;
    while ((q = strstr(p, delim)) != NULL) {
        count++;
        p = q + delim_len;
    }
    int n_tokens = count + 1;
    Token *buf = (n_tokens <= stack_cap) ? stack_buf : (Token *)PyMem_Malloc(sizeof(Token) * (size_t)n_tokens);
    if (!buf) return -1;
    p = s;
    int idx = 0;
    while ((q = strstr(p, delim)) != NULL) {
        buf[idx].ptr = p;
        buf[idx].len = (size_t)(q - p);
        idx++;
        p = q + delim_len;
    }
    buf[idx].ptr = p;
    buf[idx].len = strlen(p);
    *out_tokens = buf;
    *out_n = n_tokens;
    return 0;
}

void idx_increment_order(npy_intp *idx, const npy_intp *dims, int nd, int order_c)
{
    if (order_c) {
        for (int ax = nd - 1; ax >= 0; --ax) {
            idx[ax]++;
            if (idx[ax] < dims[ax]) break;
            idx[ax] = 0;
        }
    } else {
        for (int ax = 0; ax < nd; ++ax) {
            idx[ax]++;
            if (idx[ax] < dims[ax]) break;
            idx[ax] = 0;
        }
    }
}

typedef struct {
    size_t cap;
    size_t used;
    char **keys;
    size_t *lens;
    uint64_t *hashes;
    int *count;
    int *seen;
} LabelCountMap;

static int
labelcount_init(LabelCountMap *m, size_t cap_hint)
{
    size_t cap = next_pow2(cap_hint > 0 ? cap_hint : 1);
    m->cap = cap;
    m->used = 0;
    m->keys = (char **)calloc(cap, sizeof(char *));
    m->lens = (size_t *)calloc(cap, sizeof(size_t));
    m->hashes = (uint64_t *)calloc(cap, sizeof(uint64_t));
    m->count = (int *)calloc(cap, sizeof(int));
    m->seen = (int *)calloc(cap, sizeof(int));
    if (!m->keys || !m->lens || !m->hashes || !m->count || !m->seen) {
        free(m->keys); free(m->lens); free(m->hashes); free(m->count); free(m->seen);
        m->keys = NULL; m->lens = NULL; m->hashes = NULL; m->count = NULL; m->seen = NULL;
        return -1;
    }
    return 0;
}

static void
labelcount_free(LabelCountMap *m)
{
    free(m->keys); free(m->lens); free(m->hashes); free(m->count); free(m->seen);
    m->keys = NULL; m->lens = NULL; m->hashes = NULL; m->count = NULL; m->seen = NULL;
    m->cap = 0; m->used = 0;
}

static int
labelcount_add(LabelCountMap *m, const char *s, size_t len)
{
    uint64_t h = fnv1a64_len(s, len);
    size_t msk = m->cap - 1;
    size_t pos = (size_t)h & msk;
    while (m->keys[pos] != NULL) {
        if (m->hashes[pos] == h && m->lens[pos] == len && memcmp(m->keys[pos], s, len) == 0) {
            m->count[pos]++;
            return (int)pos;
        }
        pos = (pos + 1) & msk;
    }
    m->keys[pos] = (char *)s;
    m->lens[pos] = len;
    m->hashes[pos] = h;
    m->count[pos] = 1;
    m->seen[pos] = 0;
    m->used++;
    return (int)pos;
}

static int
labelcount_find(const LabelCountMap *m, const char *s, size_t len)
{
    uint64_t h = fnv1a64_len(s, len);
    size_t msk = m->cap - 1;
    size_t pos = (size_t)h & msk;
    while (m->keys[pos] != NULL) {
        if (m->hashes[pos] == h && m->lens[pos] == len && memcmp(m->keys[pos], s, len) == 0) return (int)pos;
        pos = (pos + 1) & msk;
    }
    return -1;
}

int make_array_unique_c(char **labels, int n, const char *delim, char **slab_io)
{
    if (n <= 1) return 0;
    size_t delim_len = (delim ? strlen(delim) : 0);
    LabelCountMap map;
    if (labelcount_init(&map, (size_t)n * 2 + 1) < 0) return -1;

    int any_dup = 0;
    for (int i = 0; i < n; ++i) {
        size_t L = strlen(labels[i]);
        int pos = labelcount_add(&map, labels[i], L);
        if (pos < 0) { labelcount_free(&map); return -1; }
    }
    for (size_t i = 0; i < map.cap; ++i) {
        if (map.keys[i] && map.count[i] > 1) { any_dup = 1; break; }
    }
    if (!any_dup) { labelcount_free(&map); return 0; }

    size_t total = 0;
    for (int i = 0; i < n; ++i) {
        size_t L = strlen(labels[i]);
        int pos = labelcount_find(&map, labels[i], L);
        int cnt = (pos >= 0) ? map.count[pos] : 1;
        if (cnt > 1) {
            char tmp[32];
            int m = snprintf(tmp, sizeof(tmp), "%d", map.seen[pos]);
            if (m < 0) { labelcount_free(&map); return -1; }
            size_t add = (size_t)m;
            if (SIZE_MAX - total < L + delim_len + add + 1) { labelcount_free(&map); return -1; }
            total += L + delim_len + add + 1;
            map.seen[pos]++; /* pre-count digits for sizing */
        } else {
            if (SIZE_MAX - total < L + 1) { labelcount_free(&map); return -1; }
            total += L + 1;
        }
    }

    for (size_t i = 0; i < map.cap; ++i) if (map.keys[i]) map.seen[i] = 0;

    char *new_slab = (char *)PyDataMem_NEW(total);
    if (!new_slab) { labelcount_free(&map); return -1; }
    char *cursor = new_slab;

    for (int i = 0; i < n; ++i) {
        size_t L = strlen(labels[i]);
        int pos = labelcount_find(&map, labels[i], L);
        int cnt = (pos >= 0) ? map.count[pos] : 1;
        char *dst = cursor;
        if (cnt > 1) {
            int idx = map.seen[pos]++;
            memcpy(dst, labels[i], L);
            dst += L;
            if (delim_len) { memcpy(dst, delim, delim_len); dst += delim_len; }
            char tmp[32];
            int m = snprintf(tmp, sizeof(tmp), "%d", idx);
            if (m < 0) { PyDataMem_FREE(new_slab); labelcount_free(&map); return -1; }
            memcpy(dst, tmp, (size_t)m); dst += (size_t)m;
        } else {
            memcpy(dst, labels[i], L);
            dst += L;
        }
        *dst = '\0';
        labels[i] = cursor;
        cursor = dst + 1;
    }

    if (*slab_io) PyDataMem_FREE(*slab_io);
    *slab_io = new_slab;
    labelcount_free(&map);
    return 0;
}
int build_combined_labels_range(LabelsBlock *lb, int start, int end, const char *delim,
                            char ***out_labels, char **out_slab, int *out_len)
{
    int k = end - start + 1;
    if (k <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid axis range for label combine");
        return -1;
    }
    size_t delim_len = (delim && delim[0]) ? strlen(delim) : 0;

    npy_intp total = 1;
    for (int ax = start; ax <= end; ++ax) {
        int n = lb->axis_len[ax];
        if (n < 0) {
            PyErr_SetString(PyExc_ValueError, "invalid axis length");
            return -1;
        }
        if (n == 0) { total = 0; break; }
        if (total > NPY_MAX_INTP / n) {
            PyErr_SetString(PyExc_OverflowError, "label combine overflow");
            return -1;
        }
        total *= n;
    }

    char **labels = NULL;
    if (total > 0) {
        labels = (char **)calloc((size_t)total, sizeof(char *));
        if (!labels) { PyErr_NoMemory(); return -1; }
    }

    size_t total_bytes = 0;
    if (total > 0) {
        int *idx = (int *)calloc((size_t)k, sizeof(int));
        if (!idx) { free(labels); PyErr_NoMemory(); return -1; }
        for (npy_intp t = 0; t < total; ++t) {
            size_t len = 0; int parts = 0;
            for (int a = 0; a < k; ++a) {
                const char *s = lb->axis_labels[start + a][idx[a]] ? lb->axis_labels[start + a][idx[a]] : "";
                size_t L = strlen(s);
                if (L > 0) { len += L; parts++; }
            }
            if (parts > 1 && delim_len) len += delim_len * (size_t)(parts - 1);
            if (SIZE_MAX - total_bytes < len + 1) { free(idx); free(labels); PyErr_NoMemory(); return -1; }
            total_bytes += len + 1;
            for (int a = k - 1; a >= 0; --a) {
                idx[a]++;
                if (idx[a] < lb->axis_len[start + a]) break;
                idx[a] = 0;
            }
        }
        free(idx);
    }

    char *slab = NULL;
    if (total_bytes > 0) {
        slab = (char *)PyDataMem_NEW(total_bytes);
        if (!slab) { free(labels); PyErr_NoMemory(); return -1; }
    }

    if (total > 0) {
        int *idx = (int *)calloc((size_t)k, sizeof(int));
        if (!idx) { if (slab) PyDataMem_FREE(slab); free(labels); PyErr_NoMemory(); return -1; }
        char *cursor = slab;
        for (npy_intp t = 0; t < total; ++t) {
            char *dst = cursor;
            int parts_written = 0;
            for (int a = 0; a < k; ++a) {
                const char *s = lb->axis_labels[start + a][idx[a]] ? lb->axis_labels[start + a][idx[a]] : "";
                size_t L = strlen(s);
                if (L == 0) continue;
                if (parts_written > 0 && delim_len) { memcpy(dst, delim, delim_len); dst += delim_len; }
                memcpy(dst, s, L); dst += L;
                parts_written++;
            }
            *dst = '\0';
            labels[t] = cursor;
            cursor = dst + 1;
            for (int a = k - 1; a >= 0; --a) {
                idx[a]++;
                if (idx[a] < lb->axis_len[start + a]) break;
                idx[a] = 0;
            }
        }
        free(idx);
    }

    *out_labels = labels;
    *out_slab = slab;
    *out_len = (int)total;
    return 0;
}

