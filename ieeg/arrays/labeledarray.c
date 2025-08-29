/* (moved down after includes/typedefs) */
/* ndarray subtype: LabeledArray with C-level labels storage (stride-safe) */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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
    PyObject *labels_cache; /* cached tuple-of-ndarray(unicode, 1d) per axis */
    char *delimiter; /* default join for label combinations */
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

/* ---------- small C helpers for speed/readability ---------- */
NPY_INLINE char *
cstr_dup(const char *s)
{
    const char *src = s ? s : "";
    size_t L = strlen(src) + 1;
    char *dst = (char *)malloc(L);
    if (!dst) return NULL;
    memcpy(dst, src, L);
    return dst;
}

NPY_INLINE int
axis_labels_copy(char **dst, char **src, int n)
{
    for (int i = 0; i < n; ++i) {
        dst[i] = cstr_dup(src[i]);
        if (!dst[i]) return -1;
    }
    return 0;
}

NPY_INLINE size_t
join_len2(const char *a, const char *sep, const char *b)
{
    const char *sa = a ? a : "";
    const char *sb = b ? b : "";
    return strlen(sa) + strlen(sep) + strlen(sb) + 1;
}

NPY_INLINE void
join_copy2(char *dst, const char *a, const char *sep, const char *b)
{
    const char *sa = a ? a : "";
    const char *sb = b ? b : "";
    size_t pa = strlen(sa);
    memcpy(dst, sa, pa);
    size_t ps = strlen(sep);
    memcpy(dst + pa, sep, ps);
    size_t pb = strlen(sb);
    memcpy(dst + pa + ps, sb, pb);
    dst[pa + ps + pb] = '\0';
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
            PyObject *s = PyObject_Str(it);
            if (!s) { Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            const char *c = PyUnicode_AsUTF8(s);
            if (!c) { Py_DECREF(s); Py_DECREF(axis_seq); Py_DECREF(labels_in); labelsblock_decref(lb); return NULL; }
            size_t L = strlen(c) + 1;
            lb->axis_labels[ax][(int)i] = (char *)malloc(L);
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
        } else if (PyArray_Check(k)) {
            /* Integer or Unicode ndarray indices. Support 1D and 2D.
               1D: replace source axis with length n
               2D: insert two axes with lengths (R, C), labels are joined strings per row/col. */
            PyArrayObject *arrk = (PyArrayObject *)PyArray_FromAny(k, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if (!arrk) { labelsblock_decref(out); return NULL; }
            int kind = PyArray_TYPE(arrk);
            PyArrayObject *ind = NULL;
            if (kind == NPY_BOOL) {
                /* 1D boolean mask along this axis */
                int ind_nd = PyArray_NDIM(arrk);
                if (ind_nd != 1) { Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_TypeError, "boolean index must be 1D for a single axis"); return NULL; }
                const npy_intp *adims = PyArray_DIMS(arrk);
                npy_intp n = adims[0];
                /* Build integer indices from mask */
                /* Result length was already set to out_len from result shape */
                int pos = 0;
                for (npy_intp i = 0; i < n && pos < out_len; ++i) {
                    /* If mask shorter than axis, treat missing as False */
                    if (i < n) {
                        npy_bool v = *(npy_bool *)PyArray_GetPtr(arrk, &i);
                        if (v) {
                            const char *s = lb->axis_labels[src_axis][(int)i];
                            size_t L = strlen(s) + 1;
                            out->axis_labels[dst_axis][pos] = (char *)malloc(L);
                            if (!out->axis_labels[dst_axis][pos]) { Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                            memcpy(out->axis_labels[dst_axis][pos], s, L);
                            pos++;
                        }
                    }
                }
                if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], pos) < 0) { Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                Py_DECREF(arrk);
                /* finalize this axis and move on */
                src_axis += 1;
                dst_axis += 1;
                continue;
            } else if (PyTypeNum_ISINTEGER(kind)) {
                ind = (PyArrayObject *)PyArray_FromAny((PyObject *)arrk, PyArray_DescrFromType(NPY_INTP), 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            } else if (kind == NPY_UNICODE || kind == NPY_STRING || kind == NPY_OBJECT) {
                /* Map strings to intp via hash */
                int ind_nd = PyArray_NDIM(arrk);
                if (ind_nd < 1 || ind_nd > 2) { Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_NotImplementedError, "string indices with ndim>2 not supported"); return NULL; }
                const npy_intp *adims = PyArray_DIMS(arrk);
                ind = (PyArrayObject *)PyArray_SimpleNew(ind_nd, adims, NPY_INTP);
                if (!ind) { Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                if (ind_nd == 1) {
                    npy_intp n = adims[0];
                    for (npy_intp i = 0; i < n; ++i) {
                        PyObject *it = PyArray_GETITEM(arrk, PyArray_GetPtr(arrk, &i));
                        if (!it) { Py_DECREF(ind); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                        Py_ssize_t pos;
                        if (labelsblock_find(lb, src_axis, it, &pos) != 0) { Py_DECREF(it); Py_DECREF(ind); Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "label not found in axis"); return NULL; }
                        Py_DECREF(it);
                        *((npy_intp *)PyArray_GetPtr(ind, &i)) = pos;
                    }
                } else {
                    npy_intp R = adims[0], C = adims[1];
                    for (npy_intp r = 0; r < R; ++r) {
                        for (npy_intp c = 0; c < C; ++c) {
                            npy_intp ij[2] = {r, c};
                            PyObject *it = PyArray_GETITEM(arrk, PyArray_GetPtr(arrk, ij));
                            if (!it) { Py_DECREF(ind); Py_DECREF(arrk); labelsblock_decref(out); return NULL; }
                            Py_ssize_t pos;
                            if (labelsblock_find(lb, src_axis, it, &pos) != 0) { Py_DECREF(it); Py_DECREF(ind); Py_DECREF(arrk); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "label not found in axis"); return NULL; }
                            Py_DECREF(it);
                            *((npy_intp *)PyArray_GetPtr(ind, ij)) = pos;
                        }
                    }
                }
            } else {
                Py_DECREF(arrk);
                labelsblock_decref(out);
                PyErr_SetString(PyExc_TypeError, "unsupported index array dtype");
                return NULL;
            }
            Py_DECREF(arrk);

            int ind_nd = PyArray_NDIM(ind);
            const npy_intp *adims = PyArray_DIMS(ind);
            if (ind_nd == 1) {
                int n = (int)adims[0];
                out->axis_len[dst_axis] = n;
                for (int j = 0; j < n && j < out_len; ++j) {
                    npy_intp jj = (npy_intp)j;
                    npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, &jj));
                    if (idxv < 0 || idxv >= lb->axis_len[src_axis]) { Py_DECREF(ind); labelsblock_decref(out); PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
                    const char *s = lb->axis_labels[src_axis][(int)idxv];
                    size_t L = strlen(s) + 1;
                    out->axis_labels[dst_axis][j] = (char *)malloc(L);
                    if (!out->axis_labels[dst_axis][j]) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                    memcpy(out->axis_labels[dst_axis][j], s, L);
                }
                if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], n) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                Py_DECREF(ind);
            } else if (ind_nd == 2) {
                /* Create two axes */
                int R = (int)adims[0], C = (int)adims[1];
                /* First new axis (rows) */
                if (dst_axis >= res_ndim) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                out->axis_len[dst_axis] = R;
                for (int r = 0; r < R && r < (int)res_shape[dst_axis]; ++r) {
                    /* join labels of row r across C with '-' */
                    size_t tot = 1; /* for NUL */
                    for (int c = 0; c < C; ++c) {
                        npy_intp ij[2] = {r, c};
                        npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, ij));
                        const char *s = lb->axis_labels[src_axis][(int)idxv];
                        tot += strlen(s) + (c ? 1 : 0);
                    }
                    char *buf = (char *)malloc(tot);
                    if (!buf) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
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
                    out->axis_labels[dst_axis][r] = buf;
                }
                if (axhash_build(&out->axis_hash[dst_axis], (const char **)out->axis_labels[dst_axis], R) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }

                /* Second new axis (cols) */
                if (dst_axis + 1 >= res_ndim) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                out->axis_len[dst_axis + 1] = C;
                out->axis_labels[dst_axis + 1] = (char **)calloc(C, sizeof(char *));
                if (!out->axis_labels[dst_axis + 1]) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                for (int c = 0; c < C && (dst_axis + 1) < res_ndim && c < (int)res_shape[dst_axis + 1]; ++c) {
                    size_t tot = 1;
                    for (int r = 0; r < R; ++r) {
                        npy_intp ij[2] = {r, c};
                        npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, ij));
                        const char *s = lb->axis_labels[src_axis][(int)idxv];
                        tot += strlen(s) + (r ? 1 : 0);
                    }
                    char *buf = (char *)malloc(tot);
                    if (!buf) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
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
                    out->axis_labels[dst_axis + 1][c] = buf;
                }
                if (axhash_build(&out->axis_hash[dst_axis + 1], (const char **)out->axis_labels[dst_axis + 1], C) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }

                Py_DECREF(ind);
                /* we consumed one source axis and produced two dst axes */
                src_axis += 1;
                dst_axis += 2;
                continue;
            } else {
                /* Generalize to k>=3 new axes */
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

                /* For each new axis p, build labels by joining across other dims */
                for (int p = 0; p < knd; ++p) {
                    int len_p = (int)adims[p];
                    for (int vp = 0; vp < len_p && (dst_axis + p) < res_ndim && vp < (int)res_shape[dst_axis + p]; ++vp) {
                        /* Odometer over other dims */
                        npy_intp idxk[NPY_MAXDIMS];
                        for (int q = 0; q < knd; ++q) idxk[q] = 0;
                        idxk[p] = vp;
                        /* First pass: total length */
                        size_t tot = 1; /* NUL */
                        int first = 1;
                        while (1) {
                            npy_intp idxv = *((npy_intp *)PyArray_GetPtr(ind, idxk));
                            const char *s = lb->axis_labels[src_axis][(int)idxv];
                            tot += strlen(s) + (first ? 0 : 1);
                            first = 0;
                            /* advance odometer over other dims */
                            int qq = knd - 1;
                            while (qq >= 0) {
                                if (qq == p) { qq--; continue; }
                                idxk[qq]++;
                                if (idxk[qq] < adims[qq]) break;
                                idxk[qq] = 0; qq--;
                            }
                            if (qq < 0) break;
                        }
                        char *buf = (char *)malloc(tot);
                        if (!buf) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
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
                        out->axis_labels[dst_axis + p][vp] = buf;
                    }
                    if (axhash_build(&out->axis_hash[dst_axis + p], (const char **)out->axis_labels[dst_axis + p], (int)adims[p]) < 0) { Py_DECREF(ind); labelsblock_decref(out); return NULL; }
                }

                Py_DECREF(ind);
                src_axis += 1;
                dst_axis += knd;
                continue;
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

NPY_INLINE void
labels_cache_clear(LabeledArrayObject *obj)
{
    if (obj) {
        Py_XDECREF(obj->labels_cache);
        obj->labels_cache = NULL;
    }
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
        ((LabeledArrayObject *)view)->delimiter = (char *)malloc(L);
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
            self->delimiter = (char *)malloc(L);
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

    /* Identify numpy.concatenate, numpy.squeeze, numpy.stack, numpy.expand_dims, numpy.take */
    static PyObject *np_concatenate = NULL;
    static PyObject *np_squeeze = NULL;
    static PyObject *np_stack = NULL;
    static PyObject *np_expand_dims = NULL;
    static PyObject *np_take = NULL;
    if (np_concatenate == NULL || np_squeeze == NULL || np_stack == NULL || np_expand_dims == NULL || np_take == NULL) {
        PyObject *numpy = PyImport_ImportModule("numpy");
        if (!numpy) return NULL;
        if (np_concatenate == NULL) np_concatenate = PyObject_GetAttrString(numpy, "concatenate");
        if (np_squeeze == NULL) np_squeeze = PyObject_GetAttrString(numpy, "squeeze");
        if (np_stack == NULL) np_stack = PyObject_GetAttrString(numpy, "stack");
        if (np_expand_dims == NULL) np_expand_dims = PyObject_GetAttrString(numpy, "expand_dims");
        if (np_take == NULL) np_take = PyObject_GetAttrString(numpy, "take");
        Py_DECREF(numpy);
        if (!np_concatenate || !np_squeeze || !np_stack || !np_expand_dims || !np_take) return NULL;
    }
    int is_concatenate = (func == np_concatenate);
    int is_squeeze = (func == np_squeeze);
    int is_stack = (func == np_stack);
    int is_expand = (func == np_expand_dims);
    int is_take = (func == np_take);
    if (!is_concatenate && !is_squeeze && !is_stack && !is_expand && !is_take) {
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
        PyObject *res = PyObject_Call(func, new_fargs, fkwargs);
        Py_DECREF(new_fargs);
        if (!res) return NULL;
        if (PyArray_Check(res)) {
            PyObject *wrapped = PyObject_CallMethod(self_obj, "__array_wrap__", "OO", res, Py_None);
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
    if (is_concatenate || is_stack || is_expand || is_take) {
        if (fkwargs && fkwargs != Py_None && PyDict_Check(fkwargs)) {
            PyObject *axis_obj = PyDict_GetItemString(fkwargs, "axis"); /* borrowed */
            if (axis_obj && axis_obj != Py_None) {
                long axl = PyLong_AsLong(axis_obj);
                if (axl == -1 && PyErr_Occurred()) { if (arrays) Py_DECREF(arrays); return NULL; }
                axis = (int)axl;
            }
        }
        if (is_take && PyTuple_Check(fargs) && PyTuple_GET_SIZE(fargs) >= 3) {
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
    if (is_take) {
        /* Handle np.take with label indices by translating strings to ints and slicing */
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
        LabelsBlock *out_lb = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
        if (!out_lb) { Py_DECREF(view); Py_DECREF(arrays); return NULL; }
        out_lb->ndim = nd;
        out_lb->refcount = 1;
        out_lb->axis_len = (int *)calloc(nd, sizeof(int));
        out_lb->axis_labels = (char ***)calloc(nd, sizeof(char **));
        out_lb->axis_hash = (AxisHash *)calloc(nd, sizeof(AxisHash));
        if (!out_lb->axis_len || !out_lb->axis_labels || !out_lb->axis_hash) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }

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
                        out_lb->axis_labels[ax][pos] = (char *)malloc(L);
                        if (!out_lb->axis_labels[ax][pos]) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
                        memcpy(out_lb->axis_labels[ax][pos], s ? s : "", L);
                    }
                }
                if (axhash_build(&out_lb->axis_hash[ax], (const char **)out_lb->axis_labels[ax], total) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
            } else {
                int n = first_lb->axis_len[ax];
                out_lb->axis_len[ax] = n;
                out_lb->axis_labels[ax] = (char **)calloc(n, sizeof(char *));
                if (!out_lb->axis_labels[ax]) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
                if (axis_labels_copy(out_lb->axis_labels[ax], first_lb->axis_labels[ax], n) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
                if (axhash_build(&out_lb->axis_hash[ax], (const char **)out_lb->axis_labels[ax], n) < 0) { labelsblock_decref(out_lb); Py_DECREF(view); Py_DECREF(arrays); return NULL; }
            }
        }

        LabeledArrayObject *vobj = (LabeledArrayObject *)view;
        LabelsBlock *old = vobj->labels_block;
        vobj->labels_block = out_lb;
        labelsblock_decref(old);
        labels_cache_clear(vobj);

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
    LabelsBlock *out_lb2 = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
    if (!out_lb2) { Py_DECREF(view2); free(drop); return NULL; }
    out_lb2->ndim = new_nd;
    out_lb2->refcount = 1;
    out_lb2->axis_len = (int *)calloc(new_nd, sizeof(int));
    out_lb2->axis_labels = (char ***)calloc(new_nd, sizeof(char **));
    out_lb2->axis_hash = (AxisHash *)calloc(new_nd, sizeof(AxisHash));
    if (!out_lb2->axis_len || !out_lb2->axis_labels || !out_lb2->axis_hash) { labelsblock_decref(out_lb2); Py_DECREF(view2); free(drop); return NULL; }
    int dst = 0;
    for (int ax = 0; ax < orig_nd; ++ax) {
        if (drop[ax]) continue;
        int n = plb->axis_len[ax];
        out_lb2->axis_len[dst] = n;
        out_lb2->axis_labels[dst] = (char **)calloc(n, sizeof(char *));
        if (!out_lb2->axis_labels[dst]) { labelsblock_decref(out_lb2); Py_DECREF(view2); free(drop); return NULL; }
        if (axis_labels_copy(out_lb2->axis_labels[dst], plb->axis_labels[ax], n) < 0) { labelsblock_decref(out_lb2); Py_DECREF(view2); free(drop); return NULL; }
        if (axhash_build(&out_lb2->axis_hash[dst], (const char **)out_lb2->axis_labels[dst], n) < 0) { labelsblock_decref(out_lb2); Py_DECREF(view2); free(drop); return NULL; }
        dst++;
    }

    LabeledArrayObject *v2 = (LabeledArrayObject *)view2;
    LabelsBlock *old2 = v2->labels_block;
    v2->labels_block = out_lb2;
    labelsblock_decref(old2);
    labels_cache_clear(v2);

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

        LabelsBlock *out_lb3 = (LabelsBlock *)calloc(1, sizeof(LabelsBlock));
        if (!out_lb3) { Py_DECREF(view3); return NULL; }
        out_lb3->ndim = nd_new;
        out_lb3->refcount = 1;
        out_lb3->axis_len = (int *)calloc(nd_new, sizeof(int));
        out_lb3->axis_labels = (char ***)calloc(nd_new, sizeof(char **));
        out_lb3->axis_hash = (AxisHash *)calloc(nd_new, sizeof(AxisHash));
        if (!out_lb3->axis_len || !out_lb3->axis_labels || !out_lb3->axis_hash) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
        for (int ax = 0, src = 0; ax < nd_new; ++ax) {
            if (ax == axis) {
                out_lb3->axis_len[ax] = 1;
                out_lb3->axis_labels[ax] = (char **)calloc(1, sizeof(char *));
                if (!out_lb3->axis_labels[ax]) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
                const char *s = "1"; size_t L = strlen(s) + 1;
                out_lb3->axis_labels[ax][0] = (char *)malloc(L);
                if (!out_lb3->axis_labels[ax][0]) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
                memcpy(out_lb3->axis_labels[ax][0], s, L);
                if (axhash_build(&out_lb3->axis_hash[ax], (const char **)out_lb3->axis_labels[ax], 1) < 0) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
            } else {
                int n = plb->axis_len[src];
                out_lb3->axis_len[ax] = n;
                out_lb3->axis_labels[ax] = (char **)calloc(n, sizeof(char *));
                if (!out_lb3->axis_labels[ax]) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
                if (axis_labels_copy(out_lb3->axis_labels[ax], plb->axis_labels[src], n) < 0) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
                if (axhash_build(&out_lb3->axis_hash[ax], (const char **)out_lb3->axis_labels[ax], n) < 0) { labelsblock_decref(out_lb3); Py_DECREF(view3); return NULL; }
                src++;
            }
        }

        LabeledArrayObject *v3 = (LabeledArrayObject *)view3;
        LabelsBlock *old3 = v3->labels_block;
        v3->labels_block = out_lb3;
        labelsblock_decref(old3);
        labels_cache_clear(v3);
        return view3;
    }
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
    PyObject *res = PyObject_Call(meth, inputs, kwargs);
    Py_DECREF(meth);
    Py_DECREF(inputs);
    if (!res) return NULL;

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
    if (a0 < 0) a0 += nd; if (a1 < 0) a1 += nd;
    if (a0 < 0 || a1 < 0 || a0 >= nd || a1 >= nd || a1 <= a0) {
        PyErr_SetString(PyExc_ValueError, "invalid levels");
        return NULL;
    }

    const npy_intp *dims = PyArray_DIMS(arr);
    int k = (int)dims[a0];

    /* Create list of slices along a0 */
    PyObject *parts = PyList_New(k);
    if (!parts) return NULL;
    for (int i = 0; i < k; ++i) {
        PyObject *idx = PyTuple_New(nd);
        if (!idx) { Py_DECREF(parts); return NULL; }
        for (int ax = 0; ax < nd; ++ax) {
            PyObject *ob;
            if (ax == a0) ob = PyLong_FromLong(i);
            else ob = (PyObject *)PySlice_New(NULL, NULL, NULL);
            if (!ob) { Py_DECREF(idx); Py_DECREF(parts); return NULL; }
            PyTuple_SET_ITEM(idx, ax, ob);
        }
        PyObject *sl = PyArray_Type.tp_as_mapping->mp_subscript(self_obj, idx);
        Py_DECREF(idx);
        if (!sl) { Py_DECREF(parts); return NULL; }
        PyList_SET_ITEM(parts, i, sl); /* steals */
    }

    /* Concatenate parts along axis a1-1 (since a0 was dropped) */
    PyObject *numpy = PyImport_ImportModule("numpy"); if (!numpy) { Py_DECREF(parts); return NULL; }
    PyObject *conc = PyObject_GetAttrString(numpy, "concatenate"); Py_DECREF(numpy);
    if (!conc) { Py_DECREF(parts); return NULL; }
    PyObject *axis_kw = Py_BuildValue("{s:i}", "axis", (int)(a1 - 1));
    if (!axis_kw) { Py_DECREF(conc); Py_DECREF(parts); return NULL; }
    PyObject *args_cat = Py_BuildValue("(O)", parts);
    if (!args_cat) { Py_DECREF(axis_kw); Py_DECREF(conc); Py_DECREF(parts); return NULL; }
    PyObject *res_base = PyObject_Call(conc, args_cat, axis_kw);
    Py_DECREF(args_cat); Py_DECREF(axis_kw); Py_DECREF(conc); Py_DECREF(parts);
    if (!res_base) return NULL;

    /* View as LabeledArray */
    PyObject *view = PyArray_View((PyArrayObject *)res_base, NULL, (PyTypeObject *)Py_TYPE(self_obj));
    Py_DECREF(res_base);
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
            int pos = 0;
            for (int i = 0; i < n0; ++i) {
                for (int j = 0; j < n1; ++j) {
                    const char *s0 = plb->axis_labels[a0][i];
                    const char *s1 = plb->axis_labels[a1][j];
                    size_t L = strlen(s0 ? s0 : "") + strlen(joiner) + strlen(s1 ? s1 : "") + 1;
                    char *buf = (char *)malloc(L);
                    if (!buf) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
                    snprintf(buf, L, "%s%s%s", s0 ? s0 : "", joiner, s1 ? s1 : "");
                    out->axis_labels[dst][pos++] = buf;
                }
            }
            if (axhash_build(&out->axis_hash[dst], (const char **)out->axis_labels[dst], n0 * n1) < 0) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
        } else {
            int n = plb->axis_len[ax];
            out->axis_len[dst] = n;
            out->axis_labels[dst] = (char **)calloc(n, sizeof(char *));
            if (!out->axis_labels[dst]) { labelsblock_decref(out); Py_DECREF(view); return NULL; }
            for (int j = 0; j < n; ++j) {
                const char *s = plb->axis_labels[ax][j]; size_t L = strlen(s ? s : "") + 1;
                out->axis_labels[dst][j] = (char *)malloc(L);
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
        size_t L = strlen(self->delimiter) + 1; v->delimiter = (char *)malloc(L);
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
    int is_float = PyTypeNum_ISFLOAT(typ);
    int is_cplx  = PyTypeNum_ISCOMPLEX(typ);

    /* Release GIL around data scan */
    NPY_BEGIN_ALLOW_THREADS;

    if (nd == 1) {
        char *ptr = base;
        for (npy_intp i0 = 0; i0 < dims[0]; ++i0, ptr += strides[0]) {
            int not_nan = 1;
            if (is_float) {
                /* npy_isnan works for single/double/extended precision */
                if (typ == NPY_FLOAT)       not_nan = !npy_isnan((double)(*(float *)ptr));
                else if (typ == NPY_DOUBLE) not_nan = !npy_isnan(*(double *)ptr);
                else                        not_nan = 1;
            } else if (is_cplx) {
                if (typ == NPY_CFLOAT)   { float *p = (float *)ptr;  not_nan = !(npy_isnan((double)p[0]) || npy_isnan((double)p[1])); }
                else if (typ == NPY_CDOUBLE) { double *p = (double *)ptr; not_nan = !(npy_isnan(p[0]) || npy_isnan(p[1])); }
                else                        not_nan = 1;
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
                    if (is_float) {
                        if (typ == NPY_FLOAT)       { float v = *(float *)ptr;  not_nan = !npy_isnan((double)v); }
                        else if (typ == NPY_DOUBLE) { double v = *(double *)ptr; not_nan = !npy_isnan(v); }
                        else                         not_nan = 1;
                    } else if (is_cplx) {
                        if (typ == NPY_CFLOAT)     { float *p = (float *)ptr;  not_nan = !(npy_isnan((double)p[0]) || npy_isnan((double)p[1])); }
                        else if (typ == NPY_CDOUBLE) { double *p = (double *)ptr; not_nan = !(npy_isnan(p[0]) || npy_isnan(p[1])); }
                        else                         not_nan = 1;
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
