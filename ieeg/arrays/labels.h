#ifndef IEEG_ARRAYS_LABELS_H
#define IEEG_ARRAYS_LABELS_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <stdint.h>
#include <stddef.h>

#ifndef NPY_INLINE
#define NPY_INLINE inline
#endif

typedef struct {
    size_t cap;
    size_t used;
    uint64_t *hashes;
    const char **keys;
    int *vals;
} AxisHash;

typedef struct LabelsBlock LabelsBlock;

typedef struct LabelsBlock {
    int ndim;
    int *axis_len;
    char ***axis_labels;
    AxisHash *axis_hash;
    unsigned char *axis_borrowed;
    unsigned char *axis_str_borrowed;
    LabelsBlock *borrowed_owner;
    PyArrayObject **axis_arr;
    PyObject **axis_index;
    char **axis_slab;
    int refcount;
} LabelsBlock;

typedef struct { const char *ptr; size_t len; } Token;

typedef struct {
    size_t cap;
    size_t used;
    char **keys;
    size_t *lens;
    uint64_t *hashes;
    int owns_keys;
} StrSet;

typedef struct {
    StrSet *set;
    int initialized;
} IntersectState;

int labelsblock_find(const LabelsBlock *lb, int axis, PyObject *value, Py_ssize_t *out);
int labelsblock_finalize_axis_from_c(LabelsBlock *lb, int ax);
void labelsblock_decref(LabelsBlock *lb);
LabelsBlock *labelsblock_alloc(int nd);
LabelsBlock *labelsblock_new_default(PyArrayObject *arr);
int axhash_build(AxisHash *ah, const char **keys, int n);
int labelsblock_set_axis_copy(LabelsBlock *lb, int ax, char **src_labels, int n);
int labelsblock_set_axis_numeric(LabelsBlock *lb, int ax, int n);
int labelsblock_set_axis_constant(LabelsBlock *lb, int ax, const char *s);

LabelsBlock *labelsblock_from_py(PyObject *labels_in, PyArrayObject *arr);
PyObject *labelsblock_to_py_tuple(const LabelsBlock *lb);
PyObject *join_axis_labels(const LabelsBlock *lb, int ax, const char *joiner);
PyArrayObject *map_label_index_array_to_int(const LabelsBlock *lb, int axis, PyArrayObject *arrk);
LabelsBlock *labelsblock_slice(const LabelsBlock *lb, PyObject *ck, PyArrayObject *result);

int build_combined_labels_range(LabelsBlock *lb, int start, int end, const char *delim,
                                char ***out_labels, char **out_slab, int *out_len);

int split_tokens(const char *s, const char *delim, size_t delim_len,
                 Token **out_tokens, int *out_n, Token *stack_buf, int stack_cap);
void idx_increment_order(npy_intp *idx, const npy_intp *dims, int nd, int order_c);

StrSet *strset_new(size_t cap_hint, int owns_keys);
int strset_add(StrSet *set, const char *s, size_t len);
void strset_free(StrSet *set);
StrSet *strset_from_tokens(Token *tokens, int ntok);
StrSet *strset_intersect_tokens(const StrSet *set, Token *tokens, int ntok);
size_t strset_join_len(const StrSet *set, size_t delim_len);
char **strset_collect_keys(const StrSet *set, int *out_n, char **stack_buf, int stack_cap);
int cmp_cstr(const void *a, const void *b);

int make_array_unique_c(char **labels, int n, const char *delim, char **slab_io);

#endif
