#ifndef IEEG_MEANVAR_CORE_H
#define IEEG_MEANVAR_CORE_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/halffloat.h>
#include <numpy/npy_math.h>

#ifndef NPY_INLINE
#define NPY_INLINE inline
#endif

static NPY_INLINE int isnan_f32(float x) { return x != x; }
static NPY_INLINE int isnan_f64(double x) { return x != x; }
static NPY_INLINE int isnan_f80(long double x) { return x != x; }

#define LOAD_F16(p) ((float)npy_half_to_float(*(npy_half *)(p)))
#define LOAD_F32(p) (*(float *)(p))
#define LOAD_F64(p) (*(double *)(p))
#define LOAD_F80(p) (*(long double *)(p))

#define STORE_F16(p, x) (*((npy_half *)(p)) = npy_float_to_half((float)(x)))
#define STORE_F32(p, x) (*((float *)(p)) = (float)(x))
#define STORE_F64(p, x) (*((double *)(p)) = (double)(x))
#define STORE_F80(p, x) (*((long double *)(p)) = (long double)(x))

void meanvar_half(char **args, const npy_intp *dimensions, const npy_intp *steps, void *extra);
void meanvar_float(char **args, const npy_intp *dimensions, const npy_intp *steps, void *extra);
void meanvar_double(char **args, const npy_intp *dimensions, const npy_intp *steps, void *extra);
void meanvar_longdouble(char **args, const npy_intp *dimensions, const npy_intp *steps, void *extra);

int meanvar_reduce_2d(PyArrayObject *arr2d, double ddof,
                      PyArrayObject **out_mean, PyArrayObject **out_var);

int std_reduce_2d(PyArrayObject *arr2d, double ddof,
                  PyArrayObject **out_std);

int meanstd_reduce_2d(PyArrayObject *arr2d, double ddof,
                      PyArrayObject **out_mean, PyArrayObject **out_std);

PyObject *get_numpy_func_cached(const char *name);

int nanmeanvar_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
                    PyObject *where_obj, double ddof,
                    PyObject **mean_out, PyObject **var_out,
                    int **axes_out, int *naxes_out);

int nanstd_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
                PyObject *where_obj, double ddof,
                PyObject **std_out, int **axes_out, int *naxes_out);

int nanmeanstd_core(PyObject *self_obj, PyObject *axis_obj, PyObject *dtype_obj,
                    PyObject *where_obj, double ddof,
                    PyObject **mean_out, PyObject **std_out,
                    int **axes_out, int *naxes_out);

#endif

