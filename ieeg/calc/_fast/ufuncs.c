#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/halffloat.h>

#ifndef NPY_INLINE
#define NPY_INLINE static inline
#endif

#define NPY_NAN_HALF (npy_half)NPY_NAN

static PyMethodDef Meandiff_Methods[] = {
    {NULL, NULL, 0, NULL}
};

NPY_INLINE void sums_and_counts_half(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, float * const suml, float * const sums, npy_intp * const countl, npy_intp * const counts) {
    for (npy_intp j = 0; j < lenl; ++j) {
        npy_half vall = *(npy_half *)(inl + j * innerstepl);
        if (vall == vall) {
            *suml += npy_half_to_float(vall);
            (*countl)++;
        }
        if (j < lens) {
            npy_half vals = *(npy_half *)(ins + j * innersteps);
            if (vals == vals) {
                *sums += npy_half_to_float(vals);
                (*counts)++;
            }
        }
    }
}

NPY_INLINE void sums_and_counts_float(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, float * const suml, float * const sums, npy_intp * const countl, npy_intp * const counts) {
    for (npy_intp j = 0; j < lenl; ++j) {
        float vall = *(float *)(inl + j * innerstepl);
        if (vall == vall) {
            *suml += vall;
            (*countl)++;
        }
        if (j < lens) {
            float vals = *(float *)(ins + j * innersteps);
            if (vals == vals) {
                *sums += vals;
                (*counts)++;
            }
        }
    }
}

NPY_INLINE void sums_and_counts_double(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, double * const suml, double * const sums, npy_intp * const countl, npy_intp * const counts) {
    for (npy_intp j = 0; j < lenl; ++j) {
        double vall = *(double *)(inl + j * innerstepl);
        if (vall == vall) {
            *suml += vall;
            (*countl)++;
        }
        if (j < lens) {
            double vals = *(double *)(ins + j * innersteps);
            if (vals == vals) {
                *sums += vals;
                (*counts)++;
            }
        }
    }
}

NPY_INLINE void sums_and_counts_longdouble(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, long double * const suml, long double * const sums, npy_intp * const countl, npy_intp * const counts) {
    for (npy_intp j = 0; j < lenl; ++j) {
        long double vall = *(long double *)(inl + j * innerstepl);
        if (vall == vall) {
            *suml += vall;
            (*countl)++;
        }
        if (j < lens) {
            long double vals = *(long double *)(ins + j * innersteps);
            if (vals == vals) {
                *sums += vals;
                (*counts)++;
            }
        }
    }
}

static void mean_diff_half(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        float sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_half(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_half(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // Calculate the difference
        *((npy_half *)out) = ((count1 > 0) && (count2 > 0)) ? npy_float_to_half(sum1 / count1 - sum2 / count2) : NPY_NAN_HALF;
    }
}

static void mean_diff_float(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        float sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_float(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_float(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // Calculate the difference
        *((float *)out) = ((count1 > 0) && (count2 > 0)) ? sum1 / count1 - sum2 / count2 : NAN;
    }
}

static void mean_diff_double(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        double sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_double(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_double(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // Calculate the difference
        *((double *)out) = ((count1 > 0) && (count2 > 0)) ? sum1 / count1 - sum2 / count2 : NAN;
    }
}

static void mean_diff_longdouble(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        long double sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_double(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_double(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }
        // Calculate the difference
        *((long double *)out) = ((count1 > 0) && (count2 > 0)) ? sum1 / count1 - sum2 / count2 : NAN;
    }
}

NPY_INLINE float sum_var_float(const float sum1, const float sum2, const npy_intp n1, const npy_intp n2) {

    // at this point, it is already known that n1 and n2 are not zero
    if (n1 == 1) {
        return (float)sqrt(sum2 / ((n2 - 1) * n2));
    } else if (n2 == 1) {
        return (float)sqrt(sum1 / ((n1 - 1) * n1));
    } else {
        const float var1 = sum1 / ((n1 - 1) * n1);
        const float var2 = sum2 / ((n2 - 1) * n2);
        return (float)sqrt(var1 + var2);
    }
}

NPY_INLINE double sum_var_double(const double sum1, const double sum2, const npy_intp n1, const npy_intp n2) {

    // at this point, it is already known that n1 and n2 are not zero
    if (n1 == 1) {
        return sqrt(sum2 / ((n2 - 1) * n2));
    } else if (n2 == 1) {
        return sqrt(sum1 / ((n1 - 1) * n1));
    } else {
        const double var1 = sum1 / ((n1 - 1) * n1);
        const double var2 = sum2 / ((n2 - 1) * n2);
        return sqrt(var1 + var2);
    }
}

NPY_INLINE long double sum_var_longdouble(const long double sum1, const long double sum2, const npy_intp n1, const npy_intp n2) {

    // at this point, it is already known that n1 and n2 are not zero
    if (n1 == 1) {
        return sqrt(sum2 / ((n2 - 1) * n2));
    } else if (n2 == 1) {
        return sqrt(sum1 / ((n1 - 1) * n1));
    } else {
        const long double var1 = sum1 / ((n1 - 1) * n1);
        const long double var2 = sum2 / ((n2 - 1) * n2);
        return sqrt(var1 + var2);
    }
}

NPY_INLINE void varsums_half(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, const float meanl, const float means, float * const varsuml, float * const varsums) {
    for (npy_intp j = 0; j < lenl; ++j) {
        npy_half vall = *(npy_half *)(inl + j * innerstepl);
        if (!npy_half_isnan(vall)) {
            float val = npy_half_to_float(vall);
            val -= meanl;
            val *= val;
            *varsuml += val;
        }
        if (j < lens) {
            npy_half vals = *(npy_half *)(ins + j * innersteps);
            if (!npy_half_isnan(vals)) {
                float val = npy_half_to_float(vals);
                val -= means;
                val *= val;
                *varsums += val;
            }
        }
    }
}

NPY_INLINE void varsums_float(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, const float meanl, const float means, float * const varsuml, float * const varsums) {
    for (npy_intp j = 0; j < lenl; ++j) {
        float vall = *(float *)(inl + j * innerstepl);
        if (vall == vall) {
            vall -= meanl;
            vall *= vall;
            *varsuml += vall;
        }
        if (j < lens) {
            float vals = *(float *)(ins + j * innersteps);
            if (vals == vals) {
                vals -= means;
                vals *= vals;
                *varsums += vals;
            }
        }
    }
}

NPY_INLINE void varsums_double(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, const double meanl, const double means, double * const varsuml, double * const varsums) {
    for (npy_intp j = 0; j < lenl; ++j) {
        double vall = *(double *)(inl + j * innerstepl);
        if (vall == vall) {
            vall -= meanl;
            vall *= vall;
            *varsuml += vall;
        }
        if (j < lens) {
            double vals = *(double *)(ins + j * innersteps);
            if (vals == vals) {
                vals -= means;
                vals *= vals;
                *varsums += vals;
            }
        }
    }
}

NPY_INLINE void varsums_longdouble(const char *inl, const char *ins, const npy_intp lenl, const npy_intp lens, const npy_intp innerstepl, const npy_intp innersteps, const long double meanl, const long double means, long double * const varsuml, long double * const varsums) {
    for (npy_intp j = 0; j < lenl; ++j) {
        long double vall = *(long double *)(inl + j * innerstepl);
        if (vall == vall) {
            vall -= meanl;
            vall *= vall;
            *varsuml += vall;
        }
        if (j < lens) {
            long double vals = *(long double *)(ins + j * innersteps);
            if (vals == vals) {
                vals -= means;
                vals *= vals;
                *varsums += vals;
            }
        }
    }
}

static void t_test_half(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input


    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        float sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_half(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_half(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // varience is zero if there is only one element, so we need to check for this
        if ((count1 == 0) || (count2 == 0) || (count1 == 1 && count2 == 1)) {
            *((npy_half *)out) = NPY_NAN_HALF;
        } else {
            float varsum1 = 0.0, varsum2 = 0.0;
            // Calculate the mean
            float mean1 = sum1 / count1;
            float mean2 = sum2 / count2;

            // Calculate the variance
            varsums_half(in1, in2, len1, len2, innerstep1, innerstep2, mean1, mean2, &varsum1, &varsum2);

            // Calculate the difference
            *((npy_half *)out) = npy_float_to_half((mean1 - mean2) / sum_var_float(varsum1, varsum2, count1, count2));
        }
    }
}

static void t_test_float(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        float sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_float(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_float(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // varience is zero if there is only one element, so we need to check for this
        if ((count1 == 0) || (count2 == 0) || (count1 == 1 && count2 == 1)) {
            *((float *)out) = NPY_NAN;
        } else {
            float varsum1 = 0.0, varsum2 = 0.0;
            // Calculate the mean
            float mean1 = sum1 / count1;
            float mean2 = sum2 / count2;

            // Calculate the variance
            varsums_float(in1, in2, len1, len2, innerstep1, innerstep2, mean1, mean2, &varsum1, &varsum2);

            // Calculate the difference
            *((float *)out) = (mean1 - mean2) / sum_var_float(varsum1, varsum2, count1, count2);
        }
    }
}

static void t_test_double(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    // outer loop
    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        double sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_double(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_double(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // varience is zero if there is only one element, so we need to check for this
        if ((count1 == 0) || (count2 == 0) || (count1 == 1 && count2 == 1)) {
            *((double *)out) = NAN;
        } else {
            double varsum1 = 0.0, varsum2 = 0.0;
            // Calculate the mean
            double mean1 = sum1 / count1;
            double mean2 = sum2 / count2;

            // Calculate the variance
            varsums_double(in1, in2, len1, len2, innerstep1, innerstep2, mean1, mean2, &varsum1, &varsum2);

            // Calculate the difference
            *((double *)out) = (mean1 - mean2) / sum_var_double(varsum1, varsum2, count1, count2);
        }
    }
}

static void t_test_longdouble(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *extra)
{
    char *in1 = args[0], *in2 = args[1], *out = args[2];

    const npy_intp nloops = dimensions[0];  // Number of outer loops
    const npy_intp len1 = dimensions[1];    // Core dimension i
    const npy_intp len2 = dimensions[2];    // Core dimension j

    const npy_intp step1 = steps[0];        // Outer loop step size for the first input
    const npy_intp step2 = steps[1];        // Outer loop step size for the second input
    const npy_intp step_out = steps[2];     // Outer loop step size for the output
    const npy_intp innerstep1 = steps[3];   // Step size of elements within the first input
    const npy_intp innerstep2 = steps[4];   // Step size of elements within the second input

    // outer loop
    for (npy_intp i = 0; i < nloops; i++, in1 += step1, in2 += step2, out += step_out) {
        long double sum1 = 0.0, sum2 = 0.0;
        npy_intp count1 = 0, count2 = 0;

        // inner loop
        if (len1 > len2) {
            sums_and_counts_longdouble(in1, in2, len1, len2, innerstep1, innerstep2, &sum1, &sum2, &count1, &count2);
        } else {
            sums_and_counts_longdouble(in2, in1, len2, len1, innerstep2, innerstep1, &sum2, &sum1, &count2, &count1);
        }

        // varience is zero if there is only one element, so we need to check for this
        if ((count1 == 0) || (count2 == 0) || (count1 == 1 && count2 == 1)) {
            *((long double *)out) = NAN;
        } else {
            long double varsum1 = 0.0, varsum2 = 0.0;
            // Calculate the mean
            long double mean1 = sum1 / count1;
            long double mean2 = sum2 / count2;

            // Calculate the variance
            varsums_longdouble(in1, in2, len1, len2, innerstep1, innerstep2, mean1, mean2, &varsum1, &varsum2);

            // Calculate the difference
            *((long double *)out) = (mean1 - mean2) / sum_var_longdouble(varsum1, varsum2, count1, count2);
        }
    }
}

static PyUFuncGenericFunction funcs[8] = {&mean_diff_half,
                                          &mean_diff_float,
                                          &mean_diff_double,
                                          &mean_diff_longdouble,
                                          &t_test_half,
                                          &t_test_float,
                                          &t_test_double,
                                          &t_test_longdouble};

static char md_types[12] = {NPY_HALF, NPY_HALF, NPY_HALF,
                            NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                            NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                            NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};

static char t_types[12] = {NPY_HALF, NPY_HALF, NPY_HALF,
                           NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                           NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                           NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufuncs",
    NULL,
    -1,
    Meandiff_Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

static char *doc = "Calculate the mean difference between two groups."
    "\n\n"
    "This function is the default statistic function for time_perm_cluster. It"
    "calculates the mean difference between two groups along the specified axis."
    "\n\n"
    "Parameters"
    "----------"
    "group1 : array, shape (..., time)"
    "    The first group of observations."
    "group2 : array, shape (..., time)"
    "    The second group of observations."
    "axis : int or tuple of ints, optional"
    "    The axis or axes along which to compute the mean difference. If None,"
    "    compute the mean difference over all axes."
    "\n\n"
    "Returns"
    "-------"
    "avg1 - avg2 : array or float"
    "    The mean difference between the two groups."
    "\n\n"
    "Examples"
    "--------"
    ">>> import numpy as np"
    ">>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]])"
    ">>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])"
    ">>> mean_diff(group1, group2, axis=1)"
    "array([ 0., 14.])"
    ">>> mean_diff(group1, group2, axis=0)"
    "array([ 0., 30.,  0.,  5.,  0.])"
    ">>> group3 = np.arange(100000, dtype=float).reshape(20000, 5)"
    ">>> mean_diff(group3, group1, axis=0)"
    "array([49997., 49968., 49999., 49995., 50001.])";

PyMODINIT_FUNC PyInit_ufuncs(void) {
    PyObject *m, *ufunc1, *ufunc2, *d;
    import_array();
    import_ufunc();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    ufunc1 = PyUFunc_FromFuncAndDataAndSignature(funcs, NULL, md_types, 4, 2, 1, PyUFunc_None, "mean_diff",
    doc, 0, "(i),(j)->()");

    ufunc2 = PyUFunc_FromFuncAndDataAndSignature(funcs + 4, NULL, t_types, 4, 2, 1, PyUFunc_None, "t_test",
    "", 0, "(i),(j)->()");

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "mean_diff", ufunc1);
    PyDict_SetItemString(d, "t_test", ufunc2);
    Py_DECREF(ufunc1);
    Py_DECREF(ufunc2);

    return m;
}
