import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def concatenate_arrays(arrays, axis=0):
    cdef list out_shape, index_slice

    if axis is None:
        axis = 0
        arrays = [np.expand_dims(ar, axis) for ar in arrays]

    out_shape = list(np.max([arr.shape for arr in arrays], axis=0))
    out_shape[axis] = sum(arr.shape[axis] for arr in arrays)

    # Create an empty array to hold the result
    new_arr = np.full(tuple(out_shape), float('nan'), dtype=float)

    # Create a list of slices that will be used to insert each array
    index_slice = [slice(None)] * new_arr.ndim
    start = 0
    for arr in arrays:
        # Insert the array into the result array
        for i in range(arr.ndim):
            if i == axis:
                index_slice[i] = slice(start, arr.shape[axis] + start)
            else:
                index_slice[i] = slice(0, arr.shape[i])
        new_arr[tuple(index_slice)] = arr
        start += arr.shape[axis]

    return new_arr
