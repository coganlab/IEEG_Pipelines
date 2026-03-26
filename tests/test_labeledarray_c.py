import numpy as np
from ieeg.arrays.labeledarray import LabeledArray
# from ieeg.arrays.label import LabeledArray # from python subclass
import faulthandler, sys
faulthandler.enable()
faulthandler.dump_traceback_later(10, repeat=True, file=sys.stderr)

labs = (('r0','r1'),('c0','c1','c2'))
a = LabeledArray(np.arange(6, dtype=np.float32).reshape(2,3), labels=labs)
print('a', a)

# slice by rows retains labels on remaining axis
s = a['r0']
print('slice_shape', s.shape)
print('slice_labels0', s.labels[0])

# slice by columns
s2 = a[:, ('c1','c2')]
print('slice2_shape', s2.shape)
print('slice2_labels1', s2.labels[1])

# ellipsis and newaxis
s3 = a['r1', ...]
print('ellipsis_shape', s3.shape)
print('ellipsis_labels', s3.labels)

s4 = a[np.newaxis, 'r0']
print('newaxis_shape', s4.shape)
print('newaxis_labels0', s4.labels[0])

# labels persist across views (transpose)
t = a.T
print('T_shape', t.shape)
print('T_labels0', t.labels[0])

# print(a.find('r0'))
print([b for b in a])

print(a.labels)
a[0] = [2,3,4]
print(a)
a[1] = np.array([5,6,7])
a[1] += 1
print(a)

def weighted_preserve_stats(data, weights, axis=None):
    """
    Multiplies data along the specified axis by weights, then rescales
    to preserve the original mean and variance.

    Parameters:
        data (np.ndarray): The input data array.
        weights (np.ndarray): The weight vector.
        axis (int): The axis along which to multiply.

    Returns:
        np.ndarray: The weighted and rescaled data.
    """
    where = ~np.isnan(data)
    kwargs = {'where': where, 'dtype': 'f4'}
    orig_mean = np.mean(data, **kwargs)
    orig_std = np.std(data, **kwargs)

    # Multiply along the specified axis
    if axis is None:
        data *= weights
    else:
        data *= weights.reshape([1 if i != axis else -1 for i in range(data.ndim)])

    # Rescale to preserve mean and variance
    weighted_mean = np.mean(data, **kwargs)
    weighted_std = np.std(data, **kwargs)
    data -= weighted_mean
    data *= orig_std / weighted_std
    data += orig_mean
    return data

print(weighted_preserve_stats(a, np.array([[1,2,3]])))

print("combine", a.combine((0,1)))

print("")
