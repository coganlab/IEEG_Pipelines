"""Tests for the Python implementation of LabeledArray."""

import numpy as np
from ieeg.arrays.labeledarray import LabeledArray as cla_c # c implementation
from ieeg.arrays.label import LabeledArray as cla

class LabeledArray(cla_c):
    """ A numpy array with labeled dimensions, acting like a dictionary.

    A numpy array with labeled dimensions. This class is useful for storing
    data that is not easily represented in a tabular format. It acts as a
    nested dictionary but its values map to elements of a stored numpy array.

    Parameters
    ----------
    input_array : array_like
        The array to store in the LabeledArray.
    labels : tuple[tuple[str, ...], ...], optional
        The labels for each dimension of the array, by default ().
    delimiter : str, optional
        The delimiter to use when combining labels, by default '-'
    **kwargs
        Additional arguments to pass to np.asarray.

    Attributes
    ----------
    labels : tuple[tuple[str, ...], ...]
        The labels for each dimension of the array.
    array : np.ndarray
        The array stored in the LabeledArray.
    """


labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
print('whole', cla(np.ones((2, 3, 4), dtype=float), labels))
print('labels', cla(np.ones((2, 3, 4), dtype=float), labels).labels)
print('la', la := cla(np.ones((2, 3, 4), dtype=float), labels))
print('la.labels', la.labels)
print('la.labels[0]', la.labels[0])
print([lab for lab in la.labels[0]])
print([a for a in la])
print('hi')
print(list(zip([lab for lab in la.labels[0]], [a for a in la])))
la['a', 'c', 'f'] = 2
print('set', la['a', 'c', 'f'])
print('to_dict', la.to_dict()) # doctest: +ELLIPSIS +SKIP

print('get', la['a', 'c'])
print('get single', np.array(la)[(0,)], la[('a',),])

print('find', la.find('a', 0))

print('take', np.take(la, np.array(['f','g']), axis=2))
print('take_along_axis', np.take_along_axis(la[:,:,0], np.array([[0, 0, 0], [1, 1, 1]]), axis=0))

print('concatenate', np.concatenate((la['a'], la['b']), axis=0))
print('swapaxes', np.swapaxes(la, 0, 1))
print('transpose', np.transpose(la, (1, 0, 2)))
la[:,:,0] = float('nan')
print('dropna', la.dropna().labels[2])
print('dropna mean', np.nanmean(la,axis=(0,1)).dropna().labels[0])

print('where mean', np.mean(la,axis=(0,1), where=np.isnan(la)))

la2 = cla.fromfile('temp')
print('la2', la2)
print('la2 dtype', la2.dtype)
print('dropna', la2.dropna())

la += 1
print('inplace', la)

print('meshgrid', np.meshgrid(*la.labels))
print('meshgrid index', la[np.ix_(*(np.arange(2) for lab in la.labels))])
print('meshgrid index control', la[:2,:2,:2])
data = {'a': {'b': {'c': 1., 'd': np.nan}}}
ad = cla.from_dict(data)
print('dropna', ad.dropna())

idx_tests = [
    (0,),
    ('a',),
    (('a','b'),),
    (slice(None),),
    (slice(None), slice(None), slice(None)),
    (slice(None), slice(None), 0),
    (slice(None), -1, slice(None)),
    ('a', slice(None), slice(None)),
    ('a', slice(None), 0),
    ('a', slice(None), ('f','g'))
]

for idx in idx_tests:
    print('idx', idx)
    print('la[idx]', la[idx])

print('la1 dict', la.__dict__)
la3 = cla_c(np.random.rand(2, 3), labels=[('a', 'b'), ('c', 'd', 'e')])
print('la3 dict', la3.__dict__)
la4 = cla(np.random.rand(2, 3), labels=[('a', 'b'), ('c', 'd', 'e')])
print('la4', la4)
print('la4 dict', la4.__dict__)
