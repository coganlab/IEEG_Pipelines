# Checked
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ieeg.arrays.api import array_namespace, xp_assert_equal, ArrayLike, is_numpy, is_cupy

try:
    from numpy.lib.array_utils import normalize_axis_tuple
except ImportError:
    import operator
    def _normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
        if type(axis) not in (tuple, list):
            try:
                axis = [operator.index(axis)]
            except TypeError:
                pass
        # Going via an iterator directly is slower than via list comprehension.
        axis = tuple([_normalize_index(ax, ndim) for ax in axis])
        if not allow_duplicate and len(set(axis)) != len(axis):
            if argname:
                raise ValueError('repeated axis in `{}` argument'.format(argname))
            else:
                raise ValueError('repeated axis')
        return axis

    def _normalize_index(index: int, ndim: int):
        if ndim < 1:
            raise ValueError("ndim must be at least 1")
        while index < 0:
            index += ndim
        if index >= ndim:
            raise IndexError("index {} is out of bounds for axis with size {}"
                             "".format(index, ndim))
        return index
    normalize_axis_tuple = _normalize_axis_tuple

try:
    import cupy as cp
    from cupy.lib.stride_tricks import as_strided as as_strided_cp
    no_cupy = False
except ImportError:
    no_cupy = True


def stitch_mats(mats: list[ArrayLike], overlaps: list[int], axis: int = 0) -> ArrayLike:
    """break up the matrices into their overlapping and non-overlapping parts
    then stitch them back together

    Parameters
    ----------
    mats : list
        The matrices to stitch together
    overlaps : list
        The number of overlapping rows between each matrix
    axis : int, optional
        The axis to stitch along, by default 0

    Returns
    -------
    np.ndarray
        The stitched matrix

    Examples
    --------
    >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> mat2 = np.array([[7, 8, 9], [10, 11, 12]])
    >>> mat3 = np.array([[13, 14, 15], [16, 17, 18]])
    >>> stitch_mats([mat1, mat2, mat3], [1, 1])
    array([[ 1,  2,  3],
           [10, 11, 12],
           [16, 17, 18]])
    >>> stitch_mats([mat1, mat2, mat3], [0, 0], axis=1)
    array([[ 1,  2,  3,  7,  8,  9, 13, 14, 15],
           [ 4,  5,  6, 10, 11, 12, 16, 17, 18]])
    >>> mat4 = np.array([[19, 20, 21], [22, 23, float("nan")]])
    >>> stitch_mats([mat3, mat4], [0], axis=1)
    array([[13., 14., 15., 19., 20., 21.],
           [16., 17., 18., 22., 23., nan]])
    """
    xp = array_namespace(*mats)
    stitches = [mats[0]]
    if len(mats) != len(overlaps) + 1:
        raise ValueError("The number of matrices must be one more than the number "
                         "of overlaps")
    for i, over in enumerate(overlaps):
        stitches = stitches[:-2] + merge(stitches[-1], mats[i + 1], over, axis)
    out = xp.concatenate(stitches, axis=axis)
    try:
        xp_assert_equal(out.astype(int), out, check_dtype=False, xp=xp)
        return out.astype(int)
    except AssertionError:
        return out


def merge(mat1: ArrayLike, mat2: ArrayLike, overlap: int, axis: int = 0) -> list[ArrayLike]:
    """Take two arrays and merge them over the overlap gradually"""
    xp = array_namespace(mat1, mat2)
    sl = [slice(None)] * mat1.ndim
    sl[axis] = slice(0, mat1.shape[axis] - overlap)
    start = mat1[tuple(sl)]
    sl[axis] = slice(mat1.shape[axis] - overlap, mat1.shape[axis])
    middle1 = xp.multiply(xp.linspace(1, 0, overlap), mat1[tuple(sl)])
    sl[axis] = slice(0, overlap)
    middle2 = xp.multiply(xp.linspace(0, 1, overlap), mat2[tuple(sl)])
    middle = xp.add(middle1, middle2)
    sl[axis] = slice(overlap, mat2.shape[axis])
    last = mat2[tuple(sl)]

    return [start, middle, last]


def make_data_same(data_fix: ArrayLike, shape: tuple | list, stack_ax: int = 0,
                   pad_ax: int = -1, make_stacks_same: bool = True,
                   rng: np.random.Generator = None) -> ArrayLike:
    """Force the last dimension of data_fix to match the last dimension of
    shape.

    Takes the two arrays and checks if the last dimension of data_fix is
    smaller than the last dimension of shape. If there's more than two
    dimensions, it will rearrange the data to match the shape. If there's only
    two dimensions, it will repeat the signal to match the shape of data_like.
    If the last dimension of data_fix is larger than the last dimension of
    shape, it will return a subset of data_fix.

    Parameters
    ----------
    data_fix : ArrayLike
        The data to reshape.
    shape : list | tuple
        The shape of data to match.

    Returns
    -------
    data_fix : ArrayLike
        The reshaped data.

    Examples
    --------
    >>> np.random.seed(0)
    >>> data_fix = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> make_data_same(data_fix, (2, 8))
    array([[ 1,  2,  3,  4,  5,  4,  3,  2],
           [ 6,  7,  8,  9, 10,  9,  8,  7]])
    >>> (newarr := make_data_same(data_fix, (2, 2), make_stacks_same=False))
    array([[1, 2],
           [3, 4],
           [6, 7],
           [8, 9]])
    >>> make_data_same(newarr, (3, 2), stack_ax=1, pad_ax=0)
    array([[1, 2],
           [3, 4],
           [6, 7]])
    >>> import cupy as cp # doctest: +SKIP
    >>> make_data_same(cp.asarray(data_fix), (2, 2),
    ... make_stacks_same=False) # doctest: +SKIP
    array([[1, 2],
           [3, 4],
           [6, 7],
           [8, 9]])
    """

    xp = array_namespace(data_fix)

    if is_cupy(xp):
        rng = None
    elif not isinstance(rng, np.random.Generator):
        rng = xp.random.default_rng(rng)

    stack_ax = list(range(len(shape)))[stack_ax]
    pad_ax = list(range(len(shape)))[pad_ax]

    # Check if the pad dimension of data_fix is smaller than the pad
    # dimension of shape
    if data_fix.shape[pad_ax] <= shape[pad_ax]:
        out = pad_to_match(xp.zeros(shape), data_fix, stack_ax)

    # When the pad dimension of data_fix is larger than the pad dimension of
    # shape, take subsets of data_fix and stack them together on the stack
    # dimension
    else:
        out = rand_offset_reshape(data_fix, shape, stack_ax, pad_ax, rng)

    if not make_stacks_same:
        return out
    elif out.shape[stack_ax] > shape[stack_ax]:  # subsample stacks if too many
        idx = rng.choice(out.shape[stack_ax], (shape[stack_ax],), False)
        out = xp.take(out, idx, axis=stack_ax)
    elif out.shape[stack_ax] < shape[stack_ax]:  # oversample stacks if too few
        n = shape[stack_ax] - out.shape[stack_ax]
        idx = rng.choice(out.shape[stack_ax], (n,), True)
        new_data = xp.take(out, idx, axis=stack_ax)
        out = xp.concatenate((out, new_data), axis=stack_ax)

    return out


def pad_to_match(sig1: ArrayLike, sig2: ArrayLike, 
                 axis: int | tuple[int, ...] = ()) -> ArrayLike:
    """Pad the second signal to match the first signal along all axes not
    specified.

    Takes the two arrays and checks if the shape of sig2 is smaller than the
    shape of sig1. For each axis not specified, it will pad the second signal
    to match the first signal along that axis.

    Parameters
    ----------
    sig1 : ArrayLike
        The data to match.
    sig2 : ArrayLike
        The data to pad.
    axis : int | tuple
        The axes along which to pad the data.

    Returns
    -------
    sig2 : ArrayLike
        The padded data.

    Examples
    --------
    >>> sig1 = np.arange(48).reshape(2, 3, 8)
    >>> sig2 = np.arange(24).reshape(2, 3, 4)
    >>> pad_to_match(sig1, sig2)
    array([[[ 0,  1,  2,  3,  2,  1,  0,  1],
            [ 4,  5,  6,  7,  6,  5,  4,  5],
            [ 8,  9, 10, 11, 10,  9,  8,  9]],
    <BLANKLINE>
           [[12, 13, 14, 15, 14, 13, 12, 13],
            [16, 17, 18, 19, 18, 17, 16, 17],
            [20, 21, 22, 23, 22, 21, 20, 21]]])
    """
    xp = array_namespace(sig1, sig2)
    # Make sure the data is the same shape
    if xp.isscalar(axis):
        axis = (axis,)
    axis = list(axis)
    for i, ax in enumerate(axis):
        axis[i] = xp.arange(sig1.ndim)[ax]
    eq = [
        e for i, e in enumerate(
            xp.equal(xp.asarray(sig1.shape), xp.asarray(sig2.shape))
        ) if i not in axis
    ]
    if not all(eq):
        for ax in axis:
            eq.insert(ax, True)
        pad_shape = [
            (0, 0) if eq[i] else (0, sig1.shape[i] - sig2.shape[i])
            for i in range(sig1.ndim)
        ]
        sig2 = xp.pad(sig2, pad_shape, mode='reflect')
    return sig2


def rand_offset_reshape(data_fix: ArrayLike, shape: tuple, stack_ax: int, pad_ax: int,
                        rng: np.random.Generator | int = None) -> ArrayLike:
    """Take subsets of data_fix and stack them together on the stack dimension

    This function takes the data and reshapes it to match the shape by taking
    subsets of data_fix and stacking them together on the stack dimension,
    randomly offsetting the start of the first subset. It is assumed that the
    padding axis 'pad_ax' is larger in data_fix.shape than in shape.

    Parameters
    ----------
    data_fix : ArrayLike
        The data to reshape.
    shape : list | tuple
        The shape of data to match.
    stack_ax : int
        The axis along which to stack the subsets.
    pad_ax : int
        The axis along which to slice the subsets.
    rng : np.random.Generator | int, optional
        The random number generator to use. If None, a default random number
        generator will be used.

    Returns
    -------
    data_fix : ArrayLike
        The reshaped data.

    Examples
    --------
    >>> data_fix = np.arange(50).reshape((5, 10))
    >>> data_fix
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
    >>> rand_offset_reshape(data_fix, (2, 4), 0, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [11, 12, 13, 14],
           [15, 16, 17, 18],
           [21, 22, 23, 24],
           [25, 26, 27, 28],
           [31, 32, 33, 34],
           [35, 36, 37, 38],
           [41, 42, 43, 44],
           [45, 46, 47, 48]])
    >>> rand_offset_reshape(data_fix, (2, 4), 1, 0, 0) # doctest: +ELLIPSIS
    array([[ 0, 20,  1, 21,  2, 22,  3, 23,  4, 24,  5, 25,  6, 26,  7, 27,
             8, 28,  9, 29],
           [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 37,
            18, 38, 19, 39]])
    """

    xp = array_namespace(data_fix)

    if not isinstance(rng, np.random.Generator):
        rng = xp.random.default_rng(rng)

    # Randomly offset the start of the first subset
    num_stack = data_fix.shape[pad_ax] // shape[pad_ax]
    if data_fix.shape[pad_ax] % shape[pad_ax] == 0:
        num_stack -= 1
    offset = rng.integers(0, data_fix.shape[pad_ax] - shape[pad_ax] * num_stack)

    # Create an array to store the output
    out_shape = [
        shape[i] if i == pad_ax else data_fix.shape[i] for i in range(data_fix.ndim)
    ]
    out_shape[stack_ax] *= num_stack
    out = xp.zeros(tuple(out_shape), dtype=data_fix.dtype)

    # Iterate over the subsets
    sl_in = [slice(None)] * data_fix.ndim
    sl_out = [slice(None)] * data_fix.ndim
    for i in range(num_stack):
        # Get the start and end indices of the subset
        start = i * shape[pad_ax] + offset
        end = start + shape[pad_ax]

        # Create a slice object for the subset
        sl_in[pad_ax] = slice(start, end)
        sl_out[stack_ax] = slice(i, None, num_stack)

        # Fill in the subset
        out[tuple(sl_out)] = data_fix[tuple(sl_in)]

    return out


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.

    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.

    .. Copied from numpy.lib.stride_tricks.sliding_window_view

    Parameters
    ----------
    x : ArrayLike
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.

    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.

    See Also
    --------
    lib.stride_tricks.as_strided: A lower-level and less safe routine for
        creating arbitrary views from custom shape and strides.
    broadcast_to: broadcast an array to a given shape.

    Notes
    -----
    For many applications using a sliding window view can be convenient, but
    potentially very slow. Often specialized solutions exist, for example:

    - `scipy.signal.fftconvolve`

    - Filtering functions in `scipy.ndimage`

    - Moving window functions provided by
      `bottleneck <https://github.com/pydata/bottleneck>`_.

    As a rough estimate, a sliding window approach with an input size of `N`
    and a window size of `W` will scale as `O(N*W)` where frequently a special
    algorithm can achieve `O(N)`. That means that the sliding window variant
    for a window size of 100 can be a 100 times slower than a more specialized
    version.

    Nevertheless, for small window sizes, when no custom algorithm exists, or
    as a prototyping and developing tool, this function can be a good solution.

    Examples
    --------
    >>> import numpy as np
    >>> from ieeg.arrays.reshape import sliding_window_view
    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    This also works in more dimensions, e.g.

    >>> i, j = np.ogrid[:3, :4]
    >>> x = 10*i + j
    >>> x.shape
    (3, 4)
    >>> x
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> shape = (2,2)
    >>> v = sliding_window_view(x, shape)
    >>> v.shape
    (2, 3, 2, 2)
    >>> v
    array([[[[ 0,  1],
             [10, 11]],
    <BLANKLINE>
            [[ 1,  2],
             [11, 12]],
    <BLANKLINE>
            [[ 2,  3],
             [12, 13]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[10, 11],
             [20, 21]],
    <BLANKLINE>
            [[11, 12],
             [21, 22]],
    <BLANKLINE>
            [[12, 13],
             [22, 23]]]])

    The axis can be specified explicitly:

    >>> v = sliding_window_view(x, 3, 0)
    >>> v.shape
    (1, 4, 3)
    >>> v
    array([[[ 0, 10, 20],
            [ 1, 11, 21],
            [ 2, 12, 22],
            [ 3, 13, 23]]])

    The same axis can be used several times. In that case, every use reduces
    the corresponding original dimension:

    >>> v = sliding_window_view(x, (2, 3), (1, 1))
    >>> v.shape
    (3, 1, 2, 3)
    >>> v
    array([[[[ 0,  1,  2],
             [ 1,  2,  3]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[10, 11, 12],
             [11, 12, 13]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[20, 21, 22],
             [21, 22, 23]]]])

    Combining with stepped slicing (`::step`), this can be used to take sliding
    views which skip elements:

    >>> x = np.arange(7)
    >>> sliding_window_view(x, 5)[:, ::2]
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6]])

    Or views which move by multiple elements

    >>> x = np.arange(7)
    >>> sliding_window_view(x, 3)[::2, :]
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6]])

    A common application of `sliding_window_view` is the calculation of running
    statistics. The simplest example is the
    `moving average <https://en.wikipedia.org/wiki/Moving_average>`_:

    >>> x = np.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> moving_average = v.mean(axis=-1)
    >>> moving_average
    array([1., 2., 3., 4.])

    Note that a sliding window approach is often **not** optimal (see Notes).
    """
    xp = array_namespace(x)
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # # first convert input to array, possibly keeping subclass
    # x = xp.asarray(x)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape

    if is_numpy(xp):
        return as_strided(x, strides=out_strides, shape=out_shape,
                          subok=subok, writeable=writeable)
    elif is_cupy(xp) and no_cupy:
        raise ModuleNotFoundError("cupy not available")
    elif is_cupy(xp):
        return as_strided_cp(x, strides=out_strides, shape=out_shape)
    else:
        try:
            return as_strided(x, strides=out_strides, shape=out_shape,
                              subok=subok, writeable=writeable)
        except Exception:
            raise NotImplementedError(f"Only numpy and cupy are supported, not {xp.__name__}")
