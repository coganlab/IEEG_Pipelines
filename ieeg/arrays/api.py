"""Utility functions to use Python Array API compatible libraries.

Copied from Scipy

For the context about the Array API see:
https://data-apis.org/array-api/latest/purpose_and_scope.html

The SciPy use case of the Array API is described on the following page:
https://data-apis.org/array-api/latest/use_cases.html#use-case-scipy
"""
import os

from types import ModuleType
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from functools import reduce

from array_api_compat import (
    array_namespace as xp_array_namespace,
    is_array_api_obj,
    size as xp_size,
    numpy as np_compat,
    device as xp_device,
    is_numpy_namespace as is_numpy,
    is_cupy_namespace as is_cupy,
    is_torch_namespace as is_torch,
    is_jax_namespace as is_jax,
    is_array_api_strict_namespace as is_array_api_strict
)
from array_api_extra import (
    at, atleast_nd, cov, create_diagonal, expand_dims, kron, nunique,
    pad, setdiff1d, sinc
)

__all__ = [
    '_asarray', 'array_namespace', 'at', 'atleast_nd', 'cov',
    'create_diagonal', 'expand_dims', 'kron', 'nunique', 'pad', 'setdiff1d',
    'sinc', 'get_xp_devices',
    'is_array_api_strict', 'is_complex', 'is_cupy', 'is_jax', 'is_numpy', 'is_torch',
    'scipy_namespace_for',
    'xp_assert_equal', 'xp_assert_less',
    'xp_copy', 'xp_copysign', 'xp_device',
    'xp_moveaxis_to_end', 'xp_ravel', 'xp_real', 'xp_sign', 'xp_size',
    'xp_take_along_axis', 'xp_vector_norm',
]


# To enable array API and strict array-like input validation
# set the environment variable SCIPY_ARRAY_API to True.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
SCIPY_ARRAY_API: str | bool = os.environ.get("SCIPY_ARRAY_API", False)

Array: TypeAlias = Any  # To be changed to a Protocol later (see array-api#589)
ArrayLike: TypeAlias = Array | npt.ArrayLike


def _check_finite(array: Array, xp: ModuleType) -> None:
    """Check for NaNs or Infs."""
    msg = "array must not contain infs or NaNs"
    try:
        if not xp.all(xp.isfinite(array)):
            raise ValueError(msg)
    except TypeError:
        raise ValueError(msg)


def array_namespace(*arrays: Array) -> ModuleType:
    """Get the array API compatible namespace for the arrays xs.

    Parameters
    ----------
    *arrays : sequence of array_like
        Arrays used to infer the common namespace.

    Returns
    -------
    namespace : module
        Common namespace.

    Notes
    -----
    Thin wrapper around `array_api_compat.array_namespace`.

    1. Check for the global switch: SCIPY_ARRAY_API. This can also be accessed
       dynamically through ``_GLOBAL_CONFIG['SCIPY_ARRAY_API']``.
    2. `_compliance_scipy` raise exceptions on known-bad subclasses. See
       its definition for more details.

    When the global switch is False, it defaults to the `numpy` namespace.
    In that case, there is no compliance check. This is a convenience to
    ease the adoption. Otherwise, arrays must comply with the new rules.

    Examples
    --------
    >>> import numpy as np
    >>> array_namespace(np.array([1, 2, 3])) # doctest: +ELLIPSIS
    <module '...numpy' from ...

    """

    _arrays = [array for array in arrays if array is not None]

    return xp_array_namespace(*_arrays)


def intersect1d(*arrays: Array, assume_unique: bool = False, xp: ModuleType | None = None) -> Array:
    """SciPy-specific replacement for `np.intersect1d` with `assume_unique` and `xp`.

    Parameters
    ----------
    *arrays : array_like
        Input arrays. Will be cast to a common type.
    assume_unique : bool, optional
        If True, the input arrays are assumed to be unique, which can speed up the calculation.
    xp : array_namespace, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    intersect1d : array
        Sorted 1D array of common elements.

    Notes
    -----
    This function is a thin wrapper around `setdiff1d` from `array_api_extra`.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([3, 4, 5, 6, 7])
    >>> intersect1d(x, y)
    array([3, 4, 5])
    >>> z = np.array([3, 4, 7, 8])
    >>> intersect1d(x, y, z)
    array([3, 4])
    """
    if xp is None:
        xp = array_namespace(*arrays)
    if hasattr(xp, 'intersect1d'):
        reduction = lambda x, y: xp.intersect1d(x, y, assume_unique=assume_unique)
        return reduce(reduction, arrays)
    if len(arrays) == 0:
        return xp.array([])

    result = xp.asarray(arrays[0])
    for array in arrays[1:]:
        result = setdiff1d(result, setdiff1d(
            result, xp.asarray(array), assume_unique=assume_unique),
                              assume_unique=assume_unique)
    return result


def split(array: Array, indices_or_sections: int | list[int], axis: int = 0, xp: ModuleType | None = None) -> list[Array]:
    """SciPy-specific replacement for `np.split` with `axis` and `xp`.

    Parameters
    ----------
    array : array_like
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided into N equal
        arrays along `axis`. If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries indicate
        where along `axis` the array is split.
    axis : int, optional
        The axis along which to split, default is 0.
    xp : array_namespace, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    subarrays : list of ndarrays
        A list of sub-arrays.

    Notes
    -----
    This function is a thin wrapper around `array_api_compat.split`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(9.0)
    >>> np.split(x, 3)
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.])]
    >>> x = np.arange(8.0).reshape(2, 4)
    >>> np.split(x, 2, axis=1)
    [array([[0., 1.],
           [4., 5.]]), array([[2., 3.],
           [6., 7.]])]
    """
    if xp is None:
        xp = array_namespace(array)
    start = 0
    if isinstance(indices_or_sections, int):
        indices = np.linspace(0, xp_size(array), indices_or_sections + 1, dtype=int)
    else:
        indices = xp.asarray(indices_or_sections)
    subarrays = []
    for end in indices:
        subarrays.append(xp.take(array, slice(start, end), axis=axis))
        start = end
    return subarrays


def _asarray(
        array: ArrayLike,
        dtype: Any = None,
        order: Literal['K', 'A', 'C', 'F'] | None = None,
        copy: bool | None = None,
        *,
        xp: ModuleType | None = None,
        check_finite: bool = False,
        subok: bool = False,
    ) -> Array:
    """SciPy-specific replacement for `np.asarray` with `order`, `check_finite`, and
    `subok`.

    Memory layout parameter `order` is not exposed in the Array API standard.
    `order` is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.

    `check_finite` is also not a keyword in the array API standard; included
    here for convenience rather than that having to be a separate function
    call inside SciPy functions.

    `subok` is included to allow this function to preserve the behaviour of
    `np.asanyarray` for NumPy based inputs.
    """
    if xp is None:
        xp = array_namespace(array)
    if is_numpy(xp):
        # Use NumPy API to support order
        if copy is True:
            array = np.array(array, order=order, dtype=dtype, subok=subok)
        elif subok:
            array = np.asanyarray(array, order=order, dtype=dtype)
        else:
            array = np.asarray(array, order=order, dtype=dtype)
    else:
        try:
            array = xp.asarray(array, dtype=dtype, copy=copy)
        except TypeError:
            coerced_xp = array_namespace(xp.asarray(3))
            array = coerced_xp.asarray(array, dtype=dtype, copy=copy)

    if check_finite:
        _check_finite(array, xp)

    return array


def xp_copy(x: Array, *, xp: ModuleType | None = None) -> Array:
    """
    Copies an array.

    Parameters
    ----------
    x : array

    xp : array_namespace

    Returns
    -------
    copy : array
        Copied array

    Notes
    -----
    This copy function does not offer all the semantics of `np.copy`, i.e. the
    `subok` and `order` keywords are not used.

    Examples
    --------
    >>> import numpy as np
    >>> xp_copy([1,2,3], xp=np)
    array([1, 2, 3])
    """
    # Note: for older NumPy versions, `np.asarray` did not support the `copy` kwarg,
    # so this uses our other helper `_asarray`.
    if xp is None:
        xp = array_namespace(x)

    return _asarray(x, copy=True, xp=xp)


def _strict_check(actual, desired, xp, *,
                  check_namespace=True, check_dtype=True, check_shape=True,
                  check_0d=True):
    __tracebackhide__ = True  # Hide traceback for py.test
    if check_namespace:
        _assert_matching_namespace(actual, desired)

    # only NumPy distinguishes between scalars and arrays; we do if check_0d=True.
    # do this first so we can then cast to array (and thus use the array API) below.
    if is_numpy(xp) and check_0d:
        _msg = ("Array-ness does not match:\n Actual: "
                f"{type(actual)}\n Desired: {type(desired)}")
        assert ((xp.isscalar(actual) and xp.isscalar(desired))
                or (not xp.isscalar(actual) and not xp.isscalar(desired))), _msg

    actual = xp.asarray(actual)
    desired = xp.asarray(desired)

    if check_dtype:
        _msg = f"dtypes do not match.\nActual: {actual.dtype}\nDesired: {desired.dtype}"
        assert actual.dtype == desired.dtype, _msg

    if check_shape:
        _msg = f"Shapes do not match.\nActual: {actual.shape}\nDesired: {desired.shape}"
        assert actual.shape == desired.shape, _msg

    desired = xp.broadcast_to(desired, actual.shape)
    return actual, desired


def _assert_matching_namespace(actual, desired):
    __tracebackhide__ = True  # Hide traceback for py.test
    actual = actual if isinstance(actual, tuple) else (actual,)
    desired_space = array_namespace(desired)
    for arr in actual:
        arr_space = array_namespace(arr)
        _msg = (f"Namespaces do not match.\n"
                f"Actual: {arr_space.__name__}\n"
                f"Desired: {desired_space.__name__}")
        assert arr_space == desired_space, _msg


def xp_assert_equal(actual, desired, *, check_namespace=True, check_dtype=True,
                    check_shape=True, check_0d=True, err_msg='', xp=None):
    """Assert that two arrays are equal.

    Parameters
    ----------
    actual : array_like
        The array to test.
    desired : array_like
        The expected array.
    check_namespace : bool, optional
        If True, check that the arrays have the same namespace.
    check_dtype : bool, optional
        If True, check that the arrays have the same dtype.
    check_shape : bool, optional
        If True, check that the arrays have the same shape.
    check_0d : bool, optional
        If True, check that the arrays have the same dimensionality.
    err_msg : str, optional
        The error message to be printed in case of failure.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    None
        If the arrays are equal, otherwise raises an AssertionError.

    Notes
    -----
    This function is a wrapper around the testing functions of different array libraries,
    providing a consistent interface for equality testing across different backends.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([1, 2, 3])
    >>> xp_assert_equal(a, b)  # No error raised
    >>> c = np.array([1, 2, 4])
    >>> xp_assert_equal(a, c)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    AssertionError: Arrays are not equal
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if xp is None:
        xp = array_namespace(actual)

    actual, desired = _strict_check(
        actual, desired, xp, check_namespace=check_namespace,
        check_dtype=check_dtype, check_shape=check_shape,
        check_0d=check_0d
    )

    if is_cupy(xp):
        return xp.testing.assert_array_equal(actual, desired, err_msg=err_msg)
    elif is_torch(xp):
        # PyTorch recommends using `rtol=0, atol=0` like this
        # to test for exact equality
        err_msg = None if err_msg == '' else err_msg
        return xp.testing.assert_close(actual, desired, rtol=0, atol=0, equal_nan=True,
                                       check_dtype=False, msg=err_msg)
    # JAX uses `np.testing`
    return np.testing.assert_array_equal(actual, desired, err_msg=err_msg)


def xp_assert_less(actual, desired, *, check_namespace=True, check_dtype=True,
                   check_shape=True, check_0d=True, err_msg='', verbose=True, xp=None):
    """Assert that all elements of an array are strictly less than another array.

    Parameters
    ----------
    actual : array_like
        The array to test.
    desired : array_like
        The array to compare against.
    check_namespace : bool, optional
        If True, check that the arrays have the same namespace.
    check_dtype : bool, optional
        If True, check that the arrays have the same dtype.
    check_shape : bool, optional
        If True, check that the arrays have the same shape.
    check_0d : bool, optional
        If True, check that the arrays have the same dimensionality.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, print arrays that are not less.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    None
        If all elements of `actual` are strictly less than all elements of `desired`,
        otherwise raises an AssertionError.

    Notes
    -----
    This function is a wrapper around the testing functions of different array libraries,
    providing a consistent interface for comparison testing across different backends.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> xp_assert_less(a, b)  # No error raised
    >>> c = np.array([3, 4, 2])
    >>> xp_assert_less(a, c)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    AssertionError: Arrays are not less
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if xp is None:
        xp = array_namespace(actual)

    actual, desired = _strict_check(
        actual, desired, xp, check_namespace=check_namespace,
        check_dtype=check_dtype, check_shape=check_shape,
        check_0d=check_0d
    )

    if is_cupy(xp):
        return xp.testing.assert_array_less(actual, desired,
                                            err_msg=err_msg, verbose=verbose)
    elif is_torch(xp):
        if actual.device.type != 'cpu':
            actual = actual.cpu()
        if desired.device.type != 'cpu':
            desired = desired.cpu()
    # JAX uses `np.testing`
    return np.testing.assert_array_less(actual, desired,
                                        err_msg=err_msg, verbose=verbose)


def is_complex(x: Array, xp: ModuleType) -> bool:
    """Check if an array has a complex floating-point data type.

    Parameters
    ----------
    x : Array
        The array to check.
    xp : module
        The array API namespace to use.

    Returns
    -------
    bool
        True if the array has a complex floating-point data type, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1+2j, 3+4j])
    >>> is_complex(x, np)
    True
    >>> y = np.array([1, 2, 3])
    >>> is_complex(y, np)
    False
    """
    return xp.isdtype(x.dtype, 'complex floating')


def get_xp_devices(xp: ModuleType) -> list[str] | list[None]:
    """Returns a list of available devices for the given namespace.

    Parameters
    ----------
    xp : module
        The array API namespace to check for available devices.

    Returns
    -------
    list of str or list of None
        A list of available device strings for the given namespace.
        For PyTorch, this might include 'cpu', 'cuda:0', etc.
        For CuPy, this might include 'cuda:0', etc.
        For JAX, this might include 'cpu:0', 'gpu:0', 'tpu:0', etc.
        For other namespaces, returns [None].

    Examples
    --------
    >>> import numpy as np
    >>> get_xp_devices(np)
    [None]

    >>> # If PyTorch is available
    >>> # import torch
    >>> # get_xp_devices(torch)
    >>> # ['cpu', 'cuda:0', ...]
    """
    devices: list[str] = []
    if is_torch(xp):
        devices += ['cpu']
        import torch # type: ignore[import]
        num_cuda = torch.cuda.device_count()
        for i in range(0, num_cuda):
            devices += [f'cuda:{i}']
        if torch.backends.mps.is_available():
            devices += ['mps']
        return devices
    elif is_cupy(xp):
        import cupy # type: ignore[import]
        num_cuda = cupy.cuda.runtime.getDeviceCount()
        for i in range(0, num_cuda):
            devices += [f'cuda:{i}']
        return devices
    elif is_jax(xp):
        import jax # type: ignore[import]
        num_cpu = jax.device_count(backend='cpu')
        for i in range(0, num_cpu):
            devices += [f'cpu:{i}']
        num_gpu = jax.device_count(backend='gpu')
        for i in range(0, num_gpu):
            devices += [f'gpu:{i}']
        num_tpu = jax.device_count(backend='tpu')
        for i in range(0, num_tpu):
            devices += [f'tpu:{i}']
        return devices

    # given namespace is not known to have a list of available devices;
    # return `[None]` so that one can use this in tests for `device=None`.
    return [None]


def scipy_namespace_for(xp: ModuleType) -> ModuleType | None:
    """Return the `scipy`-like namespace of a non-NumPy backend

    That is, return the namespace corresponding with backend `xp` that contains
    `scipy` sub-namespaces like `linalg` and `special`. If no such namespace
    exists, return ``None``. Useful for dispatching.

    Parameters
    ----------
    xp : module
        The array API namespace for which to find the corresponding SciPy-like namespace.

    Returns
    -------
    module or None
        The SciPy-like namespace for the given array API namespace, or None if no such
        namespace exists.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy
    >>> scipy_namespace = scipy_namespace_for(np)
    >>> scipy_namespace is scipy
    True

    >>> # For CuPy, if available
    >>> # import cupy as cp
    >>> # scipy_namespace = scipy_namespace_for(cp)
    >>> # scipy_namespace is cupyx.scipy
    >>> # True
    """

    if is_cupy(xp):
        import cupyx  # type: ignore[import-not-found,import-untyped]
        return cupyx.scipy

    elif is_jax(xp):
        import jax  # type: ignore[import-not-found]
        return jax.scipy

    elif is_torch(xp):
        return xp

    elif is_numpy(xp):
        import scipy
        return scipy

    return None


# temporary substitute for xp.moveaxis, which is not yet in all backends
# or covered by array_api_compat.
def xp_moveaxis_to_end(
        x: Array,
        source: int,
        /, *,
        xp: ModuleType | None = None) -> Array:
    """Move an axis to the end of the array.

    Parameters
    ----------
    x : Array
        The input array.
    source : int
        The source axis to move.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the array.

    Returns
    -------
    Array
        Array with the source axis moved to the end.

    Notes
    -----
    This is a temporary substitute for xp.moveaxis, which is not yet available in all backends
    or covered by array_api_compat.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> x.shape
    (2, 2, 2)
    >>> y = xp_moveaxis_to_end(x, 0, xp=np)
    >>> y.shape
    (2, 2, 2)
    >>> # The first axis (axis 0) is now the last axis
    >>> np.array_equal(y, np.moveaxis(x, 0, -1))
    True
    """
    xp = array_namespace(xp) if xp is None else xp
    axes = list(range(x.ndim))
    temp = axes.pop(source)
    axes = axes + [temp]
    return xp.permute_dims(x, axes)


# temporary substitute for xp.copysign, which is not yet in all backends
# or covered by array_api_compat.
def xp_copysign(x1: Array, x2: Array, /, *, xp: ModuleType | None = None) -> Array:
    """Copy the sign of x2 to the magnitude of x1.

    Parameters
    ----------
    x1 : Array
        The array containing the magnitudes.
    x2 : Array
        The array containing the signs.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    Array
        An array with the magnitude of x1 and the sign of x2.

    Notes
    -----
    This is a temporary substitute for xp.copysign, which is not yet available in all backends
    or covered by array_api_compat. This implementation does not attempt to account for special cases.

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array([-1.3, 1.5, -3.0])
    >>> x2 = np.array([1.0, -2.2, 3.0])
    >>> xp_copysign(x1, x2, xp=np)
    array([ 1.3, -1.5,  3. ])
    >>> # Equivalent to NumPy's copysign
    >>> np.copysign(x1, x2)
    array([ 1.3, -1.5,  3. ])
    """
    # no attempt to account for special cases
    xp = array_namespace(x1, x2) if xp is None else xp
    abs_x1 = xp.abs(x1)
    return xp.where(x2 >= 0, abs_x1, -abs_x1)


# partial substitute for xp.sign, which does not cover the NaN special case
# that I need. (https://github.com/data-apis/array-api-compat/issues/136)
def xp_sign(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    """Return the sign of each element in the input array.

    Parameters
    ----------
    x : Array
        The input array.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the array.

    Returns
    -------
    Array
        An array with the same shape as x, where each element has the sign of the 
        corresponding element in x. The sign is defined as:
        - 1 for positive values
        - 0 for zero
        - -1 for negative values
        - NaN for NaN values

    Notes
    -----
    This is a partial substitute for xp.sign, which does not cover the NaN special case
    in some array API implementations. See https://github.com/data-apis/array-api-compat/issues/136
    for more details.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([-5.0, 0.0, 3.0, np.nan])
    >>> xp_sign(x, xp=np)
    array([-1.,  0.,  1., nan])
    >>> # Equivalent to NumPy's sign
    >>> np.sign(x)
    array([-1.,  0.,  1., nan])
    """
    xp = array_namespace(x) if xp is None else xp
    if is_numpy(xp):  # only NumPy implements the special cases correctly
        return xp.sign(x)
    sign = xp.zeros_like(x)
    one = xp.asarray(1, dtype=x.dtype)
    sign = xp.where(x > 0, one, sign)
    sign = xp.where(x < 0, -one, sign)
    sign = xp.where(xp.isnan(x), xp.nan*one, sign)
    return sign

# maybe use `scipy.linalg` if/when array API support is added
def xp_vector_norm(x: Array, /, *,
                   axis: int | tuple[int] | None = None,
                   keepdims: bool = False,
                   ord: int | float = 2,
                   xp: ModuleType | None = None) -> Array:
    """Compute the vector norm of an array.

    Parameters
    ----------
    x : Array
        The input array.
    axis : int or tuple of ints, optional
        The axis or axes along which to compute the norm. If None, the norm is
        computed over all elements in the array.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly
        against the original array.
    ord : int or float, optional
        The order of the norm. Default is 2 (Euclidean norm).
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the array.

    Returns
    -------
    Array
        The vector norm of the input array.

    Notes
    -----
    This function attempts to use the `linalg.vector_norm` function from the array API
    if available. If not, it falls back to a simple implementation for the Euclidean norm (ord=2).
    For backends not implementing the `linalg` extension, only the Euclidean norm is supported.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([3.0, 4.0])
    >>> float(xp_vector_norm(x, xp=np))  # Euclidean norm (default)
    5.0
    >>> float(xp_vector_norm(x, ord=1, xp=np))  # L1 norm
    7.0
    >>> x = np.array([[1, 2], [3, 4]])
    >>> xp_vector_norm(x, axis=0, xp=np)  # Norm along first axis
    array([3.16227766, 4.47213595])
    >>> xp_vector_norm(x, axis=1, xp=np)  # Norm along second axis
    array([2.23606798, 5.        ])
    """
    xp = array_namespace(x) if xp is None else xp

    if SCIPY_ARRAY_API:
        # check for optional `linalg` extension
        if hasattr(xp, 'linalg'):
            return xp.linalg.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
        else:
            if ord != 2:
                raise ValueError(
                    "only the Euclidean norm (`ord=2`) is currently supported in "
                    "`xp_vector_norm` for backends not implementing the `linalg` "
                    "extension."
                )
            # return (x @ x)**0.5
            # or to get the right behavior with nd, complex arrays
            return xp.sum(xp.conj(x) * x, axis=axis, keepdims=keepdims)**0.5
    else:
        # to maintain backwards compatibility
        return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def xp_ravel(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    """Return a flattened array.

    Parameters
    ----------
    x : Array
        The input array to be flattened.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the array.

    Returns
    -------
    Array
        A 1-D array containing the elements of the input array.

    Notes
    -----
    This function is equivalent to np.ravel written in terms of the array API.
    Even though it's one line, it comes up so often that it's worth having
    this function for readability.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4]])
    >>> xp_ravel(x, xp=np)
    array([1, 2, 3, 4])
    >>> # Equivalent to NumPy's ravel
    >>> np.ravel(x)
    array([1, 2, 3, 4])
    """
    xp = array_namespace(x) if xp is None else xp
    return xp.reshape(x, (-1,))


def xp_real(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    """Return the real part of a complex array or the array itself if it's not complex.

    Parameters
    ----------
    x : Array
        The input array.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the array.

    Returns
    -------
    Array
        The real part of the input array if it has a complex data type, otherwise the input array.

    Notes
    -----
    This is a convenience wrapper of xp.real that allows non-complex input;
    see data-apis/array-api#824.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1+2j, 3+4j])
    >>> xp_real(x, xp=np)
    array([1., 3.])
    >>> y = np.array([1, 2, 3])
    >>> xp_real(y, xp=np)  # Non-complex input is returned as is
    array([1, 2, 3])
    """
    xp = array_namespace(x) if xp is None else xp
    return xp.real(x) if xp.isdtype(x.dtype, 'complex floating') else x


def xp_take_along_axis(arr: Array,
                       indices: Array, /, *,
                       axis: int = -1,
                       xp: ModuleType | None = None) -> Array:
    """Take values from an array along an axis at the indices specified.

    Parameters
    ----------
    arr : Array
        The source array.
    indices : Array
        The indices of the values to extract. This array must have the same shape
        as `arr`, excluding the axis dimension.
    axis : int, optional
        The axis over which to select values. Default is -1 (the last axis).
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    Array
        The indexed array. The shape of the output is the same as `indices`.

    Notes
    -----
    This is a dispatcher for np.take_along_axis for backends that support it;
    see data-apis/array-api/pull#816.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])
    >>> # Sort along the last axis
    >>> ai = np.argsort(a)
    >>> ai
    array([[0, 2, 1],
           [1, 2, 0]])
    >>> xp_take_along_axis(a, ai, xp=np)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> # Sort along the first axis
    >>> ai = np.argsort(a, axis=0)
    >>> ai
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> xp_take_along_axis(a, ai, axis=0, xp=np)
    array([[10, 30, 20],
           [60, 40, 50]])
    """
    xp = array_namespace(arr) if xp is None else xp
    if is_torch(xp):
        return xp.take_along_dim(arr, indices, dim=axis)
    elif is_array_api_strict(xp):
        raise NotImplementedError("Array API standard does not define take_along_axis")
    else:
        return xp.take_along_axis(arr, indices, axis)


# utility to broadcast arrays and promote to common dtype
def xp_broadcast_promote(*args, ensure_writeable=False, force_floating=False, xp=None):
    """Broadcast arrays and promote to a common data type.

    Parameters
    ----------
    *args : sequence of array_like
        The arrays to broadcast and promote.
    ensure_writeable : bool, optional
        If True, ensure that the returned arrays are writeable by making a copy if necessary.
    force_floating : bool, optional
        If True, ensure that the returned arrays have a floating-point data type.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the arrays.

    Returns
    -------
    list of Arrays
        The broadcasted and promoted arrays. The order of the arrays in the output list
        corresponds to the order of the input arrays.

    Notes
    -----
    This function performs two operations:
    1. Broadcasts all arrays to a common shape.
    2. Promotes all arrays to a common data type.

    If `force_floating` is True, the common data type will be a floating-point type,
    even if all input arrays have integer types.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([[4], [5], [6]])
    >>> c, d = xp_broadcast_promote(a, b, xp=np)
    >>> c.shape  # Broadcasted to shape (3, 3)
    (3, 3)
    >>> d.shape  # Broadcasted to shape (3, 3)
    (3, 3)
    >>> c
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    >>> d
    array([[4, 4, 4],
           [5, 5, 5],
           [6, 6, 6]])
    >>> # With force_floating=True
    >>> e, f = xp_broadcast_promote(a, b, force_floating=True, xp=np)
    >>> e.dtype  # Promoted to float
    dtype('float64')
    """
    xp = array_namespace(*args) if xp is None else xp

    args = [(xp.asarray(arg) if arg is not None else arg) for arg in args]
    args_not_none = [arg for arg in args if arg is not None]

    # determine minimum dtype
    default_float = xp.asarray(1.).dtype
    dtypes = [arg.dtype for arg in args_not_none]
    try:  # follow library's prefered mixed promotion rules
        dtype = xp.result_type(*dtypes)
        if force_floating and xp.isdtype(dtype, 'integral'):
            # If we were to add `default_float` before checking whether the result
            # type is otherwise integral, we risk promotion from lower float.
            dtype = xp.result_type(dtype, default_float)
    except TypeError:  # mixed type promotion isn't defined
        float_dtypes = [dtype for dtype in dtypes
                        if not xp.isdtype(dtype, 'integral')]
        if float_dtypes:
            dtype = xp.result_type(*float_dtypes, default_float)
        elif force_floating:
            dtype = default_float
        else:
            dtype = xp.result_type(*dtypes)

    # determine result shape
    shapes = {arg.shape for arg in args_not_none}
    shape = np.broadcast_shapes(*shapes) if len(shapes) != 1 else args_not_none[0].shape

    out = []
    for arg in args:
        if arg is None:
            out.append(arg)
            continue

        # broadcast only if needed
        # Even if two arguments need broadcasting, this is faster than
        # `broadcast_arrays`, especially since we've already determined `shape`
        if arg.shape != shape:
            arg = xp.broadcast_to(arg, shape)

        # convert dtype/copy only if needed
        if (arg.dtype != dtype) or ensure_writeable:
            arg = xp.astype(arg, dtype, copy=True)
        out.append(arg)

    return out


def xp_float_to_complex(arr: Array, xp: ModuleType | None = None) -> Array:
    """Convert a floating-point array to a complex array.

    Parameters
    ----------
    arr : Array
        The input array with floating-point data type.
    xp : module, optional
        The array API namespace to use. If not provided, the namespace is inferred from the array.

    Returns
    -------
    Array
        The input array converted to a complex data type. float32 is converted to complex64,
        and float64 (and other real floating types) are converted to complex128.

    Notes
    -----
    This function only converts arrays with floating-point data types. If the input array
    already has a complex data type, it is returned unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    >>> y = xp_float_to_complex(x, xp=np)
    >>> y.dtype
    dtype('complex64')
    >>> z = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    >>> w = xp_float_to_complex(z, xp=np)
    >>> w.dtype
    dtype('complex128')
    >>> # Complex input is returned unchanged
    >>> c = np.array([1+2j, 3+4j])
    >>> xp_float_to_complex(c, xp=np) is c
    True
    """
    xp = array_namespace(arr) if xp is None else xp
    arr_dtype = arr.dtype
    # The standard float dtypes are float32 and float64.
    # Convert float32 to complex64,
    # and float64 (and non-standard real dtypes) to complex128
    if xp.isdtype(arr_dtype, xp.float32):
        arr = xp.astype(arr, xp.complex64)
    elif xp.isdtype(arr_dtype, 'real floating'):
        arr = xp.astype(arr, xp.complex128)

    return arr

if __name__ == '__main__':
    pass
