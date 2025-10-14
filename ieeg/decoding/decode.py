try:
    import cupy as cp
except ImportError:
    cp = None

from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from typing import Optional, Callable, Iterable, Any
from ieeg.decoding.models import PcaLdaClassification, LoopwiseTransformer
from ieeg.arrays.label import LabeledArray
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.arrays.api import array_namespace, Array, is_torch, is_numpy
from ieeg.arrays.reshape import sliding_window_view
from ieeg.calc.fast import mixup, mixup2
import numpy as np
import matplotlib.pyplot as plt
from ieeg.viz.ensemble import plot_dist
from joblib import Parallel, delayed
import itertools
from tqdm import tqdm


class Decoder(MinimumNaNSplit):

    def __init__(self, categories: dict,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 min_samples: int = 1,
                 which: str = 'test',
                 model: BaseEstimator = PcaLdaClassification()):
        """Initialize the Decoder.

        Parameters
        ----------
        categories : dict
            Dictionary mapping category names to category indices.
        n_splits : int, optional
            Number of splits for cross-validation, by default 5.
        n_repeats : int, optional
            Number of repetitions for cross-validation, by default 1.
        min_samples : int, optional
            Minimum number of samples required for each category, by default 1.
        which : str, optional
            Which set to use for validation ('test' or 'train'), by default
             'test'.
        **kwargs
            Additional keyword arguments passed to the PcaLdaClassification
             model.
        """
        self.model = model
        MinimumNaNSplit.__init__(self, n_splits, n_repeats,
                                 None, min_samples, which)
        self.categories = categories
        self.current_job = "Repetitions"
        self.t = None

    def cv_cm(self, x_data: Array, labels: Array,
              normalize: str = None, obs_axs: int = -2, n_jobs: int = 1,
              average_repetitions: bool = False, window: int = None,
              shuffle: bool = False, oversample: bool = True, step: int = 1,
              # Nested CV / hyperparameter search options
              search_factory=None,
              random_state: int | None = None,
              # Parallelization options
              joblib_backend: Optional[str] = None
              ) -> Array:
        """Cross-validated confusion matrix

        Parameters
        ----------
        x_data : np.ndarray
            The data to be decoded
        labels : np.ndarray
            The labels for the data
        normalize : str, optional
            How to normalize the confusion matrix, by default None
        obs_axs : int, optional
            The axis containing the observations, by default -2
        n_jobs : int, optional
            The number of jobs to run in parallel, by default 1
        average_repetitions : bool, optional
            Whether to average the repetitions, by default True
        window : int, optional
            The window size for time sliding, by default None
        shuffle : bool, optional
            Whether to shuffle the labels, by default False
        oversample : bool, optional
            Whether to oversample the training data, by default True
        step : int, optional
            The step size for time sliding, by default 1

        Returns
        -------
        np.ndarray
            The confusion matrix

        Examples
        --------
        >>> np.random.seed(42); conds = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
        >>> model = PcaLdaClassification(0.8, 'lda')
        >>> decoder = Decoder(conds, 5, 2, model=model)
        >>> X = np.random.randn(5, 50, 5, 30)
        >>> labels = np.random.randint(1, 5, 50)
        >>> decoder.cv_cm(X, labels, normalize='true', obs_axs=1)
        array([[0.        , 0.05555556, 0.94444444, 0.        ],
               [0.        , 0.05      , 0.9       , 0.05      ],
               [0.        , 0.1       , 0.9       , 0.        ],
               [0.        , 0.09375   , 0.90625   , 0.        ]])
        >>> decoder.cv_cm(X, labels, normalize='true', window=20, step=5,
        ... obs_axs=1)[0]
        array([[0.11111111, 0.55555556, 0.27777778, 0.05555556],
               [0.1       , 0.5       , 0.35      , 0.05      ],
               [0.06666667, 0.6       , 0.33333333, 0.        ],
               [0.03125   , 0.53125   , 0.375     , 0.0625    ]])
        >>> decoder.cv_cm(X, labels, normalize='true', window=20, step=5,
        ...     shuffle=True, oversample=True, obs_axs=1)[0]
        array([[0.        , 0.        , 0.44444444, 0.55555556],
               [0.        , 0.        , 0.4       , 0.6       ],
               [0.        , 0.        , 0.36666667, 0.63333333],
               [0.        , 0.        , 0.53125   , 0.46875   ]])
        >>> model = PcaLdaClassification(0.8, 'lda')
        >>> decoder = Decoder(conds, 5, 2, model=model)
        >>> decoder.cv_cm(X, labels, normalize='true', obs_axs=1)

        >>> model = PcaLdaClassification(0.5, 'lda', loopwise=2)
        >>> decoder = Decoder(conds, 5, 2, model=model)
        >>> decoder.cv_cm(X, labels, normalize='true', obs_axs=1)
        array([[0.     , 0.     , 0.     , 1.     ],
               [0.     , 0.     , 0.     , 1.     ],
               [0.     , 0.     , 0.     , 1.     ],
               [0.     , 0.     , 0.09375, 0.90625]])
        >>> import cupy as cp
        >>> X = cp.random.randn(10, 10, 50, 100)
        >>> X[0, 0, 0, :] = np.nan
        >>> labels = cp.random.randint(1, 5, 50)
        >>> with config_context(array_api_dispatch=True):
        ...     decoder.cv_cm(X, labels, normalize='true')
        array([[0.        , 0.36666667, 0.63333333, 0.        ],
               [0.        , 0.32777778, 0.67222222, 0.        ],
               [0.        , 0.33157895, 0.66842105, 0.        ],
               [0.        , 0.35714286, 0.64285714, 0.        ]])
        >>> model = PcaLdaClassification(0.8, 'lda')
        >>> decoder = Decoder(conds, 5, 2, model=model)
        >>> with config_context(array_api_dispatch=True):
        ...     decoder.cv_cm(X, labels,  normalize='true')

        Nested hyper-parameter search with inner CV using HalvingRandomSearchCV
        ----------------------------------------------------------------------
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(2, 20, 2, 40)  # (channels, trials, ..., features)
        >>> y = np.array([0, 1] * 10)
        >>> cats = {'a': 0, 'b': 1}
        >>> model = PcaLdaClassification(0.8, 'lda')
        >>> def factory(estimator, cv, scoring):
        ...     from sklearn.model_selection import HalvingRandomSearchCV
        ...     return HalvingRandomSearchCV(
        ...         estimator=estimator,
        ...         param_distributions={'explained_variance': [0.4, 0.95]},
        ...         cv=cv,
        ...         factor=2,
        ...         random_state=0
        ...     )
        >>> dec = Decoder(cats, n_splits=5, n_repeats=1, model=model)
        >>> dec.cv_cm(X, y, obs_axs=1, search_factory=factory, normalize='true')

        >>> dec.cv_cm(X, y, obs_axs=1, search_factory=factory, window=20,
        ... step=5, normalize='true')
        """
        # # if model is a pipeline containing pca and weights is not None,
        # # change pca to weighted pca
        assert all(lab in self.categories.values() for lab in labels.tolist()), \
            "Labels must be in the categories"
        xp = array_namespace(x_data)
        def _shape_builder(data_local, xp_local):
            n_c = len(self.categories)
            shape = (self.n_repeats, self.n_splits, n_c, n_c)
            if window is not None:
                shape = ((data_local.shape[-1] - window) // step + 1,) + shape
            return shape, xp_local.int16

        mats = self._run_cv(
            x_data, labels, obs_axs, n_jobs, window, shuffle, oversample,
            step, search_factory, random_state, joblib_backend,
            mode='cm', out_shape_builder=_shape_builder
        )

        # average the repetitions
        if average_repetitions:
            mats = xp.mean(mats, axis=1)

        # normalize, sum the folds
        mats = xp.sum(mats, axis=-3)
        if normalize == 'true':
            divisor = xp.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = xp.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return mats / divisor

    def cv_accuracy(self, x_data: Array, labels: Array,
                     obs_axs: int = -2, n_jobs: int = 1,
                     average_repetitions: bool = False, window: int = None,
                     shuffle: bool = False, oversample: bool = True, step: int = 1,
                     search_factory=None, random_state: int | None = None,
                     joblib_backend: Optional[str] = None) -> Array:
        """Cross-validated balanced accuracy per fold (and window if set).

                Parameters
        ----------
        x_data : np.ndarray
            The data to be decoded
        labels : np.ndarray
            The labels for the data
        obs_axs : int, optional
            The axis containing the observations, by default -2
        n_jobs : int, optional
            The number of jobs to run in parallel, by default 1
        average_repetitions : bool, optional
            Whether to average the repetitions, by default True
        window : int, optional
            The window size for time sliding, by default None
        shuffle : bool, optional
            Whether to shuffle the labels, by default False
        oversample : bool, optional
            Whether to oversample the training data, by default True
        step : int, optional
            The step size for time sliding, by default 1

        Returns
        -------
        acc : ndarray
            Shape:
            - no window: (n_repeats, n_splits)
            - with window: (n_windows, n_repeats, n_splits)

        Examples
        --------
        >>> np.random.seed(42); conds = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}
        >>> model = PcaLdaClassification(0.8, 'lda')
        >>> decoder = Decoder(conds, 5, 2, model=model)
        >>> X = np.random.randn(5, 50, 5, 30)
        >>> labels = np.random.randint(1, 5, 50)
        >>> decoder.cv_accuracy(X, labels, obs_axs=1)
        >>> decoder.cv_accuracy(X, labels, window=20, step=5, obs_axs=1)[0]

        """
        assert all(lab in self.categories.values() for lab in labels.tolist()), \
            "Labels must be in the categories"
        xp = array_namespace(x_data)
        def _shape_builder_acc(data_local, xp_local):
            shape = (self.n_repeats, self.n_splits)
            if window is not None:
                shape = ((data_local.shape[-1] - window) // step + 1,) + shape
            return shape, xp_local.float32

        acc = self._run_cv(
            x_data, labels, obs_axs, n_jobs, window, shuffle, oversample,
            step, search_factory, random_state, joblib_backend,
            mode='score', out_shape_builder=_shape_builder_acc
        )

        # average the repetitions
        if average_repetitions:
            acc = xp.mean(acc, axis=1)

        return acc

    def _run_cv(self, x_data: Array, labels: Array, obs_axs: int, n_jobs: int,
                window: int | None, shuffle: bool, oversample: bool, step: int,
                search_factory, random_state, joblib_backend,
                mode: str, out_shape_builder) -> Array:
        assert all(lab in self.categories.values() for lab in labels.tolist()), \
            "Labels must be in the categories"
        xp = array_namespace(x_data)
        data = x_data.swapaxes(0, obs_axs)

        if shuffle:
            isnan = xp.isnan(data)
            std = float(xp.std(data[isnan], dtype='f8'))
            data[isnan] = xp.random.normal(0, 3 * std, int(xp.sum(isnan,
                                                                  dtype='i8')))
            label_stack = [labels.copy() for _ in range(self.n_repeats)]
            for i in range(self.n_repeats):
                self.shuffle_labels(data, label_stack[i], 0)
            idxs = ((self.split(data, lab), lab) for lab in label_stack)
            idxs = ((itertools.islice(s, self.n_splits),
                     itertools.repeat(l, self.n_splits))
                    for s, l in idxs)
            splits, label = zip(*idxs)
            splits = itertools.chain.from_iterable(splits)
            label = itertools.chain.from_iterable(label)
            idxs = zip(splits, label)
        else:
            idxs = ((splits, labels) for splits in self.split(data, labels))

        shape, dtype = out_shape_builder(data, xp)
        out = xp.zeros(shape, dtype=dtype)

        task_iter = (
            (train_idx, test_idx, l, data, i, self.n_splits, self.categories,
             window, step, oversample, clone(self.model), xp, search_factory,
             random_state, self.min_non_nan, mode)
            for i, ((train_idx, test_idx), l) in enumerate(idxs)
        )

        if n_jobs == 1:
            results = (_proc(*args) for args in task_iter)
        else:
            parallel_kwargs = dict(n_jobs=n_jobs, verbose=0,
                                   # require='sharedmem',
                                   return_as="generator_unordered")
            if joblib_backend is not None:
                parallel_kwargs['prefer'] = joblib_backend
            results = Parallel(**parallel_kwargs)(
                    delayed(_proc)(*args) for args in task_iter)

        if self.t is None:
            t = tqdm(desc=self.current_job, total=self.n_splits * self.n_repeats)
        else:
            t = self.t
            t.desc = self.current_job

        if window is None:
            for result, rep, fold in results:
                out[rep, fold] = result
                t.update()
        else:
            for result, rep, fold in results:
                out[:, rep, fold] = result
                t.update()

        if self.t is None:
            t.close()

        return out


def _proc(train_idx, test_idx, lab, orig_data, pid, n_splits, cats, window,
                  step, oversample, model, xp, search_factory=None,
                  random_state=None, min_non_nan_outer: int = 2,
                  mode: str = 'cm'):
    """Generic fold processor: returns cm or score per fold/window depending on mode.

    mode: 'cm' -> confusion matrix; 'score' -> estimator.score value.
    """

    def _eval(x_flat):
        x_train, x_test = (x_flat[:train_idx.shape[0]],
                           x_flat[train_idx.shape[0]:])
        est = _get_best_estimator(x_train, y_train, model, search_factory,
                                  n_splits, random_state, min_non_nan_outer)
        if mode == 'cm':
            pred = est.predict(x_test)
            return confusion_matrix(y_test, pred, label_cats, namespace=xp)
        elif mode == 'score':
            return est.score(x_test, y_test)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    label_cats = xp.asarray(list(cats.values()))
    x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, orig_data,
                                             lab, label_cats, 0, oversample,
                                             xp)

    x_stacked, in_shape = _reshape_for_loopwise(model, x_stacked)
    rep, fold = divmod(pid, n_splits)
    out = _map_over_windows(x_stacked, window, step, in_shape, _eval, xp)
    return out, rep, fold


def _reshape_for_loopwise(model, x_stacked):
    first_step = getattr(model, 'model').steps[0][1]
    if isinstance(first_step, LoopwiseTransformer):
        x_stacked = x_stacked.swapaxes(1, first_step.loop_dim)
        first_step.loop_dim = 1
        in_shape = x_stacked.shape[:2] + (-1,)
    else:
        in_shape = x_stacked.shape[:1] + (-1,)
    return x_stacked, in_shape


def _get_best_estimator(x_train, y_train, model, search_factory, n_splits,
                        random_state, min_non_nan_outer):
    if search_factory is not None:
        splitter = MinimumNaNSplit(
            n_splits=max(2, n_splits - 1),
            n_repeats=1,
            random_state=(0 if random_state is None else random_state),
            min_non_nan=max(1, int(min_non_nan_outer) - 1),
            which='train'
        )
        search = search_factory(estimator=model, cv=splitter, scoring=None)
        search.fit(x_train, y_train)
        return search.best_estimator_
    model.fit(x_train, y_train)
    return model


def _map_over_windows(x_stacked, window, step, in_shape, fn, xp):
    if window is None:
        return fn(x_stacked.reshape(in_shape))
    windowed = sliding_window_view(x_stacked, window, axis=-1, subok=True)[
               ..., ::step, :]
    outs = [fn(windowed[..., i, :].reshape(in_shape))
            for i in range(windowed.shape[-2])]
    first = outs[0]
    if hasattr(first, 'shape'):
        return xp.stack(outs, axis=0)
    return xp.asarray(outs)


def confusion_matrix(
    y_true, y_pred, labels=None, namespace=None
):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

        .. versionadded:: 0.18

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.

    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <https://en.wikipedia.org/wiki/Confusion_matrix>`_
           (Wikipedia and other references may use a different
           convention for axes).

    Examples
    --------
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]], dtype=int32)

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]], dtype=int32)

    In the binary case, we can extract true positives, etc. as follows:

    >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    >>> (tn, fp, fn, tp)
    (0, 2, 1, 1)
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]], dtype=int32)
    """
    if namespace is not None:
        xp = namespace
    elif isinstance(y_true, list) or isinstance(y_pred, list):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        xp = np
    else:
        xp = array_namespace(y_true, y_pred)

    if labels is None:
        labels, y_true_indices = xp.unique(y_true, return_inverse=True)
    else:
        labels = xp.array(labels)
        y_true_indices = xp.searchsorted(labels, y_true)

    y_pred_indices = xp.searchsorted(labels, y_pred)

    n_labels = labels.shape[0]
    cm = xp.zeros((n_labels, n_labels), dtype=xp.int32)
    xp.add.at(cm, (y_true_indices, y_pred_indices), 1)
    return cm


def nan_common_denom(array: LabeledArray, sort: bool = True,
                     trials_ax: int = 1, min_trials: int = 0,
                     ch_ax: int = 0, crop_trials: bool = True,
                     verbose: bool = False) -> LabeledArray:
    """Remove trials with NaNs from all channels.

    This function processes a LabeledArray to remove trials containing NaN
     values, with options for sorting, specifying axes, and setting minimum
      trial counts.

    Parameters
    ----------
    array : LabeledArray
        The input array to process.
    sort : bool, optional
        Whether to sort trials by NaN presence, by default True.
    trials_ax : int, optional
        The axis containing trials, by default 1.
    min_trials : int, optional
        Minimum number of trials to keep, by default 0.
    ch_ax : int, optional
        The axis containing channels, by default 0.
    crop_trials : bool, optional
        Whether to crop trials to the minimum number, by default True.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    LabeledArray
        The processed array with NaN trials removed.

    Examples
    --------
    >>> import numpy as np
    >>> from ieeg.arrays.label import LabeledArray
    >>> data = np.array([[1, 2, np.nan], [4, 5, 6], [7, np.nan, 9]])
    >>> labels = [['ch1', 'ch2', 'ch3'], ['trial1', 'trial2', 'trial3']]
    >>> array = LabeledArray(data, labels)
    >>> processed_array = nan_common_denom(array, sort=True, trials_ax=1,
    ... ch_ax=0, min_trials=3, crop_trials=True, verbose=True)
    Lowest trials 2 at ch1
    Channels excluded (too few trials): ['ch1', 'ch3']
    """
    others = [i for i in range(array.ndim) if ch_ax != i != trials_ax]
    isn = np.isnan(array.__array__())
    nan_trials = np.any(isn, axis=tuple(others))

    # Sort the trials by whether they are nan or not
    if sort:
        order = np.argsort(nan_trials, axis=1)
        old_shape = list(order.shape)
        new_shape = [1 if ch_ax != i != trials_ax else old_shape.pop(0)
                     for i in range(array.ndim)]
        order = np.reshape(order, new_shape)
        data = np.take_along_axis(array, order, axis=trials_ax)
        # data = LabeledArray(data, copy(array.labels))
    else:
        data = array

    ch_tnum = array.shape[trials_ax] - np.sum(nan_trials, axis=1)
    ch_min = ch_tnum.min()
    if verbose:
        print(f"Lowest trials {ch_min} at "
              f"{array.labels[ch_ax][ch_tnum.argmin()]}")

    ntrials = max(ch_min, min_trials)
    if ch_min < min_trials:
        # data = data.take(np.where(ch_tnum >= ntrials)[0], ch_idx)
        ch = np.array(array.labels[ch_ax])[ch_tnum < ntrials].tolist()
        if verbose:
            print(f"Channels excluded (too few trials): {ch}")

    # data = data.take(np.arange(ntrials), trials_idx)
    idx = [np.arange(ntrials) if i == trials_ax and crop_trials
           else np.arange(s) for i, s in enumerate(array.shape)]
    idx[ch_ax] = np.where([ch_tnum >= ntrials])[1]

    return data[np.ix_(*idx)]


def sample_fold(train_idx: Array, test_idx: Array,
                x_data: Array, labels: Array, unique: Array,
                axis: int, oversample: bool, xp) -> tuple[Array, Array, Array]:
    """Sample a fold of data for cross-validation.

    Parameters
    ----------
    train_idx : Array
        Indices of the training data.
    test_idx : Array
        Indices of the test data.
    x_data : Array
        The data to be sampled.
    labels : Array
        Labels corresponding to the data.
    unique : Array
        Unique labels to be used for oversampling.
    axis : int
        Axis along which to stack the data.
    oversample : bool
        Whether to oversample the training data.
    xp : module
        The array namespace (numpy or cupy).

    Returns
    -------
    tuple[Array, Array, Array]
        Stacked data, training labels, and test labels.
    """

    # make first and only copy of x_data
    idx_stacked = xp.concatenate((train_idx, test_idx))
    idx = tuple(slice(None) if i != axis else idx_stacked
                for i in range(x_data.ndim))
    x_stacked = x_data[idx]
    y_stacked = labels[idx_stacked]

    # define train and test as views of x_stacked
    sep = train_idx.shape[0]
    y_train, y_test = y_stacked[:sep], y_stacked[sep:]

    if not oversample:
        return x_stacked, y_train, y_test

    idx1 = tuple(slice(None) if i != axis else slice(None, sep)
                 for i in range(x_data.ndim))
    idx2 = tuple(slice(None) if i != axis else slice(sep, None)
                 for i in range(x_data.ndim))
    x_train, x_test = x_stacked[idx1], x_stacked[idx2]
    mixup2(x_train, y_train, axis)
    # idx = [slice(None) for _ in range(x_data.ndim)]
    # for i in unique:
    #     # fill in train data nans with random combinations of
    #     # existing train data trials (mixup)
    #     isin = y_train == i
    #     idx[axis] = isin
    #     out = x_train[tuple(idx)]
    #     if out.size != 0:
    #         mixup(out, axis)
    #         if is_torch(xp):
    #             idx3 = tuple(None if j != axis else
    #                          slice(None) for j in range(x_data.ndim))
    #             x_train.masked_scatter_(isin[idx3], out)
    #         else:
    #             x_train[tuple(idx)] = out

    # fill in test data nans with noise from distribution
    is_nan = xp.isnan(x_test)
    if is_torch(xp):
        normal = xp.distributions.normal.Normal(0, 1, dtype=x_data.dtype)
        x_test.masked_scatter_(is_nan, normal.sample((xp.sum(is_nan),)))
    else:
        x_test[is_nan] = xp.random.normal(0, 1, int(xp.sum(is_nan)))

    return x_stacked, y_train, y_test


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    """Flatten features in an array.

    This function swaps the first axis with the observation axis and reshapes
    the array to flatten all dimensions except the first one.

    Parameters
    ----------
    arr : np.ndarray
        The input array to flatten.
    obs_axs : int, optional
        The axis containing observations, by default -2.

    Returns
    -------
    np.ndarray
        The flattened array with shape (n_observations, n_features).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> arr = np.random.rand(4, 3, 2)
    >>> flatten_features(arr, obs_axs=-2)
    array([[0.5488135 , 0.71518937, 0.43758721, 0.891773  , 0.56804456,
            0.92559664, 0.77815675, 0.87001215],
           [0.60276338, 0.54488318, 0.96366276, 0.38344152, 0.07103606,
            0.0871293 , 0.97861834, 0.79915856],
           [0.4236548 , 0.64589411, 0.79172504, 0.52889492, 0.0202184 ,
            0.83261985, 0.46147936, 0.78052918]])
    """
    out = arr.swapaxes(0, obs_axs)
    return out.reshape(out.shape[0], -1)


def classes_from_labels(labels: np.ndarray, delim: str = '-', which: int = 0,
                        crop: slice = slice(None), cats: dict = None
                        ) -> tuple[dict, np.ndarray]:
    """Extract class IDs from string labels.

    This function processes string labels to extract class IDs using a
     delimiter, and returns a dictionary mapping class names to indices and an
      array of class indices.

    Parameters
    ----------
    labels : np.ndarray
        Array of string labels to process.
    delim : str, optional
        Delimiter to split the labels, by default '-'.
    which : int, optional
        Which part of the split label to use, by default 0.
    crop : slice, optional
        Slice to apply to each label part, by default slice(None).
    cats : dict, optional
        Existing category mapping to use. If None, a new mapping is created.

    Returns
    -------
    tuple[dict, np.ndarray]
        A tuple containing:
        - Dictionary mapping class names to indices
        - Array of class indices corresponding to the input labels

    Examples
    --------
    >>> labels = np.array(['cat-dog', 'dog-cat', 'cat-bird'])
    >>> classes_from_labels(labels, delim='-')
    ({'cat': 0, 'dog': 1}, array([0, 1, 0]))
    """
    class_ids = np.array([k.split(delim, )[which][crop] for k in labels])
    if cats is None:
        classes = {k: i for i, k in enumerate(np.unique(class_ids))}
        return classes, np.array([classes[k] for k in class_ids])
    else:
        return cats, np.array([cats[k] for k in class_ids])


def flatten_list(nested_list: list[list[str] | str]) -> list[str]:
    """Flatten a nested list of strings.

    This function takes a list that may contain both strings and lists of
    strings, and returns a single flat list containing all the strings.

    Parameters
    ----------
    nested_list : list[list[str] | str]
        A list containing strings and/or lists of strings.

    Returns
    -------
    list[str]
        A flattened list containing all strings from the input.

    Examples
    --------
    >>> flatten_list(['a', ['b', 'c'], 'd', 'd'])
    ['a', 'b', 'c', 'd']
    """
    result = []
    for item in nested_list:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, (list, tuple, set)):
            result.extend(flatten_list(item))
        else:
            raise TypeError(f"Unsupported type: {type(item)}")
    return sorted(set(result), key=result.index)


def plot_all_scores(all_scores: dict[str, np.ndarray],
                    conds: list[str], idxs: dict[str, list[int]],
                    colors: list[list[float]], suptitle: str = None,
                    fig: plt.Figure = None, axs: plt.Axes = None,
                    ylims: tuple[float, float] = (0.1, 0.8), **plot_kwargs
                    ) -> tuple[plt.Figure, plt.Axes]:
    """Plot scores for different conditions and categories.

    This function creates plots of scores for different experimental conditions
    and categories, setting up appropriate axes and labels.

    Parameters
    ----------
    all_scores : dict[str, np.ndarray]
        Dictionary mapping score names to score arrays.
    conds : list[str]
        List of condition names to plot.
    idxs : dict[str, list[int]]
        Dictionary mapping category names to indices.
    colors : list[list[float]]
        List of colors for each category.
    suptitle : str, optional
        Super title for the figure, by default None.
    fig : plt.Figure, optional
        Existing figure to plot on, by default None.
    axs : plt.Axes, optional
        Existing axes to plot on, by default None.
    ylims : tuple[float, float], optional
        Y-axis limits, by default (0.1, 0.8).
    **plot_kwargs
        Additional keyword arguments passed to plot_dist.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects containing the plots.
    """
    names = list(idxs.keys())
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, len(conds))
    elif axs is None:
        axs = fig.get_axes()
    if len(conds) == 1:
        axs = [axs]
    for color, name, idx in zip(colors, names, idxs.values()):
        for cond, ax in zip(conds, axs):
            if isinstance(cond, list):
                cond = "-".join(cond)
            ax.set_title(cond)
            if cond == 'resp':
                times = (-0.9, 0.9)
                ax.set_xlabel("Time from response (s)")
            else:
                times = (-0.4, 1.4)
                if 'aud' in cond:
                    ax.set_xlabel("Time from stim (s)")
                elif 'go' in cond:
                    ax.set_xlabel("Time from go (s)")
                else:
                    raise ValueError("Condition not recognized")
            pl_sc = np.reshape(all_scores["-".join([name, cond])],
                               (all_scores["-".join([name, cond])].shape[0],
                                -1)).T
            plot_dist(pl_sc, mode='std', times=times,
                      color=color, label=name, ax=ax,
                      **plot_kwargs)
            if name is names[-1]:
                ax.legend()
                ax.set_title(cond)
                ax.set_ylim(*ylims)

    axs[0].set_ylabel("Accuracy (%)")
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig, axs
