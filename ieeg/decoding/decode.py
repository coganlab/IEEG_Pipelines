try:
    import cupy as cp
except ImportError:
    cp = None

from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import make_scorer
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal


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
              parameter_grid=None,
              random_state: int | None = None
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
        >>> dec = Decoder(cats, n_splits=5, n_repeats=1, model=model)
        >>> grid = {'explained_variance': [0.4, 0.95]}
        >>> dec.cv_cm(X, y, obs_axs=1, parameter_grid=grid, normalize='true')
        >>> dec.cv_cm(X, y, obs_axs=1, parameter_grid=grid, window=20,
        ... step=5, normalize='true')
        """
        # # if model is a pipeline containing pca and weights is not None,
        # # change pca to weighted pca
        assert all(lab in self.categories.values() for lab in labels.tolist()), \
            "Labels must be in the categories"
        xp = array_namespace(x_data)
        config = _ProcessConfig(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            oversample=oversample,
            categories=self.categories,
            namespace=xp,
            min_non_nan=self.min_non_nan,
            mode=Mode.CM,
            which=self.which,
            state=random_state,
            parameter_grid=parameter_grid,
            window=window,
            step=step
        )

        mats = self._run_cv(
            x_data, labels, obs_axs, n_jobs, shuffle, config
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
                     parameter_grid=None, random_state: int | None = None
                    ) -> Array:
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
        config = _ProcessConfig(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            oversample=oversample,
            categories=self.categories,
            namespace=xp,
            min_non_nan=self.min_non_nan,
            mode=Mode.SCORE,
            which=self.which,
            state=random_state,
            parameter_grid=parameter_grid,
            window=window,
            step=step
        )

        acc = self._run_cv(
            x_data, labels, obs_axs, n_jobs, shuffle, config
        )

        # average the repetitions
        if average_repetitions:
            acc = xp.mean(acc, axis=1)

        return acc

    def _run_cv(self, x_data: Array, labels: Array, obs_axs: int, n_jobs: int,
                shuffle: bool, config: '_ProcessConfig') -> Array:
        assert all(lab in config.categories.values() for lab in labels.tolist()), \
            "Labels must be in the categories"
        xp = config.namespace
        data = x_data.swapaxes(0, obs_axs)

        if shuffle:
            isnan = xp.isnan(data)
            std = float(xp.std(data[isnan], dtype='f8'))
            data[isnan] = xp.random.normal(0, 3 * std, int(xp.sum(isnan,
                                                                  dtype='i8')))
            label_stack = [labels.copy() for _ in range(config.n_repeats)]
            for i in range(config.n_repeats):
                self.shuffle_labels(data, label_stack[i], 0)
            idxs = ((self.split(data, lab), lab) for lab in label_stack)
            idxs = ((itertools.islice(s, config.n_splits),
                     itertools.repeat(l, config.n_splits))
                    for s, l in idxs)
            splits, label = zip(*idxs)
            splits = itertools.chain.from_iterable(splits)
            label = itertools.chain.from_iterable(label)
            idxs = zip(splits, label)
        else:
            idxs = ((splits, labels) for splits in self.split(data, labels))

        shape, dtype = config.shape_builder(data)
        out = xp.zeros(shape, dtype=dtype)

        # Precompute sliding-window view once if needed
        total = config.n_splits * config.n_repeats
        if config.window is not None:
            data_w = sliding_window_view(data, config.window, axis=-1, subok=True)[..., ::config.step, :]
            n_windows = int(data_w.shape[-2])
            # Build tasks across (split, window)
            task_iter = (
                (train_idx, test_idx, l, data, i, clone(self.model), data_w, w)
                for i, ((train_idx, test_idx), l) in enumerate(idxs)
                for w in range(n_windows)
            )
            total *= n_windows
        else:
            task_iter = (
                (train_idx, test_idx, l, data, i, clone(self.model), None, None)
                for i, ((train_idx, test_idx), l) in enumerate(idxs)
            )

        if n_jobs == 1:
            results = (config.proc(*args) for args in task_iter)
        else:
            parallel_kwargs = dict(n_jobs=n_jobs, verbose=0,
                                   require='sharedmem',
                                   return_as="generator_unordered")
            results = Parallel(**parallel_kwargs)(
                    delayed(config.proc)(*args) for args in task_iter)

        if self.t is None:
            t = tqdm(desc=self.current_job, total=total)
        else:
            t = self.t
            t.desc = self.current_job

        if config.window is None:
            for result, rep, fold in results:
                out[rep, fold] = result
                t.update()
        else:
            for result, rep, fold, w in results:
                out[w, rep, fold] = result
                t.update()

        if self.t is None:
            t.close()

        return out


class Mode(Enum):
    CM = 'cm'
    SCORE = 'score'


@dataclass(slots=True)
class _ProcessConfig:
    n_splits: int
    n_repeats: int
    oversample: bool
    categories: dict[str, int]
    namespace: object
    min_non_nan: int
    mode: Mode
    which: str = 'train'
    state: int | None = None
    parameter_grid: dict | list[dict] = None
    window: int = None
    step: int = None
    shape_builder: Callable = field(init=False)
    eval: Callable = field(init=False)
    trainer: Callable = field(init=False)
    labels_vec: Array = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.mode is Mode.CM:
            self.shape_builder = self._shape_builder
            self.eval = self._eval_cm
        elif self.mode is Mode.SCORE:
            self.shape_builder = self._shape_builder_acc
            self.eval = self._eval_score
        else:
            raise ValueError("mode must be Mode.CM or Mode.SCORE")

        # bind trainer once to avoid per-call branching
        if self.parameter_grid is not None:
            self.trainer = self._train_search
        else:
            self.trainer = self._train_fit
        # cache category labels vec for CM
        self.labels_vec = self.namespace.asarray(list(self.categories.values()))

    def _shape_builder_acc(self, data_local: Array):
        shape = (self.n_repeats, self.n_splits)
        if self.window is not None:
            shape = ((data_local.shape[-1] - self.window) // self.step + 1,) + shape
        return shape, self.namespace.float32

    def _shape_builder(self, data_local: Array):
        n_c = len(self.categories)
        shape = (self.n_repeats, self.n_splits, n_c, n_c)
        if self.window is not None:
            shape = ((data_local.shape[-1] - self.window) // self.step + 1,) + shape
        return shape, self.namespace.uint16

    def _eval_cm(self, est: BaseEstimator, x_test: Array, y_test: Array):
        pred = est.predict(x_test)
        return confusion_matrix(y_test, pred, self.labels_vec,
                                namespace=self.namespace)

    def _eval_score(self, est: BaseEstimator, x_test: Array, y_test: Array):
        return est.score(x_test, y_test)

    def search_factory(self, estimator: BaseEstimator):
        # Bayesian hyperparameter search using skopt.BayesSearchCV
        try:
            from skopt import BayesSearchCV
            from skopt.space import Categorical, Integer, Real
        except Exception as e:
            raise ImportError("scikit-optimize is required for BayesSearchCV. Install 'scikit-optimize'.") from e
        splitter = MinimumNaNSplit(
            n_splits=max(2, self.n_splits - 1),
            n_repeats=1,
            random_state=(0 if self.state is None else self.state),
            min_non_nan=max(2, int(self.min_non_nan) - 1),
            which=self.which
        )
        # Convert simple grids (lists) to Categorical spaces; leave others as-is
        spaces = self.parameter_grid
        if isinstance(self.parameter_grid, dict):
            converted = {}
            for k, v in self.parameter_grid.items():
                if isinstance(v, list):
                    converted[k] = Categorical(v)
                else:
                    converted[k] = v
            spaces = converted
        elif isinstance(self.parameter_grid, list):
            converted_list = []
            for space in self.parameter_grid:
                if isinstance(space, dict):
                    conv = {}
                    for k, v in space.items():
                        conv[k] = Categorical(v) if isinstance(v, list) else v
                    converted_list.append(conv)
                else:
                    converted_list.append(space)
            spaces = converted_list

        return BayesSearchCV(
            estimator=estimator,
            search_spaces=spaces,
            cv=splitter,
            # Avoid nested parallelism; outer level should handle parallelism
            n_jobs=5,
            scoring=make_scorer(balanced_accuracy_score),
            n_iter=25,
            n_points=5
        )

    def _train_search(self, estimator: BaseEstimator, x_train: Array, y_train: Array) -> BaseEstimator:
        search = self.search_factory(estimator)
        # Disable sklearn metadata routing to avoid passing unsupported 'groups'
        # into skopt.BayesSearchCV.fit
        with config_context(enable_metadata_routing=False):
            search.fit(x_train, y_train)
        return search.best_estimator_

    def _train_fit(self, estimator: BaseEstimator, x_train: Array, y_train: Array) -> BaseEstimator:
        estimator.fit(x_train, y_train)
        return estimator


    def proc(self, train_idx: Array, test_idx: Array, lab: Array,
             orig_data: Array, pid: int, model: BaseEstimator,
             data_w: Array | None = None, w: int | None = None):
        """Generic fold processor: returns cm or score per fold/window depending on mode.

        mode: 'cm' -> confusion matrix; 'score' -> estimator.score value.
        """

        def _eval(train: Array, test: Array):
            # fit or inner-search using bound trainer
            est = self.trainer(model, train, y_train)
            return self.eval(est, test, y_test)

        xp = self.namespace
        # Build train/test views directly; oversampling/flattening are in-pipeline
        idx_tr = (train_idx,) + tuple(slice(None) for _ in range(orig_data.ndim - 1))
        idx_te = (test_idx,) + tuple(slice(None) for _ in range(orig_data.ndim - 1))
        X_train = orig_data[idx_tr]
        X_test = orig_data[idx_te]
        y_train = lab[train_idx]
        y_test = lab[test_idx]
        rep, fold = divmod(pid, self.n_splits)
        # Mapper contracts: for no-window simply call _eval once; for windowed,
        # use precomputed data_w and selected window index w.
        if self.window is None:
            out = _eval(X_train, X_test)
            return out, rep, fold
        else:
            if data_w is None or w is None:
                raise ValueError("Windowed processing expects precomputed windows and window index.")
            # Index precomputed window view for train/test
            idx_tr_w = (train_idx,) + tuple(slice(None) for _ in range(data_w.ndim - 1))
            idx_te_w = (test_idx,) + tuple(slice(None) for _ in range(data_w.ndim - 1))
            Xw_tr = data_w[idx_tr_w][..., w, :]
            Xw_te = data_w[idx_te_w][..., w, :]
            out = _eval(Xw_tr, Xw_te)
            return out, rep, fold, w


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


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False):
    """Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.

    The best value is 1 and the worst value is 0 when ``adjusted=False``.

    Read more in the :ref:`User Guide <balanced_accuracy_score>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, while keeping perfect performance at a score
        of 1.

    Returns
    -------
    balanced_accuracy : float
        Balanced accuracy score.

    See Also
    --------
    average_precision_score : Compute average precision (AP) from prediction
        scores.
    precision_score : Compute the precision score.
    recall_score : Compute the recall score.
    roc_auc_score : Compute Area Under the Receiver Operating Characteristic
        Curve (ROC AUC) from prediction scores.

    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.

    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.

    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    np.float64(0.625)
    """
    xp = array_namespace(y_true, y_pred)
    C = confusion_matrix(y_true, y_pred, namespace=xp)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = xp.diag(C) / C.sum(axis=1)
    if xp.any(xp.isnan(per_class)):
        # warnings.warn("y_pred contains classes not in y_true")
        per_class = per_class[~np.isnan(per_class)]
    score = xp.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

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


def sample_fold(*args, **kwargs):
    raise NotImplementedError("sample_fold is obsolete; pipeline handles oversample/flatten.")


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


if __name__ == "__main__":
    # Minimal Dask-backed test run for cv_accuracy
    from dask.distributed import Client
    import joblib

    # Prefer LocalCluster for a lightweight test
    from dask.distributed import LocalCluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)


    # Synthetic data
    rng = np.random.RandomState(0)
    # Shape: (channels, trials, extra_dim, features)
    X = rng.randn(2, 60, 2, 32).astype(np.float32)
    y = rng.randint(0, 2, size=60)
    cats = {'a': 0, 'b': 1}

    # Model and decoder
    model = PcaLdaClassification(0.8, 'lda')
    dec = Decoder(cats, n_splits=3, n_repeats=1, model=model)

    # Run with Dask joblib backend if available
    print("Running cv_accuracy with joblib_backend='dask' (falls back if unavailable)...")
    with joblib.parallel_backend('dask'):
        acc = dec.cv_accuracy(
            X, y, obs_axs=1, n_jobs=-1,
            window=None, shuffle=False, oversample=True
        )
    print("cv_accuracy shape:", acc.shape)
    print("cv_accuracy sample:", np.asarray(acc))

    with joblib.parallel_backend('dask'):
        dec = Decoder(cats, n_splits=5, n_repeats=10, model=model)
        acc = dec.cv_cm(X, y, obs_axs=1, parameter_grid={
            'explained_variance': (0.4, 0.99)}, normalize='true')

    print(acc)

    # Clean up Dask client/cluster
    if client is not None:
        client.close()
        if 'cluster' in locals():
            cluster.close()
