from typing import Any, Generator

from numpy import ndarray
from sklearn import config_context
try:
    import cupy as cp
except ImportError:
    cp = None

from ieeg.decoding.models import PcaLdaClassification
from ieeg.arrays.label import LabeledArray
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.arrays.api import array_namespace, Array, is_torch, is_numpy
from ieeg.arrays.reshape import sliding_window_view
from ieeg.calc.fast import mixup
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
                 **kwargs):
        # self.model = PcaLdaClassification(**kwargs)
        self.kwargs = kwargs
        MinimumNaNSplit.__init__(self, n_splits, n_repeats,
                                 None, min_samples, which)
        self.categories = categories
        self.current_job = "Repetitions"

    def cv_cm(self, x_data: Array, labels: Array,
              normalize: str = None, obs_axs: int = -2, n_jobs: int = 1,
              average_repetitions: bool = True, window: int = None,
              shuffle: bool = False, oversample: bool = True, step: int = 1) -> Array:
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
        >>> decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
        ...             5, 10, explained_variance=0.8, da_type='lda')
        >>> X = np.random.randn(100, 50, 100)
        >>> labels = np.random.randint(1, 5, 50)
        >>> decoder.cv_cm(X, labels, normalize='true')
        >>> decoder = Decoder({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
        ...             5, 10, explained_variance=0.8, da_type='lda')
        >>> decoder.cv_cm(X, labels, normalize='true', window=20, step=5)
        >>> decoder.cv_cm(X, labels, normalize='true', window=20, step=5,
        ...     shuffle=True)
        >>> import cupy as cp
        >>> X = cp.random.randn(100, 100, 50, 100)
        >>> X[0, 0, 0, :] = np.nan
        >>> labels = cp.random.randint(1, 5, 50)
        >>> with config_context(array_api_dispatch=True):
        ...     decoder.cv_cm(X, labels, normalize='true')

        """
        assert all(l in self.categories.values() for l in labels), \
            "Labels must be in the categories"
        xp = array_namespace(x_data)
        n_cats = len(self.categories)
        out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)
        if window is not None:
            out_shape = ((x_data.shape[-1] - window) // step + 1,) + out_shape
        mats = xp.zeros(out_shape, dtype=xp.int16)
        data = x_data.swapaxes(0, obs_axs)

        if shuffle:
            isnan = xp.isnan(data)
            std = float(xp.nanstd(data, dtype='f8'))
            data[isnan] = xp.random.normal(0, 3 * std, int(xp.sum(isnan, dtype='i8')))
            # shuffled label pool
            label_stack = [labels.copy() for _ in range(self.n_repeats)]
            for i in range(self.n_repeats):
                self.shuffle_labels(data, label_stack[i], 0)

            # build the test/train indices from the shuffled labels for each
            # repetition, then chain together the repetitions
            # splits = (train, test)
            idxs = ((self.split(data, l), l) for l in label_stack)
            idxs = ((itertools.islice(s, self.n_splits),
                     itertools.repeat(l, self.n_splits))
                    for s, l in idxs)
            splits, label = zip(*idxs)
            splits = itertools.chain.from_iterable(splits)
            label = itertools.chain.from_iterable(label)
            idxs = zip(splits, label)

        else:
            idxs = ((splits, labels) for splits in self.split(data, labels))

        # loop over folds and repetitions
        if n_jobs == 1:
            results = (proc(train_idx, test_idx, l, data, i,
                            self.n_splits, self.categories, window, step,
                            oversample, self.kwargs)
                       for i, ((train_idx, test_idx), l) in enumerate(idxs))
        else:
            results = Parallel(n_jobs=n_jobs, verbose=0, require='sharedmem',
                               return_as="generator_unordered")(
                    delayed(proc)(train_idx, test_idx, l, data, i,
                                  self.n_splits, self.categories, window, step,
                                  oversample, self.kwargs)
                    for i, ((train_idx, test_idx), l) in enumerate(idxs))

        # Collect the results
        t = tqdm(desc=self.current_job, total=self.n_splits * self.n_repeats)
        for result, rep, fold in results:
            mats[..., rep, fold, :, :] = result
            t.update()
        t.close()

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

def proc(train_idx, test_idx, l, orig_data, pid, n_splits, cats, window, step, oversample, model_kwargs):
    xp = array_namespace(orig_data)
    label_cats = xp.asarray(list(cats.values()))
    x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, orig_data, l, label_cats, 0, oversample, xp)
    model = PcaLdaClassification(**model_kwargs)

    def _fit_predict(x_flat):
        x_train, x_test = x_flat[:train_idx.shape[0]], x_flat[train_idx.shape[0]:]
        # fit model and score results
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return confusion_matrix(y_test, pred, label_cats, namespace=xp)

    rep, fold = divmod(pid, n_splits)
    if window is None:
        x_flattened = x_stacked.reshape(x_stacked.shape[0], -1)
        return _fit_predict(x_flattened), rep, fold

    windowed = sliding_window_view(x_stacked, window, axis=-1, subok=True)[..., ::step, :]
    swapped = xp.moveaxis(windowed.swapaxes(-1, -2).reshape(
        windowed.shape[0], -1, windowed.shape[-2]),-1, 0)

    if is_numpy(xp):
        func = np.vectorize(_fit_predict,
                            signature='(a,b) -> (d,d)',
                            otypes=[xp.uint8])
        out = func(swapped)
    else:
        out = xp.zeros((windowed.shape[-2], label_cats.shape[0],
                        label_cats.shape[0]), dtype=xp.uint8)
        for i in range(windowed.shape[-2]):
            x_window = windowed[..., i, :]
            out[i] = _fit_predict(x_window.reshape(x_window.shape[0], -1))

    return out, rep, fold


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
           [1, 0, 2]])

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    In the binary case, we can extract true positives, etc. as follows:

    >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    >>> (tn, fp, fn, tp)
    (np.int64(0), np.int64(2), np.int64(1), np.int64(1))
    >>> confusion_matrix(y_true, y_pred)
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


def get_scores(array, decoder: Decoder, idxs: list[list[int]], conds: list[str],
               names: list[str], on_gpu: bool = False,
               **decoder_kwargs) -> Generator[ndarray, Any, None]:
    ax = array.ndim - 2
    for i, idx in enumerate(idxs):
        all_conds = flatten_list(conds)
        x_data = extract(array, all_conds, ax, idx, decoder.n_splits,
                         False)

        for cond in conds:
            if isinstance(cond, list):
                X = concatenate_conditions(x_data, cond, 0, ax-1)
                cond = "-".join(cond)
            else:
                X = x_data[cond,].dropna()
            cats, labels = classes_from_labels(X.labels[ax-2],
                                               crop=slice(0, 4),
                                               cats=decoder.categories)

            # Decoding
            decoder.current_job = "-".join([names[i], cond])
            if on_gpu:
                if cp is None:
                    raise ImportError("CuPy is not installed.")
                with config_context(array_api_dispatch=True,
                                    enable_metadata_routing=True,
                                    skip_parameter_validation=True):
                    data = cp.asarray(X.__array__())
                    labels = cp.asarray(labels)
                    score = decoder.cv_cm(data, labels, **decoder_kwargs)
                yield score.get()
            else:
                yield decoder.cv_cm(X.__array__(), labels, **decoder_kwargs)


def extract(array: LabeledArray, conds: list[str], trial_ax: int,
            idx: list[int] = slice(None), common: int = 5,
            crop_nan: bool = False) -> LabeledArray:
    """Extract data from GroupData object"""
    reduced = array[conds,][:,:,idx]
    reduced = reduced.dropna()
    # also sorts the trials by nan or not
    reduced = nan_common_denom(reduced, True, trial_ax, common, 2, crop_nan)
    comb = reduced.combine((1, trial_ax))
    return comb.dropna()


def nan_common_denom(array: LabeledArray, sort: bool = True, trials_ax: int = 1, min_trials: int = 0,
                     ch_ax: int = 0, crop_trials: bool = True, verbose: bool = False):
    """Remove trials with NaNs from all channels"""
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
        data = np.take_along_axis(array.__array__(), order, axis=trials_ax)
        data = LabeledArray(data, array.labels.copy())
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
    # mixup2(x_train, labels[:sep], axis)
    idx = [slice(None) for _ in range(x_data.ndim)]
    for i in unique:
        # fill in train data nans with random combinations of
        # existing train data trials (mixup)
        isin = y_train == i
        idx[axis] = isin
        out = x_train[tuple(idx)]
        if out.size != 0:
            mixup(out, axis)
            if is_torch(xp):
                idx3 = tuple(None if j != axis else
                             slice(None) for j in range(x_data.ndim))
                x_train.masked_scatter_(isin[idx3], out)
            else:
                x_train[tuple(idx)] = out


    # fill in test data nans with noise from distribution
    is_nan = xp.isnan(x_test)
    if is_torch(xp):
        normal = xp.distributions.normal.Normal(0, 1, dtype=x_data.dtype)
        x_test.masked_scatter_(is_nan, normal.sample((xp.sum(is_nan),)))
    else:
        x_test[is_nan] = xp.random.normal(0, 1, int(xp.sum(is_nan)))

    return x_stacked, y_train, y_test

def concatenate_conditions(data, conditions, axis=1, trial_axis=2):
    """Concatenate data for all conditions"""
    concatenated_data = np.take(data, conditions[0], axis=axis)
    for condition in conditions[1:]:
        cond_data = np.take(data, condition, axis=axis)
        concatenated_data = concatenated_data.concatenate(cond_data, axis=trial_axis - 1)
    return concatenated_data


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    out = arr.swapaxes(0, obs_axs)
    return out.reshape(out.shape[0], -1)


def classes_from_labels(labels: np.ndarray, delim: str = '-', which: int = 0,
                        crop: slice = slice(None), cats: dict = None) -> tuple[dict, np.ndarray]:
    class_ids = np.array([k.split(delim, )[which][crop] for k in labels])
    if cats is None:
        classes = {k: i for i, k in enumerate(np.unique(class_ids))}
        return classes, np.array([classes[k] for k in class_ids])
    else:
        return cats, np.array([cats[k] for k in class_ids])


def scale(X, xmax: float, xmin: float):
    return (X - xmin) / (xmax - xmin)


def flatten_list(nested_list: list[list[str] | str]) -> list[str]:
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    return flat_list


def decode_and_score(decoder, data, labels, scorer='acc', **decoder_kwargs):
    """Perform decoding and scoring"""
    mats = decoder.cv_cm(data.__array__(), labels, **decoder_kwargs)
    if scorer == 'acc':
        score = np.mean(mats.T[np.eye(len(decoder.categories)).astype(bool)].T, axis=-1)
    else:
        raise NotImplementedError("Only accuracy is implemented")
    return score


def plot_all_scores(all_scores: dict[str, np.ndarray],
                    conds: list[str], idxs: dict[str, list[int]],
                    colors: list[list[float]], suptitle: str = None,
                    fig: plt.Figure = None, axs: plt.Axes = None,
                    ylims: tuple[float, float]= (0.1, 0.8), **plot_kwargs
                    ) -> tuple[plt.Figure, plt.Axes]:
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

