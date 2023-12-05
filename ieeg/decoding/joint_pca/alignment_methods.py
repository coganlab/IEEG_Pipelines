""" Various methods for aligning microECoG datasets across patients.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from sklearn.decomposition import PCA
from functools import reduce
from ieeg.decoding.joint_pca.utils import cnd_avg, label2str
from numba import njit


class JointPCADecomp:
    def __init__(self, n_components=40, dim_red=PCA):
        """
        Initializes JointPCADecomp class with the number of latent
        components and the method for dimensionality reduction.

        Parameters
        ----------
        n_components : int, optional
            Number of components for dimensionality reduction i.e.
            dimensionality of latent space. Defaults to 40.
        dim_red : Callable, optional
            Dimensionality reduction function. Must implement
            sklearn-style fit_transform() function. Defaults to PCA.
        """

        self.n_components = n_components
        self.dim_red = dim_red

    def fit(self, X, y):
        """
        Learns source-specific (e.g. patient-specific) transformations to
        the shared latent space and stores transformations in self.transforms.

        Parameters
        ----------
        X : list of ndarray
            List of features from multiple sources to compute shared latent
            space.
        y : list of ndarray
            List of labels corresponding to feature sources. Must be the same
            length as features.
        """

        transforms = get_joint_PCA_transforms(X, y,
                                              n_components=self.n_components,
                                              dim_red=self.dim_red)
        self.transforms = transforms

    def transform(self, X, idx=-1):
        """
        Applies learned transformations to input data. Supports transforming
        a single, specified dataset or all-source datasets at once.

        Parameters
        ----------
        X : ndarray or list of ndarray
            Features to transform. If a list, the length must be equal to the
            number of learned transforms (i.e. transforming all sources).
            If an ndarray, a source-specific transformation is applied to the
            data, with the source specified by the idx input.
        idx : int, optional
            Index of saved transform list to apply to single source data,
            or -1 if applying to all sources. Defaults to -1.

        Raises
        ------
        IndexError
            Error if idx is too large to select a learned transform from the
            saved list.
        RuntimeError
            Error if fit() has not been called before calling transform().

        Returns
        -------
        ndarray or tuple
            Transformed data from single session if idx is not -1, or a tuple
            of containing:
                Transformed data from all sources input to the fit() method.
                Length will be equal to the number of learned transformations.
        """

        if not self._check_fit():
            raise RuntimeError('Must call fit() before transforming data.')
        if idx == -1:
            return self._transform_multiple(X)
        if idx >= len(self.transforms):
            raise IndexError('Input idx is greater than the number of learned '
                             'transforms. For transformation of data from a '
                             'specific session, provide the input idx as the '
                             'index of the session in the input list. If '
                             'transforming multiple sessions, set idx=-1 '
                             '(default).')
        return self._transform_single(X, idx)

    def fit_transform(self, X, y):
        """Fits the model with X and y and applies the learned transformations
        to X.

        Parameters
        ----------
        X : list of ndarray
            List of features from multiple sources to compute shared latent
            space.
        y : list of ndarray
            List of labels corresponding to feature sources. Must be the same
            length as features.

        Returns
        -------
        tuple
            Tuple containing transformed ndarray data from all sources input to
            the fit() method. Length will be equal to the number of learned
            transformations.
        """

        self.fit(X, y)
        return self.transform(X)

    def _transform_multiple(self, X):
        """
        Uses learned latent space transformations to transform data from all
        sources used to fit the decomposition.

        Parameters
        ----------
        X : list of ndarray
            List of features from multiple sources to compute shared latent
            space.

        Returns
        -------
        tuple of ndarray
            Tuple containing transformed ndarray data from all sources input to
             the fit() method. Length will be equal to the number of learned
             transformations.
        """

        transformed_lst = [0]*len(X)
        for i, (feats, transform) in enumerate(zip(X, self.transforms)):
            transform_feats = feats.reshape(-1, feats.shape[-1]) @ transform
            transformed_lst[i] = transform_feats.reshape(feats.shape[:-1] +
                                                         (-1,))
        return (*transformed_lst,)

    def _transform_single(self, X, idx):
        """
        Applies learned latent space transformations to data from a single
        source specified by idx.

        Parameters
        ----------
        X : ndarray
            Features to transform.
        idx : int
            Index of learned transforms to apply to X.

        Returns
        -------
        ndarray
            Features transformed to shared latent space.
        """

        transform = self.transforms[idx]
        transform_feats = X.reshape(-1, X.shape[-1]) @ transform
        return transform_feats.reshape(X.shape[:-1] + (-1,))

    def _check_fit(self):
        """Checks if the joint PCA decomposition has been fit to data.

        Returns
        -------
        boolean
            True if fit() has been called, False otherwise.
        """

        try:
            self.transforms
        except AttributeError:
            return False
        return True


class CCAAlign():

    def __init__(self, type='class', return_space='b_to_a'):
        self.type = type
        self.return_space = return_space

    def fit(self, X_a, X_b, y_a, y_b):
        L_a, L_b = reshape_latent_dynamics(X_a, X_b, y_a, y_b, type=self.type)
        M_a, M_b = CCA_align(L_a.T, L_b.T)
        self.M_a = M_a
        self.M_b = M_b

    def transform(self, X):
        if not self._check_fit():
            raise RuntimeError('Must call fit() before transforming data.')
        if self.return_space in ['b_to_a', 'a_to_b']:
            return self._transform_single(X)
        return self._transform_shared(X)

    def _transform_single(self, X):
        if self.return_space == 'b_to_a':
            return X @ self.M_b @ np.linalg.pinv(self.M_a)
        return X @ self.M_a @ np.linalg.pinv(self.M_b)

    def _transform_shared(self, X):
        return X[0] @ self.M_a, X[1] @ self.M_b

    def _check_fit(self):
        """Checks if the joint PCA decomposition has been fit to data.

        Returns
        -------
        boolean
            True if fit() has been called, False otherwise.
        """
        try:
            self.M_a
            self.M_b
        except AttributeError:
            return False
        return True


def get_joint_PCA_transforms(features, labels, n_components=40, dim_red=PCA):
    """
    Calculates a shared latent space across features from multiple patients
    or recording sessions.

    Uses the method described by Pandarinath et al. in
    https://www.nature.com/articles/s41592-018-0109-9 (2018) for pre-computing
    session specific read-in matrices (see Methods: Modifications to the LFADS
    algorithm for stitching together data from multiple recording sessions)

    Parameters
    ----------
    features : list
        List of features from multiple sources to compute shared latent space.
    labels : list
        List of labels corresponding to feature sources. Must be the same
        length as features.
    n_components : int, optional
        Number of components for dimensionality reduction i.e. dimensionality
        of latent space. Defaults to 40.
    dim_red : Callable, optional
        Dimensionality reduction function. Must implement sklearn-style
        fit_transform() function. Defaults to PCA.

    Returns
    -------
    tuple
        Tuple containing transformation matrices to shared latent space for
        each input source. Length will be equal to the length of the input
        feature list.
    """
    # process labels for easy comparison of label sequences
    labels = [label2str(labs) for labs in labels]

    # condition average firing rates for all datasets
    cnd_avg_data = [0]*len(features)
    for i, (feats, labs) in enumerate(zip(features, labels)):
        cnd_avg_data[i] = cnd_avg(feats, labs)

    # only use same conditions across datasets
    shared_lab = reduce(np.intersect1d, labels)
    cnd_avg_data = [cnd_avg_data[i][np.isin(np.unique(lab), shared_lab,
                                            assume_unique=True)] for i, lab
                    in enumerate(labels)]

    # combine all datasets into one matrix (n_conditions x n_timepoints x
    # sum channels)
    cross_pt_mat = np.concatenate(cnd_avg_data, axis=-1)
    # reshape to 2D with channels as final dim
    cross_pt_mat = cross_pt_mat.reshape(-1, cross_pt_mat.shape[-1])

    # perform dimensionality reduction on channel dim of combined matrix
    latent_mat = dim_red(n_components=n_components).fit_transform(cross_pt_mat)

    # calculate per pt channel -> factor transformation matrices
    pt_latent_trans = [0]*len(cnd_avg_data)
    for i, pt_ca in enumerate(cnd_avg_data):
        pt_ca = pt_ca.reshape(-1, pt_ca.shape[-1])  # isolate channel dim
        latent_trans = np.linalg.pinv(pt_ca) @ latent_mat  # lst_sq soln
        pt_latent_trans[i] = latent_trans
        # latent_trans = np.linalg.pinv(latent_mat) @ pt_ca  # lst_sq soln
        # pt_latent_trans[i] = latent_trans.T
        # pt_latent_trans[i] = np.linalg.pinv(latent_trans)

    return (*pt_latent_trans,)


def reshape_latent_dynamics(X_a, X_b, y_a, y_b, type='class'):
    if type == 'class':
        L_a, L_b = extract_latent_dynamics_by_class(X_a, X_b, y_a, y_b)
    elif type == 'trial':
        L_a, L_b = extract_latent_dynamics_by_trial_subselect(X_a, X_b, y_a,
                                                              y_b)
    else:
        raise ValueError('type must be "class" or "trial".')

    # fold timepoints into trials (isolate latent dimensionality as 2nd dim)
    L_a, L_b = L_a.reshape(-1, L_a.shape[-1]), L_b.reshape(-1, L_b.shape[-1])
    return L_a, L_b


def extract_latent_dynamics_by_class(X_a, X_b, y_a, y_b):
    # process labels for easy comparison of label sequences
    y_a, y_b = label2str(y_a), label2str(y_b)
    # average trials within class
    L_a, L_b = cnd_avg(X_a, y_a), cnd_avg(X_b, y_b)

    # only align via shared classes between datasets
    _, y_shared_a, y_shared_b = np.intersect1d(np.unique(y_a), np.unique(y_b),
                                               assume_unique=True,
                                               return_indices=True)
    L_a, L_b = L_a[y_shared_a], L_b[y_shared_b]

    return L_a, L_b


def extract_latent_dynamics_by_trial_subselect(X_a, X_b, y_a, y_b):
    y_a, y_b = label2str(y_a), label2str(y_b)
    L_a, L_b = shared_trial_subselect(X_a, X_b, y_a, y_b)
    return L_a, L_b


def shared_trial_subselect(X_a, X_b, y_a, y_b):
    L_a, L_b = [], []
    # subselect same amount of trials for each class
    for c in np.intersect1d(y_a, y_b):
        # shuffle trial order within class
        curr_a = np.random.permutation(np.where(y_a == c)[0])
        curr_b = np.random.permutation(np.where(y_b == c)[0])
        min_shared = min(curr_a.shape[0], curr_b.shape[0])

        L_a.append(X_a[curr_a[:min_shared]])
        L_b.append(X_b[curr_b[:min_shared]])
    L_a, L_b = np.vstack(L_a), np.vstack(L_b)
    return L_a, L_b


def CCA_align_by_class(X_a, X_b, y_a, y_b, return_space='b_to_a'):
    """
    CCA Alignment between 2 datasets with correspondence by averaging within
    class conditions.

    The number of features must be the same for datasets A and B. For example,
    if the datasets have different feature sizes, you can use PCA to reduce
    both datasets to the same number of PCs first.

    Parameters
    ----------
    X_a : ndarray
        Data matrix for dataset A of shape (n_trials_a, n_timepoints,
        n_features)
    X_b : ndarray
        Data matrix for dataset B of shape (n_trials_b, n_timepoints,
        n_features)
    y_a : ndarray
        Label matrix for dataset A of shape (n_trials_a, ...). The first
        dimension must be the trial dimension. This can be a 1D array, or a 2D
        array if each trial has multiple labels (e.g. a sequence of phonemes).
        Label sequences are converted to a single string so that only the same
        label sequences have correspondence between the datasets.
    y_b : ndarray
        Label matrix for dataset B of shape (n_trials_a, ...). See y_a for more
        details.
    return_space : str, optional
        How to perform alignment. Dataset B can be aligned to A, and vice versa
        ('b_to_a' and 'a_to_b', respectively), or both datasets can be aligned
        to a shared space ('shared'). Defaults to 'b_to_a'.

    Returns
    -------
    tuple
        Tuple containing aligned data matrix for dataset A and dataset B.
    """

    parse_return_type(return_space)

    # convert labels to strings for seqeunce comparison
    y_a = label2str(y_a)
    y_b = label2str(y_b)

    # group trials by label type
    L_a = cnd_avg(X_a, y_a)
    L_b = cnd_avg(X_b, y_b)

    # find common labels between datasets for alignment
    _, y_shared_a, y_shared_b = np.intersect1d(np.unique(y_a), np.unique(y_b),
                                               assume_unique=True,
                                               return_indices=True)
    L_a = L_a[y_shared_a]
    L_b = L_b[y_shared_b]

    # fold timepoints into trials
    L_a = np.reshape(L_a, (-1, L_a.shape[-1]))
    L_b = np.reshape(L_b, (-1, L_b.shape[-1]))

    # calculate manifold directions with CCA
    M_a, M_b = CCA_align(L_a.T, L_b.T)

    # align in put data with manifold transformation matrices
    if return_space == 'b_to_a':
        return X_a, X_b @ M_b @ np.linalg.pinv(M_a)
    elif return_space == 'a_to_b':
        return X_a @ M_a @ np.linalg.pinv(M_b), X_b
    return X_a @ M_a, X_b @ M_b


def CCA_align_by_trial_subselect(X_a, X_b, y_a, y_b, return_space='b_to_a'):
    """
    CCA Alignment between 2 datasets with correspondence via subselection of
    trials within shared clases.

    The number of features must be the same for datasets A and B. For example,
    if the datasets have different feature sizes, you can use PCA to reduce
    both datasets to the same number of PCs first.

    Parameters
    ----------
    X_a : ndarray
        Data matrix for dataset A of shape (n_trials_a, n_timepoints,
        n_features)
    X_b : ndarray
        Data matrix for dataset B of shape (n_trials_b, n_timepoints,
        n_features)
    y_a : ndarray
        Label matrix for dataset A of shape (n_trials_a, ...). The first
        dimension must be the trial dimension. This can be a 1D array, or a 2D
        array if each trial has multiple labels (e.g. a sequence of phonemes).
        Label sequences are converted to a single string so that only the same
        label sequences have correspondence between the datasets.
    y_b : ndarray
        Label matrix for dataset B of shape (n_trials_a, ...). See y_a for more
        details.
    return_space : str, optional
        How to perform alignment. Dataset B can be aligned to A, and vice versa
        ('b_to_a' and 'a_to_b', respectively), or both datasets can be aligned
        to a shared space ('shared'). Defaults to 'b_to_a'.

    Returns
    -------
    tuple
        Tuple containing aligned data matrix for dataset A and dataset B.
    """

    parse_return_type(return_space)

    y_a = label2str(y_a)
    y_b = label2str(y_b)

    L_a, L_b = [], []
    # subselect same amount of trials for each class
    for c in np.intersect1d(y_a, y_b):
        # shuffle trial order within class
        curr_a = np.random.permutation(np.where(y_a == c)[0])
        curr_b = np.random.permutation(np.where(y_b == c)[0])
        min_shared = min(curr_a.shape[0], curr_b.shape[0])

        L_a.append(X_a[curr_a[:min_shared]])
        L_b.append(X_b[curr_b[:min_shared]])

    # combine subselected trials
    L_a = np.vstack(L_a)
    L_b = np.vstack(L_b)

    # fold timepoints into trials
    L_a = np.reshape(L_a, (-1, L_a.shape[-1]))
    L_b = np.reshape(L_b, (-1, L_b.shape[-1]))

    # calculate alignment
    M_a, M_b = CCA_align(L_a.T, L_b.T)

    # align in put data with manifold transformation matrices
    if return_space == 'b_to_a':
        return X_a, X_b @ M_b @ np.linalg.pinv(M_a)
    elif return_space == 'a_to_b':
        return X_a @ M_a @ np.linalg.pinv(M_b), X_b
    return X_a @ M_a, X_b @ M_b


def parse_return_type(return_space):
    """
    Checks the CCA alignment return type is valid.

    Parameters
    ----------
    return_space : str
        String detailing how to perform alignment.

    Raises
    ------
    ValueError
        Error if return_space is not 'b_to_a', 'a_to_b', or 'shared'
    """

    if return_space not in ['b_to_a', 'a_to_b', 'shared']:
        raise ValueError('return_space must be "b_to_a" or "a_to_b" or'
                         '"shared".')


@njit
def CCA_align(L_a, L_b):
    """Canonical Correlation Analysis (CCA) alignment between 2 datasets.

    From: https://www.nature.com/articles/s41593-019-0555-4#Sec11.
    Returns manifold directions to transform L_a and L_b into a common space
    (e.g. L_a_new.T = L_a.T @ M_a, L_b_new.T = L_b.T @ M_b).
    To transform into a specific patient space, for example putting everything
    in patient A's space, use L_(b->a).T = L_b.T @ M_b @ (M_a)^-1, where L_a
    and L_(b->a) will be aligned in the same space.

    Parameters
    ----------
    L_a : ndarray
        Latent dynamics array for dataset A of shape (m, T),
        where m is the number of latent dimensions and T is the number of
        timepoints.
    L_b : ndarray
        Latent dynamics array for dataset B of shape (m, T)

    Returns
    -------
    tuple
        Tuple containing:
        M_a : ndarray
            Manifold directions for dataset A of shape (m, m)
        M_b : ndarray
            Manifold directions for dataset B of shape (m, m)

    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.RandomState(seed=0)
    >>> L_a = np.random.randn(5, 10)
    >>> L_b = np.random.randn(5, 10)
    >>> M_a, M_b = CCA_align(L_a, L_b)
    >>> M_a.shape
    (5, 5)
    >>> (L_b.T @ M_b).shape
    (10, 5)
    """
    # QR decomposition
    Q_a, R_a = np.linalg.qr(L_a.T)
    Q_b, R_b = np.linalg.qr(L_b.T)

    # SVD on q inner product
    U, S, Vt = np.linalg.svd(Q_a.T @ Q_b)

    # calculate manifold directions
    M_a = np.linalg.pinv(R_a) @ U
    M_b = np.linalg.pinv(R_b) @ Vt.T

    return M_a, M_b


if __name__ == '__main__':
    from timeit import timeit
    L_a = np.random.randn(50, 1000)
    L_b = np.random.randn(50, 1000)
    M_a, M_b = CCA_align(L_a, L_b)
    print(timeit('CCA_align(L_a, L_b)', globals=globals(), number=1000))