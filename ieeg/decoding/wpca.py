from sklearn.utils.validation import check_array
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip, stable_cumsum
from sklearn.decomposition._pca import _infer_dimension
from ieeg.arrays.api import array_namespace

def check_array_with_weights(X, weights, **kwargs):
    if weights is None:
        return check_array(X, **kwargs), weights

    weights = check_array(weights, **kwargs)
    kwargs_allow_nonfinite = dict(kwargs)
    kwargs_allow_nonfinite.update(force_all_finite=False)
    X = check_array(X, **kwargs_allow_nonfinite)
    if X.shape != weights.shape:
        raise ValueError("Shape of `X` and `weights` should match")
    if not np.all(np.isfinite(X) | (weights == 0)):
        raise ValueError("Input contains NaN or infinity without "
                         "a corresponding zero in `weights`.")
    return X, weights

def weighted_mean(x, w=None, axis=None):
    """Compute the weighted mean along the given axis

    The result is equivalent to (x * w).sum(axis) / w.sum(axis),
    but large temporary arrays are not created.

    Parameters
    ----------
    x : array_like
        data for which mean is computed
    w : array_like (optional)
        weights corresponding to each data point. If supplied, it must be the
        same shape as x
    axis : int or None (optional)
        axis along which mean should be computed

    Returns
    -------
    mean : np.ndarray
        array representing the weighted mean along the given axis
    """
    if w is None:
        return np.mean(x, axis)

    xp = array_namespace(x, w)

    x = xp.asarray(x)
    w = xp.asarray(w)

    if x.shape != w.shape:
        raise NotImplementedError("Broadcasting is not implemented: "
                                  "x and w must be the same shape.")

    if axis is None:
        wx_sum = xp.einsum('i,i', np.ravel(x), np.ravel(w))
    else:
        try:
            axis = tuple(axis)
        except TypeError:
            axis = (axis,)

        if len(axis) != len(set(axis)):
            raise ValueError("duplicate value in 'axis'")

        trans = sorted(set(range(x.ndim)).difference(axis)) + list(axis)
        operand = "...{0},...{0}".format(''.join(chr(ord('i') + i)
                                                 for i in range(len(axis))))
        wx_sum = xp.einsum(operand,
                           np.transpose(x, trans),
                           np.transpose(w, trans))

    return wx_sum / xp.sum(w, axis)

class WPCA(PCA):
    """Weighted Principal Component Analysis

    This is a direct implementation of weighted PCA based on the eigenvalue
    decomposition of the weighted covariance matrix following
    Delchambre (2014) [1]_.

    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept

    xi : float (optional)
        Degree of weight enhancement.

    regularization : float (optional)
        Control the strength of ridge regularization used to compute the
        transform.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

    See Also
    --------
    - PCA
    - sklearn.decomposition.PCA

    References
    ----------
    .. [1] Delchambre, L. MNRAS 2014 446 (2): 3545-3555 (2014)
           http://arxiv.org/abs/1412.4533
    """
    def __init__(self, n_components=None, xi=0, regularization=None,
                 copy_data=True):
        self.n_components = n_components
        self.xi = xi
        self.regularization = regularization
        self.copy_data = copy_data

    def _center_and_weight(self, X, weights, fit_mean=False):
        """Compute centered and weighted version of X.

        If fit_mean is True, then also save the mean to self.mean_
        """
        X, weights = check_array_with_weights(X, weights, dtype=float,
                                              copy=self.copy_data)
        xp = array_namespace(X, weights)

        if fit_mean:
            self.mean_ = weighted_mean(X, weights, axis=0)

        # now let X <- (X - mean) * weights
        X -= self.mean_

        if weights is not None:
            X *= weights
        else:
            weights = xp.ones_like(X)

        return X, weights

    def fit(self, X, y=None, weights=None):
        """Compute principal components for X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # let X <- (X - mean) * weights
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_full(X, weights, self.n_components, array_namespace(X, weights))
        return self

    def _fit_precentered(self, X, weights):
        """fit pre-centered data"""
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        xp = array_namespace(X, weights)

        # TODO: filter NaN warnings
        covar = xp.dot(X.T, X)
        covar /= xp.dot(weights.T, weights)
        covar[xp.isnan(covar)] = 0

        # enhance weights if desired
        if self.xi != 0:
            Ws = weights.sum(0)
            covar *= xp.outer(Ws, Ws) ** self.xi

        # eigvals = (X.shape[1] - n_components, X.shape[1] - 1)
        evals, evecs = xp.linalg.eigh(covar)
        self.components_ = evecs[:, ::-1].T
        self.explained_variance_ = evals[::-1]
        self.explained_variance_ratio_ = evals[::-1] / covar.trace()

    def transform(self, X, weights=None):
        """Apply dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=False)
        return self._transform_precentered(X, weights)

    def _transform_precentered(self, X, weights):
        """transform pre-centered data"""
        xp = array_namespace(X, weights)
        # TODO: parallelize this?
        Y = xp.zeros((X.shape[0], self.components_.shape[0]))
        for i in range(X.shape[0]):
            cW = self.components_ * weights[i]
            cWX = xp.dot(cW, X[i])
            cWc = xp.dot(cW, cW.T)
            if self.regularization is not None:
                cWc += xp.diag(self.regularization / self.explained_variance_)
            Y[i] = xp.linalg.solve(cWc, cWX)
        return Y

    def fit_transform(self, X, y=None, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_full(X, weights, self.n_components, array_namespace(X, weights))
        return self._transform_precentered(X, weights)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        X = check_array(X)
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X, weights=None):
        """Reconstruct the data using the PCA model

        This is equivalent to calling transform followed by inverse_transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.transform(X, weights=weights))

    def fit_reconstruct(self, X, weights=None):
        """Fit the model and reconstruct the data using the PCA model

        This is equivalent to calling fit_transform()
        followed by inverse_transform().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse of the Gaussian errorbar.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights=weights))\

    def _fit_full(self, X, weights, n_components, xp):
        """Fit the model by computing full SVD on X."""
        n_samples, n_features = X.shape

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                f"n_components={n_components} must be between 0 and "
                f"min(n_samples, n_features)={min(n_samples, n_features)} with "
                f"svd_solver={self._fit_svd_solver!r}"
            )

        self.mean_ = xp.mean(X, axis=0)
        # When X is a scipy sparse matrix, self.mean_ is a numpy matrix, so we need
        # to transform it to a 1D array. Note that this is not the case when X
        # is a scipy sparse array.
        # TODO: remove the following two lines when scikit-learn only depends
        # on scipy versions that no longer support scipy.sparse matrices.
        self.mean_ = xp.reshape(xp.asarray(self.mean_), (-1,))

        # In the following, we center the covariance matrix C afterwards
        # (without centering the data X first) to avoid an unnecessary copy
        # of X. Note that the mean_ attribute is still needed to center
        # test data in the transform method.
        #
        # Note: at the time of writing, `xp.cov` does not exist in the
        # Array API standard:
        # https://github.com/data-apis/array-api/issues/43
        #
        # Besides, using `numpy.cov`, as of numpy 1.26.0, would not be
        # memory efficient for our use case when `n_samples >> n_features`:
        # `numpy.cov` centers a copy of the data before computing the
        # matrix product instead of subtracting a small `(n_features,
        # n_features)` square matrix from the gram matrix X.T @ X, as we do
        # below.
        x_is_centered = False
        C = X.T @ X
        C -= (
            n_samples
            * xp.reshape(self.mean_, (-1, 1))
            * xp.reshape(self.mean_, (1, -1))
        )
        C /= xp.dot(weights.T, weights)
        C[xp.isnan(C)] = 0

        # enhance weights if desired
        if self.xi != 0:
            Ws = weights.sum(0)
            C *= xp.outer(Ws, Ws) ** self.xi
        eigenvals, eigenvecs = xp.linalg.eigh(C)

        # When X is a scipy sparse matrix, the following two datastructures
        # are returned as instances of the soft-deprecated numpy.matrix
        # class. Note that this problem does not occur when X is a scipy
        # sparse array (or another other kind of supported array).
        # TODO: remove the following two lines when scikit-learn only
        # depends on scipy versions that no longer support scipy.sparse
        # matrices.
        eigenvals = xp.reshape(xp.asarray(eigenvals), (-1,))
        eigenvecs = xp.asarray(eigenvecs)

        eigenvals = xp.flip(eigenvals, axis=0)
        eigenvecs = xp.flip(eigenvecs, axis=1)

        # The covariance matrix C is positive semi-definite by
        # construction. However, the eigenvalues returned by xp.linalg.eigh
        # can be slightly negative due to numerical errors. This would be
        # an issue for the subsequent sqrt, hence the manual clipping.
        eigenvals[eigenvals < 0.0] = 0.0
        explained_variance_ = eigenvals

        # Re-construct SVD of centered X indirectly and make it consistent
        # with the other solvers.
        S = xp.sqrt(eigenvals * (n_samples - 1))
        Vt = eigenvecs.T
        U = None

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt, u_based_decision=False)

        components_ = Vt

        # Get variance explained by singular values
        total_var = xp.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = xp.asarray(S, copy=True)  # Store the singular values.

        # Postprocess the number of components required
        if n_components == "mle":
            n_components = _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = xp.mean(explained_variance_[n_components:])
        else:
            self.noise_variance_ = 0.0

        self.n_samples_ = n_samples
        self.n_components_ = n_components
        # Assign a copy of the result of the truncation of the components in
        # order to:
        # - release the memory used by the discarded components,
        # - ensure that the kept components are allocated contiguously in
        #   memory to make the transform method faster by leveraging cache
        #   locality.
        self.components_ = xp.asarray(components_[:n_components, :], copy=True)

        # We do the same for the other arrays for the sake of consistency.
        self.explained_variance_ = xp.asarray(
            explained_variance_[:n_components], copy=True
        )
        self.explained_variance_ratio_ = xp.asarray(
            explained_variance_ratio_[:n_components], copy=True
        )
        self.singular_values_ = xp.asarray(singular_values_[:n_components], copy=True)

        return U, S, Vt, X, x_is_centered, xp
