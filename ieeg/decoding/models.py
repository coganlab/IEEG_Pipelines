# %% IMPORT PACKAGES

import numpy as np
from numpy.linalg import inv as inv  # Used in kalman filter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.spatial.distance import cdist
import math
from sklearn.pipeline import Pipeline
from sklearn import linear_model  # For Wiener Filter and Wiener Cascade
from sklearn.svm import SVR, SVC  # For support vector regression (SVR)
from sklearn.decomposition import PCA  # For PCA decomposition (PCA - LDA)
from sklearn import \
    discriminant_analysis as da  # For LDA decomposition (PCA - LDA)
from sklearn.base import BaseEstimator, TransformerMixin, clone  # For weighted PCA (WPCA - LDA)
# from ieeg.decoding.wpca import WPCA
from joblib import Memory
from sklearn.metrics import accuracy_score
from ieeg.calc.fast import mixup2
from ieeg.arrays.api import array_namespace, is_torch
from typing import Optional

# Used for naive bayes decoder
try:
    import statsmodels.api as sm
except ImportError:
    print(
        "\nWARNING: statsmodels is not installed. You will be unable to use "
        "the Naive Bayes Decoder")
    pass

try:
    from sklearnex import patch_sklearn

    # The names match scikit-learn estimators
    patch_sklearn()
except ImportError:
    print(
        "\nWARNING: sklearnex is not installed. You will be unable to use the "
        "PCA decoder acceleration")
    pass
# Import XGBoost if the package is installed
try:
    import xgboost as xgb  # For xgboost
except ImportError:
    print(
        "\nWARNING: Xgboost package is not installed. You will be unable to"
        "use the xgboost decoder")
    pass

# Import functions for Keras if Keras is installed
# Note that Keras has many more built-in functions that I have not imported
# because I have not used them but if you want to modify the decoders with
# other functions (e.g. regularization), import them here
try:
    import keras

    keras_v1 = int(keras.__version__[0]) <= 1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
except ImportError:
    print(
        "\nWARNING: Keras package is not installed. You will be unable to use"
        "all neural net decoders")
    pass

# pytorch imports
try:
    from ieeg.decoding.models_torch import SimpleDecoder, CNNTransformer
except ImportError:
    print(
        "\nWARNING: PyTorch package is not installed. You will be unable to use"
        "all PyTorch decoders")
    pass

# Optional: skorch + torchvision for PyTorch vision models with sklearn API
try:
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models
except ImportError:
    torch = None
    nn = None
    tv_models = None
    print("\nWARNING: PyTorch and/or torchvision not available. PyTorch decoders will be unavailable.")

try:
    from skorch import NeuralNetClassifier
    from skorch.callbacks import GradientNormClipping, EarlyStopping, LRScheduler, Callback

except Exception:
    NeuralNetClassifier = None
    GradientNormClipping = None
    EarlyStopping = None
    LRScheduler = None
    print(
        "\nWARNING: skorch/torchvision not available. ResNetTokenClassifier will be unavailable.")


# %% DECODER FUNCTIONS


# %% WIENER FILTER

class WienerFilterRegression(object):
    """Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self):
        return

    def fit(self, X_flat_train, y_train):
        """Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        # Initialize linear regression model
        self.model = linear_model.LinearRegression()

        # Train the model
        self.model.fit(X_flat_train, y_train)

    def predict(self, X_flat_test):
        """Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted


# %% WIENER CASCADE

class WienerCascadeRegression(object):
    """Class for the Wiener Cascade Decoder

    Parameters
    ----------
    degree: integer, optional, default 3
        The degree of the polynomial used for the static nonlinearity
    """

    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X_flat_train, y_train):

        """Train Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs = y_train.shape[1]  # Number of outputs
        models = []  # Initialize list of models (there will be a separate
        # model for each output)
        for i in range(num_outputs):  # Loop through outputs
            # Fit linear portion of model
            regr = linear_model.LinearRegression()  # Call the linear portion
            # of the model "regr"
            regr.fit(X_flat_train, y_train[:, i])  # Fit linear
            y_train_predicted_linear = regr.predict(
                X_flat_train)  # Get outputs of linear portion of model
            # Fit nonlinear portion of model
            p = np.polyfit(y_train_predicted_linear, y_train[:, i],
                           self.degree)
            # Add model for this output (both linear and nonlinear parts)
            # to the list "models"
            models.append([regr, p])
        self.model = models

    def predict(self, X_flat_test):

        """Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs = len(
            self.model)  # Number of outputs being predicted. Recall from the
        # "fit" function that self.model is a list of models
        y_test_predicted = np.empty([X_flat_test.shape[0],
                                     num_outputs])  # Initialize matrix that
        # contains predicted outputs
        for i in range(num_outputs):  # Loop through outputs
            [regr, p] = self.model[
                i]  # Get the linear (regr) and nonlinear (p) portions of the
            # trained model
            # Predictions on test set
            y_test_predicted_linear = regr.predict(
                X_flat_test)  # Get predictions on the linear portion of
            # the model
            y_test_predicted[:, i] = np.polyval(p,
                                                y_test_predicted_linear)
            # Run the linear predictions through the nonlinearity to get
            # the final predictions
        return y_test_predicted


# %% KALMAN FILTER

class KalmanFilterRegression(object):
    """Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in
    kinematic states. It effectively allows changing the weight of the new
    neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on
    that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of
    -cursor-motion-using-a-kalman-filter.pdf) with the exception of the
    addition of the parameter C. The original implementation has previously
    been coded in Matlab by Dan Morris
    (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self, C=1):
        self.C = C

    def fit(self, X_kf_train, y_train):
        """Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) ,
        n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        # First we'll rename and reformat the variables to be in a more
        # standard kalman filter nomenclature (specifically that from Wu et
        # al, 2003):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        # number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using
        # least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to
        # the next
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]
        A = X2 * X1.T * inv(X1 * X1.T)  # Transition matrix
        W = (X2 - A * X1) * (X2 - A * X1).T / (
                nt - 1) / self.C  # Covariance of transition matrix. Note we
        # divide by nt-1 since only nt-1 points were used in the computation
        # (that's the length of X1 and X2). We also introduce the extra
        # parameter C here.

        # Calculate the measurement matrix (from x_t to z_t) using
        # least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z * X.T * (inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * (
            (Z - H * X).T)) / nt  # Covariance of measurement matrix
        params = [A, W, H, Q]
        self.model = params

    def predict(self, X_kf_test, y_test):
        """Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) ,
        n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other
            decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),
        n_outputs]
            The predicted outputs
        """

        # Extract parameters
        A, W, H, Q = self.model

        # First we'll rename and reformat the variables to be in a more
        # standard kalman filter nomenclature (specifically that from Wu et
        # al):
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_test.T)
        Z = np.matrix(X_kf_test.T)

        # Initializations
        num_states = X.shape[0]  # Dimensionality of the state
        states = np.empty(
            X.shape)  # Keep track of states over time (states is what will
        # be returned as y_test_predicted)
        P_m = np.matrix(np.zeros([num_states, num_states]))
        P = np.matrix(np.zeros([num_states, num_states]))
        state = X[:, 0]  # Initial state
        states[:, 0] = np.copy(np.squeeze(state))

        # Get predicted state for every time bin
        for t in range(X.shape[1] - 1):
            # Do first part of state update - based on transition matrix
            P_m = A * P * A.T + W
            state_m = A * state

            # Do second part of state update - based on measurement matrix
            K = P_m * H.T * inv(H * P_m * H.T + Q)  # Calculate Kalman gain
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m
            state = state_m + K * (Z[:, t + 1] - H * state_m)
            states[:, t + 1] = np.squeeze(
                state)  # Record state at the timestep
        y_test_predicted = states.T
        return y_test_predicted


# %% DENSE (FULLY-CONNECTED) NEURAL NETWORK

class DenseNNRegression(object):
    """Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------

    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give
        you a single hidden layer with 400 units) If you want multiple layers,
        input a vector (e.g. units=[400,200]) will give you 2 hidden layers
        with 400 and 200 units, repsectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

        # If "units" is an integer, put it in the form of a vector
        try:  # Check if it's a vector
            units[0]
        except IndexError:
            # If it's not a vector, create a vector of the number of
            # units for each layer
            units = [units]
        self.units = units

        # Determine the number of hidden layers (based on "units" that the
        # user entered)
        self.num_layers = len(units)

    def fit(self, X_flat_train, y_train):

        """Train DenseNN Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add first hidden layer
        model.add(Dense(self.units[0],
                        input_dim=X_flat_train.shape[1]))  # Add dense layer
        model.add(Activation('relu'))  # Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout != 0:
            # Dropout some units if proportion of dropout != 0
            model.add(Dropout(self.dropout))

        # Add any additional hidden layers (beyond the 1st)
        for layer in range(
                self.num_layers - 1):  # Loop through additional layers
            model.add(Dense(self.units[layer + 1]))  # Add dense layer
            model.add(Activation('relu'))  # Add nonlinear (tanh) activation
            if self.dropout != 0:
                # Dropout some units if proportion of dropout != 0
                model.add(Dropout(self.dropout))

        # Add dense connections to all outputs
        model.add(Dense(
            y_train.shape[1]))  # Add final dense layer (connected to outputs)

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='adam',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_flat_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_flat_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_flat_test):

        """Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted


# %% SIMPLE RECURRENT NEURAL NETWORK

class SimpleRNNRegression(object):
    """Class for the simple recurrent neural network decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):

        """Train SimpleRNN Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add recurrent layer
        if keras_v1:
            model.add(SimpleRNN(self.units, input_shape=(
                X_train.shape[1], X_train.shape[2]), dropout_W=self.dropout,
                                dropout_U=self.dropout,
                                activation='relu'))  # Within recurrent
            # layer, include dropout
        else:
            model.add(SimpleRNN(self.units, input_shape=(
                X_train.shape[1], X_train.shape[2]), dropout=self.dropout,
                                recurrent_dropout=self.dropout,
                                activation='relu'))  # Within recurrent
            # layer, include dropout
        if self.dropout != 0:
            # Dropout some units (recurrent layer output units)
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='rmsprop',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_test):

        """Predict outcomes using trained SimpleRNN Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)  # Make predictions
        return y_test_predicted


# %% GATED RECURRENT UNIT (GRU) DECODER

class GRURegression(object):
    """Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):

        """Train GRU Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add recurrent layer
        if keras_v1:
            model.add(GRU(self.units,
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          dropout_W=self.dropout,
                          dropout_U=self.dropout))  # Within recurrent layer,
            # include dropout
        else:
            model.add(GRU(self.units,
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          dropout=self.dropout,
                          recurrent_dropout=self.dropout))
        if self.dropout != 0:
            # Dropout some units (recurrent layer output units)
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='rmsprop',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_test):

        """Predict outcomes using trained GRU Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)  # Make predictions
        return y_test_predicted


# %% LONG SHORT TERM MEMORY (LSTM) DECODER

class LSTMRegression(object):
    """Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):

        """Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units,
                           input_shape=(X_train.shape[1], X_train.shape[2]),
                           dropout_W=self.dropout,
                           dropout_U=self.dropout))  # Within recurrent layer,
            # include dropout
        else:
            model.add(LSTM(self.units,
                           input_shape=(X_train.shape[1], X_train.shape[2]),
                           dropout=self.dropout,
                           recurrent_dropout=self.dropout))  # Within recurrent
            # layer, include dropout
        if self.dropout != 0:
            # Dropout some units (recurrent layer output units)
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='rmsprop',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_test):

        """Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)  # Make predictions
        return y_test_predicted


# %% EXTREME GRADIENT BOOSTING (XGBOOST)

class XGBoostRegression(object):
    """Class for the XGBoost Decoder

    Parameters
    ----------
    max_depth: integer, optional, default=3
        the maximum depth of the trees

    num_round: integer, optional, default=300
        the number of trees that are fit

    eta: float, optional, default=0.3
        the learning rate

    gpu: integer, optional, default=-1
        if the gpu version of xgboost is installed, this can be used to select
        which gpu to use
        for negative values (default), the gpu is not used
    """

    def __init__(self, max_depth=3, num_round=300, eta=0.3, gpu=-1):
        self.max_depth = max_depth
        self.num_round = num_round
        self.eta = eta
        self.gpu = gpu

    def fit(self, X_flat_train, y_train):

        """Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs = y_train.shape[1]  # Number of outputs

        # Set parameters for XGBoost
        param = {'objective': "reg:linear",  # for linear output
                 'eval_metric': "logloss",  # loglikelihood loss
                 'max_depth': self.max_depth,
                 # this is the only parameter we have set, it's one of the way
                 # or regularizing
                 'eta': self.eta,
                 'seed': 2925,  # for reproducibility
                 'silent': 1}
        if self.gpu < 0:
            param['nthread'] = -1  # with -1 it will use all available threads
        else:
            param['gpu_id'] = self.gpu
            param['updater'] = 'grow_gpu'

        models = []  # Initialize list of models (there will be a separate
        # model for each output)
        for y_idx in range(num_outputs):  # Loop through outputs
            dtrain = xgb.DMatrix(X_flat_train, label=y_train[:, y_idx])
            # Put in correct format for XGB
            bst = xgb.train(param, dtrain, self.num_round)  # Train model
            models.append(bst)  # Add fit model to list of models

        self.model = models

    def predict(self, X_flat_test):

        """Predict outcomes using trained XGBoost Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        dtest = xgb.DMatrix(X_flat_test)  # Put in XGB format
        num_outputs = len(self.model)  # Number of outputs
        y_test_predicted = np.empty([X_flat_test.shape[0],
                                     num_outputs])  # Initialize matrix of
        # predicted outputs
        for y_idx in range(num_outputs):  # Loop through outputs
            bst = self.model[y_idx]  # Get fit model for this output
            y_test_predicted[:, y_idx] = bst.predict(dtest)  # Make prediction
        return y_test_predicted


# %% SUPPORT VECTOR REGRESSION

class SVRegression(object):
    """Class for the Support Vector Regression (SVR) Decoder
    This simply leverages the scikit-learn SVR

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self, max_iter=-1, C=3.0):
        self.max_iter = max_iter
        self.C = C
        return

    def fit(self, X_flat_train, y_train):

        """Train SVR Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs = y_train.shape[1]  # Number of outputs
        models = []  # Initialize list of models (there will be a separate
        # model for each output)
        for y_idx in range(num_outputs):  # Loop through outputs
            model = SVR(C=self.C,
                        max_iter=self.max_iter)  # Initialize SVR model
            model.fit(X_flat_train, y_train[:, y_idx])  # Train the model
            models.append(model)  # Add fit model to list of models
        self.model = models

    def predict(self, X_flat_test):

        """Predict outcomes using trained SVR Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs = len(self.model)  # Number of outputs
        y_test_predicted = np.empty([X_flat_test.shape[0],
                                     num_outputs])  # Initialize matrix of
        # predicted outputs
        for y_idx in range(num_outputs):  # Loop through outputs
            model = self.model[y_idx]  # Get fit model for that output
            y_test_predicted[:, y_idx] = model.predict(
                X_flat_test)  # Make predictions
        return y_test_predicted


# GLM helper function for the NaiveBayesDecoder
def glm_run(Xr, Yr, X_range):
    X2 = sm.add_constant(Xr)

    poiss_model = sm.GLM(Yr, X2, family=sm.families.Poisson())
    try:
        glm_results = poiss_model.fit()
        Y_range = glm_results.predict(sm.add_constant(X_range))
    except np.linalg.LinAlgError:
        print("\nWARNING: LinAlgError")
        Y_range = np.mean(Yr) * np.ones([X_range.shape[0], 1])

    return Y_range


class NaiveBayesRegression(object):
    """Class for the Naive Bayes Decoder

    Parameters
    ----------
    encoding_model: string, default='quadratic'
        what encoding model is used

    res:int, default=100
        resolution of predicted values
        This is the number of bins to divide the outputs into (going from
        minimum to maximum) larger values will make decoding slower
    """

    def __init__(self, encoding_model='quadratic', res=100):
        self.encoding_model = encoding_model
        self.res = res
        return

    def fit(self, X_b_train, y_train):

        """Train Naive Bayes Decoder

        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """

        # %% FIT TUNING CURVE
        # First, get the output values (x/y position or velocity) that we will
        # be creating tuning curves over
        # Create the range for x and y (position/velocity) values
        input_x_range = np.arange(np.min(y_train[:, 0]),
                                  np.max(y_train[:, 0]) + .01, np.round(
                (np.max(y_train[:, 0]) - np.min(y_train[:, 0])) / self.res))
        input_y_range = np.arange(np.min(y_train[:, 1]),
                                  np.max(y_train[:, 1]) + .01, np.round(
                (np.max(y_train[:, 1]) - np.min(y_train[:, 1])) / self.res))
        # Get all combinations of x/y values
        input_mat = np.meshgrid(input_x_range, input_y_range)
        # Format so that all combinations of x/y values are in 2 columns (first
        # column x, second column y). This is called "input_xy"
        xs = np.reshape(input_mat[0],
                        [input_x_range.shape[0] * input_y_range.shape[0], 1])
        ys = np.reshape(input_mat[1],
                        [input_x_range.shape[0] * input_y_range.shape[0], 1])
        input_xy = np.concatenate((xs, ys), axis=1)

        # If quadratic model:
        #   -make covariates have squared components and mixture of x and y
        #   -do same thing for "input_xy", which are the values for creating
        #   the tuning curves
        if self.encoding_model == 'quadratic':
            input_xy_modified = np.empty([input_xy.shape[0], 5])
            input_xy_modified[:, 0] = input_xy[:, 0] ** 2
            input_xy_modified[:, 1] = input_xy[:, 0]
            input_xy_modified[:, 2] = input_xy[:, 1] ** 2
            input_xy_modified[:, 3] = input_xy[:, 1]
            input_xy_modified[:, 4] = input_xy[:, 0] * input_xy[:, 1]
            y_train_modified = np.empty([y_train.shape[0], 5])
            y_train_modified[:, 0] = y_train[:, 0] ** 2
            y_train_modified[:, 1] = y_train[:, 0]
            y_train_modified[:, 2] = y_train[:, 1] ** 2
            y_train_modified[:, 3] = y_train[:, 1]
            y_train_modified[:, 4] = y_train[:, 0] * y_train[:, 1]

        # Create tuning curves

        num_nrns = X_b_train.shape[
            1]  # Number of neurons to fit tuning curves for
        tuning_all = np.zeros([num_nrns, input_xy.shape[
            0]])  # Matrix that stores tuning curves for all neurons

        # Loop through neurons and fit tuning curves
        for j in range(num_nrns):  # Neuron number

            if self.encoding_model == 'linear':
                tuning = glm_run(y_train, X_b_train[:, j:j + 1], input_xy)
            if self.encoding_model == 'quadratic':
                tuning = glm_run(y_train_modified, X_b_train[:, j:j + 1],
                                 input_xy_modified)
            # Enter tuning curves into matrix
            tuning_all[j, :] = np.squeeze(tuning)

        # Save tuning curves to be used in "predict" function
        self.tuning_all = tuning_all
        self.input_xy = input_xy

        # Get information about the probability of being in one state
        # (position/velocity) based on the previous state. Here we're
        # calculating the standard deviation of the change in state
        # (velocity/acceleration) in the training set
        n = y_train.shape[0]
        dx = np.zeros([n - 1, 1])
        for i in range(n - 1):
            # Change in state across time steps
            dx[i] = np.sqrt((y_train[i + 1, 0] - y_train[i, 0]) ** 2 + (
                    y_train[i + 1, 1] - y_train[i, 1]) ** 2)
        std = np.sqrt(np.mean(
            dx ** 2))  # dx is only positive. this gets approximate stdev of
        # distribution (if it was positive and negative)
        self.std = std  # Save for use in "predict" function

        # Get probability of being in each state - we are not using this since
        # it did not help decoding performance
        # n_x=np.empty([input_xy.shape[0]])
        # for i in range(n):
        #     loc_idx=np.argmin(cdist(y_train[0:1,:],input_xy))
        #     n_x[loc_idx]=n_x[loc_idx]+1
        # p_x=n_x/n
        # self.p_x=p_x

    def predict(self, X_b_test, y_test):

        """Predict outcomes using trained tuning curves

        Parameters
        ----------
        X_b_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        y_test: numpy 2d array of shape [n_samples,n_outputs]
            The actual outputs
            This parameter is necesary for the NaiveBayesDecoder  (unlike most
            other decoders) because the first value is nececessary for
            initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        # Get values saved in "fit" function
        tuning_all = self.tuning_all
        input_xy = self.input_xy
        std = self.std

        # Get probability of going from one state to the next
        dists = squareform(pdist(input_xy,
                                 'euclidean'))
        # Distance between all states in "input_xy" Probability of going from
        # one state to the next, based on the above calculated distances
        # The probability is calculated based on the distances coming from a
        # Gaussian with standard deviation of std
        prob_dists = norm.pdf(dists, 0, std)

        # Initializations
        loc_idx = np.argmin(
            cdist(y_test[0:1, :], input_xy))  # The index of the first location
        num_nrns = tuning_all.shape[0]  # Number of neurons
        y_test_predicted = np.empty(
            [X_b_test.shape[0], 2])  # Initialize matrix of predicted outputs
        num_ts = X_b_test.shape[0]  # Number of time steps we are predicting

        # Loop across time and decode
        for t in range(num_ts):
            rs = X_b_test[t, :]
            # Number of spikes at this time point (in the interval
            # we've specified including bins_before and bins_after)

            probs_total = np.ones([tuning_all[0, :].shape[
                                       0]])
            # Vector that stores the probabilities of being in any state based
            # on the neural activity (does not include probabilities of going
            # from one state to the next)
            for j in range(num_nrns):  # Loop across neurons
                lam = np.copy(tuning_all[j, :])
                # Expected spike counts given the tuning curve
                r = rs[j]  # Actual spike count
                probs = np.exp(-lam) * lam ** r / math.factorial(r)
                # Probability of the given neuron's spike count given tuning
                # curve (assuming poisson distribution)
                probs_total = np.copy(probs_total * probs)
                # Update the probability across neurons (probabilities are
                # multiplied across neurons due to the independence assumption)
            prob_dists_vec = np.copy(prob_dists[loc_idx, :])
            # Probability of going to all states from the previous state
            probs_final = probs_total * prob_dists_vec
            # Get final probability (multiply probabilities based on spike
            # count and previous state)
            # probs_final=probs_total*prob_dists_vec*self.p_x
            # #Get final probability when including p(x), i.e. prior about
            # being in states, which we're not using
            loc_idx = np.argmax(probs_final)
            # Get the index of the current state (that w/ the highest
            # probability)
            y_test_predicted[t, :] = input_xy[loc_idx, :]
            # The current predicted output

        return y_test_predicted  # Return predictions


# %% ALIASES for Regression

WienerFilterDecoder = WienerFilterRegression
WienerCascadeDecoder = WienerCascadeRegression
KalmanFilterDecoder = KalmanFilterRegression
DenseNNDecoder = DenseNNRegression
SimpleRNNDecoder = SimpleRNNRegression
GRUDecoder = GRURegression
LSTMDecoder = LSTMRegression
XGBoostDecoder = XGBoostRegression
SVRDecoder = SVRegression
NaiveBayesDecoder = NaiveBayesRegression


# %% CLASSIFICATION


class WienerFilterClassification(object):
    """Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn logistic regression.
    """

    def __init__(self, C=1):
        self.C = C
        return

    def fit(self, X_flat_train, y_train):
        """Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # if self.C>0:
        self.model = linear_model.LogisticRegression(C=self.C,
                                                     multi_class='auto')
        # Initialize linear regression model
        # else:
        # self.model=linear_model.LogisticRegression(penalty='none',
        # solver='newton-cg') #Initialize linear regression model
        self.model.fit(X_flat_train, y_train)  # Train the model

    def predict(self, X_flat_test):
        """Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted


# %% SUPPORT VECTOR REGRESSION

class SVClassification(object):
    """Class for the Support Vector Classification Decoder
    This simply leverages the scikit-learn SVM

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self, max_iter=-1, C=3.0):
        self.max_iter = max_iter
        self.C = C
        return

    def fit(self, X_flat_train, y_train):
        """Train SVR Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = SVC(C=self.C, max_iter=self.max_iter)  # Initialize model
        model.fit(X_flat_train, y_train)  # Train the model
        self.model = model

    def predict(self, X_flat_test):
        """Predict outcomes using trained SV Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        model = self.model  # Get fit model for that output
        y_test_predicted = model.predict(X_flat_test)  # Make predictions
        return y_test_predicted


# %% DENSE (FULLY-CONNECTED) NEURAL NETWORK

class DenseNNClassification(object):
    """Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------

    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give
        you a single hidden layer with 400 units). If you want multiple layers,
        input a vector (e.g. units=[400,200]) will give you 2 hidden layers
        with 400 and 200 units, repsectively. The vector can either be a list
        or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

        # If "units" is an integer, put it in the form of a vector
        try:  # Check if it's a vector
            units[0]
        except IndexError:
            # If it's not a vector, create a vector of the number of
            # units for each layer
            units = [units]
        self.units = units

        # Determine the number of hidden layers (based on "units" that the
        # user entered)
        self.num_layers = len(units)

    def fit(self, X_flat_train, y_train):

        """Train DenseNN Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # Use one-hot coding for y
        if y_train.ndim == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1] == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))

        model = Sequential()  # Declare model
        # Add first hidden layer
        model.add(Dense(self.units[0],
                        input_dim=X_flat_train.shape[1]))  # Add dense layer
        model.add(Activation('relu'))  # Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout != 0:
            # Dropout some units if proportion of dropout != 0
            model.add(Dropout(self.dropout))

        # Add any additional hidden layers (beyond the 1st)
        for layer in range(
                self.num_layers - 1):  # Loop through additional layers
            model.add(Dense(self.units[layer + 1]))  # Add dense layer
            # Add nonlinear (tanh) activation - can also make
            model.add(Activation('tanh'))
            # relu
            if self.dropout != 0:
                # Dropout some units if proportion of dropout != 0
                model.add(Dropout(self.dropout))

        # Add dense connections to all outputs
        model.add(Dense(
            y_train.shape[1]))  # Add final dense layer (connected to outputs)
        model.add(Activation('softplus'))

        # Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_flat_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_flat_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
            self.model = model

    def predict(self, X_flat_test):

        """Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(
            X_flat_test)  # Make predictions

        y_test_predicted = np.argmax(y_test_predicted_raw, axis=1)

        return y_test_predicted


# %% SIMPLE RNN DECODER

class SimpleRNNClassification(object):
    """Class for the RNN decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):

        """Train GRU Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # Use one-hot coding for y
        if y_train.ndim == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1] == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))

        model = Sequential()  # Declare model
        # Add recurrent layer

        # %% MAKE RELU ACTIVATION BELOW LIKE IN REGRESSION?????
        if keras_v1:
            model.add(SimpleRNN(self.units, input_shape=(
                X_train.shape[1], X_train.shape[2]), dropout_W=self.dropout,
                                dropout_U=self.dropout))
            # Within recurrent layer, include dropout
        else:
            model.add(SimpleRNN(self.units, input_shape=(
                X_train.shape[1], X_train.shape[2]), dropout=self.dropout,
                                recurrent_dropout=self.dropout))
            # Within recurrent layer, include dropout
        if self.dropout != 0:
            # Dropout some units (recurrent layer output units)
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softplus'))

        # Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_test):

        """Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_test)  # Make predictions
        y_test_predicted = np.argmax(y_test_predicted_raw, axis=1)

        return y_test_predicted


# %% GATED RECURRENT UNIT (GRU) DECODER

class GRUClassification(object):
    """Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):

        """Train GRU Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # Use one-hot coding for y
        if y_train.ndim == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1] == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))

        model = Sequential()  # Declare model
        # Add recurrent layer
        if keras_v1:
            model.add(GRU(self.units,
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          dropout_W=self.dropout,
                          dropout_U=self.dropout))
            # Within recurrent layer, include dropout
        else:
            model.add(GRU(self.units,
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          dropout=self.dropout,
                          recurrent_dropout=self.dropout))
            # Within recurrent layer, include dropout
        if self.dropout != 0:
            # Dropout some units (recurrent layer output units)
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softplus'))

        # Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_test):

        """Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_test)  # Make predictions
        y_test_predicted = np.argmax(y_test_predicted_raw, axis=1)

        return y_test_predicted


# %% LONG SHORT TERM MEMORY (LSTM) DECODER

class LSTMClassification(object):
    """Class for the LSTM decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):

        """Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # Use one-hot coding for y
        if y_train.ndim == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1] == 1:
            y_train = np_utils.to_categorical(y_train.astype(int))

        model = Sequential()  # Declare model
        # Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units,
                           input_shape=(X_train.shape[1], X_train.shape[2]),
                           dropout_W=self.dropout,
                           dropout_U=self.dropout))
            # Within recurrent layer, include dropout
        else:
            model.add(LSTM(self.units,
                           input_shape=(X_train.shape[1], X_train.shape[2]),
                           dropout=self.dropout,
                           recurrent_dropout=self.dropout))
            # Within recurrent layer, include dropout
        if self.dropout != 0:
            # Dropout some units (recurrent layer output units)
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softplus'))

        # Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])  # Set loss function and optimizer
        if keras_v1:
            model.fit(X_train, y_train, nb_epoch=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        else:
            model.fit(X_train, y_train, epochs=self.num_epochs,
                      verbose=self.verbose)  # Fit the model
        self.model = model

    def predict(self, X_test):

        """Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_test)  # Make predictions
        y_test_predicted = np.argmax(y_test_predicted_raw, axis=1)

        return y_test_predicted


# %% EXTREME GRADIENT BOOSTING (XGBOOST)

class XGBoostClassification(object):
    """Class for the XGBoost Decoder

    Parameters
    ----------
    max_depth: integer, optional, default=3
        the maximum depth of the trees

    num_round: integer, optional, default=300
        the number of trees that are fit

    eta: float, optional, default=0.3
        the learning rate

    gpu: integer, optional, default=-1
        if the gpu version of xgboost is installed, this can be used to select
        which gpu to use
        for negative values (default), the gpu is not used
    """

    def __init__(self, max_depth=3, num_round=300, eta=0.3, gpu=-1):
        self.max_depth = max_depth
        self.num_round = num_round
        self.eta = eta
        self.gpu = gpu

    def fit(self, X_flat_train, y_train):

        """Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data
            correctly

        y_train: numpy 1d array of shape (n_samples), with integers
            representing classes or 2d array of shape [n_samples, n_outputs] in
             1-hot form. This is the outputs that are being predicted
        """

        # turn to categorial (not 1-hat)
        if (y_train.ndim == 2):
            if (y_train.shape[1] == 1):
                y_train = np.reshape(y_train, -1)
            else:
                y_train = np.argmax(y_train, axis=1, out=None)

        # Get number of classes
        n_classes = len(np.unique(y_train))

        # Set parameters for XGBoost
        param = {'objective': "multi:softmax",  # or softprob
                 'eval_metric': "mlogloss",  # loglikelihood loss
                 # 'eval_metric': "merror",
                 'max_depth': self.max_depth,
                 # this is the only parameter we have set, it's one of the way
                 # or regularizing
                 'eta': self.eta,
                 'num_class': n_classes,  # y_train.shape[1],
                 'seed': 2925,  # for reproducibility
                 'silent': 1}
        if self.gpu < 0:
            param['nthread'] = -1  # with -1 it will use all available threads
        else:
            param['gpu_id'] = self.gpu
            param['updater'] = 'grow_gpu'

        dtrain = xgb.DMatrix(X_flat_train,
                             label=y_train)  # Put in correct format for XGB
        bst = xgb.train(param, dtrain, self.num_round)  # Train model

        self.model = bst

    def predict(self, X_flat_test):

        """Predict outcomes using trained XGBoost Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 1d array with integers as classes
            The predicted outputs
        """

        dtest = xgb.DMatrix(X_flat_test)  # Put in XGB format
        bst = self.model  # Get fit model
        y_test_predicted = bst.predict(dtest)  # Make prediction
        return y_test_predicted

# %% Looping pipeline for classification

class LoopwiseTransformer(BaseEstimator, TransformerMixin):
    """Transformer that applies loop-wise transformations.

    This transformer applies a base transformer to each loop dimension
    of the input data independently. It is useful for scenarios where
    the input data has a loop structure, such as time series data with
    multiple time steps or trials, and you want to apply the same
    transformation to each loop independently.

    Parameters
    ----------
    base_transformer : object
        The base transformer to apply to each loop dimension.
        This should be a scikit-learn compatible transformer.
    loop_dim : int, optional, default=1
        The dimension along which the loops are defined in the input data.
        For example, if the input data has shape (n_samples, n_loops, n_features),
        then loop_dim=1 means that the loops are along the second dimension.
     """
    def __init__(self, base_transformer, loop_dim: int = 1):
        self.base_transformer = base_transformer
        self.loop_dim = loop_dim

    def idx(self, ndim, i):
        """Create an index for the loop dimension."""
        return tuple(
            slice(None) if d != self.loop_dim else i
            for d in range(ndim)
        )

    def fit(self, X, y=None, **fit_params):
        # Handle degenerate or already-flattened case
        if X.ndim <= self.loop_dim:
            transformer = clone(self.base_transformer)
            weights = fit_params.pop('weights', None)
            if weights is not None:
                transformer.fit(X, y, weights=weights, **fit_params)
            else:
                transformer.fit(X, y, **fit_params)
            self.transformers_ = [transformer]
            return self
        n_loops = X.shape[self.loop_dim]
        self.transformers_ = []
        # Extract loop-wise weights if provided
        weights = fit_params.pop('weights', None)
        for i in range(n_loops):
            transformer = clone(self.base_transformer)
            if weights is not None:
                transformer.fit(X[self.idx(X.ndim, i)], y,
                                weights=weights[self.idx(weights.ndim, i)],
                                **fit_params)
            else:
                transformer.fit(X[self.idx(X.ndim, i)], y, **fit_params)
            self.transformers_.append(transformer)
        return self

    def transform(self, X):
        # If data is 2D or loop axis missing, just pass through single transformer
        if X.ndim <= self.loop_dim or len(self.transformers_) == 1:
            return self.transformers_[0].transform(X)
        transformed = [t.transform(X[self.idx(X.ndim, i)])
                       for i, t in enumerate(self.transformers_)]
        # Harmonize feature dimension across loops, then concatenate along feature axis
        n_feats = [arr.shape[-1] for arr in transformed]
        target = min(n_feats)
        xp = array_namespace(transformed[0])
        if any(n != target for n in n_feats):
            for i, arr in enumerate(transformed):
                if arr.shape[-1] > target:
                    transformed[i] = arr[..., :target]
                elif arr.shape[-1] < target:
                    pad = xp.zeros((arr.shape[0], target - arr.shape[-1]), dtype=arr.dtype)
                    if hasattr(xp, 'concatenate'):
                        transformed[i] = xp.concatenate((arr, pad), axis=-1)
                    else:
                        # Torch fallback
                        transformed[i] = xp.cat((arr, pad), dim=-1)
        # Flatten loop dimension into features: (samples, sum(features_per_loop))
        if hasattr(xp, 'concatenate'):
            out = xp.concatenate(transformed, axis=-1)
        else:
            out = xp.cat(transformed, dim=-1)
        # sklearn expects numpy arrays
        if is_torch(xp):
            return out.detach().cpu().numpy()
        return out

    def set_output(self, *, transform=None):
        """Set the output type of the transformer."""
        if hasattr(self, 'transformers_'):
            for transformer in self.transformers_:
                transformer.set_output(transform=transform)
        elif hasattr(self.base_transformer, 'set_output'):
            self.base_transformer.set_output(transform=transform)
        else:
            raise ValueError("Base transformer does not support set_output.")
        return self

class FlattenFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, samples_axis: int = 0, loop_axis: int | None = None):
        self.samples_axis = samples_axis
        self.loop_axis = loop_axis

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        xp = array_namespace(X)
        ndim = X.ndim
        sa = self.samples_axis % ndim
        la = None if self.loop_axis is None else (self.loop_axis % ndim)
        if la is not None and la == sa:
            la = None  # loop dim cannot coincide with samples dim

        # Build permutation to [samples, (loop), others...]
        axes = list(range(ndim))
        perm = [sa]
        if la is not None:
            perm.append(la)
        for ax in axes:
            if ax != sa and ax != la:
                perm.append(ax)
        Xp = xp.moveaxis(X, list(range(ndim)), perm)
        if la is not None:
            Xr = Xp.reshape(Xp.shape[0], Xp.shape[1], -1)
        else:
            Xr = Xp.reshape(Xp.shape[0], -1)
        return Xr


class OversampleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, samples_axis: int = 0, alpha: float = 1.):
        self.samples_axis = samples_axis
        self.alpha = alpha

    def fit(self, X, y=None, **fit_params):
        # no params to learn
        return self

    def transform(self, X):
        # Validation/test: fill NaNs with random normal noise based on data std
        xp = array_namespace(X)
        Xc = X.clone() if is_torch(xp) and hasattr(X, "clone") else X.copy()
        self.norm(Xc, xp)
        return Xc

    def fit_transform(self, X, y=None, **fit_params):
        xp = array_namespace(X)
        # copy to avoid in-place modifications on caller data
        Xc = X.clone() if is_torch(xp) and hasattr(X, "clone") else X.copy()
        if y is None:
            # Validation/test: fill NaNs with random normal noise
            self.norm(Xc, xp)
        # Training: label-aware mixup to impute NaNs
        else:
            if is_torch(xp):
                # Ensure labels tensor on same device/dtype compatible
                if 'torch' in globals() and torch is not None:
                    y_t = y if hasattr(y, 'device') else torch.as_tensor(y, device=Xc.device)
                else:
                    raise RuntimeError("PyTorch not available but tensor input detected.")
                mixup2(Xc, y_t, self.samples_axis, alpha=self.alpha)
            else:
                mixup2(Xc, xp.asarray(y), self.samples_axis, alpha=self.alpha)
        return Xc

    @staticmethod
    def norm(Xc, xp):
        # Replace any non-finite values (NaN/Inf) with random normal noise
        try:
            mask = ~xp.isfinite(Xc)
        except Exception:
            mask = xp.isnan(Xc)
        if mask.any():
            if is_torch(xp):
                if 'torch' in globals() and torch is not None:
                    Xc[mask] = torch.randn((int(mask.sum()),),
                                           device=Xc.device, dtype=Xc.dtype)
                else:
                    raise RuntimeError("PyTorch not available but tensor input detected.")
            else:
                Xc[mask] = xp.random.normal(0., 1., int(mask.sum()))


# %% PRINCIPAL COMPONENT ANALYSIS - Covariance Reducing CLASSIFIER

class CovarianceReducingClassifier(BaseEstimator):
    """Base class for PCA + downstream estimator pipelines (sklearn-compatible).

    Subclasses should override _default_estimator and set estimator_step_name
    to the name used in the Pipeline (e.g., 'discriminant' or 'clf').
    """

    model: Pipeline

    def __init__(self, pca: BaseEstimator, classifier: BaseEstimator,
                 memory=None, samples_axis: int = 0, oversample: bool = True):
        self.pca = pca
        self.classifier = classifier
        self.memory = memory
        self.samples_axis = samples_axis
        self.oversample = oversample
        steps = []
        if oversample:
            steps.append(('oversample', OversampleTransformer(samples_axis=self.samples_axis)))
        loop_axis = None
        if isinstance(self.pca, LoopwiseTransformer):
            loop_axis = self.pca.loop_dim
        steps.append(('flatten', FlattenFeaturesTransformer(samples_axis=self.samples_axis, loop_axis=loop_axis)))
        if self.pca is not None:
            steps.append(('pca', self.pca))
        steps.append(('classifier', self.classifier))
        self.model = Pipeline(steps=steps, memory=self.memory or Memory())

    def fit(self, X, y=None, **params):
        if params:
            self.model.set_params(**params)
        self.model.fit(X, y)

    def predict(self, X, **params):
        return self.model.predict(X, **params)

    def score(self, X, y, sample_weight=None, **params):

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight, **params)

# %% PRINCIPAL COMPONENT ANALYSIS - LINEAR DISCRIMINANT CLASSIFIER

class PcaLdaClassification(CovarianceReducingClassifier):
    """Class for the PCA - LDA Classifier

        Parameters
        ----------
    explained variance: integer, optional, default=80
        the number of modes that explain the cumulative variance of the dataset

    da_type: string, optional, de    fault=lda
        type of discriminant analysis; lda or qda

    """

    def __init__(self, explained_variance=0.8, da_type='lda', PCA_kwargs={},
                 loopwise: int = None, DA_kwargs={}, samples_axis: int = 0,
                 oversample: bool = True):

        self.explained_variance = explained_variance
        self.da_type = da_type
        self.loopwise = loopwise
        self.PCA_kwargs = PCA_kwargs
        self.DA_kwargs = DA_kwargs

        # choose discriminant type
        if (da_type == 'lda'):
            # linear discriminant analysis
            da_model = da.LinearDiscriminantAnalysis(**DA_kwargs)
        else:
            # Quadratic discriminant analysis
            da_model = da.QuadraticDiscriminantAnalysis(**DA_kwargs)

        PCA_kwargs['n_components'] = explained_variance
        if loopwise is not None:
            pca_transformer = LoopwiseTransformer(
                PCA(**PCA_kwargs), loop_dim=loopwise)
        else:
            pca_transformer = PCA(**PCA_kwargs)

        super().__init__(pca=pca_transformer, classifier=da_model,
                         memory=Memory(), samples_axis=samples_axis,
                         oversample=oversample)


    # %% PRINCIPAL COMPONENT ANALYSIS Wrapper for classification function

class PcaEstimateDecoder(CovarianceReducingClassifier):
    """Class for the PCA - SVM Classifier

    Parameters
    ----------
    explained_variance: float, optional, default=0.8
        the cumulative explained variance ratio required for PCA

    clf: object, optional, default=SVC()
        the classifier object to use for classification

    clf_params: dict, optional
        Additional parameters to be passed to the classifier (e.g.,
        {'param_name': value})

    """
    def __init__(self, explained_variance=0.8, clf_params={},
                 PCA_kwargs={}, samples_axis: int = 0,
                 oversample: bool = True):
        self.explained_variance = explained_variance
        self.clf_params = clf_params
        self.PCA_kwargs = PCA_kwargs

        PCA_kwargs['n_components'] = explained_variance
        pca_transformer = PCA(**PCA_kwargs)
        clf_instance = SVC(**clf_params)
        super().__init__(pca=pca_transformer, classifier=clf_instance,
                         memory=Memory(), samples_axis=samples_axis,
                         oversample=oversample)

# %% TorchVision ResNet classifier (sklearn-compatible via skorch)
if torch is not None and NeuralNetClassifier is not None and tv_models is not None:
    # Masked Batch Normalization that respects masks
    class MaskedBatchNorm2d(nn.Module):
        """BatchNorm2d that computes statistics only over valid (non-masked) positions."""
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.track_running_stats = track_running_stats
            
            if self.affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            
            if self.track_running_stats:
                self.register_buffer('running_mean', torch.zeros(num_features))
                self.register_buffer('running_var', torch.ones(num_features))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            else:
                self.register_parameter('running_mean', None)
                self.register_parameter('running_var', None)
                self.register_parameter('num_batches_tracked', None)
        
        def forward(self, x, mask=None):
            # x: (N, C, H, W), mask: (N, C, H, W) or None
            # mask: True for valid, False for NaN
            if mask is None:
                # No mask - use standard batch norm
                return nn.functional.batch_norm(
                    x, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training or not self.track_running_stats, self.momentum, self.eps
                )
            
            # Masked batch norm: compute statistics only over valid positions
            N, C, H, W = x.shape
            # Ensure mask matches x shape
            if mask.shape != x.shape:
                if mask.shape[1] == 1:
                    mask = mask.expand_as(x)
                elif mask.shape[1] != C:
                    raise ValueError(f"Mask channels {mask.shape[1]} must match x channels {C} or be 1")
            
            # Compute mean and var per channel over valid positions only
            if self.training:
                # Training: compute batch statistics over valid positions
                mean = []
                var = []
                for c in range(C):
                    ch_data = x[:, c, :, :]  # (N, H, W)
                    ch_mask = mask[:, c, :, :]  # (N, H, W)
                    valid_data = ch_data[ch_mask]  # Flattened valid values
                    if valid_data.numel() > 0:
                        ch_mean = valid_data.mean()
                        ch_var = valid_data.var(unbiased=False)
                    else:
                        # All positions masked for this channel - use running stats or 0/1
                        if self.track_running_stats:
                            ch_mean = self.running_mean[c]
                            ch_var = self.running_var[c]
                        else:
                            ch_mean = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                            ch_var = torch.tensor(1.0, device=x.device, dtype=x.dtype)
                    mean.append(ch_mean)
                    var.append(ch_var)
                mean = torch.stack(mean)  # (C,)
                var = torch.stack(var)  # (C,)
                
                # Update running statistics
                if self.track_running_stats:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
            else:
                # Eval: use running statistics
                mean = self.running_mean
                var = self.running_var
            
            # Normalize: (x - mean) / sqrt(var + eps)
            mean = mean.view(1, C, 1, 1)
            var = var.view(1, C, 1, 1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            
            # Apply affine transform if enabled
            if self.affine:
                x_norm = x_norm * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
            
            # Preserve NaNs where mask is False (masked positions remain NaN)
            # This allows NaNs to exist if all values in a dimension are masked
            x_norm = torch.where(mask, x_norm, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
            
            return x_norm
    
    # Helper function for masked convolution
    def _masked_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, mask=None):
        """Convolution that handles NaNs properly - NaNs propagate naturally.
        
        If all values in a convolution window are NaN, output will be NaN (correct behavior).
        We don't need special handling - standard convolution with NaNs produces NaN output.
        """
        if mask is None:
            return nn.functional.conv2d(x, weight, bias, stride, padding, dilation, groups)
        
        # Standard convolution - NaNs will propagate naturally
        # If all values in a convolution window are NaN, output will be NaN (correct)
        result = nn.functional.conv2d(x, weight, bias, stride, padding, dilation, groups)
        return result
    
    # Helper function for masked mean/sum operations
    def _masked_mean(x, mask, dim, keepdim=False):
        """Compute mean only over valid (non-masked) positions.
        
        If all positions are masked for a particular output location, result is NaN (correct).
        """
        if mask is None:
            return x.mean(dim=dim, keepdim=keepdim)
        # Use masked sum: sum valid values, divide by count of valid values
        # Temporarily use 0 for masked positions in sum (they don't contribute)
        # This is only for computation - NaNs remain in original data
        x_masked = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        counts = mask.sum(dim=dim, keepdim=keepdim)
        # If count is 0 (all masked), result should be NaN
        result = x_masked.sum(dim=dim, keepdim=keepdim) / counts.clamp(min=1)
        # Set to NaN where count was 0 (all positions masked)
        result = torch.where(counts > 0, result, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
        return result
    
    def _masked_sum(x, mask, dim, keepdim=False):
        """Compute sum only over valid (non-masked) positions.
        
        If all positions are masked, result is 0 (sum of nothing).
        """
        if mask is None:
            return x.sum(dim=dim, keepdim=keepdim)
        # Temporarily use 0 for masked positions in sum (they don't contribute)
        x_masked = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        return x_masked.sum(dim=dim, keepdim=keepdim)
    
    # Custom gradient clipping that clips infinite values but preserves NaNs
    class SafeGradientNormClipping(Callback):
        """Gradient clipping that clips infinite values but does not modify NaNs.
        
        NaNs should be handled by masking in the forward pass, not by gradient clipping.
        This callback only clips infinite gradient values to prevent overflow.
        """
        def __init__(self, clip_value: float = 1.0):
            self.clip_value = clip_value
        
        def on_grad_computed(self, net, named_parameters, **kwargs):
            # Clip infinite gradients but preserve NaNs (they indicate a problem upstream)
            for name, param in named_parameters:
                if param.grad is not None:
                    # Only clip infinite values, preserve NaNs
                    if torch.isinf(param.grad).any():
                        # Clip infinite values to clip_value
                        param.grad = torch.where(
                            torch.isinf(param.grad),
                            torch.sign(param.grad) * self.clip_value,
                            param.grad
                        )
                        # Also clamp to prevent any remaining extreme values
                        param.grad = torch.clamp(param.grad, -self.clip_value, self.clip_value)
    
    class ImageStandardizeTransformer(BaseEstimator, TransformerMixin):
        """Standardize NCHW inputs per-channel using training set statistics.
        
        Handles MaskedTensor inputs by computing statistics only over valid (non-masked) values.
        """
        def __init__(self, eps: float = 1e-6):
            self.eps = eps
            self.mean_ = None
            self.std_ = None
        
        def fit(self, X, y=None, **fit_params):
            xp = array_namespace(X)
            # Handle MaskedTensor or regular tensor
            mask = None
            if hasattr(X, 'get_mask'):
                # MaskedTensor: extract data and mask
                try:
                    from torch.masked import MaskedTensor
                    if isinstance(X, MaskedTensor):
                        mask = X.get_mask()
                        Xc = X.get_data()
                        xp = array_namespace(Xc)
                except (ImportError, AttributeError):
                    pass
            elif hasattr(X, '_nan_mask'):
                # Fallback: mask stored as attribute
                mask = X._nan_mask
                Xc = X
            else:
                Xc = X
            
            # X shape: (N, C, H, W)
            if mask is not None and is_torch(xp):
                # Compute mean/std only over valid (non-masked) values
                Xc = torch.as_tensor(Xc, dtype=torch.float32)
                mask = torch.as_tensor(mask, dtype=torch.bool)
                # Per-channel mean over valid values (masked values ignored)
                mean = []
                for c in range(Xc.shape[1]):
                    ch_data = Xc[:, c, :, :]
                    ch_mask = mask[:, c, :, :]
                    if ch_mask.any():
                        # Use nanmean to handle NaNs properly (mask ensures we only use valid values)
                        valid_data = ch_data[ch_mask]
                        mean.append(valid_data.nanmean().item() if torch.isnan(valid_data).any() else valid_data.mean().item())
                    else:
                        mean.append(0.0)
                mean = torch.tensor(mean, dtype=torch.float32, device=Xc.device)
                # Per-channel std over valid values
                var = []
                for c in range(Xc.shape[1]):
                    ch_data = Xc[:, c, :, :]
                    ch_mask = mask[:, c, :, :]
                    if ch_mask.any():
                        valid_data = ch_data[ch_mask]
                        centered = (valid_data - mean[c]) ** 2
                        var.append(centered.nanmean().item() if torch.isnan(centered).any() else centered.mean().item())
                    else:
                        var.append(1.0)
                var = torch.tensor(var, dtype=torch.float32, device=Xc.device)
                self.mean_ = mean
                self.std_ = torch.sqrt(var) + self.eps
            else:
                # When oversample=True, OversampleTransformer handles NaNs via mixup/norm
                # Compute statistics normally (NaNs should already be handled)
                # Per-channel mean/std over N,H,W
                mean = Xc.mean(axis=(0, 2, 3))
                var = ((Xc - mean[None, :, None, None]) ** 2).mean(axis=(0, 2, 3))
                # Store as float32 to avoid upcasting the input to float64
                self.mean_ = xp.asarray(mean, dtype='f4')
                self.std_ = xp.sqrt(xp.asarray(var, dtype='f4')) + self.eps
            return self
        
        def transform(self, X):
            xp = array_namespace(X)
            # Handle MaskedTensor or regular tensor
            mask = None
            is_masked = False
            if hasattr(X, 'get_mask'):
                try:
                    from torch.masked import MaskedTensor
                    if isinstance(X, MaskedTensor):
                        mask = X.get_mask()
                        X = X.get_data()
                        is_masked = True
                        xp = array_namespace(X)
                except (ImportError, AttributeError):
                    pass
            elif hasattr(X, '_nan_mask'):
                mask = X._nan_mask
                is_masked = True
            
            # When oversample=True, OversampleTransformer already handled NaNs
            # When oversample=False, NaNs are preserved and masked
            
            X = torch.as_tensor(X, dtype=torch.float32) if is_torch(xp) else xp.asarray(X, dtype='f4')
            mean_t = torch.as_tensor(self.mean_, dtype=torch.float32, device=X.device) if is_torch(xp) else self.mean_
            std_t = torch.as_tensor(self.std_, dtype=torch.float32, device=X.device) if is_torch(xp) else self.std_
            
            # Standardize: NaNs remain NaNs, mask will handle them
            X = X - mean_t[None, :, None, None]
            X = X / std_t[None, :, None, None]
            
            # Re-wrap in MaskedTensor if input was masked
            if is_masked and mask is not None:
                try:
                    from torch.masked import MaskedTensor
                    X = MaskedTensor(X, mask)
                except (ImportError, AttributeError):
                    X._nan_mask = mask
            
            return X

    class ResNetInputTransformer(BaseEstimator, TransformerMixin):
        """Permute and reshape SEEG tensor into NCHW for ResNet.
        
        Expects an array with at least sample, channel, frequency, and time axes.
        Produces shape (N, C, H, W) where:
          - C = number of channels (from channel_axis)
          - W = time (from time_axis)
          - H = product of remaining feature dims (e.g., frequency and others)
        
        If oversample=False and NaNs are present, returns a PyTorch MaskedTensor.
        """
        def __init__(self, samples_axis: int = 0, channel_axis: int = -3,
                     freq_axis: int = -2, time_axis: int = -1, oversample: bool = True):
            self.samples_axis = samples_axis
            self.channel_axis = channel_axis
            self.freq_axis = freq_axis
            self.time_axis = time_axis
            self.oversample = oversample
        
        def fit(self, X, y=None, **fit_params):
            return self
        
        def transform(self, X):
            xp = array_namespace(X)
            ndim = X.ndim
            sa = self.samples_axis % ndim
            ca = self.channel_axis % ndim
            ta = self.time_axis % ndim
            axes = list(range(ndim))
            others = [ax for ax in axes if ax not in (sa, ca, ta, ca)]
            # Order: samples, channels, others..., time
            perm = [sa, ca] + others + [ta]
            Xt = xp.transpose(X, axes=perm)
            N = Xt.shape[0]
            C = int(Xt.shape[1])
            H = int(np.prod([int(s) for s in Xt.shape[2:-1]]) or 1)
            W = int(Xt.shape[-1])
            # Ensure float32
            Xn = xp.asarray(Xt, dtype='f4').reshape((N, C, H, W))
            
            # Handle NaNs based on oversample setting
            if not self.oversample:
                # Check for NaNs and create masked tensor if present
                try:
                    finite = xp.isfinite(Xn)
                    has_nans = not xp.all(finite)
                except Exception:
                    has_nans = False
                
                if has_nans and is_torch(xp):
                    # Convert to PyTorch tensor if not already
                    if not hasattr(Xn, 'device'):
                        Xn = torch.as_tensor(Xn, dtype=torch.float32)
                    
                    # Create mask: True for valid (non-NaN) values, False for NaN
                    mask = torch.isfinite(Xn)
                    
                    # DO NOT replace NaNs - keep them in the data, mask will handle them
                    # Clip extremes only on finite values
                    Xn = torch.where(mask, torch.clamp(Xn, -1e6, 1e6), Xn)
                    
                    # Create MaskedTensor with original data (including NaNs) and mask
                    try:
                        from torch.masked import MaskedTensor
                        Xn = MaskedTensor(Xn, mask)
                    except (ImportError, AttributeError):
                        # Fallback: store mask as attribute for later use
                        Xn._nan_mask = mask
            else:
                # When oversample=True, OversampleTransformer handles NaNs via mixup/norm
                # Just clip extremes here (don't replace NaNs)
                try:
                    Xn = xp.clip(Xn, -1e6, 1e6)
                except Exception:
                    pass
            return Xn
    
    class SEEGResNet(nn.Module):
        """ResNet backbone adapted for single-channel SEEG 'images'."""
        def __init__(self, num_classes: int, base: str = 'resnet18',
                     pretrained: bool = False, dropout: float = 0.0,
                     in_channels: int = 1,
                     use_amp: bool = True,
                     amp_dtype: "torch.dtype" = None):
            super().__init__()
            self.use_amp = use_amp
            # Prefer BF16 when supported for better stability, else FP16
            if amp_dtype is None:
                try:
                    self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    self.amp_dtype = torch.float16
            else:
                self.amp_dtype = amp_dtype
            # Select backbone
            if base == 'resnet18':
                backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            elif base == 'resnet34':
                backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            elif base == 'resnet50':
                backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            else:
                raise ValueError(f"Unsupported ResNet base: {base}")
            # Adapt first conv to multi-channel input
            old_conv = backbone.conv1
            self.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride, padding=old_conv.padding, bias=False)
            # He init for new conv
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            # Replace batch norm with masked batch norm to handle NaNs properly
            self.bn1 = MaskedBatchNorm2d(old_conv.out_channels, 
                                         eps=backbone.bn1.eps,
                                         momentum=backbone.bn1.momentum,
                                         affine=backbone.bn1.affine,
                                         track_running_stats=backbone.bn1.track_running_stats)
            # Copy weights from original batch norm if affine
            if backbone.bn1.affine:
                self.bn1.weight.data.copy_(backbone.bn1.weight.data)
                self.bn1.bias.data.copy_(backbone.bn1.bias.data)
            if backbone.bn1.track_running_stats:
                self.bn1.running_mean.copy_(backbone.bn1.running_mean)
                self.bn1.running_var.copy_(backbone.bn1.running_var)
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.avgpool = backbone.avgpool
            in_features = backbone.fc.in_features
            if dropout and dropout > 0:
                self.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, num_classes)
                )
            else:
                self.fc = nn.Linear(in_features, num_classes)
        
        def forward(self, x):
            # Extract mask once globally - mask: True for valid, False for NaN
            mask = None
            if hasattr(x, 'get_mask'):
                try:
                    from torch.masked import MaskedTensor
                    if isinstance(x, MaskedTensor):
                        mask = x.get_mask()
                        x = x.get_data()  # Extract data with NaNs still present
                except (ImportError, AttributeError):
                    pass
            elif hasattr(x, '_nan_mask'):
                mask = x._nan_mask
                x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            
            use_amp_now = self.use_amp and x.is_cuda
            # autocast only affects CUDA; safe no-op on CPU
            with torch.cuda.amp.autocast(enabled=use_amp_now, dtype=self.amp_dtype):
                # Convolution: NaNs will propagate naturally
                # If all values in a convolution window are NaN, output will be NaN (correct)
                x = _masked_conv2d(x, self.conv1.weight, None, 
                                  stride=self.conv1.stride, padding=self.conv1.padding,
                                  dilation=self.conv1.dilation, groups=self.conv1.groups, mask=mask)
                
                # Update mask for batch norm (conv preserves spatial dimensions with padding)
                # Conv output channels may differ from input channels
                if mask is not None:
                    if mask.shape[1] == x.shape[1]:
                        bn_mask = mask
                    else:
                        # Broadcast first channel's mask to all output channels
                        bn_mask = mask[:, 0:1, :, :].expand(-1, x.shape[1], -1, -1)
                    x = self.bn1(x, mask=bn_mask)
                    # Update mask: after batch norm, mask is same shape as x
                    mask = bn_mask
                else:
                    x = self.bn1(x)
                
                x = self.relu(x)
                x = self.maxpool(x)
                # Update mask after maxpool (spatial dimensions reduced)
                if mask is not None:
                    # Maxpool reduces spatial dimensions - compute mask for output
                    # Use maxpool on mask (True=valid, False=NaN) - output is valid if any input was valid
                    mask = torch.nn.functional.max_pool2d(mask.float(), 
                                                          kernel_size=self.maxpool.kernel_size,
                                                          stride=self.maxpool.stride,
                                                          padding=self.maxpool.padding).bool()
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                # Average pooling: use masked mean if mask exists
                if mask is not None:
                    # Adaptive avgpool preserves spatial structure, then we flatten
                    x = self.avgpool(x)
                    # For avgpool output, compute mask (valid if any spatial position was valid)
                    # avgpool typically produces (N, C, 1, 1), so mask should be (N, C, 1, 1)
                    mask_pooled = mask.any(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
                    # Preserve NaNs where all spatial positions were masked
                    x = torch.where(mask_pooled, x, torch.tensor(float('nan'), device=x.device, dtype=x.dtype))
                else:
                    x = self.avgpool(x)
                
                x = torch.flatten(x, 1)
                x = self.fc(x)
                
                # Final check: NaNs can exist for specific trials if all channels/spatial positions were NaN
                # Only raise error if ALL values are NaN (indicating a fundamental problem)
                if torch.isnan(x).all():
                    raise RuntimeError("All values are NaN in final output - fundamental problem detected")
            # Ensure float32 output for downstream sklearn/skorch (predict_proba expects float32)
            return x.float()
    
    class ResNetTokenClassifier(BaseEstimator):
        """Sklearn-compatible classifier using torchvision ResNet via skorch.
        
        Pipeline: (optional) mixup oversample -> NCHW transform -> ResNet classifier.
        """
        def __init__(self,
                     base: str = 'resnet18',
                     pretrained: bool = False,
                     dropout: float = 0.0,
                     max_epochs: int = 20,
                     lr: float = 1e-3,
                     batch_size: int = 64,
                     device: str = 'auto',
                     optimizer=torch.optim.AdamW,
                     samples_axis: int = 0,
                     channel_axis: int = -3,
                     freq_axis: int = -2,
                     time_axis: int = -1,
                     oversample: bool = True,
                     alpha: float = 1.0,
                     use_amp: bool = True,
                     amp_dtype: "torch.dtype" = None,
                     early_stopping: bool = True,
                     es_patience: int = 20,
                     es_threshold: float = 0.0,
                     es_load_best: bool = True,
                     # LR scheduling (ReduceLROnPlateau)
                     lr_schedule: str = 'plateau',  # 'plateau' or 'none'
                     lr_factor: float = 0.5,
                     lr_patience: int = 10,
                     lr_min_lr: float = 1e-6,
                     lr_threshold: float = 1e-3,
                     lr_cooldown: int = 2,
                     # Regularization and loss
                     optimizer_weight_decay: float = 3e-4,
                     label_smoothing: float = 0.1,
                     class_weight: str | None = None):
            self.base = base
            self.pretrained = pretrained
            self.dropout = dropout
            self.max_epochs = max_epochs
            self.lr = lr
            self.batch_size = batch_size
            self.device = device
            self.optimizer = optimizer
            self.samples_axis = samples_axis
            self.channel_axis = channel_axis
            self.freq_axis = freq_axis
            self.time_axis = time_axis
            self.oversample = oversample
            self.alpha = alpha
            self.use_amp = use_amp
            self.amp_dtype = amp_dtype
            self.early_stopping = early_stopping
            self.es_patience = es_patience
            self.es_threshold = es_threshold
            self.es_load_best = es_load_best
            self.lr_schedule = lr_schedule
            self.lr_factor = lr_factor
            self.lr_patience = lr_patience
            self.lr_min_lr = lr_min_lr
            self.lr_threshold = lr_threshold
            self.lr_cooldown = lr_cooldown
            self.optimizer_weight_decay = optimizer_weight_decay
            self.label_smoothing = label_smoothing
            self.class_weight = class_weight
            self.model = None
        
        def _infer_in_channels(self, X):
            ndim = X.ndim
            ca = self.channel_axis % ndim
            return int(X.shape[ca])
        
        def _build_pipeline(self, n_classes: int, in_channels: int):
            steps = []
            if self.oversample:
                steps.append(('oversample', OversampleTransformer(samples_axis=self.samples_axis, alpha=self.alpha)))
            steps.append(('to_image', ResNetInputTransformer(
                samples_axis=self.samples_axis,
                channel_axis=self.channel_axis,
                freq_axis=self.freq_axis,
                time_axis=self.time_axis,
                oversample=self.oversample
            )))
            # Standardize per-channel after shaping to NCHW
            steps.append(('standardize', ImageStandardizeTransformer()))
            # Configure skorch net
            # Determine AMP dtype default if not provided
            if self.amp_dtype is None:
                try:
                    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    amp_dtype = torch.float16
            else:
                amp_dtype = self.amp_dtype
            callbacks_list = []
            # Use custom gradient clipping that only clips infinite values, not NaNs
            callbacks_list.append(SafeGradientNormClipping(1.0))
            if self.early_stopping and EarlyStopping is not None:
                try:
                    callbacks_list.append(EarlyStopping(
                        monitor='valid_loss',
                        patience=self.es_patience,
                        threshold=self.es_threshold,
                        lower_is_better=True,
                        load_best=self.es_load_best
                    ))
                except TypeError:
                    # Fallback for older skorch without load_best arg
                    callbacks_list.append(EarlyStopping(
                        monitor='valid_loss',
                        patience=self.es_patience,
                        threshold=self.es_threshold,
                        lower_is_better=True
                    ))
            if self.lr_schedule == 'plateau' and LRScheduler is not None:
                callbacks_list.append(LRScheduler(
                    policy='ReduceLROnPlateau',
                    monitor='valid_loss',
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    min_lr=self.lr_min_lr,
                    threshold=self.lr_threshold,
                    cooldown=self.lr_cooldown
                ))
            net = NeuralNetClassifier(
                module=SEEGResNet,
                module__num_classes=n_classes,
                module__base=self.base,
                module__pretrained=self.pretrained,
                module__dropout=self.dropout,
                module__in_channels=in_channels,
                module__use_amp=self.use_amp,
                module__amp_dtype=amp_dtype,
                max_epochs=self.max_epochs,
                lr=self.lr,
                optimizer=self.optimizer,
                optimizer__weight_decay=self.optimizer_weight_decay,
                batch_size=self.batch_size,
                device=('cuda' if (self.device in ('auto', 'cuda') and torch.cuda.is_available()) else 'cpu'),
                iterator_train__shuffle=True,
                criterion=torch.nn.CrossEntropyLoss,
                criterion__label_smoothing=self.label_smoothing,
                callbacks=callbacks_list
            )
            steps.append(('classifier', net))
            return Pipeline(steps=steps, memory=Memory())
        
        def fit(self, X, y=None, **params):
            if y is None:
                raise ValueError("ResNetTokenClassifier requires labels y for classification.")
            n_classes = int(len(np.unique(y)))
            in_channels = self._infer_in_channels(X)
            self.model = self._build_pipeline(n_classes, in_channels)
            # Optional class weighting for imbalanced labels
            if self.class_weight == 'balanced' and torch is not None:
                # Compute inverse frequency weights
                classes, counts = np.unique(y, return_counts=True)
                total = counts.sum()
                weights = (total / (len(classes) * counts)).astype('f4')
                # Map to class order used by the criterion (sorted classes)
                order = np.argsort(classes)
                w_sorted = weights[order]
                device = ('cuda' if (self.device in ('auto', 'cuda') and torch.cuda.is_available()) else 'cpu')
                w_tensor = torch.tensor(w_sorted, dtype=torch.float32, device=device)
                self.model.set_params(classifier__criterion__weight=w_tensor)
            if params:
                self.model.set_params(**params)
            self.model.fit(X, y)
            return self

        def predict(self, X, **params):
            if self.model is None:
                raise RuntimeError("Model not fitted. Call fit before predict.")
            return self.model.predict(X, **params)

        def score(self, X, y, sample_weight=None, **params):
            return accuracy_score(y, self.predict(X, **params), sample_weight=sample_weight)
else:
    class ResNetTokenClassifier(BaseEstimator):
        def __init__(self, *args, **kwargs):
            raise ImportError("ResNetTokenClassifier requires torch, torchvision, and skorch to be installed.")


# %% Conformer-like sEEG classifier (inspired by seegnificant and EEG-Conformer)
# References:
# - seegnificant (NeurIPS 2024): https://gmentz.github.io/seegnificant
# - EEG-Conformer (TNSRE 2023): https://github.com/eeyhsong/EEG-Conformer
if torch is not None and NeuralNetClassifier is not None:
    class DebugCrossEntropyLoss(nn.Module):
        """CrossEntropyLoss wrapper that prints target/input stats when debug=True."""
        def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0,
                     debug: bool = False, num_classes: int | None = None):
            super().__init__()
            self.inner = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                             label_smoothing=label_smoothing)
            self.debug = debug
            self.num_classes = num_classes
        
        def forward(self, input: "torch.Tensor", target: "torch.Tensor"):
            t = target
            if t.dtype != torch.long:
                # Cast to indices as expected by CE
                t = t.to(torch.long)
            if self.debug:
                try:
                    # Basic stats on target and logits
                    tmin = int(t.min().item())
                    tmax = int(t.max().item())
                    nunique = int(torch.unique(t).numel())
                    in_min = float(input.min().item())
                    in_max = float(input.max().item())
                    in_mean = float(input.mean().item())
                    print(f"[CE DEBUG] target dtype={t.dtype} shape={tuple(t.shape)} "
                          f"min={tmin} max={tmax} nunique={nunique} "
                          f"num_classes={self.num_classes if self.num_classes is not None else 'NA'} | "
                          f"logits min={in_min:.6g} max={in_max:.6g} mean={in_mean:.6g}")
                except Exception:
                    pass
            return self.inner(input, t)
    
    class ConformerInputTransformer(BaseEstimator, TransformerMixin):
        """Permute and reshape to (N, C, F, T) for SEEGConformer.
        
        - samples_axis: index of samples/trials dimension
        - channel_axis: index of channel/electrode dimension
        - freq_axis: index of frequency or feature dimension to optionally keep
        - time_axis: index of time dimension
        
        Any remaining feature dims (besides C, F, T) are folded into F.
        
        If oversample=False and NaNs are present, returns a PyTorch MaskedTensor.
        """
        def __init__(self, samples_axis: int = 0, channel_axis: int = -3,
                     freq_axis: int = -2, time_axis: int = -1, oversample: bool = True):
            self.samples_axis = samples_axis
            self.channel_axis = channel_axis
            self.freq_axis = freq_axis
            self.time_axis = time_axis
            self.oversample = oversample
        
        def fit(self, X, y=None, **fit_params):
            return self
        
        def transform(self, X):
            xp = array_namespace(X)
            ndim = X.ndim
            sa = self.samples_axis % ndim
            ca = self.channel_axis % ndim
            fa = self.freq_axis % ndim
            ta = self.time_axis % ndim
            axes = list(range(ndim))
            others = [ax for ax in axes if ax not in (sa, ca, fa, ta)]
            # Order to [samples, channels, others..., freq, time]
            perm = [sa, ca] + others + [fa, ta]
            Xt = xp.transpose(X, axes=perm)
            N, C = int(Xt.shape[0]), int(Xt.shape[1])
            F = int(np.prod([int(s) for s in Xt.shape[2:-1]]) or 1)
            T = int(Xt.shape[-1])
            Xn = xp.asarray(Xt, dtype='f4').reshape((N, C, F, T))
            
            # Handle NaNs based on oversample setting
            if not self.oversample:
                # Check for NaNs and create masked tensor if present
                try:
                    finite = xp.isfinite(Xn)
                    has_nans = not xp.all(finite)
                except Exception:
                    has_nans = False
                
                if has_nans and is_torch(xp):
                    # Convert to PyTorch tensor if not already
                    if not hasattr(Xn, 'device'):
                        Xn = torch.as_tensor(Xn, dtype=torch.float32)
                    
                    # Create mask: True for valid (non-NaN) values, False for NaN
                    mask = torch.isfinite(Xn)
                    
                    # DO NOT replace NaNs - keep them in the data, mask will handle them
                    # Clip extremes only on finite values
                    Xn = torch.where(mask, torch.clamp(Xn, -1e6, 1e6), Xn)
                    
                    # Create MaskedTensor with original data (including NaNs) and mask
                    try:
                        from torch.masked import MaskedTensor
                        Xn = MaskedTensor(Xn, mask)
                    except (ImportError, AttributeError):
                        # Fallback: store mask as attribute for later use
                        Xn._nan_mask = mask
            else:
                # When oversample=True, OversampleTransformer handles NaNs via mixup/norm
                # Just clip extremes here (don't replace NaNs)
                try:
                    Xn = xp.clip(Xn, -1e6, 1e6)
                except Exception:
                    pass
            return Xn
    
    class ConformerStandardizeTransformer(BaseEstimator, TransformerMixin):
        """Standardize NCFT inputs per-channel over N,F,T (mean 0, std 1).
        
        Handles MaskedTensor inputs by computing statistics only over valid (non-masked) values.
        """
        def __init__(self, eps: float = 1e-6):
            self.eps = eps
            self.mean_ = None
            self.std_ = None
        
        def fit(self, X, y=None, **fit_params):
            xp = array_namespace(X)
            # Handle MaskedTensor or regular tensor
            mask = None
            if hasattr(X, 'get_mask'):
                # MaskedTensor: extract data and mask
                try:
                    from torch.masked import MaskedTensor
                    if isinstance(X, MaskedTensor):
                        mask = X.get_mask()
                        Xc = X.get_data()
                        xp = array_namespace(Xc)
                except (ImportError, AttributeError):
                    pass
            elif hasattr(X, '_nan_mask'):
                # Fallback: mask stored as attribute
                mask = X._nan_mask
                Xc = X
            else:
                Xc = X
            
            # X shape: (N, C, F, T)
            if mask is not None and is_torch(xp):
                # Compute mean/std only over valid (non-masked) values
                Xc = torch.as_tensor(Xc, dtype=torch.float32)
                mask = torch.as_tensor(mask, dtype=torch.bool)
                # Per-channel mean over valid values (masked values ignored)
                mean = []
                for c in range(Xc.shape[1]):
                    ch_data = Xc[:, c, :, :]
                    ch_mask = mask[:, c, :, :]
                    if ch_mask.any():
                        # Use nanmean to handle NaNs properly (mask ensures we only use valid values)
                        valid_data = ch_data[ch_mask]
                        mean.append(valid_data.nanmean().item() if torch.isnan(valid_data).any() else valid_data.mean().item())
                    else:
                        mean.append(0.0)
                mean = torch.tensor(mean, dtype=torch.float32, device=Xc.device)
                # Per-channel std over valid values
                var = []
                for c in range(Xc.shape[1]):
                    ch_data = Xc[:, c, :, :]
                    ch_mask = mask[:, c, :, :]
                    if ch_mask.any():
                        valid_data = ch_data[ch_mask]
                        centered = (valid_data - mean[c]) ** 2
                        var.append(centered.nanmean().item() if torch.isnan(centered).any() else centered.mean().item())
                    else:
                        var.append(1.0)
                var = torch.tensor(var, dtype=torch.float32, device=Xc.device)
                self.mean_ = mean
                self.std_ = torch.sqrt(var) + self.eps
            else:
                # When oversample=True, OversampleTransformer handles NaNs via mixup/norm
                # Compute statistics normally (NaNs should already be handled)
                mean = Xc.mean(axis=(0, 2, 3))  # per-channel mean
                var = ((Xc - mean[None, :, None, None]) ** 2).mean(axis=(0, 2, 3))
                self.mean_ = xp.asarray(mean, dtype='f4')
                self.std_ = xp.sqrt(xp.asarray(var, dtype='f4')) + self.eps
            return self
        
        def transform(self, X):
            xp = array_namespace(X)
            # Handle MaskedTensor or regular tensor
            mask = None
            is_masked = False
            if hasattr(X, 'get_mask'):
                try:
                    from torch.masked import MaskedTensor
                    if isinstance(X, MaskedTensor):
                        mask = X.get_mask()
                        X = X.get_data()
                        is_masked = True
                        xp = array_namespace(X)
                except (ImportError, AttributeError):
                    pass
            elif hasattr(X, '_nan_mask'):
                mask = X._nan_mask
                is_masked = True
            
            # When oversample=True, OversampleTransformer already handled NaNs
            # When oversample=False, NaNs are preserved and masked
            
            X = torch.as_tensor(X, dtype=torch.float32) if is_torch(xp) else xp.asarray(X, dtype='f4')
            mean_t = torch.as_tensor(self.mean_, dtype=torch.float32, device=X.device) if is_torch(xp) else self.mean_
            std_t = torch.as_tensor(self.std_, dtype=torch.float32, device=X.device) if is_torch(xp) else self.std_
            
            # Standardize: NaNs remain NaNs, mask will handle them
            X = X - mean_t[None, :, None, None]
            X = X / std_t[None, :, None, None]
            
            # Re-wrap in MaskedTensor if input was masked
            if is_masked and mask is not None:
                try:
                    from torch.masked import MaskedTensor
                    X = MaskedTensor(X, mask)
                except (ImportError, AttributeError):
                    X._nan_mask = mask
            
            return X
    
    class SEEGConformer(nn.Module):
        """Convolutional-Transformer for sEEG decoding.
        
        Pipeline:
        1) Frequency reduction (mean over F) -> (N, C, T)
        2) Temporal depthwise Conv1d per channel -> tokens (dim=d_model)
        3) Self-attention in time (per channel)
        4) Aggregate over time -> (N, C, d_model)
        5) Add channel positional encodings (+ optional coord MLP on 3D positions)
        6) Self-attention across channels
        7) Global channel pooling -> classifier
        """
        def __init__(self,
                     num_classes: int,
                     d_model: int = 128,
                     nhead_time: int = 8,
                     depth_time: int = 2,
                     nhead_space: int = 8,
                     depth_space: int = 2,
                     kernel_size: int = 9,
                     dropout: float = 0.1,
                     coord_embed_dim: int = 32,
                     channel_positions: Optional[np.ndarray] = None,
                     use_amp: bool = True,
                     amp_dtype: "torch.dtype" = None,
                     debug: bool = False,
                     debug_max_prints: int = 5):
            super().__init__()
            self.use_amp = use_amp
            if amp_dtype is None:
                try:
                    self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    self.amp_dtype = torch.float16
            else:
                self.amp_dtype = amp_dtype
            self.d_model = d_model
            self.dropout = nn.Dropout(dropout)
            self.kernel_size = kernel_size
            # Temporal encoder (defined lazy after seeing C)
            self.temporal_conv = None  # created in forward when C known
            encoder_layer_t = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead_time, dim_feedforward=4*d_model,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            self.time_encoder = nn.TransformerEncoder(encoder_layer_t, num_layers=depth_time)
            # Spatial/channel encoder
            encoder_layer_s = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead_space, dim_feedforward=4*d_model,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            self.space_encoder = nn.TransformerEncoder(encoder_layer_s, num_layers=depth_space)
            # Channel index embedding (lazy initialized with C)
            self.channel_embed = None
            # Optional coordinate embedding
            self.coord_proj = None
            self.register_buffer('channel_positions_buf', None, persistent=False)
            if channel_positions is not None:
                # Expect (C, 3); actual C checked at first forward
                pos = torch.as_tensor(channel_positions, dtype=torch.float32)
                self.register_buffer('channel_positions_buf', pos, persistent=False)
                self.coord_proj = nn.Sequential(
                    nn.Linear(3, coord_embed_dim),
                    nn.GELU(),
                    nn.Linear(coord_embed_dim, d_model)
                )
            # Classifier
            self.classifier = nn.Linear(d_model, num_classes)
            # Debug config
            self.debug = debug
            self._debug_max_prints = debug_max_prints
            self._debug_prints = 0
        
        def _dbg(self, name: str, t: "torch.Tensor"):
            if not self.debug:
                return
            if self._debug_prints >= self._debug_max_prints:
                return
            try:
                numel = t.numel()
                n_nan = torch.isnan(t).sum().item()
                n_inf = torch.isinf(t).sum().item()
                finite = torch.isfinite(t)
                if finite.any():
                    t_f = t[finite]
                    t_min = t_f.min().item()
                    t_max = t_f.max().item()
                    t_mean = t_f.mean().item()
                else:
                    t_min = float('nan')
                    t_max = float('nan')
                    t_mean = float('nan')
                print(f"[SEEGConformer DEBUG] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                      f"numel={numel} nan={n_nan} inf={n_inf} min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g}")
            except Exception as _:
                # Avoid crashing due to debug
                pass
            finally:
                self._debug_prints += 1
        
        def _ensure_temporal(self, C: int, device: "torch.device | None" = None):
            if self.temporal_conv is None:
                # Depthwise temporal conv per channel
                self.temporal_conv = nn.Conv1d(
                    in_channels=C, out_channels=C*self.d_model,
                    kernel_size=self.kernel_size, padding=self.kernel_size // 2,
                    groups=C, bias=False
                )
                nn.init.kaiming_normal_(self.temporal_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.channel_embed is None:
                self.channel_embed = nn.Embedding(C, self.d_model)
            # Ensure lazily created modules are on the same device as the inputs/net
            if device is not None:
                self.temporal_conv.to(device)
                self.channel_embed.to(device)
                if self.coord_proj is not None:
                    self.coord_proj.to(device)
        
        def forward(self, x):
            # Extract mask once globally - mask: True for valid, False for NaN
            mask = None
            if hasattr(x, 'get_mask'):
                try:
                    from torch.masked import MaskedTensor
                    if isinstance(x, MaskedTensor):
                        mask = x.get_mask()
                        x = x.get_data()  # Extract data with NaNs still present
                except (ImportError, AttributeError):
                    pass
            elif hasattr(x, '_nan_mask'):
                mask = x._nan_mask
                x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            
            # x: (N, C, F, T)
            N, C, F, T = x.shape
            use_amp_now = self.use_amp and x.is_cuda
            # Keep input reduction in AMP
            with torch.cuda.amp.autocast(enabled=use_amp_now, dtype=self.amp_dtype):
                # Create lazy modules and move them to the right device
                self._ensure_temporal(C, device=x.device)
                # 1) Frequency reduction using masked mean - compute mean only over valid frequencies
                if mask is not None:
                    # mask shape: (N, C, F, T) - True for valid values
                    # Use masked mean: sum valid values, divide by count of valid values
                    # If all frequencies are NaN for a (N, C, T) position, result will be NaN (correct)
                    x = _masked_mean(x, mask, dim=2, keepdim=False)  # (N, C, T)
                    # Update mask: (N, C, T) is valid if any frequency was valid
                    mask = mask.any(dim=2)  # (N, C, T)
                else:
                    x = x.mean(dim=2)  # (N, C, T)
                self._dbg("after_freq_mean", x)
            # Compute numerically sensitive blocks in float32 to avoid NaNs
            with torch.cuda.amp.autocast(enabled=False):
                # 2) Temporal convolution: NaNs will propagate naturally
                # If all timepoints are NaN for a specific (N, C) combination, output will be NaN (correct)
                x = self.temporal_conv(x.float())  # (N, C*d_model, T)
                self._dbg("after_temporal_conv", x)
                # Update mask after temporal conv: output shape is (N, C*d_model, T)
                # Each output channel corresponds to an input channel, so expand mask accordingly
                if mask is not None:
                    # mask was (N, C, T), now need (N, C*d_model, T)
                    # Each input channel produces d_model output channels
                    mask = mask.unsqueeze(2).expand(-1, -1, self.d_model, -1)  # (N, C, d_model, T)
                    mask = mask.reshape(N, C * self.d_model, T)  # (N, C*d_model, T)
                
                # reshape to (N*C, T, d_model)
                x = x.view(N, C, self.d_model, T).permute(0, 1, 3, 2).reshape(N*C, T, self.d_model).float()
                # Update mask for reshaped x: (N*C, T, d_model)
                if mask is not None:
                    mask = mask.view(N, C, self.d_model, T).permute(0, 1, 3, 2).reshape(N*C, T, self.d_model)
                
                # 3) Time self-attention per channel
                # Transformer encoder: NaNs will propagate naturally
                x = self.time_encoder(x)  # (N*C, T, d_model)
                self._dbg("after_time_encoder", x)
                # Mask unchanged after transformer (same shape)
                
                # 4) Aggregate time using masked mean
                if mask is not None:
                    x = _masked_mean(x, mask, dim=1, keepdim=False)  # (N*C, d_model)
                else:
                    x = x.mean(dim=1)  # (N*C, d_model)
                x = x.view(N, C, self.d_model)  # (N, C, d_model)
                # Update mask after time aggregation: (N, C, d_model)
                if mask is not None:
                    mask = mask.any(dim=1)  # (N*C,) -> valid if any timepoint was valid
                    mask = mask.view(N, C)  # (N, C)
                    mask = mask.unsqueeze(2).expand(-1, -1, self.d_model)  # (N, C, d_model)
                
                # 5) Add channel positional embeddings (and optional coords)
                ch_ids = torch.arange(C, device=x.device)
                ch_emb = self.channel_embed(ch_ids).unsqueeze(0).float()  # (1, C, d_model)
                x = x + ch_emb
                if self.coord_proj is not None and self.channel_positions_buf is not None:
                    if self.channel_positions_buf.shape[0] == C:
                        coord_emb = self.coord_proj(self.channel_positions_buf).unsqueeze(0).float()  # (1, C, d_model)
                        x = x + coord_emb
                x = self.dropout(x)
                # Mask unchanged after embeddings/dropout (same shape)
                
                # 6) Space/channel self-attention
                # Transformer encoder: NaNs will propagate naturally
                x = self.space_encoder(x)  # (N, C, d_model)
                self._dbg("after_space_encoder", x)
                # Mask unchanged after transformer (same shape)
                
                # 7) Global channel pooling using masked mean
                if mask is not None:
                    x = _masked_mean(x, mask, dim=1, keepdim=False)  # (N, d_model)
                else:
                    x = x.mean(dim=1)  # (N, d_model)
                
                # Classifier
                x = self.classifier(x)
                self._dbg("logits", x)
                
                # Final check: NaNs can exist for specific trials if all channels were NaN
                # Only raise error if ALL values are NaN (indicating a fundamental problem)
                if torch.isnan(x).all():
                    raise RuntimeError("All values are NaN in final output - fundamental problem detected")
            # Cast to float32 for compatibility with sklearn/skorch conversion
            return x.float()
    
    class SEEGConformerClassifier(BaseEstimator):
        """Sklearn-compatible classifier wrapping SEEGConformer via skorch.
        
        Accepts optional `channel_positions` (numpy array (C,3)) for spatial encoding.
        """
        def __init__(self,
                     d_model: int = 128,
                     nhead_time: int = 8,
                     depth_time: int = 2,
                     nhead_space: int = 8,
                     depth_space: int = 2,
                     kernel_size: int = 9,
                     dropout: float = 0.1,
                     coord_embed_dim: int = 32,
                     channel_positions: Optional[np.ndarray] = None,
                     max_epochs: int = 30,
                     lr: float = 1e-3,
                     batch_size: int = 128,
                     device: str = 'auto',
                     optimizer=torch.optim.AdamW,
                     samples_axis: int = 0,
                     channel_axis: int = -3,
                     freq_axis: int = -2,
                     time_axis: int = -1,
                     oversample: bool = True,
                     alpha: float = 1.0,
                     debug: bool = False,
                     debug_max_prints: int = 5):
            self.d_model = d_model
            self.nhead_time = nhead_time
            self.depth_time = depth_time
            self.nhead_space = nhead_space
            self.depth_space = depth_space
            self.kernel_size = kernel_size
            self.dropout = dropout
            self.coord_embed_dim = coord_embed_dim
            self.channel_positions = channel_positions
            self.max_epochs = max_epochs
            self.lr = lr
            self.batch_size = batch_size
            self.device = device
            self.optimizer = optimizer
            self.samples_axis = samples_axis
            self.channel_axis = channel_axis
            self.freq_axis = freq_axis
            self.time_axis = time_axis
            self.oversample = oversample
            self.alpha = alpha
            self.debug = debug
            self.debug_max_prints = debug_max_prints
            self.model = None
        
        def _build_pipeline(self, n_classes: int):
            steps = []
            if self.oversample:
                steps.append(('oversample', OversampleTransformer(samples_axis=self.samples_axis, alpha=self.alpha)))
            steps.append(('to_tensor', ConformerInputTransformer(
                samples_axis=self.samples_axis,
                channel_axis=self.channel_axis,
                freq_axis=self.freq_axis,
                time_axis=self.time_axis,
                oversample=self.oversample
            )))
            # Standardize after shaping to (N, C, F, T)
            steps.append(('standardize', ConformerStandardizeTransformer()))
            callbacks_list = []
            # Use custom gradient clipping that only clips infinite values, not NaNs
            callbacks_list.append(SafeGradientNormClipping(1.0))
            net = NeuralNetClassifier(
                module=SEEGConformer,
                module__num_classes=n_classes,
                module__d_model=self.d_model,
                module__nhead_time=self.nhead_time,
                module__depth_time=self.depth_time,
                module__nhead_space=self.nhead_space,
                module__depth_space=self.depth_space,
                module__kernel_size=self.kernel_size,
                module__dropout=self.dropout,
                module__coord_embed_dim=self.coord_embed_dim,
                module__channel_positions=self.channel_positions,
                module__use_amp=True,
                module__amp_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
                module__debug=self.debug,
                module__debug_max_prints=self.debug_max_prints,
                max_epochs=self.max_epochs,
                lr=self.lr,
                optimizer=self.optimizer,
                batch_size=self.batch_size,
                device=('cuda' if (self.device in ('auto', 'cuda') and torch.cuda.is_available()) else 'cpu'),
                iterator_train__shuffle=True,
                callbacks=callbacks_list,
                # Debugging loss to inspect target/inputs just before loss computation
                criterion=DebugCrossEntropyLoss,
                criterion__ignore_index=-100,
                criterion__label_smoothing=0.0,
                criterion__debug=self.debug,
                criterion__num_classes=n_classes
            )
            steps.append(('classifier', net))
            return Pipeline(steps=steps, memory=Memory())
        
        def fit(self, X, y=None, **params):
            if y is None:
                raise ValueError("SEEGConformerClassifier requires labels y for classification.")
            n_classes = int(len(np.unique(y)))
            self.model = self._build_pipeline(n_classes)
            if params:
                self.model.set_params(**params)
            self.model.fit(X, y)
            return self
        
        def predict(self, X, **params):
            if self.model is None:
                raise RuntimeError("Model not fitted. Call fit before predict.")
            return self.model.predict(X, **params)
        
        def score(self, X, y, sample_weight=None, **params):
            return accuracy_score(y, self.predict(X, **params), sample_weight=sample_weight)
else:
    class SEEGConformerClassifier(BaseEstimator):
        def __init__(self, *args, **kwargs):
            raise ImportError("SEEGConformerClassifier requires torch and skorch to be installed.")


if __name__ == "__main__":
    pca = PcaLdaClassification()

