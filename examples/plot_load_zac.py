"""
Joint pca decoding
===================================

Takes high gamma filtered data with event labels from multiple subjects and
performs joint pca decoding
"""

# %% Imports
from ieeg.navigate import channel_outlier_marker, trial_ieeg
from ieeg.io import raw_from_layout
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.decoding.joint_pca.alignment_methods import JointPCADecomp
from ieeg.calc.mat import Labels
from bids import BIDSLayout
import mne
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %% Load Data
bids_root = mne.datasets.epilepsy_ecog.data_path()
# sample_path = mne.datasets.sample.data_path()
layout = BIDSLayout(bids_root)
raw = raw_from_layout(layout, subject="pt1", preload=True,
                      extension=".vhdr")

# %% Some preprocessing

# Mark channel outliers as bad
channel_outlier_marker(raw, 5)

# Exclude bad channels
raw.drop_channels(raw.info['bads'])
good = raw.copy()
good.load_data()

# Remove intermediates from mem
del raw

# CAR
good.set_eeg_reference()

# %% High Gamma Filter

ev1 = trial_ieeg(good, ["PD", "G16", "SLT1-3"], (-1, 2), preload=True)
base = trial_ieeg(good, "onset", (-1, 0.5), preload=True)

gamma.extract(ev1, copy=False, n_jobs=1)
gamma.extract(base, copy=False, n_jobs=1)
crop_pad(ev1, "500ms")
crop_pad(base, "500ms")

del good
# %% Zac do your stuff

data1 = ev1.get_data(slice(0, 40)).swapaxes(1, 2)
data2 = ev1.get_data(slice(41, 81)).swapaxes(1, 2)
phon_labels = Labels([["AD1-4, ATT1,2", "PD", "G16"]]).T

# %% Decoding
n_iter = 5
cv = KFold(n_splits=3, shuffle=True, random_state=42)

y_true_all, y_pred_all = [], []
for _ in range(n_iter):  # repeat K-fold evaluation n_iter times
    for train_idx, test_idx in cv.split(data1, phon_labels):
        # split X1 into train and test
        X1_train, y1_train = data1[train_idx], phon_labels[train_idx]
        X1_test, y1_test = data1[test_idx], phon_labels[test_idx]
        # X2, y2 are only pooled with X1 for training, so no splitting
        X2, y2 = data2, phon_labels
        
        # Jointly decompose X1 and X2 to shared subspace via PCA
        jointPCA = JointPCADecomp(n_components=3)
        X1_train, X2 = jointPCA.fit_transform([X1_train, X2], [y1_train, y2])
        # Transform X1_test to shared subspace
        X1_test = jointPCA.transform(X1_test, idx=0)

        # Reshape data into trials x features (features = PCs*time)
        X1_train = X1_train.reshape(X1_train.shape[0], -1)
        X1_test = X1_test.reshape(X1_test.shape[0], -1)
        X2 = X2.reshape(X2.shape[0], -1)

        # pool X1 training data and X2 data together
        X_train = np.concatenate((X1_train, X2), axis=0)
        y_train = np.concatenate((y1_train, y2), axis=0)

        # classification via a linear SVM
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X1_test)

        # store results
        y_true_all.extend(y1_test)
        y_pred_all.extend(y_pred)

# %% Results - Confusion Matrix
cmat = confusion_matrix(y_true_all, y_pred_all)
disp = ConfusionMatrixDisplay(cmat, display_labels=phon_labels)
disp.plot()

