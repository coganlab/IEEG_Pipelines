""" Alignment Util Fcns

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import pickle

import numpy as np

from ieeg.calc.mat import Labels


def cnd_avg(data, labels):
    """Averages data trials along first axis by condition type present in
    labels.

    Args:
        data (ndarray): Data matrix with shape (n_trials, ...). The first
            dimension must be the trial dimension. Number and shape of other
            dimensions is arbitrary.
        labels (ndarray): Label array with shape (n_trials,).

    Returns:
        ndarray: Data matrix averaged within conditions with shape
        (n_conditions, ...).
    """
    data_shape = data.shape
    class_shape = (len(np.unique(labels)),) + data_shape[1:]
    data_by_class = np.zeros(class_shape)
    for i, seq in enumerate(np.unique(labels)):
        data_by_class[i] = np.mean(data[labels == seq], axis=0)
    return data_by_class


def label2str(labels):
    if not isinstance(labels, Labels):
        labels = Labels(labels, delim='')

    # Converts a 2D array of label sequences into a 1D array of label strings.
    if len(labels.shape) > 1:
        labels = labels.join(0)
    return labels.astype(str)


def save_pkl(data, filename):
    with open(filename, 'wb+') as f:
        pickle.dump(data, f, protocol=-1)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def decoding_data_from_dict(data_dict, pt, p_ind, lab_type='phon',
                            algn_type='phon_seq'):
    D_tar, lab_tar, lab_tar_full = get_features_labels(data_dict[pt], p_ind,
                                                       lab_type, algn_type)

    pre_data = []
    for p_pt in data_dict[pt]['pre_pts']:
        D_curr, lab_curr, lab_curr_full = get_features_labels(data_dict[p_pt],
                                                              p_ind, lab_type,
                                                              algn_type)
        pre_data.append((D_curr, lab_curr, lab_curr_full))

    return (D_tar, lab_tar, lab_tar_full), pre_data


def get_features_labels(data, p_ind, lab_type, algn_type):
    lab_full = data['y_full_' + algn_type[:-4]]
    if p_ind == -1:  # collapsed across all phonemes
        D = data['X_collapsed']
        lab = data['y_' + lab_type + '_collapsed']
        lab_full = np.tile(lab_full, (3, 1))  # label repeat for shape match
    else:  # individual phoneme
        D = data['X' + str(p_ind)]
        lab = data['y' + str(p_ind)]
    if lab_type == 'artic':  # convert from phonemes to articulator label
        lab = phon_to_artic_seq(lab)
    return D, lab, lab_full


def phon_to_artic_seq(phon_seq):
    phon_to_artic_conv = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4}
    flat_seq = phon_seq.flatten()
    artic_conv = np.array([phon_to_artic(phon_idx, phon_to_artic_conv) for
                           phon_idx in flat_seq])
    return np.reshape(artic_conv, phon_seq.shape)


def phon_to_artic(phon_idx, phon_to_artic_conv):
    return phon_to_artic_conv[phon_idx]
