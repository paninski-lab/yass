"""
Code for generating Phy input files:
http://phy-contrib.readthedocs.io/en/latest/template-gui/
"""

from datetime import datetime
import os.path as path

import numpy as np

from ..util import load_asset, load_yaml
from ..geometry import n_steps_neigh_channels


def params(path_to_config):
    """
    Generate phy's params.py from YASS' config.yaml
    """
    template = load_asset('phy/params.py')
    config = load_yaml(path_to_config)

    timestamp = datetime.now().strftime('%B %-d, %Y at %H:%M')
    dat_path = path.join(config['data']['root_folder'],
                         config['data']['recordings'])
    n_channels_dat = config['recordings']['n_channels']
    dtype = config['recordings']['dtype']
    sample_rate = config['recordings']['sampling_rate']

    params = template.format(timestamp=timestamp,
                             dat_path=dat_path,
                             n_channels_dat=n_channels_dat,
                             dtype=dtype,
                             offset=0,
                             sample_rate=sample_rate,
                             hp_filtered='True')

    return params


def amplitudes(spike_train):
    """
    amplitudes.npy - [nSpikes, ] double vector with the amplitude scaling
    factor that was applied to the template when extracting that spike
    """
    n_spikes, _ = spike_train.shape
    return np.ones(n_spikes)


def channel_map(n_channels):
    """
    Generate channel_map numpy.array. For n_channels, it generates a
    numpy.array as follows: [0, ..., n_channels - 1]
    """
    return np.arange(n_channels)


def pc_feature_ind(n_spikes, n_templates, n_channels, geom, neigh_channels,
                   spike_train, templates):
    """
    pc_feature_ind.npy - [nTemplates, nPCFeatures] uint32 matrix specifying
    which pcFeatures are included in the pc_features matrix.
    """

    # get main channel for each template
    templates_mainc = np.argmax(np.max(templates, axis=1), axis=0)

    # main channel for each spike based on templates_mainc
    spikes_mainc = np.zeros(n_spikes, 'int32')

    for j in range(n_spikes):
        spikes_mainc[j] = templates_mainc[spike_train[j, 1]]

    # number of neighbors to consider
    # NOTE: is the '2' ok to be hardcoded?
    neighbors = n_steps_neigh_channels(neigh_channels, 2)
    nneigh = np.max(np.sum(neighbors, 0))

    # ordered neighboring channels w.r.t. each channel
    c_idx = np.zeros((n_channels, nneigh), 'int32')

    for c in range(n_channels):
        c_idx[c] = (np.argsort(np.sum(np.square(geom - geom[c]), axis=1))
                    [:nneigh])

    pc_feature_ind = np.zeros((n_templates, nneigh), 'int32')

    for k in range(n_templates):
        pc_feature_ind[k] = c_idx[templates_mainc[k]]

    return pc_feature_ind


def similar_templates(templates):
    """
    similar_templates.npy - [nTemplates, nTemplates] single matrix giving the
    similarity score (larger is more similar) between each pair of templates
    """
    _, _, n_templates = templates.shape
    return np.corrcoef(np.reshape(templates, [-1, n_templates]).T)


def template_features(n_spikes, n_templates, score):
    """
    template_features.npy - [nSpikes, nTempFeatures] single matrix giving the
    magnitude of the projection of each spike onto nTempFeatures other
    features. Which other features is specified in template_feature_ind.npy
    """
    k_neigh = np.min((5, n_templates))

    template_features = np.zeros((n_spikes, k_neigh))

    # for j in range(n_spikes):

    #     ch_idx = c_idx[spikes_mainc[j]]
    #     kk = spike_train[j, 1]

    #     for k in range(k_neigh):
    #         template_features[j] = np.sum(
    #             np.multiply(score[j].T,
    #             templates_low_dim[ch_idx][:, :, template_feature_ind[kk, k]]))

    return template_features


def template_feature_ind(n_templates):
    """
    template_feature_ind.npy - [nTemplates, nTempFeatures] uint32 matrix
    specifying which templateFeatures are included in the template_features
    matrix.
    """
    k_neigh = np.min((5, n_templates))
    template_feature_ind = np.zeros((n_templates, k_neigh), 'int32')

    for k in range(n_templates):
        template_feature_ind[k] = np.argsort(-similar_templates[k])[:k_neigh]

    return templates_ind


def templates_ind(n_templates, n_channels):
    """
    templates_ind.npy - [nTemplates, nTempChannels] double matrix specifying
    the channels on which each template is defined. In the case of Kilosort
    templates_ind is just the integers from 0 to nChannels-1, since templates
    are defined on all channels.
    """
    templates_ind = np.zeros((n_templates, n_channels), 'int32')

    for k in range(n_templates):
        templates_ind[k] = np.arange(n_channels)

    return templates_ind


def whitening_matrices(n_channels):
    """
    whitening_mat.npy - [nChannels, nChannels] double whitening matrix applied
    to the data during automatic spike sorting

    whitening_mat_inv.npy - [nChannels, nChannels] double, the inverse of the
    whitening matrix.
    """
    # return whitening matrix and the inverse
    return np.eye(n_channels), np.eye(n_channels)
