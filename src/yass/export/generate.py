"""
Code for generating Phy input files:
http://phy-contrib.readthedocs.io/en/latest/template-gui/
"""

from datetime import datetime
import os.path as path

import numpy as np

from ..util import load_asset, load_yaml


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


def whitening_matrices(n_channels):
    """
    whitening_mat.npy - [nChannels, nChannels] double whitening matrix applied
    to the data during automatic spike sorting

    whitening_mat_inv.npy - [nChannels, nChannels] double, the inverse of the
    whitening matrix.
    """
    # return whitening matrix and the inverse
    return np.eye(n_channels), np.eye(n_channels)
