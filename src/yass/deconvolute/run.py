import numpy as np

from .deconvolute import Deconvolution
from .. import read_config


def run(spike_train, spikes_left, templates):
    """Run deconvolution

    Parameters
    ----------
    spike_train
    spikes_left
    templates

    Returns
    ------
    spike_train
    """
    CONFIG = read_config()
    deconv = Deconvolution(CONFIG, np.transpose(templates, [1, 0, 2]),
                           spikes_left, filename='whiten.bin')
    spikes_deconv = deconv.fullMPMU()

    spikes_all = np.concatenate((spikes_deconv, spike_train))

    idx_sort = np.argsort(spikes_all[:, 0])
    spikes_all = spikes_all[idx_sort]

    idx_keep = np.zeros(spikes_all.shape[0], 'bool')

    for k in range(templates.shape[2]):
        idx_c = np.where(spikes_all[:, 1] == k)[0]
        idx_keep[idx_c[np.concatenate(([True], np.diff(spikes_all[idx_c,0]) > 1))]] = 1

    spikes_all = spikes_all[idx_keep]

    return spikes_all
