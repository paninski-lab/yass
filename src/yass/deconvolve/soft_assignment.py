"""Providing a class for dealing with soft assignments of spikes at the end."""


import copy as copy
import numpy as np
from tqdm import tqdm

from yass.cluster.cluster import read_spikes
from yass.deconvolve.template import WaveForms
from yass.deconvolve.merge import template_spike_dist_linear_align


def get_soft_assignments(templates, templates_upsampled, spike_train,
        spike_train_upsampled, filename_residual, n_similar_units=2): 
    """Given templates and spikes determines collision templates.

    params:
    -------
    templates: np.ndarray
        Has shape (# units, # channels, # time samples).
    n_similar_units: int
        Number of similar units that the spikes should be compare against.
    """

    n_spikes = spike_train.shape[0]
    temp = WaveForms(templates.transpose([0, 2, 1])
    pdist = temp.pair_dist() 

    soft_assignments = np.zeros([n_spikes, n_similar_units])
    sim_unit_map = np.zeros([temp.n_unit, n_similar_units]).astype(np.int)

    for unit in tqdm(range(temp.n_unit), "Computing soft assignments"):

        spt_idx = np.where(spike_train[:, 1] == unit)[0]
        spt = spike_train[spt_idx, 0]
        # Get all upsampled ids
        units = spike_train_upsampled[spt_idx, 1]

        spikes, success_idx = read_spikes(
            filename_residual, spt, temp.n_channel, spike_size=temp.n_time,
            units, templates_upsampled, residual_flag=True)

        sim_units = pdist[unit].argsort()[:n_similar_units]
        sim_unit_map[unit] = sim_units

        # Get distances of spikes to both similar units.
        dist_features = template_spike_dist_linear_align(
                templates=templates[sim_units],
                spikes=spikes)

        def softmax(x):
            """Sape must be (N, d)"""
            e = np.exp(x)
            return e / e.sum(axis=1)[:, None]
        # Note that we are actually doing soft-min by using negative distance.
        assignments = softmax(- dist_features.T)
        soft_assignments[spt_idx[success_idx], :] = assignments

    return soft_assignments, sim_unit_map
