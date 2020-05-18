"""Provides a set of utility functions that is used for evaluations.

This includes finding main channel and signal to noise ration for templates.
"""

import numpy as np


def main_channels(template):
    """Computes the main channel of a list of templates.

    Parameters
    ----------
    template: numpy.ndarray
        The shape of the array should be (T, C, K) where T indicates
        time samples, C number of channels and K total number of
        units/clusters.
    """
    return np.argsort(np.max(
        np.abs(template), axis=0), axis=0).T


def temp_snr(templates):
    """Computes the PNR of a list of templates.

    Parameters
    ----------
    template: numpy.ndarray
        The shape of the array should be (T, C, K) where T indicates
        time samples, C number of channels and K total number of
        units/clusters.
    """
    tot = templates.shape[2]
    res = np.zeros(tot)
    for unit, c in enumerate(main_channels(templates)[:, -1]):
        res[unit] = np.linalg.norm(templates[:, c, unit], np.inf)
    return res
