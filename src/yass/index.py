"""
Functions for indexing recordings
"""
from __future__ import division

import numpy as np


class Indexer(object):
    """Eficicnet indexing for large mult-channel recordings

    Parameters
    ----------
    path: str
        Path to mult-channel recordings file
    n_channels: int
        Number of channels
    mode: str ('long', 'wide'), optional
        'long' if the dimensions are (n observations x n channels), 'wide'
        if they are (n channels x n observations), defaults to 'long'
    dtype: str, optional
        Recordings dtype, defaults to 'int16'

    Examples
    --------

    .. literalinclude:: ../../examples/indexer.py

    Notes
    -----
    Indexer uses a `numpy.memmap` under the hood
    """

    def __init__(self, path, n_channels, mode='long', dtype='int16'):
        self.path = path
        self.n_channels = n_channels
        self.mode = mode

        obs_total = len(np.memmap(path, dtype=dtype))
        self.obs_per_channel = obs_total/n_channels

        if mode not in ('long', 'wide'):
            raise ValueError("Invalid {} mode, it should be eiter 'long' or "
                             "'wide'")

        if not self.obs_per_channel.is_integer():
            raise ValueError
        else:
            self.obs_per_channel = int(self.obs_per_channel)

        shape = ((self.obs_per_channel, n_channels) if mode == 'long' else
                 (n_channels, self.obs_per_channel))

        self.data = np.memmap(path, dtype=dtype, shape=shape)

    def read(self, observations, channels='all'):
        """Read a range of observations from specific channels

        Parameters
        ----------
        observations: tuple
            (lowest, highest) indexes
        channels: tuple, optional
            Channels to read. If 'all', reads all channels. Channels index
            start from 0

        Returns
        -------

        Examples
        --------
        """
        channels = range(self.n_channels) if channels == 'all' else channels
        first, last = observations

        if self.mode == 'long':
            return self.data[first:last, channels]
        else:
            return self.data[channels, first:last]
