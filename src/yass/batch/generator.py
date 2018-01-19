from __future__ import division
import numbers
from math import ceil, floor
import logging
from itertools import chain


import numpy as np

ACTIVATE_HUMAN = True


def human_size(nbytes):
    """
    Notes
    -----
    Source: https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes
    # noqa
    """

    if not ACTIVATE_HUMAN:
        return nbytes

    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    i = 0

    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1

    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')

    return '%s %s' % (f, suffixes[i])


def human_bytes(size):
    """
    Given a human-readable byte string (e.g. 2G, 10GB, 30MB, 20KB),
    return the number of bytes.  Will return 0 if the argument has
    unexpected form.

    Notes
    -----
    Based on: https://gist.github.com/beugley/ccd69945346759eb6142272a6d69b4e0
    """
    if isinstance(size, numbers.Integral):
        return size

    if (size[-1] == 'B'):
        size = size[:-1]

    if (size.isdigit()):
        bytes_ = int(size)

    else:

        bytes_ = size[:-1]
        unit = size[-1]

        if (bytes_.isdigit()):
            bytes_ = int(bytes_)

            if (unit == 'G'):
                bytes_ *= 1073741824
            elif (unit == 'M'):
                bytes_ *= 1048576
            elif (unit == 'K'):
                bytes_ *= 1024
            else:
                raise ValueError('Invalid units')
        else:
            raise ValueError('Invalid value')

    return bytes_


class IndexGenerator(object):
    """

    Parameters
    ----------
    max_memory: int or str
        Max memory to use in each batch, interpreted as bytes if int,
        if string, it can be any of {N}KB, {N}MB or {N}GB
    """

    def __init__(self, n_observations, n_channels, dtype, max_memory):
        self.n_observations = n_observations
        self.n_channels = n_channels
        self.max_memory = human_bytes(max_memory)
        self.itemsize = np.dtype(dtype).itemsize

        self.logger = logging.getLogger(__name__)

        self.logger.debug('Max memory: {}. Itemsize: {} bytes'
                          .format(human_size(self.max_memory),
                                  self.itemsize))

    @property
    def can_allocate_one_complete_channel(self):
        """
        Check wheter there is enough memory to allocate all observations
        in a single channel
        """
        channel_size = self.n_observations * self.itemsize
        return channel_size > self.max_memory

    def single_channel(self, force_complete_channel_batch=True, from_time=None,
                       to_time=None, channels='all'):
        """
        Traverse temporal observations in the selected channels, where
        every batch data strictly comes from one channel

        Parameters
        ----------
        force_complete_channel_batch: bool, optional
            If True, every index generated will correspond to all the
            observations in a single channel, hence
            n_batches = n_selected_channels, defaults to True. If True
            from_time and to_time must be None

        from_time: int, optional
            Starting time, defaults to None

        to_time: int, optional
            Ending time, defaults to None

        channels: int, tuple or str, optional
            A tuple with the channel indexes or 'all' to traverse all channels,
            defaults to 'all'

        Notes
        -----
        If both from_time and to_time are None, all the observations along
        the time axis will be traversed
        """
        from_time = from_time if from_time is not None else 0
        to_time = to_time if to_time is not None else self.n_observations
        t_total = to_time - from_time

        # size of selected osbervations in a single channel
        channel_size = t_total * self.itemsize

        channel_indexes = (channels if channels != 'all'
                           else range(self.n_channels))
        channel_indexes = ([channel_indexes]
                           if isinstance(channel_indexes, numbers.Integral)
                           else channel_indexes)

        # traversing all observations in every channel, n_batches = n_channels
        if force_complete_channel_batch:
            if channel_size >= self.max_memory:
                raise ValueError('Cannot traverse all observations in a '
                                 'channel at once, each channel has a size of'
                                 ' {} but maximum memory is {}'
                                 .format(human_size(channel_size),
                                         human_size(self.max_memory)))
            else:
                self.logger.debug('Loading all observations per channel, '
                                  'each channel has size of {} '
                                  '({} observations)'
                                  .format(human_size(channel_size),
                                          t_total))

            for ch in channel_indexes:
                yield (slice(from_time, to_time, None), ch)

        # partial onservations per batch
        else:
            generators = chain(*(self.multi_channel(from_time, to_time, ch)
                                 for ch in channel_indexes))
            for gen in generators:
                yield gen

    def multi_channel(self, from_time=None, to_time=None, channels='all'):
        """
        Traverse a temporal window in chunks for selected channels,
        chunk size is calculated depending on maximum memory. Each batch
        includes a chunk of observations from all selected channels

        Notes
        -----
        If both from_time and to_time are None, all the observations along
        the time axis will be traversed
        """
        from_time = from_time if from_time is not None else 0
        to_time = to_time if to_time is not None else self.n_observations

        # TODO: support channel slices, lists of channels are slow to index
        channel_indexes = (channels if channels != 'all'
                           else slice(0, self.n_channels, None))

        if channels == 'all':
            channel_indexes_length = self.n_channels
        elif isinstance(channel_indexes, numbers.Integral):
            channel_indexes_length = 1
        else:
            channel_indexes_length = len(channel_indexes)

        # get the value of t and channels for the subset to traverse
        t_total = to_time - from_time
        channels_total = (self.n_channels if channels == 'all' else
                          channel_indexes_length)

        bytes_total = t_total * channels_total * self.itemsize
        obs_total = t_total * channels_total

        self.logger.debug('Observations per channel: {:,}. Number of channels:'
                          ' {:,}. Total observations: {:,} Size to traverse: '
                          '{}'.format(t_total, channels_total, obs_total,
                                      human_size(bytes_total)))

        # find how many observations (for every channel selected) we can fit
        # in memory
        obs_channel_batch = int(floor(self.max_memory /
                                      (channels_total*self.itemsize)))
        obs_batch = obs_channel_batch * channels_total
        bytes_batch = obs_batch * self.itemsize

        self.logger.debug('Max observations per batch: {:,} ({}), {:,} '
                          'max observations per channel'
                          .format(obs_batch, human_size(bytes_batch),
                                  obs_channel_batch))

        n_batches = int(ceil(obs_total / obs_batch))

        self.logger.debug('Number of batches: {:,}'.format(n_batches))

        obs_residual = obs_total - obs_batch * (n_batches - 1)
        obs_channel_residual = int(obs_residual / channels_total)
        bytes_residual = obs_residual * self.itemsize

        self.logger.debug('Last batch with {:,} observations ({}), {:,} '
                          'observations per channel'
                          .format(obs_residual, human_bytes(bytes_residual),
                                  obs_channel_residual))

        # generate list with batch sizes (n of observations per channel
        # per batch)
        last_i = n_batches - 1

        for i in range(n_batches):
            start = from_time + i * obs_channel_batch

            if i < last_i:
                end = start + obs_channel_batch
                yield (slice(start, end, None), channel_indexes)
            else:
                end = start + obs_channel_residual
                yield (slice(start, end), channel_indexes)
