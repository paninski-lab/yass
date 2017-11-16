from __future__ import division
import numbers
from math import ceil, floor
import logging


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
    """

    def __init__(self, observations, n_channels, dtype, max_memory):
        self.observations = observations
        self.n_channels = n_channels
        self.max_memory = human_bytes(max_memory)
        self.itemsize = np.dtype(dtype).itemsize

        self.logger = logging.getLogger(__name__)

        self.logger.info('Max memory: {} bytes'.format(self.max_memory))

    def channelwise(self, from_time=None, to_time=None, channels='all',
                    complete_channel_batch=True):
        """
        Traverse temporal observations in the selected channels

        Parameters
        ----------
        from_time: int, optional
            Starting time, defaults to None

        to_time: int, optional
            Ending time, defaults to None

        channels: tuple or str, optional
            A tuple with the channel indexes or 'all' to traverse all channels,
            defaults to 'all'

        Notes
        -----
        One of from_time/to_time can be None (behavior is similar to what you
        would do when indexing arrays (e.g. array[:10] or array[10:])), but
        you cannot leave both as None
        """
        if not from_time and not to_time:
            raise ValueError('from_time and to_time cannot be both None')

        # size of all observations is any channel
        channel_size = self.observations * self.itemsize

        if complete_channel_batch and channel_size > self.max_memory:
            raise ValueError('Cannot traverse all observations in a '
                             'channel at once, each channel has a size of'
                             ' {} but maximum memory is {}'
                             .format(human_size(channel_size),
                                     human_size(self.max_memory)))

        channels = channels if channels != 'all' else range(self.n_channels)

        for ch in channels:
            yield (slice(from_time, to_time, None), ch)

    def temporalwise(self, from_time=None, to_time=None, channels='all'):
        """
        Traverse a temporal window in chunks for selected channels,
        chunk size is calculated depending on maximum memory. Each batch
        includes a chunk of observations from all selected channels
        """
        if not from_time and not to_time:
            raise ValueError('from_time and to_time cannot be both None')

        from_time = from_time if from_time else 0
        to_time = to_time if to_time else self.observations

        # TODO: support channel slices, lists of channels are slow to index
        channel_indexes = (channels if channels != 'all'
                           else slice(0, self.n_channels, None))

        # get the value of t and channels for the subset to traverse
        t_total = to_time - from_time
        channels_total = (self.n_channels if channels == 'all' else
                          len(channel_indexes))

        bytes_total = t_total * channels_total * self.itemsize
        obs_total = t_total * channels_total

        self.logger.info('Observations per channel: {}. Number of channels: '
                         '{}. Total observations: {} Size to traverse: {}'
                         .format(t_total, channels_total, obs_total,
                                 human_size(bytes_total)))

        # find how many observations (for every channel selected) we can fit
        # in memory
        obs_channel_batch = int(floor(self.max_memory /
                                (channels_total*self.itemsize)))
        obs_batch = obs_channel_batch * channels_total
        bytes_batch = obs_batch * self.itemsize

        self.logger.info('Observations per batch: {} ({}), {} observations '
                         'per channel'.format(obs_batch,
                                              human_size(bytes_batch),
                                              obs_channel_batch))

        n_batches = int(ceil(obs_total / obs_batch))

        self.logger.info('Number of batches: {}'.format(n_batches))

        obs_residual = obs_total - obs_batch * (n_batches - 1)
        obs_channel_residual = int(obs_residual / channels_total)
        bytes_residual = obs_residual * self.itemsize

        self.logger.info('Last batch with {} observations ({}), {} '
                         'observations per channel'
                         .format(obs_residual, human_bytes(bytes_residual),
                                 obs_channel_residual))

        # generate list with batch sizes (n of observations per channel
        # per batch)
        last_i = n_batches - 1

        for i in range(n_batches):
            start = i * obs_channel_batch

            if i < last_i:
                end = start + obs_channel_batch
                yield (slice(start, end, None), channel_indexes)
            else:
                end = start + obs_channel_residual
                yield (slice(start, end), channel_indexes)
