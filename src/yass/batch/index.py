from __future__ import division
import numbers
from math import ceil, floor
import logging


import numpy as np

ACTIVATE_HUMAN = False


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


class BatchIndexer(object):
    """
    """

    def __init__(self, observations, n_channels, dtype, max_memory):
        self.observations = observations
        self.n_channels = n_channels
        self.max_memory = human_bytes(max_memory)
        self.itemsize = np.dtype(dtype).itemsize

        self.logger = logging.getLogger(__name__)

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

        channels = channels if channels != 'all' else range(self.channels)

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
        channels = (channels if channels != 'all'
                    else slice(0, self.channels, None))

        total_t = to_time - from_time
        n_channels = self.channels if channels != 'all' else len(channels)
        total_bytes = total_t * n_channels * self.itemsize

        self.logger.info('Observations per channel: {}. Number of channels: '
                         '{} Size to traverse: {}'
                         .format(total_t, n_channels,
                                 human_size(total_bytes)))

        n_batches = int(ceil(total_bytes/self.max_memory))

        if n_batches == 1:
            self.logger.info('One batch of size: {}'
                             .format(human_size(total_bytes)))
            return [(slice(from_time, to_time, None), channels)]
        else:
            self.logger.info('Number of batches: {}'
                             .format(n_batches))

            residual_bytes = (total_bytes % n_batches) * n_channels
            batch_bytes = int(floor(total_bytes/n_batches))

            self.logger.info('Batch size: {}'
                             .format(human_size(batch_bytes)))

            self.logger.info('Residual: {}'
                             .format(human_size(residual_bytes)))

            # compute the number of per-channel observations in a single batch
            batch_bytes_per_channel = int(floor(batch_bytes/n_channels))

            self.logger.info('Per-channel bytes in each batch: {}'
                             .format(human_bytes(batch_bytes_per_channel)))




        # for low, high in channels:
        #     yield (slice(None, None, None), channels)
