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

    def __init__(self, observations, channels, dtype, max_memory):
        self.observations = observations
        self.channels = channels
        self.max_memory = human_bytes(max_memory)
        self.itemsize = np.dtype(dtype).itemsize

        self.logger = logging.getLogger(__name__)

    def channelwise(self, times='all', channels='all',
                    complete_channel_batch=True):
        """
        Traverse temporal observations in the selected channels
        """
        if times == 'all':

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
                yield (slice(None, None, None), ch)

        else:
            raise NotImplementedError('This feature has not been implemented')

    def temporalwise(self, times='all', channels='all'):
        """
        Traverse a temporal window in chunks for selected channels,
        chunk size is calculated depending on maximum memory
        """
        channels = (channels if channels != 'all'
                    else slice(0, self.channels, None))

        min_t = 0 if times == 'all' else times[0]
        max_t = self.observations if times == 'all' else times[1]

        total_t = max_t - min_t + 1
        total_channels = self.channels if channels != 'all' else len(channels)
        total_bytes = total_t * total_channels * self.itemsize

        self.logger.info('Size to traverse: {}'
                         .format(human_size(total_bytes)))

        batches = int(ceil(total_bytes/self.max_memory))

        if batches == 1:
            self.logger.info('One batch of size: {}'
                             .format(human_size(total_bytes)))
            return [(slice(min_t, max_t, None), channels)]
        else:
            self.logger.info('Number of batches: {}'
                             .format(batches))

            residual = total_bytes % self.max_memory
            batch_size = int(floor(total_bytes/self.max_memory))

            self.logger.info('Batch size: {}'
                             .format(human_size(batch_size)))

            self.logger.info('Residual: {}'
                             .format(human_size(residual)))



        # for low, high in channels:
        #     yield (slice(None, None, None), channels)
