import os
import numpy as np


# generate some sample data
# this generates data whose contiguous bytes contain observations from the
# same channel
data_wide = np.array(np.arange(10000)).reshape(10, 1000)
data_wide.tofile('data_wide.bin')

# this generates data whose contiguous bytes contain the nth observation
# of all channels
data_long = data_wide.T
data_long.tofile('data_long.bin')


# TODO: where to close the file?
# TODO: add performance considerations when reading long data
class BinaryReader(object):

    def __init__(self, path_to_file, dtype, n_channels, data_shape):
        """
        Reading batches from large array binary files on disk, similar to
        numpy.memmap. It is essentially just a wrapper around Python
        files API to read through large array binary file using the
        array[:,:] syntax.

        """
        filesize = os.path.getsize(path_to_file)

        if data_shape not in ('long', 'wide'):
            raise ValueError('data_shape must be either "long" or "wide"')

        if not (filesize / n_channels).is_integer():
            raise ValueError('Wrong file size: not divisible by number of '
                             'channels')

        self.data_shape = data_shape
        self.dtype = dtype if not isinstance(dtype, str) else np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.n_observations = filesize / n_channels / self.dtype.itemsize
        self.n_channels = n_channels
        self.f = open(path_to_file, 'rb')
        self.channel_size_byte = self.itemsize * self.n_observations

    def _read_n_bytes_from(self, f, n, start):
        f.seek(int(start))
        return f.read(n)

    def _read_wide_data(self, obs_start, obs_end, ch_start, ch_end):
        obs_to_read = obs_end - obs_start

        # compute bytes where reading starts in every channel:
        # where channel starts + offset due to obs_start
        start_bytes = [start * self.channel_size_byte +
                       obs_start * self.itemsize
                       for start in range(ch_start, ch_end)]

        # bytes to read in every channel
        to_read_bytes = obs_to_read * self.itemsize

        batch = [np.frombuffer(self._read_n_bytes_from(self.f, to_read_bytes,
                                                       start),
                               dtype=self.dtype)
                 for start in start_bytes]

        return np.array(batch).T

    def _read_long_data(self, obs_start, obs_end, ch_start, ch_end):

        obs_start_byte = obs_start * self.itemsize
        obs_to_read = obs_end - obs_start

        # compute the byte size of going from the n -1 observation from channel
        # k to the n observation of channel k
        jump_size_byte = self.itemsize * self.n_channels

        # compute start poisition (first observation on first desired channel)
        # = observation 0 in first selected channel + offset to move to
        # first desired observation in first selected channel
        start_byte = obs_start_byte * ch_start + jump_size_byte * obs_start

        # how many consecutive bytes
        ch_to_read = ch_end - ch_start
        to_read_bytes = self.itemsize * ch_to_read

        # compute seek poisitions (first observation on desired channels)
        start_bytes = [start_byte + jump_size_byte * offset for offset in
                       range(obs_to_read)]

        batch = [np.frombuffer(self._read_n_bytes_from(self.f, to_read_bytes,
                                                       start),
                               dtype=self.dtype)
                 for start in start_bytes]
        batch = np.array(batch)

        return batch

    def __getitem__(self, key):

        if not isinstance(key, tuple):
            raise ValueError('Must pass two slide objects i.e. obj[:,:]')

        obs, ch = key

        if obs.step is not None or ch.step is not None:
            raise ValueError('Step size not supported')

        if self.data_shape == 'long':
            return self._read_long_data(obs.start, obs.stop, ch.start, ch.stop)
        else:
            return self._read_wide_data(obs.start, obs.stop, ch.start, ch.stop)


data_wide.T[1:10, 2:3]
data_long[1:10, 2:3]

wide = BinaryReader('data_wide.bin', 'int64', 10, 'wide')
wide[1:10, 2:3]

long = BinaryReader('data_long.bin', 'int64', 10, 'long')
long[1:10, 2:3]
