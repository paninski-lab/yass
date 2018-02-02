import numpy as np


class BufferGenerator(object):
    """Utility class to generate buffers around numpy 2D arrays

    Parameters
    ----------
    n_observations: int
        Number of observations in the full dataset

    data_format: str
        Data format, either 'wide' or 'long'

    buffer_size: int
        Buffer size (in number of observations) to be added at the beginning
        and at the end
    """

    def __init__(self, n_observations, data_format, buffer_size):
        self.n_observations = n_observations
        self.data_format = data_format
        self.buffer_size = buffer_size

    def _add_zero_buffer(self, data, size, option):
        """Add zeros to an array

        Parameters
        ----------
        data: np.ndarray (2D)
            The data that will be modified

        size: int
            Number of observations to add

        option: str ('start', 'end')
            Where to add the buffer

        Returns
        -------
        numpy.ndarray
            An array with zero buffer
        """
        rows, cols = data.shape
        buff_shape = ((size, cols) if self.data_format == 'long'
                      else (rows, size))
        buff = np.zeros(buff_shape)

        append = np.vstack if self.data_format == 'long' else np.hstack

        if option == 'start':
            return append([buff, data])
        elif option == 'end':
            return append([data, buff])

    def update_key_with_buffer(self, key):
        """
        Updates a slice object to include a buffer in the first axis

        Parameters
        ----------
        key: tuple
            A tuple of slice objects

        Returns
        -------
        slice: tuple, size 2
            A new slice object update to include the buffer at the beginning
            and end of the slice

        missing_buffer: tuple, size 2
            A tuple indicating if there are missing observations at the start
            or end of the slice to complete the buffer size
        """
        t_slice, ch_slice = key
        t_start, t_end = t_slice.start, t_slice.stop

        t_start_new = t_start - self.buffer_size
        t_end_new = t_end + self.buffer_size

        buffer_missing_start = 0
        buffer_missing_end = 0

        if t_start_new < 0:
            buffer_missing_start = abs(t_start_new)
            t_start_new = 0

        if t_end_new > self.n_observations:
            buffer_missing_end = t_end_new - self.n_observations
            t_end_new = self.n_observations

        return ((slice(t_start_new, t_end_new, None), ch_slice),
                (buffer_missing_start, buffer_missing_end))

    def add_buffer(self, data, start, end):
        """Add zero buffer

        Parameters
        ----------
        data: numpy.ndarray
            Data to be modified

        start: int
            How many zeros add before the data (left for 'wide' data, top
            for 'long data')

        end: int
            How many zeros add after the data (right for 'wide' data and
            bottom for 'long')
        """
        data = self._add_zero_buffer(data, start, 'start')
        data = self._add_zero_buffer(data, end, 'end')
        return data
