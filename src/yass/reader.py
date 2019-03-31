import os
import numpy as np

class READER(object):

    def __init__(self, bin_file, dtype, n_sec_chunk, CONFIG, buffer=None):

        # frequently used parameters
        self.n_sec_chunk = n_sec_chunk
        self.n_channels = CONFIG.recordings.n_channels
        self.sampling_rate = CONFIG.recordings.sampling_rate
        self.rec_len = CONFIG.rec_len

        # save bin_file
        self.bin_file = bin_file
        self.dtype = np.dtype(dtype)

        # define a size of buffer if not defined
        if buffer is None:
            self.buffer = int(max(self.sampling_rate/100, 200))
        else:
            self.buffer = buffer

        # batch sizes
        indexes = np.arange(0, self.rec_len, self.sampling_rate*self.n_sec_chunk)
        indexes = np.hstack((indexes, self.rec_len))

        idx_list = []
        for k in range(len(indexes) - 1):
            idx_list.append([indexes[k], indexes[k + 1]])
        self.idx_list = np.int64(np.vstack(idx_list))
        self.n_batches = len(self.idx_list)


    def read_data(self, data_start, data_end, channels=None):

        with open(self.bin_file, "rb") as fin:
            # Seek position and read N bytes
            fin.seek(data_start*self.dtype.itemsize*self.n_channels, os.SEEK_SET)
            data = np.fromfile(
                fin, dtype=self.dtype,
                count=(data_end - data_start)*self.n_channels)
        fin.close()
        
        data = data.reshape(-1, self.n_channels)
        if channels is not None:
            data = data[:, channels]

        return data

    def read_data_batch(self, batch_id, add_buffer=False, channels=None):

        # batch start and end
        data_start, data_end = self.idx_list[batch_id]

        # add buffer if asked
        if add_buffer:
            data_start -= self.buffer
            data_end += self.buffer

            # if start is below zero, put it back to 0 and and zeros buffer
            if data_start < 0:
                left_buffer_size = 0 - data_start
                data_start = 0
            else:
                left_buffer_size = 0

            # if end is above rec_len, put it back to rec_len and and zeros buffer
            if data_end > self.rec_len:
                right_buffer_size = data_end - self.rec_len
                data_end = self.rec_len
            else:
                right_buffer_size = 0

        # read data
        data = self.read_data(data_start, data_end, channels)

        # add leftover buffer with zeros if necessary
        if add_buffer:
            left_buffer = np.zeros(
                (left_buffer_size, self.n_channels),
                dtype=self.dtype)
            right_buffer = np.zeros(
                (right_buffer_size, self.n_channels),
                dtype=self.dtype)
            if channels is not None:
                left_buffer = left_buffer[:, channels]
                right_buffer = right_buffer[:, channels]
            data = np.concatenate((left_buffer, data, right_buffer), axis=0)

        return data

    def read_data_batch_batch(self, batch_id, n_sec_chunk_small, add_buffer=False, channels=None):
        
        data = self.read_data_batch(batch_id, add_buffer, channels)
        
        T, C = data.shape
        T_mini = self.sampling_rate*n_sec_chunk_small
        buffer = self.buffer

        if add_buffer:
            T = T - 2*buffer
        else:
            buffer = 0

        # batch sizes
        indexes = np.arange(0, T, T_mini)
        indexes = np.hstack((indexes, T))
        indexes += buffer

        n_mini_batches = len(indexes) - 1

        if n_mini_batches*T_mini > T:
            T_left = n_mini_batches*T_mini - T

            pad_zeros = np.zeros((T_left, C),
                dtype=self.dtype)

            data = np.zeros((data, pad_zeros), axis=1)
       
        data_batched = np.zeros((n_mini_batches, T_mini + 2*buffer, C))
        for k in range(n_mini_batches):
            data_batched[k] = data[indexes[k]-buffer:indexes[k + 1]+buffer]
                
        return data_batched, indexes-buffer
