import numpy as np


data_wide = np.array([np.arange(1, 101) * i for i in range(1, 11)])
data_wide.tofile('data_wide.bin')

data_long = data_wide.T
data_long.tofile('data_long.bin')

f_wide = open('data_wide.bin', 'rb')
f_long = open('data_long.bin', 'rb')

n_observations, n_channels = data.shape
dtype = data.dtype
itemsize = data.dtype.itemsize

channel_size_byte = itemsize * n_observations

obs_start, obs_end = 0, 10
ch_start, ch_end = 0, 3

obs_start_byte = obs_start * itemsize
obs_end_byte = obs_end * itemsize

obs_to_read = obs_end - obs_start

# When reading data in wide format...

# compute bytes where reading starts in every channel:
# where channel starts + offset due to obs_start
start_bytes = [start * channel_size_byte + obs_start * itemsize
               for start in range(ch_start, ch_end)]

# bytes to read in every channel
to_read_bytes = obs_to_read * itemsize


def read_n_bytes_from(f, n, start):
    f.seek(start)
    return f.read(n)


batch = [np.frombuffer(read_n_bytes_from(f_wide, to_read_bytes, start), dtype=dtype)
         for start in start_bytes]

batch = np.array(batch).T
batch

data_wide[obs_start:obs_end, ch_start:ch_end]

np.frombuffer(read_n_bytes_from(f_wide, 10000, 0), dtype=dtype)

# When reading data in long format...

# compute the byte size of going from the n -1 observation from channel k
# to the n observation of channel k
jump_size_byte = itemsize * n_channels

# compute start poisition (first observation on first desired channel)
# = observation 0 in first selected channel + offset to move to
# first desired observation in first selected channel
start_byte = obs_start_byte * ch_start + jump_size_byte * obs_start

# how many consecutive bytes
ch_to_read = ch_end - ch_start
to_read_bytes = itemsize * ch_to_read

# compute seek poisitions (first observation on desired channels)
start_bytes = [start_byte + jump_size_byte * offset for offset in
               range(obs_to_read)]


batch = [np.frombuffer(read_n_bytes_from(f_long, to_read_bytes, start), dtype=dtype)
         for start in start_bytes]
batch = np.array(batch)
batch.T
