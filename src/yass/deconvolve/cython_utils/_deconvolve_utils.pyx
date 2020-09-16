import numpy as np
cimport numpy as np

def shift_channels_cython(float[:,:] signal,
                          int[:] shifts):

    cdef int n_chan = signal.shape[0]
    cdef int size = signal.shape[1]

    cdef int max_shift = np.max(shifts)
    cdef int shifted_signal_size = size + max_shift

    cdef float[:,:] shifted_signal = np.zeros((n_chan, shifted_signal_size), dtype='float32')

    cdef Py_ssize_t x
    cdef int offset
    for x in range(n_chan):
        offset = shifts[x]
        shifted_signal[x, offset:offset+size] = signal[x,:]

    return shifted_signal.base
