import numpy as np
from scipy import sparse
import logging

from ..geometry import n_steps_neigh_channels


def get_templates(spike_train_clear, batch_size, buff, n_batches, n_channels,
                  spike_size, path_to_rec, dtype):
    """

    Returns
    -------
    spike_train_clear
    templates
        n channels x (2*(spike_size+template_max_shift)+1) x
    """
    logger = logging.getLogger(__name__)

    wfile = open(path_to_rec, 'rb')
    dsize = np.dtype(dtype).itemsize
    flattenedLength = dsize*batch_size*n_channels

    K = np.max(spike_train_clear[:, 1])+1

    templates = np.zeros((n_channels, 2*spike_size+1, K))
    weights = np.zeros(K)

    for i in range(n_batches):
        logger.info("extracting waveforms from batch {} out of {} batches"
            .format(i+1, n_batches))
        wfile.seek(flattenedLength*i)
           
        wrec = wfile.read(flattenedLength)
        wrec = np.fromstring(wrec, dtype=dtype)
        wrec = np.reshape(wrec, (-1, n_channels))
        wrec = wrec.astype('float32')

        idx_batch = np.logical_and(spike_train_clear[:,0] > i*batch_size + buff, spike_train_clear[:,0] < (i+1)*batch_size - buff)

        if np.sum(idx_batch) > 0:

            spike_train_batch = spike_train_clear[idx_batch]
            spt_batch = spike_train_batch[:,0] - i*batch_size
            L_batch = spike_train_batch[:,1]

            wf = np.zeros((spt_batch.shape[0], templates.shape[1], n_channels))
            for j in range(spt_batch.shape[0]):
                wf[j] = wrec[(spt_batch[j]-spike_size):(spt_batch[j]+spike_size+1)]

            for k in range(K):
                templates[:, :, k] += np.sum(wf[L_batch == k], axis=0).T
                weights[k] += np.sum(L_batch == k)
    templates = templates/weights[np.newaxis, np.newaxis, :]

    wfile.close()

    return templates