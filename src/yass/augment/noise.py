import logging


import numpy as np

from yass.geometry import order_channels_by_distance
from yass.batch import RecordingsReader


def noise_cov(path_to_data, neighbors, geom, temporal_size):
    """Compute noise temporal and spatial covariance

    Parameters
    ----------
    path_to_data: str
        Path to recordings data

    neighbors: numpy.ndarray
        Neighbors matrix

    geom: numpy.ndarray
        Cartesian coordinates for the channels

    temporal_size:
        Waveform size

    Returns
    -------
    spatial_SIG: numpy.ndarray

    temporal_SIG: numpy.ndarray
    """
    logger = logging.getLogger(__name__)

    logger.debug('Computing noise_cov. Neighbors shape: {}, geom shape: {} '
                 'temporal_size: {}'.format(neighbors.shape, geom.shape,
                                            temporal_size))

    c_ref = np.argmax(np.sum(neighbors, 0))
    ch_idx = np.where(neighbors[c_ref])[0]
    ch_idx, temp = order_channels_by_distance(c_ref, ch_idx, geom)

    rec = RecordingsReader(path_to_data, loader='array')
    rec = rec[:, ch_idx]

    T, C = rec.shape
    idxNoise = np.zeros((T, C))
    R = int((temporal_size-1)/2)

    for c in range(C):

        idx_temp = np.where(rec[:, c] > 3)[0]

        for j in range(-R, R+1):

            idx_temp2 = idx_temp + j
            idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
                                                 idx_temp2 < T)]
            rec[idx_temp2, c] = np.nan

        idxNoise_temp = (rec[:, c] == rec[:, c])
        rec[:, c] = rec[:, c]/np.nanstd(rec[:, c])

        rec[~idxNoise_temp, c] = 0
        idxNoise[idxNoise_temp, c] = 1

    spatial_cov = np.divide(np.matmul(rec.T, rec),
                            np.matmul(idxNoise.T, idxNoise))

    w, v = np.linalg.eig(spatial_cov)
    spatial_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    spatial_whitener = np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)
    rec = np.matmul(rec, spatial_whitener)

    noise_wf = np.zeros((1000, temporal_size))
    count = 0

    while count < 1000:

        tt = np.random.randint(T-temporal_size)
        cc = np.random.randint(C)
        temp = rec[tt:(tt+temporal_size), cc]
        temp_idxnoise = idxNoise[tt:(tt+temporal_size), cc]

        if np.sum(temp_idxnoise == 0) == 0:
            noise_wf[count] = temp
            count += 1

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    logger.debug('spatial_SIG shape: {} temporal_SIG shape: {}'
                 .format(spatial_SIG.shape, temporal_SIG.shape))

    return spatial_SIG, temporal_SIG
