import logging


import numpy as np

from yass.geometry import order_channels_by_distance
from yass.batch import RecordingsReader


def noise_cov(path_to_data, neighbors, geom, temporal_size,
              sample_size=1000):
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

    # reference channel: channel with max number of neighbors
    c_ref = np.argmax(np.sum(neighbors, 0))
    # neighbors for the reference channel
    ch_idx = np.where(neighbors[c_ref])[0]
    # ordered neighbors for reference channel
    ch_idx, temp = order_channels_by_distance(c_ref, ch_idx, geom)

    # read the selected channels
    rec = RecordingsReader(path_to_data, loader='array')
    rec = rec[:, ch_idx]

    T, C = rec.shape
    idxNoise = np.zeros((T, C))
    R = int((temporal_size-1)/2)

    # go through every neighboring channel
    for c in range(C):

        # get obserations where observation is above 3
        idx_temp = np.where(rec[:, c] > 3)[0]

        # shift every index found
        for j in range(-R, R+1):

            # shift
            idx_temp2 = idx_temp + j

            # remove indexes outside range [0, T]
            idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
                                                 idx_temp2 < T)]

            # set surviving indexes to nan
            rec[idx_temp2, c] = np.nan

        # noise indexes are the ones that are not nan
        # FIXME: compare to np.nan instead
        idxNoise_temp = (rec[:, c] == rec[:, c])

        # standarize data, ignoring nans
        rec[:, c] = rec[:, c]/np.nanstd(rec[:, c])

        # set non noise indexes to 0 in the recordings
        rec[~idxNoise_temp, c] = 0

        # save noise indexes
        idxNoise[idxNoise_temp, c] = 1

    # compute spatial covariance, output: (n_channels, n_channels)
    spatial_cov = np.divide(np.matmul(rec.T, rec),
                            np.matmul(idxNoise.T, idxNoise))

    # compute spatial sig
    w_spatial, v_spatial = np.linalg.eig(spatial_cov)
    spatial_SIG = np.matmul(np.matmul(v_spatial,
                                      np.diag(np.sqrt(w_spatial))),
                            v_spatial.T)

    # apply spatial whitening to recordings
    spatial_whitener = np.matmul(np.matmul(v_spatial,
                                           np.diag(1/np.sqrt(w_spatial))),
                                 v_spatial.T)
    rec = np.matmul(rec, spatial_whitener)

    # generate noise waveform
    noise_wf = np.zeros((sample_size, temporal_size))
    count = 0

    # repeat until you get sample_size noise snippets
    while count < sample_size:

        # random number for the start of the noise snippet
        t_start = np.random.randint(T-temporal_size)
        # random channel
        ch = np.random.randint(C)

        t_slice = slice(t_start, t_start+temporal_size)

        # get a snippet from the recordings and the noise flags for the same
        # location
        snippet = rec[t_slice, ch]
        snipped_idx_noise = idxNoise[t_slice, ch]

        # check if there is any signal observation in the snippet
        signal_in_snippet = snipped_idx_noise.any()

        # if all snippet is noise..
        if not signal_in_snippet:
            # add the snippet and increase count
            noise_wf[count] = snippet
            count += 1

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    logger.debug('spatial_SIG shape: {} temporal_SIG shape: {}'
                 .format(spatial_SIG.shape, temporal_SIG.shape))

    return spatial_SIG, temporal_SIG
