"""
Utility functions for noise processing
"""
import numpy as np


# FIXME: i dont think is is being used
def isolate_noise(recording, spike_size, threshold):
    """
    Isolate noise from signal using a threshold value and filling with
    zeros the sourrounding observations in a (-spike_size, spike_size)
    window

    Parameters
    ----------
    recording: matrix
        Multi-cannel recordings (n observations x n channels)
    threshold: float
        Threshold value, values below or equal to this will be considered
        noise
    spike_size: int
        Spike size, used to build a window around high values

    Returns
    -------
    matrix
        Matrix with non-noise values replaced by zeros
    matrix
        Matrix indicating wheter the value was identified as noise (as
        opposed to being filled with zeros)
    """
    # FIXME: this function is mutating recording which is not a good idea
    n_observations, n_channels = recording.shape
    is_noise_matrix = np.zeros((n_observations, n_channels))

    # iterate over the number of channels
    for c in range(n_channels):

        # get indices where recording value is > 3 (this won't be considered
        # noise)
        (high_values_idx,) = np.where(recording[:, c] > 3)

        # iterate over the index found using a window based on the spike size
        for j in range(-spike_size, spike_size+1):

            # shift the high value indexes, this will give you the jth
            # overvation for very high value position
            position = high_values_idx + j

            # filter to remove values outside the boundaries of the recordings
            valid_idx = np.logical_and(position >= 0,
                                       position < n_observations)
            position_valid = position[valid_idx]

            # replace with nans in every position around the high values
            recording[position_valid, c] = np.nan

        # at this point channel c has its high value entries removed, get
        # a bool array for the values that are not nan, this are noise values
        is_noise = ~np.isnan(recording[:, c])

        # divide observations over the standar deviation
        recording[:, c] = recording[:, c]/np.nanstd(recording[:, c])

        # replace high values (not noise) in the recording with zeros
        recording[~is_noise, c] = 0
        # populate column with the bool "is noise" array for the current
        # channel
        is_noise_matrix[is_noise, c] = 1

    return recording, is_noise_matrix


def covariance(recording, temporal_size, spatial_distance, spike_size):
    """Compute the covariance matrix

    Parameters
    ----------
    recording: matrix
        Multi-cannel recordings (n observations x n channels)
    temporal_size:
    spatial_distance:
    spike_size: int
        Spike size
    """
    n_observations, n_channels = recording.shape

    # build a matrix whose i-th column contain flags to indicate whether an
    # observation is considered noise or not for the j-th channel
    recording, is_noise_matrix = isolate_noise(recording, spike_size,
                                               threshold=3)

    spatial_cov = np.divide(np.matmul(recording.T, recording),
                            np.matmul(is_noise_matrix.T, is_noise_matrix))

    w, v = np.linalg.eig(spatial_cov)

    spatial_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)
    spatial_whitener = np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))),
                                 v.T)
    recording = np.matmul(recording, spatial_whitener)

    noise_wf = np.zeros((1000, temporal_size))
    count = 0

    while count < 1000:
        tt = np.random.randint(n_observations-temporal_size)
        cc = np.random.randint(n_channels)
        temp = recording[tt:(tt+temporal_size), cc]
        temp_idxnoise = is_noise_matrix[tt:(tt+temporal_size), cc]
        if np.sum(temp_idxnoise == 0) == 0:
            noise_wf[count] = temp
            count += 1

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    return spatial_SIG, temporal_SIG
