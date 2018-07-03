import os
import numpy as np
import logging


from yass.augment.choose import choose_templates
from yass.augment.crop import crop_and_align_templates
from yass.augment.noise import noise_cov
from yass.augment.util import (make_noisy, make_clean, make_collided,
                               make_misaligned, make_noise)
from yass.templates.util import get_templates


def make_training_data(CONFIG, spike_train, chosen_templates_indexes, min_amp,
                       nspikes, data_folder, noise_ratio=10, collision_ratio=1,
                       misalign_ratio=1, misalign_ratio2=1,
                       multi_channel=True):
    """Makes training sets for detector, triage and autoencoder

    Parameters
    ----------
    CONFIG: yaml file
        Configuration file
    spike_train: numpy.ndarray
        [number of spikes, 2] Ground truth for training. First column is the
        spike time, second column is the spike id
    chosen_templates_indexes: list
        List of chosen templates' id's
    min_amp: float
        Minimum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel
    nspikes: int
        Number of isolated spikes to generate. This is different from the
        total number of x_detect
    data_folder: str
        Folder storing the standarized data (if not exist, run preprocess to
        automatically generate)
    noise_ratio: int
        Ratio of number of noise to isolated spikes. For example, if
        n_isolated_spike=1000, noise_ratio=5, then n_noise=5000
    collision_ratio: int
        Ratio of number of collisions to isolated spikes.
    misalign_ratio: int
        Ratio of number of spatially and temporally misaligned spikes to
        isolated spikes
    misalign_ratio2: int
        Ratio of number of only-spatially misaligned spikes to isolated spikes
    multi_channel: bool
        If True, generate training data for multi-channel neural
        network. Otherwise generate single-channel data

    Returns
    -------
    x_detect: numpy.ndarray
        [number of detection training data, temporal length, number of
        channels] Training data for the detect net.
    y_detect: numpy.ndarray
        [number of detection training data] Label for x_detect

    x_triage: numpy.ndarray
        [number of triage training data, temporal length, number of channels]
        Training data for the triage net.
    y_triage: numpy.ndarray
        [number of triage training data] Label for x_triage
    x_ae: numpy.ndarray
        [number of ae training data, temporal length] Training data for the
        autoencoder: noisy spikes
    y_ae: numpy.ndarray
        [number of ae training data, temporal length] Denoised x_ae

    Notes
    -----
    * Detection training data
        * Multi channel
            * Positive examples: Clean spikes + noise, Collided spikes + noise
            * Negative examples: Temporally misaligned spikes + noise, Noise

    * Triage training data
        * Multi channel
            * Positive examples: Clean spikes + noise
            * Negative examples: Collided spikes + noise,
                spatially misaligned spikes  + noise
    """

    logger = logging.getLogger(__name__)

    path_to_data = os.path.join(data_folder, 'preprocess', 'standarized.bin')

    n_spikes, _ = spike_train.shape

    # make sure standarized data already exists
    if not os.path.exists(path_to_data):
        raise ValueError('Standarized data does not exist in: {}, this is '
                         'needed to generate training data, run the '
                         'preprocesor first to generate it'
                         .format(path_to_data))

    logger.info('Getting templates...')

    # add weight of one to every spike
    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    # get templates
    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    K, _, n_channels = templates_uncropped.shape

    logger.info('Got templates ndarray of shape: {}'
                .format(templates_uncropped.shape))

    # choose good templates (user selected and amplitude above threshold)
    # TODO: maybe the minimum_amplitude parameter should be selected by the
    # user
    templates_uncropped = choose_templates(templates_uncropped,
                                           chosen_templates_indexes,
                                           minimum_amplitude=4)

    if templates_uncropped.shape[0] == 0:
        raise ValueError("Coulndt find any good templates...")

    logger.info('Good looking templates of shape: {}'
                .format(templates_uncropped.shape))

    templates = crop_and_align_templates(templates_uncropped,
                                         CONFIG.spike_size,
                                         CONFIG.neigh_channels,
                                         CONFIG.geom)

    # make training data set
    R = CONFIG.spike_size
    amps = np.max(np.abs(templates), axis=1)

    # make clean augmented spikes
    nk = int(np.ceil(nspikes/K))
    max_amp = np.max(amps)*1.5
    nneigh = templates.shape[2]
    max_shift = 2*R

    # make clean spikes
    x_clean = make_clean(templates, min_amp, max_amp, nk)

    # make collided spikes
    x_collision = make_collided(x_clean, collision_ratio, templates,
                                R, multi_channel, nneigh)

    # make misaligned spikes
    (x_temporally_misaligned,
     x_spatially_misaligned) = make_misaligned(x_clean,
                                               templates, max_shift,
                                               misalign_ratio,
                                               misalign_ratio2,
                                               multi_channel,
                                               nneigh)

    # determine noise covariance structure
    spatial_SIG, temporal_SIG = noise_cov(path_to_data,
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    # make noise
    noise = make_noise(x_clean, noise_ratio, templates, spatial_SIG,
                       temporal_SIG)

    # make labels
    y_clean_1 = np.ones((x_clean.shape[0]))
    y_collision_1 = np.ones((x_collision.shape[0]))

    y_misaligned_0 = np.zeros((x_temporally_misaligned.shape[0]))
    y_noise_0 = np.zeros((noise.shape[0]))
    y_collision_0 = np.zeros((x_collision.shape[0]))

    if multi_channel:
        y_misaligned2_0 = np.zeros((x_spatially_misaligned.shape[0]))

    mid_point = int((x_clean.shape[1]-1)/2)
    MID_POINT_IDX = slice(mid_point - R, mid_point + R + 1)

    x_clean_noisy = make_noisy(x_clean, noise)
    x_collision_noisy = make_noisy(x_collision, noise)
    x_temporally_misaligned_noisy = make_noisy(x_temporally_misaligned,
                                               noise)
    x_spatially_misaligned_noisy = make_noisy(x_spatially_misaligned,
                                              noise)

    #############
    # Detection #
    #############

    if multi_channel:
        x = np.concatenate((x_clean_noisy, x_collision_noisy,
                            x_temporally_misaligned_noisy, noise))
        x_detect = x[:, MID_POINT_IDX, :]

        y_detect = np.concatenate((y_clean_1, y_collision_1,
                                   y_misaligned_0, y_noise_0))
    else:
        x = np.concatenate((x_clean_noisy, x_temporally_misaligned_noisy,
                            noise))
        x_detect = x[:, MID_POINT_IDX, 0]

        y_detect = np.concatenate((y_clean_1,
                                   y_misaligned_0, y_noise_0))

    ##########
    # Triage #
    ##########

    if multi_channel:
        x = np.concatenate((x_clean_noisy, x_collision_noisy))
        x_triage = x[:, MID_POINT_IDX, :]

        y_triage = np.concatenate((y_clean_1, y_collision_0))
    else:
        x = np.concatenate((x_clean_noisy, x_collision_noisy,))
        x_triage = x[:, MID_POINT_IDX, 0]

        y_triage = np.concatenate((y_clean_1,
                                   y_collision_0))

    ###############
    # Autoencoder #
    ###############

    # TODO: need to abstract this part of the code, create a separate
    # function and document it
    neighbors_ae = np.ones((n_channels, n_channels), 'int32')

    templates_ae = crop_and_align_templates(templates_uncropped,
                                            CONFIG.spike_size,
                                            neighbors_ae,
                                            CONFIG.geom)

    tt = templates_ae.transpose(1, 0, 2).reshape(templates_ae.shape[1], -1)
    tt = tt[:, np.ptp(tt, axis=0) > 2]
    max_amp = np.max(np.ptp(tt, axis=0))

    y_ae = np.zeros((nk*tt.shape[1], tt.shape[0]))

    for k in range(tt.shape[1]):
        amp_now = np.ptp(tt[:, k])
        amps_range = (np.arange(nk)*(max_amp-min_amp)
                      / nk+min_amp)[:, np.newaxis, np.newaxis]

        y_ae[k*nk:(k+1)*nk] = ((tt[:, k]/amp_now)[np.newaxis, :]
                               * amps_range[:, :, 0])

    noise_ae = np.random.normal(size=y_ae.shape)
    noise_ae = np.matmul(noise_ae, temporal_SIG)

    x_ae = y_ae + noise_ae
    x_ae = x_ae[:, MID_POINT_IDX]
    y_ae = y_ae[:, MID_POINT_IDX]

    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae
