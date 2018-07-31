import os

import numpy as np
import logging

from yass.templates.crop import crop_and_align_templates
from yass.templates import TemplatesProcessor
from yass.augment.noise import noise_cov
from yass.augment import util


def training_data(CONFIG, spike_train, chosen_templates_indexes, min_amp,
                  max_amp, n_isolated_spikes, data_folder, noise_ratio=10,
                  collision_ratio=1, misalign_ratio=1, misalign_ratio2=1,
                  multi_channel=True, return_metadata=False):
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
    max_amp: float
        Maximum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel
    n_isolated_spikes: int
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
            * Negative examples: Collided spikes + noise
    """
    logger = logging.getLogger(__name__)

    path_to_standarized = os.path.join(data_folder, 'preprocess',
                                       'standarized.bin')

    # load 4x templates
    processor = TemplatesProcessor.from_spike_train(CONFIG,
                                                    4 * CONFIG.spike_size,
                                                    spike_train,
                                                    path_to_standarized)

    processor.choose_with_indexes(chosen_templates_indexes, inplace=True)
    # TODO: make this a parameter
    processor.choose_with_minimum_amplitude(4, inplace=True)

    templates_uncropped = processor.templates

    # TODO: fix the 3 * spike_size
    processor.align(CONFIG.spike_size, inplace=True)
    templates = (processor.crop_spatially(CONFIG.neigh_channels, CONFIG.geom)
                 .values)

    # TODO: remove, this data can be obtained from other variables
    K, _, n_channels = templates_uncropped.shape

    # make training data set
    R = CONFIG.spike_size

    logger.debug('Output will be of size %s', 2 * R + 1)

    # make clean augmented spikes
    nk = int(np.ceil(n_isolated_spikes/K))
    max_shift = 2*R

    # make spikes from templates
    x_templates = util.make_from_templates(templates, min_amp, max_amp, nk)

    # make collided spikes - max shift is set to R since 2 * R + 1 will be
    # the final dimension for the spikes
    x_collision = util.make_collided(x_templates, collision_ratio,
                                     multi_channel, max_shift=R,
                                     return_metadata=return_metadata)

    # make misaligned spikes
    (x_temporally_misaligned,
     x_spatially_misaligned) = util.make_misaligned(x_templates,
                                                    max_shift,
                                                    misalign_ratio,
                                                    misalign_ratio2,
                                                    multi_channel)

    # determine noise covariance structure
    spatial_SIG, temporal_SIG = noise_cov(path_to_standarized,
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    # make noise
    noise_shape = (int(x_templates.shape[0] * noise_ratio),
                   x_templates.shape[1], x_templates.shape[2])
    noise = util.make_noise(noise_shape, spatial_SIG, temporal_SIG)

    # make labels
    y_clean_1 = np.ones((x_templates.shape[0]))
    y_collision_1 = np.ones((x_collision.shape[0]))

    y_misaligned_0 = np.zeros((x_temporally_misaligned.shape[0]))
    y_noise_0 = np.zeros((noise.shape[0]))
    y_collision_0 = np.zeros((x_collision.shape[0]))

    mid_point = int((x_templates.shape[1]-1)/2)
    MID_POINT_IDX = slice(mid_point - R, mid_point + R + 1)

    # TODO: replace _make_noisy for new function
    x_templates_noisy = util._make_noisy(x_templates, noise)
    x_collision_noisy = util._make_noisy(x_collision, noise)
    x_temporally_misaligned_noisy = util._make_noisy(x_temporally_misaligned,
                                                     noise)

    #############
    # Detection #
    #############

    if multi_channel:
        x = np.concatenate((x_templates_noisy, x_collision_noisy,
                            x_temporally_misaligned_noisy, noise))
        x_detect = x[:, MID_POINT_IDX, :]

        y_detect = np.concatenate((y_clean_1, y_collision_1,
                                   y_misaligned_0, y_noise_0))
    else:
        x = np.concatenate((x_templates_noisy, x_temporally_misaligned_noisy,
                            noise))
        x_detect = x[:, MID_POINT_IDX, 0]

        y_detect = np.concatenate((y_clean_1,
                                   y_misaligned_0, y_noise_0))
    ##########
    # Triage #
    ##########

    if multi_channel:
        x = np.concatenate((x_templates_noisy, x_collision_noisy))
        x_triage = x[:, MID_POINT_IDX, :]

        y_triage = np.concatenate((y_clean_1, y_collision_0))
    else:
        x = np.concatenate((x_templates_noisy, x_collision_noisy,))
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

    # FIXME: y_ae is no longer used, autoencoder was replaced by PCA
    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae


def spikes(templates, min_amplitude, max_amplitude, path_to_data,
           n_per_template, geom,
           make_from_templates=True,
           make_spatially_misaligned=True,
           make_temporally_misaligned=True,
           make_collided=True,
           make_noise=True):
    """
    Make spikes, it creates several types of spikes from templates with a range
    of amplitudes

    Parameters
    ----------
    templates: numpy.ndarray, (n_templates, waveform_length, n_channels)
        Templates used to generate the spikes

    min_amplitude: float
        Minimum amplitude for the spikes

    max_amplitude: float
        Maximum amplitude for the spikes

    path_to_data: str
        Path to the data used to generate the templates (used to estimate)
        noise covariance

    n_per_template: int
        How many spikes to generate per template. This along with
        min_amplitude and max_amplitude are used to generate spikes covering
        the desired amplitude range

    geom: numpy.ndarray
        Geometry matrix

    make_from_templates: bool
        Whether to return spikes generated from the templates (these are
        the same as the templates but with different amplitudes)

    make_spatially_misaligned: bool
        Whether to return spatially misaligned spikes (by shuffling channels)

    make_temporally_misaligned: bool
        Whether to return temporally misaligned spikes (by shifting along
        the temporal axis)

    make_collided: bool
        Whether to return collided spikes

    make_noise: bool
        Whether to return pure noise


    Returns
    -------
    x_all: numpy.ndarray, (n_templates * n_per_template, waveform_length,
    n_channels)
        All generated spikes

    x_all_noisy: numpy.ndarray, (n_templates * n_per_template, waveform_length,
    n_channels)
        Noisy versions of all generated spikes

    the_amplitudes: numpy.ndarray, (n_templates * n_per_template,)
        Amplitudes for all generated spikes

    slices: dictionary
        Dictionary where the keys are the kind of spikes ('from templates',
        'spatially misaligned', 'temporally misaligned', 'collided', noise')
        and the values are slice objects with the location for each kind
        of spike
    """
    # NOTE: is the order importante here, maybe it's better to first compute
    # from templates, then take those and misalign spatially
    # (all templates in all channels) then take those and misalign temporally
    # and finally produce collided spikes

    # TODO: add multi_channel parameter and options for hardcoded parameter

    _, waveform_length, n_neigh = templates.shape

    # TODO: remove this
    neigh_channels = np.ones((n_neigh, n_neigh), dtype=bool)

    # make spikes
    x_templates = util.make_from_templates(templates, min_amplitude,
                                           max_amplitude, n_per_template)

    n_spikes, _, _ = x_templates.shape

    x_all, keys = [], []

    if make_from_templates:
        x_all.append(x_templates)
        keys.append('from templates')

    if make_spatially_misaligned:
        x_spatially = util.make_spatially_misaligned(x_templates,
                                                     n_per_spike=1)
        x_all.append(x_spatially)
        keys.append('spatially misaligned')

    if make_temporally_misaligned:
        x_temporally = util.make_temporally_misaligned(x_templates,
                                                       n_per_spike=1,
                                                       multi_channel=True)
        x_all.append(x_temporally)
        keys.append('temporally misaligned')

    if make_collided:
        # TODO: refactor this as it has redundant logic with misaligned
        x_collided = util.make_collided(x_templates, n_per_spike=1,
                                        multi_channel=True)
        x_all.append(x_collided)
        keys.append('collided')

    if make_noise:
        x_zero = np.zeros((n_spikes, waveform_length, n_neigh))
        x_all.append(x_zero)
        keys.append('noise')

    x_all = np.concatenate(x_all, axis=0)

    # add noise
    spatial_SIG, temporal_SIG = noise_cov(path_to_data,
                                          neigh_channels,
                                          geom,
                                          waveform_length)

    x_all_noisy = util.add_noise(x_all, spatial_SIG, temporal_SIG)

    # compute amplitudes
    the_amplitudes = util.amplitudes(x_all)

    # return a dictionary with slices for every type of spike generated
    slices = {k: slice(n_spikes * i, n_spikes * (i + 1)) for k, i
              in zip(keys, range(len(x_all)))}

    return x_all, x_all_noisy, the_amplitudes, slices
