import os

import numpy as np
import logging


from yass.templates import TemplatesProcessor
from yass.augment.noise import noise_cov
from yass.augment import util
import yass.array as yarr
from yass.geometry import order_channels_by_distance
from yass.batch import RecordingsReader


def load_templates(data_folder, spike_train, CONFIG, chosen_templates_indexes):
    """
    Parameters
    ----------
    data_folder: str
        Folder storing the standarized data (if not exist, run preprocess to
        automatically generate)
    spike_train: numpy.ndarray
        [number of spikes, 2] Ground truth for training. First column is the
        spike time, second column is the spike id
    chosen_templates_indexes: list
        List of chosen templates' id's
    """
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

    # TODO: fix the 3 * spike_size
    processor.align(CONFIG.spike_size, inplace=True)

    return processor.templates


def training_data_triage(templates, minimum_amplitude, maximum_amplitude,
                         n_clean_per_template,
                         n_collided_per_spike,
                         max_shift, min_shift,
                         spatial_SIG, temporal_SIG,
                         from_templates_kwargs,
                         collided_kwargs):
    """Make training data for triage network


    Notes
    -----
    """
    K, _, n_channels = templates.shape

    # make spikes from templates
    x_templates = util.make_from_templates(templates, minimum_amplitude,
                                           maximum_amplitude,
                                           n_clean_per_template,
                                           **from_templates_kwargs)

    x_collision = util.make_collided(x_templates, n_collided_per_spike,
                                     multi_channel=True,
                                     max_shift=max_shift,
                                     min_shift=min_shift,
                                     **collided_kwargs)

    # make labels
    ones = np.ones((x_templates.shape[0]))
    zeros = np.zeros((x_collision.shape[0]))

    x_templates_noisy = util.add_noise(x_templates, spatial_SIG, temporal_SIG)
    x_collision_noisy = util.add_noise(x_collision, spatial_SIG, temporal_SIG)

    x_triage = yarr.concatenate((x_templates_noisy, x_collision_noisy))
    y_triage = yarr.concatenate((ones, zeros))

    return x_triage, y_triage


def training_data_detect(templates,
                         minimum_amplitude,
                         maximum_amplitude,
                         n_clean_per_template,
                         n_collided_per_spike,
                         n_temporally_misaligned_per_spike,
                         n_spatially_misaliged_per_spike,
                         n_noise,
                         spatial_SIG,
                         temporal_SIG,
                         from_templates_kwargs={},
                         collided_kwargs={},
                         temporally_misaligned_kwargs={},
                         add_noise_kwargs={'reject_cancelling_noise': False}):
    """Make training data for detector network


    Notes
    -----
    Recordings are passed through the detector network which identifies
    spikes (clean and collided), it rejects noise and misaligned spikes
    (temporally and spatially)
    """

    # make spikes from templates
    x_templates = util.make_from_templates(templates, minimum_amplitude,
                                           maximum_amplitude,
                                           n_clean_per_template,
                                           **from_templates_kwargs)

    # now spatially misalign (shuffle channels)
    fn = util.make_spatially_misaligned

    x_spatially2 = fn(x_templates, n_per_spike=n_spatially_misaliged_per_spike,
                      force_first_channel_shuffle=True)

    x_spatially = fn(x_templates, n_per_spike=1,
                     force_first_channel_shuffle=False)

    x_collision = util.make_collided(x_templates, x_spatially,
                                     n_collided_per_spike,
                                     **collided_kwargs)

    # create temporally misaligned spikes
    fn = util.make_temporally_misaligned
    x_misalign = fn(x_spatially, n_temporally_misaligned_per_spike,
                    **temporally_misaligned_kwargs)

    x_noise = util.make_noise(n_noise, spatial_SIG, temporal_SIG)

    x_templates_noisy, _ = util.add_noise(x_templates, spatial_SIG,
                                          temporal_SIG,
                                          **add_noise_kwargs)

    x_collision_noisy, _ = util.add_noise(x_collision, spatial_SIG,
                                          temporal_SIG,
                                          **add_noise_kwargs)
    x_misaligned_noisy, _ = util.add_noise(x_misalign, spatial_SIG,
                                           temporal_SIG,
                                           **add_noise_kwargs)

    x_spatially2_noisy, _ = util.add_noise(x_spatially2, spatial_SIG,
                                           temporal_SIG,
                                           **add_noise_kwargs)

    X = yarr.concatenate((x_templates_noisy, x_collision_noisy,
                          x_misaligned_noisy, x_noise, x_spatially2_noisy))

    # make labels
    ones = np.ones(len(x_templates_noisy) + len(x_collision_noisy))
    zeros = np.zeros(len(x_misaligned_noisy) + len(x_noise) +
                     len(x_spatially2_noisy))

    y = np.concatenate((ones, zeros))

    return X, y


def training_data(CONFIG, templates_uncropped, min_amp, max_amp,
                  n_isolated_spikes,
                  path_to_standarized, noise_ratio=10,
                  collision_ratio=1, misalign_ratio=1, misalign_ratio2=1,
                  multi_channel=True, return_metadata=False):
    """Makes training sets for detector, triage and autoencoder

    Parameters
    ----------
    CONFIG: yaml file
        Configuration file
    min_amp: float
        Minimum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel
    max_amp: float
        Maximum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel
    n_isolated_spikes: int
        Number of isolated spikes to generate. This is different from the
        total number of x_detect
    path_to_standarized: str
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
    # FIXME: should we add collided spikes with the first spike non-centered
    # tod the detection training set?

    logger = logging.getLogger(__name__)

    # STEP1: Load recordings data, and select one channel and random (with the
    # right number of neighbors, then swap the channels so the first one
    # corresponds to the selected channel, then the nearest neighbor, then the
    # second nearest and so on... this is only used for estimating noise
    # structure

    # ##### FIXME: this needs to be removed, the user should already
    # pass data with the desired channels
    rec = RecordingsReader(path_to_standarized, loader='array')
    channel_n_neighbors = np.sum(CONFIG.neigh_channels, 0)
    max_neighbors = np.max(channel_n_neighbors)
    channels_with_max_neighbors = np.where(channel_n_neighbors
                                           == max_neighbors)[0]
    logger.debug('The following channels have %i neighbors: %s',
                 max_neighbors, channels_with_max_neighbors)

    # reference channel: channel with max number of neighbors
    channel_selected = np.random.choice(channels_with_max_neighbors)

    logger.debug('Selected channel %i', channel_selected)

    # neighbors for the reference channel
    channel_neighbors = np.where(CONFIG.neigh_channels[channel_selected])[0]

    # ordered neighbors for reference channel
    channel_idx, _ = order_channels_by_distance(channel_selected,
                                                channel_neighbors,
                                                CONFIG.geom)
    # read the selected channels
    rec = rec[:, channel_idx]
    # ##### FIXME:end of section to be removed

    # STEP 2: load templates

    processor = TemplatesProcessor(templates_uncropped)

    # swap channels, first channel is main channel, then nearest neighbor
    # and so on, only keep neigh_channels
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
    # the final dimension for the spikes. one of the spikes is kept with the
    # main channel, the other one is shifted and channels are changed
    x_collision = util.make_collided(x_templates, collision_ratio,
                                     multi_channel, max_shift=R,
                                     min_shift=5,
                                     return_metadata=return_metadata)

    # make misaligned spikes
    x_temporally_misaligned = util.make_temporally_misaligned(
        x_templates, misalign_ratio,
        multi_channel=multi_channel,
        max_shift=max_shift)

    # now spatially misalign those
    x_misaligned = util.make_spatially_misaligned(x_temporally_misaligned,
                                                  n_per_spike=misalign_ratio2)

    # determine noise covariance structure
    spatial_SIG, temporal_SIG = noise_cov(rec,
                                          temporal_size=templates.shape[1],
                                          window_size=templates.shape[1],
                                          sample_size=1000,
                                          threshold=3.0)

    # make noise
    n_noise = int(x_templates.shape[0] * noise_ratio)
    noise = util.make_noise(n_noise, spatial_SIG, temporal_SIG)

    # make labels
    y_clean_1 = np.ones((x_templates.shape[0]))
    y_collision_1 = np.ones((x_collision.shape[0]))

    y_misaligned_0 = np.zeros((x_misaligned.shape[0]))
    y_noise_0 = np.zeros((noise.shape[0]))
    y_collision_0 = np.zeros((x_collision.shape[0]))

    mid_point = int((x_templates.shape[1]-1)/2)
    MID_POINT_IDX = slice(mid_point - R, mid_point + R + 1)

    # TODO: replace _make_noisy for new function
    x_templates_noisy = util._make_noisy(x_templates, noise)
    x_collision_noisy = util._make_noisy(x_collision, noise)
    x_misaligned_noisy = util._make_noisy(x_misaligned,
                                          noise)

    #############
    # Detection #
    #############

    if multi_channel:
        x = yarr.concatenate((x_templates_noisy, x_collision_noisy,
                              x_misaligned_noisy, noise))
        x_detect = x[:, MID_POINT_IDX, :]

        y_detect = np.concatenate((y_clean_1, y_collision_1,
                                   y_misaligned_0, y_noise_0))
    else:
        x = yarr.concatenate((x_templates_noisy, x_misaligned_noisy,
                              noise))
        x_detect = x[:, MID_POINT_IDX, 0]

        y_detect = yarr.concatenate((y_clean_1, y_misaligned_0, y_noise_0))
    ##########
    # Triage #
    ##########

    if multi_channel:
        x = yarr.concatenate((x_templates_noisy, x_collision_noisy))
        x_triage = x[:, MID_POINT_IDX, :]

        y_triage = yarr.concatenate((y_clean_1, y_collision_0))
    else:
        x = yarr.concatenate((x_templates_noisy, x_collision_noisy,))
        x_triage = x[:, MID_POINT_IDX, 0]

        y_triage = yarr.concatenate((y_clean_1, y_collision_0))

    ###############
    # Autoencoder #
    ###############

    # # TODO: need to abstract this part of the code, create a separate
    # # function and document it
    # neighbors_ae = np.ones((n_channels, n_channels), 'int32')

    # templates_ae = crop_and_align_templates(templates_uncropped,
    #                                         CONFIG.spike_size,
    #                                         neighbors_ae,
    #                                         CONFIG.geom)

    # tt = templates_ae.transpose(1, 0, 2).reshape(templates_ae.shape[1], -1)
    # tt = tt[:, np.ptp(tt, axis=0) > 2]
    # max_amp = np.max(np.ptp(tt, axis=0))

    # y_ae = np.zeros((nk*tt.shape[1], tt.shape[0]))

    # for k in range(tt.shape[1]):
    #     amp_now = np.ptp(tt[:, k])
    #     amps_range = (np.arange(nk)*(max_amp-min_amp)
    #                   / nk+min_amp)[:, np.newaxis, np.newaxis]

    #     y_ae[k*nk:(k+1)*nk] = ((tt[:, k]/amp_now)[np.newaxis, :]
    #                            * amps_range[:, :, 0])

    # noise_ae = np.random.normal(size=y_ae.shape)
    # noise_ae = np.matmul(noise_ae, temporal_SIG)

    # x_ae = y_ae + noise_ae
    # x_ae = x_ae[:, MID_POINT_IDX]
    # y_ae = y_ae[:, MID_POINT_IDX]

    x_ae = None
    y_ae = None

    # FIXME: y_ae is no longer used, autoencoder was replaced by PCA
    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae


def spikes(templates, min_amplitude, max_amplitude,
           n_per_template, spatial_sig, temporal_sig,
           make_from_templates=True,
           make_spatially_misaligned=True,
           make_temporally_misaligned=True,
           make_collided=True,
           make_noise=True,
           return_metadata=True,
           templates_kwargs=dict(),
           collided_kwargs=dict(n_per_spike=1, min_shift=5),
           temporally_misaligned_kwargs=dict(n_per_spike=1),
           spatially_misaligned_kwargs=dict(n_per_spike=1,
                                            force_first_channel_shuffle=True),
           add_noise_kwargs={'reject_cancelling_noise': False},
           ):
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

    n_per_template: int
        How many spikes to generate per template. This along with
        min_amplitude and max_amplitude are used to generate spikes covering
        the desired amplitude range

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

    return_metadata: bool, optional
        Return metadata in the generated spikes


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

    spatial_SIG

    temporal_SIG
    """

    # NOTE: is the order importante here, maybe it's better to first compute
    # from templates, then take those and misalign spatially
    # (all templates in all channels) then take those and misalign temporally
    # and finally produce collided spikes

    # TODO: add multi_channel parameter and options for hardcoded parameter
    # FIXME: verify that the templates are in the right format, main channel
    # nearest neighbor...

    _, waveform_length, n_neigh = templates.shape

    waveform_length_sig, _ = temporal_sig.shape
    n_neigh_sig, _ = spatial_sig.shape

    if waveform_length != waveform_length_sig:
        raise ValueError("Templates waveform length ({}) doesnt match "
                         "temporal sig dimension ({})"
                         .format(waveform_length, waveform_length_sig))

    if n_neigh != n_neigh_sig:
        raise ValueError("Templates waveform length ({}) doesnt match "
                         "temporal sig dimension ({})"
                         .format(n_neigh, n_neigh_sig))

    # make spikes
    x_templates = util.make_from_templates(templates, min_amplitude,
                                           max_amplitude, n_per_template,
                                           **templates_kwargs)
    kwargs = spatially_misaligned_kwargs
    x_spatially = (util.make_spatially_misaligned(x_templates, **kwargs))

    n_spikes, _, _ = x_templates.shape

    x_all, x_all_noisy, keys, lengths = [], [], [], []

    if make_from_templates:
        (x_templates_noisy,
         x_templates_sub) = util.add_noise(x_templates,
                                           spatial_sig,
                                           temporal_sig,
                                           **add_noise_kwargs)

        x_all.append(x_templates_sub)
        x_all_noisy.append(x_templates_noisy)
        keys.append('from templates')
        lengths.append(len(x_templates_sub))

    if make_spatially_misaligned:
        (x_spatially_noisy,
         x_spatially_sub) = util.add_noise(x_spatially,
                                           spatial_sig,
                                           temporal_sig,
                                           **add_noise_kwargs)
        x_all.append(x_spatially_sub)
        x_all_noisy.append(x_spatially_noisy)
        keys.append('spatially misaligned')
        lengths.append(len(x_spatially_sub))

    if make_temporally_misaligned:
        kwargs = temporally_misaligned_kwargs
        x_temporally = (util.make_temporally_misaligned(x_spatially, **kwargs))

        (x_temporally_noisy,
         x_temporally) = util.add_noise(x_temporally,
                                        spatial_sig,
                                        temporal_sig,
                                        **add_noise_kwargs)

        x_all.append(x_temporally)
        x_all_noisy.append(x_temporally_noisy)
        keys.append('temporally misaligned')
        lengths.append(len(x_temporally))

    if make_collided:
        x_collided = util.make_collided(x_templates, x_spatially,
                                        **collided_kwargs)

        (x_collided_noisy,
         x_collided) = util.add_noise(x_collided,
                                      spatial_sig,
                                      temporal_sig,
                                      **add_noise_kwargs)

        x_all.append(x_collided)
        x_all_noisy.append(x_collided_noisy)
        keys.append('collided')
        lengths.append(len(x_collided))

    if make_noise:
        x_zero = np.zeros((n_spikes, waveform_length, n_neigh))

        (x_zero_noisy,
         x_zero) = util.add_noise(x_zero,
                                  spatial_sig,
                                  temporal_sig,
                                  **add_noise_kwargs)

        x_all.append(x_zero)
        x_all_noisy.append(x_zero_noisy)
        keys.append('noise')
        lengths.append(len(x_zero))

    x_all = np.concatenate(x_all, axis=0)
    x_all_noisy = np.concatenate(x_all_noisy, axis=0)

    # compute amplitudes
    the_amplitudes = util.amplitudes(x_all)

    def previous(lengths, i):
        if i == 0:
            return 0
        else:
            return sum(lengths[:i]) + 1

    def following(lengths, i):
        return previous(lengths, i) + lengths[i]

    # return a dictionary with slices for every type of spike generated
    slices = {k: slice(previous(lengths, i), following(lengths, i))
              for k, i
              in zip(keys, range(len(lengths)))}

    # FIXME: shoudld not return sigs
    return (x_all, x_all_noisy, the_amplitudes, slices, spatial_sig,
            temporal_sig)
