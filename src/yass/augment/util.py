"""Utility functions for augmenting data
"""
import random
import logging
import numpy as np
import logging
from yass import _get_debug_mode

logger = logging.getLogger(__name__)


def make_noisy(x, the_noise):
    """Make a noisy version of x
    """
    noise_sample = the_noise[np.random.choice(the_noise.shape[0],
                                              x.shape[0],
                                              replace=False)]
    return x + noise_sample


def sample_from_zero_axis(x, axis=0):
    """Sample from a certain axis
    """
    idx = np.random.choice(x.shape[0], 1, replace=True)
    return x[idx], idx


def amplitudes(x):
    """Compute amplitudes
    """
    return np.max(np.abs(x), axis=(1, 2))


def make_clean(templates, min_amplitude, max_amplitude, n_per_template):
    """Make clean spikes from templates

    Parameters
    ----------
    templates: numpy.ndarray, (n_templates, waveform_length, n_channels)
        Templates

    min_amplitude: float
        Minimum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel

    max_amplitude: float
        Maximum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel

    n_per_template: int
        How many spikes to generate per template

    Returns
    -------
    numpy.ndarray (n_templates * n_per_template, waveform_length, n_channels)
        Clean spikes
    """
    logger = logging.getLogger(__name__)

    logger.debug('templates shape: %s, min amplitude: %s, '
                 'max_amplitude: %s', templates.shape, min_amplitude,
                 max_amplitude)

    n_templates, waveform_length, n_neighbors = templates.shape

    x_clean = np.zeros((n_per_template * n_templates,
                        waveform_length, n_neighbors))

    d = max_amplitude - min_amplitude
    amps_range = (min_amplitude + np.arange(n_per_template) * d/n_per_template)
    amps_range = amps_range[:, np.newaxis, np.newaxis]

    # go over every template
    for k in range(n_templates):

        # get current template and scale it
        current = templates[k]
        amp = np.max(np.abs(current))
        scaled = (current/amp)[np.newaxis, :, :]

        # create n clean spikes by scaling the template along the range
        x_clean[k * n_per_template: (k + 1) * n_per_template] = (scaled
                                                                 * amps_range)

    return x_clean


def make_collided(x_clean, collision_ratio, multi_channel, amp_tolerance=0.2,
                  max_shift='auto'):
    """Make collided spikes

    Parameters
    ----------
    x_clean
    collision_ratio
    multi_channel
    amp_tolerance: float, optional
        Maximum relative difference in amplitude between the collided spikes,
        defaults to 0.2
    max_shift: int or string, optional
        Maximum amount of shift for the collided spike. If 'auto', it sets
        to half the waveform length in x_clean
    """
    logger = logging.getLogger(__name__)

    n_clean, wf_length, n_neighbors = x_clean.shape

    if max_shift == 'auto':
        max_shift = int((wf_length - 1) / 2)

    logger.debug('Making collided spikes with collision_ratio: %s max shift: '
                 '%s, multi_channel: %s, amp_tolerance: %s, clean spikes with'
                 ' shape: %s', collision_ratio, max_shift, multi_channel,
                 amp_tolerance, x_clean.shape)

    x_collision = np.zeros((n_clean*int(collision_ratio),
                            wf_length,
                            n_neighbors))

    if _get_debug_mode():
        logger.info('Running in debug mode...')
        x_to_collide_all = np.zeros(x_collision.shape)

    n_collided, _, _ = x_collision.shape

    amps = amplitudes(x_clean)

    for j in range(n_collided):

        # random shifting
        shift = random.randint(-max_shift, max_shift)

        # sample a clean spike
        x_collision[j], i = sample_from_zero_axis(x_clean)

        # get amplitude for sampled x_clean and compute bounds
        amp = amps[i]
        lower = amp * (1.0 - amp_tolerance)
        upper = amp * (1.0 + amp_tolerance)

        # draw another clean spike
        scale_factor = np.linspace(lower, upper, num=50)[random.randint(0, 49)]
        x_to_collide, i = sample_from_zero_axis(x_clean)
        x_to_collide = scale_factor * x_to_collide / amps[i]
        # FIXME: remove this
        x_to_collide = x_to_collide[0, :, :]

        if multi_channel:
            shuffled_neighs = np.random.choice(n_neighbors, n_neighbors,
                                               replace=False)
            x_to_collide = x_to_collide[:, shuffled_neighs]

        if _get_debug_mode():
            to_add = x_to_collide_all
        else:
            to_add = x_collision

        if shift > 0:
            to_add[j, :(wf_length-shift)] += x_to_collide[shift:]
        elif shift < 0:
            to_add[j, (-shift):] += x_to_collide[:(wf_length+shift)]
        else:
            to_add[j] += x_to_collide

    if _get_debug_mode():
        return x_collision, x_to_collide_all
    else:
        return x_collision


def make_misaligned(x_clean, templates, max_shift, misalign_ratio,
                    multi, misalign_ratio2, nneigh):
    """Make temporally and spatially misaligned spikes from clean templates

    Parameters
    ----------
    """

    ################################
    # temporally misaligned spikes #
    ################################

    x_temporally = make_temporally_misaligned(x_clean, misalign_ratio,
                                              multi, max_shift)

    ###############################
    # spatially misaligned spikes #
    ###############################

    if multi:
        x_spatially = np.zeros(
            (x_clean.shape[0]*int(misalign_ratio2), templates.shape[1],
                templates.shape[2]))

        for j in range(x_spatially.shape[0]):
            x_spatially[j] = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)][:, :, np.random.choice(
                    nneigh, nneigh, replace=False)])

    return x_temporally, x_spatially


def make_temporally_misaligned(x_clean, n_per_spike, multi, max_shift='auto'):
    """Make temporally shifted spikes from clean spikes
    """
    n_spikes, waveform_length, n_neigh = x_clean.shape
    n_out = int(n_spikes * n_per_spike)

    if max_shift == 'auto':
        max_shift = int(waveform_length / 2)

    x_temporally = np.zeros((n_out, waveform_length, n_neigh))

    logger.debug('Making spikes with max_shift: %i, output shape: %s',
                 max_shift, x_temporally.shape)

    temporal_shifts = np.random.randint(-max_shift, max_shift, size=n_out)

    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    for j in range(n_out):
        shift = temporal_shifts[j]
        if multi:
            x_clean2 = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)][:, :, np.random.choice(
                    n_neigh, n_neigh, replace=False)])
            x_clean2 = np.squeeze(x_clean2)
        else:
            x_clean2 = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)])
            x_clean2 = np.squeeze(x_clean2)

        if shift > 0:
            x_temporally[j, :(x_temporally.shape[1]-shift)] += x_clean2[shift:]

        elif shift < 0:
            x_temporally[
                j, (-shift):] += x_clean2[:(x_temporally.shape[1]+shift)]
        else:
            x_temporally[j] += x_clean2

    return x_temporally


def make_noise(x_clean, n_per_spike, spatial_SIG, temporal_SIG):
    """Make noise
    """
    n_spikes, waveform_length, n_neigh = x_clean.shape
    n_out = int(n_spikes * n_per_spike)

    # get noise
    noise = np.random.normal(size=(n_out, waveform_length, n_neigh))

    for c in range(n_neigh):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           (n_out, waveform_length, n_neigh))

    return the_noise
