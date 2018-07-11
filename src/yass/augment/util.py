"""Utility functions for augmenting data
"""
import random
import numpy as np
from yass import DEBUG_MODE


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


def make_clean(templates, min_amp, max_amp, nk):
    """Make clean spikes from templates

    Parameters
    ----------
    templates: numpy.ndarray, (n_templates, waveform_length, n_channels)
        Templates

    min_amp: float
        Minimum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel

    max_amp: float
        Maximum value allowed for the maximum absolute amplitude of the
        isolated spike on its main channel

    nk: int
        (n_templates * nk ) spikes will be produced

    Returns
    -------
    numpy.ndarray (n_templates * nk, waveform_length, n_channels)
        Clean spikes
    """
    n_templates, waveform_length, n_neighbors = templates.shape

    x_clean = np.zeros((nk * n_templates, waveform_length, n_neighbors))

    d = max_amp - min_amp
    amps_range = (min_amp + np.arange(nk) * d/nk)[:, np.newaxis, np.newaxis]

    for k in range(n_templates):

        current = templates[k]
        amp = np.max(np.abs(current))
        scaled = (current/amp)[np.newaxis, :, :]

        # create n clean spikes by scaling the template along the range
        x_clean[k * nk: (k + 1) * nk] = scaled * amps_range

    return x_clean


def make_collided(x_clean, collision_ratio, templates, max_shift,
                  multi_channel, nneigh, amp_tolerance=0.2):
    """Make collided spikes

    Parameters
    ----------
    x_clean
    collision_ratio
    templates
    max_shift
    multi_channel
    """
    # FIXME: nneigh can be removed

    n_clean, _, _ = x_clean.shape
    _, wf_length, n_neighbors = templates.shape

    x_collision = np.zeros((n_clean*int(collision_ratio),
                            wf_length,
                            n_neighbors))

    if DEBUG_MODE:
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
            shuffled_neighs = np.random.choice(nneigh, nneigh, replace=False)
            x_to_collide = x_to_collide[:, shuffled_neighs]

        if DEBUG_MODE:
            to_add = x_to_collide_all
        else:
            to_add = x_collision

        if shift > 0:
            to_add[j, :(wf_length-shift)] += x_to_collide[shift:]
        elif shift < 0:
            to_add[j, (-shift):] += x_to_collide[:(wf_length+shift)]
        else:
            to_add[j] += x_to_collide

    if DEBUG_MODE:
        return x_collision, x_to_collide_all
    else:
        return x_collision


def make_misaligned(x_clean, templates, max_shift, misalign_ratio,
                    multi, misalign_ratio2, nneigh):
    """Make temporally and spatially misaligned spikes
    """

    ################################
    # temporally misaligned spikes #
    ################################

    x_temporally = np.zeros(
        (x_clean.shape[0]*int(misalign_ratio), templates.shape[1],
            templates.shape[2]))

    temporal_shifts = np.random.randint(
        max_shift*2, size=x_temporally.shape[0]) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    for j in range(x_temporally.shape[0]):
        shift = temporal_shifts[j]
        if multi:
            x_clean2 = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)][:, :, np.random.choice(
                    nneigh, nneigh, replace=False)])
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


def make_noise(x_clean, noise_ratio, templates, spatial_SIG, temporal_SIG):
    """Make noise
    """

    # get noise
    noise = np.random.normal(
        size=[x_clean.shape[0]*int(noise_ratio), templates.shape[1],
              templates.shape[2]])

    for c in range(noise.shape[2]):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, noise.shape[2]))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           [noise.shape[0],
                            x_clean.shape[1], x_clean.shape[2]])

    return the_noise
