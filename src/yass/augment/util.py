"""Utility functions for augmenting data
"""
import numpy as np


def make_noisy(x, the_noise):
    """Make a noisy version of x
    """
    noise_sample = the_noise[np.random.choice(the_noise.shape[0],
                                              x.shape[0],
                                              replace=False)]
    return x + noise_sample


def make_clean(templates, min_amp, max_amp, n):
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

    n: int
        (n_templates * n ) spikes will be produced

    Returns
    -------
    numpy.ndarray (n_templates * n, waveform_length, n_channels)
        Clean spikes
    """
    n_templates, waveform_length, n_neighbors = templates.shape

    x_clean = np.zeros((n * n_templates, waveform_length, n_neighbors))

    for k in range(n_templates):

        current = templates[k]
        amp = np.max(np.abs(current))

        amps_range = (np.arange(n)*(max_amp-min_amp)
                      / n+min_amp)[:, np.newaxis, np.newaxis]

        x_clean[k * n: (k+1) * n] = (current/amp)[np.newaxis, :, :]*amps_range

    return x_clean


def make_collided(x_clean, collision_ratio, templates, R, multi,
                  nneigh):
    """Make collided spikes
    """
    x_collision = np.zeros(
        (x_clean.shape[0]*int(collision_ratio), templates.shape[1],
         templates.shape[2]))
    max_shift = 2*R

    temporal_shifts = np.random.randint(
        max_shift*2, size=x_collision.shape[0]) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    amp_per_data = np.max(x_clean[:, :, 0], axis=1)

    for j in range(x_collision.shape[0]):
        shift = temporal_shifts[j]

        x_collision[j] = np.copy(x_clean[np.random.choice(
            x_clean.shape[0], 1, replace=True)])
        idx_candidate = np.where(
            amp_per_data > np.max(x_collision[j, :, 0])*0.3)[0]
        idx_match = idx_candidate[np.random.randint(
            idx_candidate.shape[0], size=1)[0]]
        if multi:
            x_clean2 = np.copy(x_clean[idx_match][:, np.random.choice(
                nneigh, nneigh, replace=False)])
        else:
            x_clean2 = np.copy(x_clean[idx_match])

        if shift > 0:
            x_collision[j, :(x_collision.shape[1]-shift)] += x_clean2[shift:]

        elif shift < 0:
            x_collision[
                j, (-shift):] += x_clean2[:(x_collision.shape[1]+shift)]
        else:
            x_collision[j] += x_clean2

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
