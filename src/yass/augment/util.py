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


def clean_spikes(templates, min_amp, max_amp, nk):
    """Make clean spikes
    """
    K = templates.shape[0]

    x_clean = np.zeros((nk * K, templates.shape[1], templates.shape[2]))

    for k in range(K):
        tt = templates[k]
        amp_now = np.max(np.abs(tt))
        amps_range = (np.arange(nk)*(max_amp-min_amp)
                      / nk+min_amp)[:, np.newaxis, np.newaxis]
        x_clean[k*nk:(k+1)*nk] = (tt/amp_now)[np.newaxis, :, :]*amps_range

    return x_clean


def collided_spikes(x_clean, collision_ratio, templates, R, multi,
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


def misaligned_spikes(x_clean, templates, max_shift, misalign_ratio,
                      multi, misalign_ratio2, nneigh):
    """temporally and spatially misaligned spikes
    """

    x_misaligned = np.zeros(
        (x_clean.shape[0]*int(misalign_ratio), templates.shape[1],
            templates.shape[2]))

    temporal_shifts = np.random.randint(
        max_shift*2, size=x_misaligned.shape[0]) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    for j in range(x_misaligned.shape[0]):
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
            x_misaligned[j, :(x_misaligned.shape[1]-shift)] += x_clean2[shift:]

        elif shift < 0:
            x_misaligned[
                j, (-shift):] += x_clean2[:(x_misaligned.shape[1]+shift)]
        else:
            x_misaligned[j] += x_clean2

    ################################
    # spatially misaligned spikes #
    ##############################
    if multi:
        x_misaligned2 = np.zeros(
            (x_clean.shape[0]*int(misalign_ratio2), templates.shape[1],
                templates.shape[2]))
        for j in range(x_misaligned2.shape[0]):
            x_misaligned2[j] = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)][:, :, np.random.choice(
                    nneigh, nneigh, replace=False)])

    return x_misaligned, x_misaligned2


def noise(x_clean, noise_ratio, templates, spatial_SIG, temporal_SIG):
    """make noise
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
