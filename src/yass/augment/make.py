import os
import numpy as np
import logging


from .choose import choose_templates
from .crop import crop_templates
from .noise import noise_cov
from ..process.templates import get_templates
from ..util import load_yaml


# TODO: documentation
# TODO: comment code, it's not clear what it does
def make_training_data(CONFIG, spike_train, chosen_templates, min_amp,
                       nspikes, data_folder):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """

    logger = logging.getLogger(__name__)

    path_to_data = os.path.join(data_folder, 'standarized.bin')
    path_to_config = os.path.join(data_folder, 'standarized.yaml')

    # make sure standarized data already exists
    if not os.path.exists(path_to_data):
        raise ValueError('Standarized data does not exist in: {}, this is '
                         'needed to generate training data, run the '
                         'preprocesor first to generate it'
                         .format(path_to_data))

    PARAMS = load_yaml(path_to_config)

    logger.info('Getting templates...')

    # get templates
    templates, _ = get_templates(spike_train, path_to_data, CONFIG.spikeSize)

    templates = np.transpose(templates, (2, 1, 0))

    logger.info('Got templates ndarray of shape: {}'.format(templates.shape))

    # choose good templates (good looking and big enough)
    templates = choose_templates(templates, chosen_templates)

    if templates.shape[0] == 0:
        raise ValueError("Coulndt find any good templates...")

    logger.info('Good looking templates of shape: {}'.format(templates.shape))

    # align and crop templates
    templates = crop_templates(templates, CONFIG.spikeSize,
                               CONFIG.neighChannels, CONFIG.geom)

    # determine noise covariance structure
    spatial_SIG, temporal_SIG = noise_cov(path_to_data,
                                          PARAMS['dtype'],
                                          CONFIG.recordings.n_channels,
                                          PARAMS['data_format'],
                                          CONFIG.neighChannels,
                                          CONFIG.geom,
                                          templates.shape[1])

    # make training data set
    K = templates.shape[0]
    R = CONFIG.spikeSize
    amps = np.max(np.abs(templates), axis=1)

    # make clean augmented spikes
    nk = int(np.ceil(nspikes/K))
    max_amp = np.max(amps)*1.5
    nneigh = templates.shape[2]

    ################
    # clean spikes #
    ################
    x_clean = np.zeros((nk*K, templates.shape[1], templates.shape[2]))
    for k in range(K):
        tt = templates[k]
        amp_now = np.max(np.abs(tt))
        amps_range = (np.arange(nk)*(max_amp-min_amp)
                      / nk+min_amp)[:, np.newaxis, np.newaxis]
        x_clean[k*nk:(k+1)*nk] = (tt/amp_now)[np.newaxis, :, :]*amps_range

    #############
    # collision #
    #############
    x_collision = np.zeros(x_clean.shape)
    max_shift = 2*R

    temporal_shifts = np.random.randint(max_shift*2, size=nk*K) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    amp_per_data = np.max(x_clean[:, :, 0], axis=1)
    for j in range(nk*K):
        shift = temporal_shifts[j]

        x_collision[j] = np.copy(x_clean[j])
        idx_candidate = np.where(amp_per_data > amp_per_data[j]*0.3)[0]
        idx_match = idx_candidate[np.random.randint(
            idx_candidate.shape[0], size=1)[0]]
        x_clean2 = np.copy(x_clean[idx_match][:, np.random.choice(
            nneigh, nneigh, replace=False)])

        if shift > 0:
            x_collision[j, :(x_collision.shape[1]-shift)] += x_clean2[shift:]

        elif shift < 0:
            x_collision[
                j, (-shift):] += x_clean2[:(x_collision.shape[1]+shift)]
        else:
            x_collision[j] += x_clean2

    #####################
    # misaligned spikes #
    #####################
    x_misaligned = np.zeros(x_clean.shape)

    temporal_shifts = np.random.randint(max_shift*2, size=nk*K) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    for j in range(nk*K):
        shift = temporal_shifts[j]
        x_clean2 = np.copy(x_clean[j][:, np.random.choice(
            nneigh, nneigh, replace=False)])

        if shift > 0:
            x_misaligned[j, :(x_collision.shape[1]-shift)] += x_clean2[shift:]

        elif shift < 0:
            x_misaligned[
                j, (-shift):] += x_clean2[:(x_collision.shape[1]+shift)]
        else:
            x_misaligned[j] += x_clean2

    #########
    # noise #
    #########

    # get noise
    noise = np.random.normal(size=x_clean.shape)
    for c in range(noise.shape[2]):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)

        reshaped_noise = np.reshape(noise, (-1, noise.shape[2]))
    noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                       [x_clean.shape[0], x_clean.shape[1], x_clean.shape[2]])

    y_clean = np.ones((x_clean.shape[0]))
    y_col = np.ones((x_clean.shape[0]))
    y_misalinged = np.zeros((x_clean.shape[0]))
    y_noise = np.zeros((x_clean.shape[0]))

    mid_point = int((x_clean.shape[1]-1)/2)

    # get training set for detection
    x = np.concatenate((
        x_clean + noise,
        x_collision + noise[np.random.permutation(noise.shape[0])],
        x_misaligned + noise[np.random.permutation(noise.shape[0])],
        noise
    ))

    x_detect = x[:, (mid_point-R):(mid_point+R+1), :]
    y_detect = np.concatenate((y_clean, y_col, y_misalinged, y_noise))

    # get training set for triage
    x = np.concatenate((
        x_clean + noise,
        x_collision + noise[np.random.permutation(noise.shape[0])],
    ))
    x_triage = x[:, (mid_point-R):(mid_point+R+1), :]
    y_triage = np.concatenate((y_clean, np.zeros((x_clean.shape[0]))))

    # ge training set for auto encoder
    ae_shift_max = 1
    temporal_shifts_ae = np.random.randint(
        ae_shift_max*2+1, size=x_clean.shape[0]) - ae_shift_max
    y_ae = np.zeros((x_clean.shape[0], 2*R+1))
    x_ae = np.zeros((x_clean.shape[0], 2*R+1))
    for j in range(x_ae.shape[0]):
        y_ae[j] = x_clean[j, (mid_point-R+temporal_shifts_ae[j]):
                          (mid_point+R+1+temporal_shifts_ae[j]), 0]
        x_ae[j] = x_clean[j, (mid_point-R+temporal_shifts_ae[j]):
                          (mid_point+R+1+temporal_shifts_ae[j]), 0]+noise[
            j, (mid_point-R+temporal_shifts_ae[j]):
            (mid_point+R+1+temporal_shifts_ae[j]), 0]

    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae
