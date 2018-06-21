import os
import numpy as np
import logging


from yass.augment.choose import choose_templates
from yass.augment.crop import crop_templates
from yass.augment.noise import noise_cov
from yass.templates.util import get_templates
from yass.util import load_yaml


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


def make_training_data(CONFIG, spike_train, chosen_templates, min_amp,
                       nspikes, data_folder, noise_ratio=10, collision_ratio=1,
                       misalign_ratio=1, misalign_ratio2=1, multi=True):
    """Makes training sets for detector, triage and autoencoder

    Parameters
    ----------
    CONFIG: yaml file
        Configuration file
    spike_train: numpy.ndarray
        [number of spikes, 2] Ground truth for training. First column is the
        spike time, second column is the spike id
    chosen_templates: list
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
    multi: bool
        If multi= True, generate training data for multi-channel neural
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

    """

    logger = logging.getLogger(__name__)

    path_to_data = os.path.join(data_folder, 'preprocess', 'standarized.bin')
    path_to_config = os.path.join(data_folder, 'preprocess',
                                  'standarized.yaml')

    # make sure standarized data already exists
    if not os.path.exists(path_to_data):
        raise ValueError('Standarized data does not exist in: {}, this is '
                         'needed to generate training data, run the '
                         'preprocesor first to generate it'
                         .format(path_to_data))

    PARAMS = load_yaml(path_to_config)

    logger.info('Getting templates...')

    # get templates
    templates, _ = get_templates(
        np.hstack((spike_train,
                   np.ones((spike_train.shape[0], 1), 'int32'))),
        path_to_data,
        CONFIG.resources.max_memory,
        4*CONFIG.spike_size)

    templates = np.transpose(templates, (2, 1, 0))

    logger.info('Got templates ndarray of shape: {}'.format(templates.shape))

    # choose good templates (good looking and big enough)
    templates = choose_templates(templates, chosen_templates)
    templates_uncropped = np.copy(templates)

    if templates.shape[0] == 0:
        raise ValueError("Coulndt find any good templates...")

    logger.info('Good looking templates of shape: {}'.format(templates.shape))

    # align and crop templates
    templates = crop_templates(templates, CONFIG.spike_size,
                               CONFIG.neigh_channels, CONFIG.geom)

    # make training data set
    K = templates.shape[0]
    R = CONFIG.spike_size
    amps = np.max(np.abs(templates), axis=1)

    # make clean augmented spikes
    nk = int(np.ceil(nspikes/K))
    max_amp = np.max(amps)*1.5
    nneigh = templates.shape[2]
    max_shift = 2*R

    # make clean spikes
    x_clean = clean_spikes(templates, min_amp, max_amp, nk)

    # make collided spikes
    x_collision = collided_spikes(x_clean, collision_ratio, templates,
                                  R, multi, nneigh)

    # make misaligned spikes
    x_misaligned, x_misaligned2 = misaligned_spikes(x_clean,
                                                    templates, max_shift,
                                                    misalign_ratio,
                                                    misalign_ratio2,
                                                    multi,
                                                    nneigh)

    # determine noise covariance structure
    spatial_SIG, temporal_SIG = noise_cov(path_to_data,
                                          PARAMS['dtype'],
                                          CONFIG.recordings.n_channels,
                                          PARAMS['data_order'],
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    # make noise
    the_noise = noise(x_clean, noise_ratio, templates, spatial_SIG,
                      temporal_SIG)

    # make labels
    y_clean = np.ones((x_clean.shape[0]))
    y_col = np.ones((x_collision.shape[0]))

    y_misaligned = np.zeros((x_misaligned.shape[0]))
    y_noise = np.zeros((the_noise.shape[0]))

    if multi:
        y_misaligned2 = np.zeros((x_misaligned2.shape[0]))

    mid_point = int((x_clean.shape[1]-1)/2)

    x_clean_noisy = make_noisy(x_clean)
    x_collision_noisy = make_noisy(x_collision)
    x_misaligned_noisy = make_noisy(x_misaligned)
    x_misaligned2_noisy = make_noisy(x_misaligned2)

    # get training set for detection
    if multi:
        x = np.concatenate((x_clean_noisy, x_collision_noisy,
                            x_misaligned_noisy, the_noise))

        x_detect = x[:, (mid_point-R):(mid_point+R+1), :]
        y_detect = np.concatenate((y_clean, y_col, y_misaligned, y_noise))
    else:
        x = np.concatenate((x_clean_noisy, x_misaligned_noisy, the_noise))
        x_detect = x[:, (mid_point-R):(mid_point+R+1), 0]
        y_detect = np.concatenate((y_clean, y_misaligned, y_noise))

    # get training set for triage
    if multi:
        x = np.concatenate((x_clean_noisy, x_collision_noisy,
                            x_misaligned2_noisy))

        x_triage = x[:, (mid_point-R):(mid_point+R+1), :]
        y_triage = np.concatenate(
            (y_clean, np.zeros((x_collision.shape[0])), y_misaligned2))
    else:
        x = np.concatenate((x_clean_noisy, x_collision_noisy,))
        x_triage = x[:, (mid_point-R):(mid_point+R+1), 0]
        y_triage = np.concatenate((y_clean, np.zeros((x_collision.shape[0]))))

    ###############
    # Autoencoder #
    ###############

    n_channels = templates_uncropped.shape[2]
    templates_ae = crop_templates(templates_uncropped, CONFIG.spike_size,
                                  np.ones((n_channels, n_channels), 'int32'),
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

    the_noise_ae = np.random.normal(size=y_ae.shape)
    the_noise_ae = np.matmul(the_noise_ae, temporal_SIG)

    x_ae = y_ae + the_noise_ae
    x_ae = x_ae[:, (mid_point-R):(mid_point+R+1)]
    y_ae = y_ae[:, (mid_point-R):(mid_point+R+1)]

    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae
