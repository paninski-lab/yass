import os
import numpy as np
import logging


from yass.augment.choose import choose_templates
from yass.augment.crop import crop_templates
from yass.augment.noise import noise_cov
from yass.templates.util import get_templates
from yass.util import load_yaml

# TODO: documentation
# TODO: comment code, it's not clear what it does


def make_training_data_2(CONFIG, spike_train, chosen_templates, min_amp,
                       nspikes, templates, data_folder, noise_ratio=10, 
                       collision_ratio=1, misalign_ratio=1, misalign_ratio2=1, 
                       max_memory='5GB', multi=True):
    """[Description]
    Parameters
    ----------
    CONFIG: yaml file
        Configuration file
    spike_train: numpy.ndarray
        [number of spikes, 2] Ground truth for training. First column is the spike time, second column is the spike id  
    chosen_templates: list
        List of chosen templates' id's  
    min_amp: float
        Minimum value allowed for the maximum absolute amplitude of the isolated spike on its main channel  
    nspikes: int
        Number of isolated spikes to generate. This is different from the total number of x_detect   
    data_folder: str
        Folder storing the standarized data (if not exist, run preprocess to automatically generate)     
    noise_ratio: int
        Ratio of number of noise to isolated spikes. For example, if n_isolated_spike=1000, noise_ratio=5, then n_noise=5000     
    collision_ratio: int
        Ratio of number of collisions to isolated spikes.    
    misalign_ratio: int
        Ratio of number of spatially and temporally misaligned spikes to isolated spikes 
    misalign_ratio2: int
        Ratio of number of only-spatially misaligned spikes to isolated spikes      
    multi: bool
        If multi= True, generate training data for multi-channel neural network. Otherwise generate single-channel data

    Returns
    -------
    x_detect: numpy.ndarray
        [number of detection training data, temporal length, number of channels] Training data for the detect net.
    y_detect: numpy.ndarray
        [number of detection training data] Label for x_detect

    x_triage: numpy.ndarray
        [number of triage training data, temporal length, number of channels] Training data for the triage net.
    y_triage: numpy.ndarray
        [number of triage training data] Label for x_triage


    x_ae: numpy.ndarray
        [number of ae training data, temporal length] Training data for the autoencoder: noisy spikes
    y_ae: numpy.ndarray
        [number of ae training data, temporal length] Denoised x_ae

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

    # get templates - making templates from spike train
    templates, _ = get_templates(
        spike_train, path_to_data, max_memory, CONFIG, 4*CONFIG.spike_size)
                          
    templates = np.transpose(templates, (2, 1, 0))
  
    # choose good templates (good looking and big enough)
    templates = choose_templates(templates, chosen_templates)
    #templates = choose_templates(templates, np.arange(templates.shape[0]))

    print ("Sub selected # of templates: ", templates.shape)

    logger.info('Got templates ndarray of shape: {}'.format(templates.shape))


    if templates.shape[0] == 0:
        raise ValueError("Coulndt find any good templates...")

    logger.info('Good looking templates of shape: {}'.format(templates.shape))

    # align and crop templates;
    # Cat: original size of tempalte is 4 x cross size; 
    #      need to make it smaller
    templates = crop_templates(templates, CONFIG.spike_size,
                               CONFIG.neigh_channels, CONFIG.geom)

    # determine noise covariance structure
    # Cat: spatial covariance matrix is 7 x 7
    # 2 matrices spatial and temporal
    spatial_SIG, temporal_SIG = noise_cov(path_to_data,
                                          PARAMS['dtype'],
                                          CONFIG.recordings.n_channels,
                                          PARAMS['data_order'],
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    # Cat: Make augmented data

    # make training data set
    K = templates.shape[0]
    R = CONFIG.spike_size
    amps = np.max(np.abs(templates), axis=1)  # Cat: max amplitude on main chan

    # make clean augmented spikes
    nk = int(np.ceil(nspikes/K))    # get average # spikes / templates
    max_amp = np.max(amps)*1.5
    nneigh = templates.shape[2]

    ################
    # clean spikes #
    ################
    # placeholder for clean spikes
    # generate augmented spikes for eacah tempalte
    # arange tempaltes from min to max amplitude
    # clean spikes, no noise
    x_clean = np.zeros((nk*K, templates.shape[1], templates.shape[2]))
    for k in range(K):
        tt = templates[k]
        amp_now = np.max(np.abs(tt))
        amps_range = (np.arange(nk)*(max_amp-min_amp)           #Uniformly generate augmented spike sizes
                      / nk+min_amp)[:, np.newaxis, np.newaxis]
        x_clean[k*nk:(k+1)*nk] = (tt/amp_now)[np.newaxis, :, :]*amps_range


    #############
    # collision #
    #############
    # collision_ratio: # of clean spikes to # of collision spikes
    x_collision = np.zeros(
        (x_clean.shape[0]*int(collision_ratio), templates.shape[1],
         templates.shape[2]))
    max_shift = 2*R

    # main spike kept as above (max chan is first)
    # collission spike is randomly tempporally and spatially shifted
    # have to ensure temporal shift is sufficient otherwise difficult to detect
    temporal_shifts = np.random.randint(
        max_shift*2, size=x_collision.shape[0]) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    # make sure that secondary/collission spike is sufficiently large
    # otherwise tough to detect very small collissions
    amp_per_data = np.max(x_clean[:, :, 0], axis=1)
    
    #make collission data; candidate collission spike has to be > 0.3 clean spike
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

    ##############################################
    # temporally and spatially misaligned spikes #
    ##############################################
    
    # Cat: another negative case with just reference spikes that are 
    # misalgined, i.e. offcentre, or whose main channel is not first channel
    # train to discard these
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

    ###############################
    # spatially misaligned spikes #
    ###############################
    # Cat: another case where just spatially misaligning
    # need this case for triage nn; 
    # misalgined2 different than above
    if multi:
        x_misaligned2 = np.zeros(
            (x_clean.shape[0]*int(misalign_ratio2), templates.shape[1],
                templates.shape[2]))
        for j in range(x_misaligned2.shape[0]):
            x_misaligned2[j] = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)][:, :, np.random.choice(
                    nneigh, nneigh, replace=False)])

    #########
    # noise #
    #########
    # get noise
    # Cat: generate gaussian with mean=0; covariance same as above;
    noise = np.random.normal(
        size=[x_clean.shape[0]*int(noise_ratio), templates.shape[1],
              templates.shape[2]])
    for c in range(noise.shape[2]):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)

        reshaped_noise = np.reshape(noise, (-1, noise.shape[2]))
    noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                       [noise.shape[0], x_clean.shape[1], x_clean.shape[2]])

    y_clean = np.ones((x_clean.shape[0]))
    y_col = np.ones((x_collision.shape[0]))  #for detection collissions are 1s
    y_misaligned = np.zeros((x_misaligned.shape[0]))
    if multi:
        y_misaligned2 = np.zeros((x_misaligned2.shape[0]))
    y_noise = np.zeros((noise.shape[0]))

    mid_point = int((x_clean.shape[1]-1)/2)

    # Cat: Label data 
    # get training set for detection
    # stacking all data together:  clean, collissions, misaligned, noise
    # everything gets noise addded
    if multi:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_collision +
            noise[np.random.choice(
                noise.shape[0], x_collision.shape[0], replace=False)],
            x_misaligned +
            noise[np.random.choice(
                noise.shape[0], x_misaligned.shape[0], replace=False)],
            noise
        ))

        x_detect = x[:, (mid_point-R):(mid_point+R+1), :]
        y_detect = np.concatenate((y_clean, y_col, y_misaligned, y_noise))
    else:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_misaligned +
            noise[np.random.choice(
                noise.shape[0], x_misaligned.shape[0], replace=False)],
            noise
        ))
        x_detect = x[:, (mid_point-R):(mid_point+R+1), 0]
        y_detect = np.concatenate((y_clean, y_misaligned, y_noise))

    # get training set for triage
    if multi:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_collision +
            noise[np.random.choice(
                noise.shape[0], x_collision.shape[0], replace=False)],
            x_misaligned2 +
            noise[np.random.choice(
                noise.shape[0], x_misaligned2.shape[0], replace=False)],
        ))
        x_triage = x[:, (mid_point-R):(mid_point+R+1), :]
        y_triage = np.concatenate(
            (y_clean, np.zeros((x_collision.shape[0])), y_misaligned2))
    else:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_collision +
            noise[np.random.choice(
                noise.shape[0], x_collision.shape[0], replace=False)],
        ))
        x_triage = x[:, (mid_point-R):(mid_point+R+1), 0]
        y_triage = np.concatenate((y_clean, np.zeros((x_collision.shape[0]))))

    # ge training set for auto encoder
    # AE is single channel data
    ae_shift_max = 1
    temporal_shifts_ae = np.random.randint(
        ae_shift_max*2+1, size=x_clean.shape[0]) - ae_shift_max
    y_ae = np.zeros((x_clean.shape[0], 2*R+1))   # denoised version
    x_ae = np.zeros((x_clean.shape[0], 2*R+1))   # noise version
    
    # training goal: get matrix that tries to reduce noise version to 
    # denoised version - which is just the template...
    for j in range(x_ae.shape[0]):
        y_ae[j] = x_clean[j, (mid_point-R+temporal_shifts_ae[j]):
                          (mid_point+R+1+temporal_shifts_ae[j]), 0]
        x_ae[j] = x_clean[j, (mid_point-R+temporal_shifts_ae[j]):
                          (mid_point+R+1+temporal_shifts_ae[j]), 0]+noise[
            j, (mid_point-R+temporal_shifts_ae[j]):
            (mid_point+R+1+temporal_shifts_ae[j]), 0]

    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae



def make_training_data(CONFIG, spike_train, chosen_templates, min_amp,
                       nspikes, data_folder, noise_ratio=10, collision_ratio=1,
                       misalign_ratio=1, misalign_ratio2=1, multi=True):
    """[Description]
    Parameters
    ----------
    CONFIG: yaml file
        Configuration file
    spike_train: numpy.ndarray
        [number of spikes, 2] Ground truth for training. First column is the spike time, second column is the spike id  
    chosen_templates: list
        List of chosen templates' id's  
    min_amp: float
        Minimum value allowed for the maximum absolute amplitude of the isolated spike on its main channel  
    nspikes: int
        Number of isolated spikes to generate. This is different from the total number of x_detect   
    data_folder: str
        Folder storing the standarized data (if not exist, run preprocess to automatically generate)     
    noise_ratio: int
        Ratio of number of noise to isolated spikes. For example, if n_isolated_spike=1000, noise_ratio=5, then n_noise=5000     
    collision_ratio: int
        Ratio of number of collisions to isolated spikes.    
    misalign_ratio: int
        Ratio of number of spatially and temporally misaligned spikes to isolated spikes 
    misalign_ratio2: int
        Ratio of number of only-spatially misaligned spikes to isolated spikes      
    multi: bool
        If multi= True, generate training data for multi-channel neural network. Otherwise generate single-channel data

    Returns
    -------
    x_detect: numpy.ndarray
        [number of detection training data, temporal length, number of channels] Training data for the detect net.
    y_detect: numpy.ndarray
        [number of detection training data] Label for x_detect

    x_triage: numpy.ndarray
        [number of triage training data, temporal length, number of channels] Training data for the triage net.
    y_triage: numpy.ndarray
        [number of triage training data] Label for x_triage


    x_ae: numpy.ndarray
        [number of ae training data, temporal length] Training data for the autoencoder: noisy spikes
    y_ae: numpy.ndarray
        [number of ae training data, temporal length] Denoised x_ae

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

    # get templates - making templates from spike train
    templates, _ = get_templates(
        spike_train, path_to_data, 4*CONFIG.spike_size)

    templates = np.transpose(templates, (2, 1, 0))

    logger.info('Got templates ndarray of shape: {}'.format(templates.shape))

    # choose good templates (good looking and big enough)
    templates = choose_templates(templates, chosen_templates)

    if templates.shape[0] == 0:
        raise ValueError("Coulndt find any good templates...")

    logger.info('Good looking templates of shape: {}'.format(templates.shape))

    # align and crop templates;
    # Cat: original size of tempalte is 4 x cross size; 
    #      need to make it smaller
    templates = crop_templates(templates, CONFIG.spike_size,
                               CONFIG.neigh_channels, CONFIG.geom)

    # determine noise covariance structure
    # Cat: spatial covariance matrix is 7 x 7
    # 2 matrices spatial and temporal
    spatial_SIG, temporal_SIG = noise_cov(path_to_data,
                                          PARAMS['dtype'],
                                          CONFIG.recordings.n_channels,
                                          PARAMS['data_order'],
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    # Cat: Make augmented data

    # make training data set
    K = templates.shape[0]
    R = CONFIG.spike_size
    amps = np.max(np.abs(templates), axis=1)  # Cat: max amplitude on main chan

    # make clean augmented spikes
    nk = int(np.ceil(nspikes/K))    # get average # spikes / templates
    max_amp = np.max(amps)*1.5
    nneigh = templates.shape[2]

    ################
    # clean spikes #
    ################
    # placeholder for clean spikes
    # generate augmented spikes for eacah tempalte
    # arange tempaltes from min to max amplitude
    # clean spikes, no noise
    x_clean = np.zeros((nk*K, templates.shape[1], templates.shape[2]))
    for k in range(K):
        tt = templates[k]
        amp_now = np.max(np.abs(tt))
        amps_range = (np.arange(nk)*(max_amp-min_amp)           #Uniformly generate augmented spike sizes
                      / nk+min_amp)[:, np.newaxis, np.newaxis]
        x_clean[k*nk:(k+1)*nk] = (tt/amp_now)[np.newaxis, :, :]*amps_range


    #############
    # collision #
    #############
    # collision_ratio: # of clean spikes to # of collision spikes
    x_collision = np.zeros(
        (x_clean.shape[0]*int(collision_ratio), templates.shape[1],
         templates.shape[2]))
    max_shift = 2*R

    # main spike kept as above (max chan is first)
    # collission spike is randomly tempporally and spatially shifted
    # have to ensure temporal shift is sufficient otherwise difficult to detect
    temporal_shifts = np.random.randint(
        max_shift*2, size=x_collision.shape[0]) - max_shift
    temporal_shifts[temporal_shifts < 0] = temporal_shifts[
        temporal_shifts < 0]-5
    temporal_shifts[temporal_shifts >= 0] = temporal_shifts[
        temporal_shifts >= 0]+6

    # make sure that secondary/collission spike is sufficiently large
    # otherwise tough to detect very small collissions
    amp_per_data = np.max(x_clean[:, :, 0], axis=1)
    
    #make collission data; candidate collission spike has to be > 0.3 clean spike
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

    ##############################################
    # temporally and spatially misaligned spikes #
    ##############################################
    
    # Cat: another negative case with just reference spikes that are 
    # misalgined, i.e. offcentre, or whose main channel is not first channel
    # train to discard these
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

    ###############################
    # spatially misaligned spikes #
    ###############################
    # Cat: another case where just spatially misaligning
    # need this case for triage nn; 
    # misalgined2 different than above
    if multi:
        x_misaligned2 = np.zeros(
            (x_clean.shape[0]*int(misalign_ratio2), templates.shape[1],
                templates.shape[2]))
        for j in range(x_misaligned2.shape[0]):
            x_misaligned2[j] = np.copy(x_clean[np.random.choice(
                x_clean.shape[0], 1, replace=True)][:, :, np.random.choice(
                    nneigh, nneigh, replace=False)])

    #########
    # noise #
    #########
    # get noise
    # Cat: generate gaussian with mean=0; covariance same as above;
    noise = np.random.normal(
        size=[x_clean.shape[0]*int(noise_ratio), templates.shape[1],
              templates.shape[2]])
    for c in range(noise.shape[2]):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)

        reshaped_noise = np.reshape(noise, (-1, noise.shape[2]))
    noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                       [noise.shape[0], x_clean.shape[1], x_clean.shape[2]])

    y_clean = np.ones((x_clean.shape[0]))
    y_col = np.ones((x_collision.shape[0]))  #for detection collissions are 1s
    y_misaligned = np.zeros((x_misaligned.shape[0]))
    if multi:
        y_misaligned2 = np.zeros((x_misaligned2.shape[0]))
    y_noise = np.zeros((noise.shape[0]))

    mid_point = int((x_clean.shape[1]-1)/2)

    # Cat: Label data 
    # get training set for detection
    # stacking all data together:  clean, collissions, misaligned, noise
    # everything gets noise addded
    if multi:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_collision +
            noise[np.random.choice(
                noise.shape[0], x_collision.shape[0], replace=False)],
            x_misaligned +
            noise[np.random.choice(
                noise.shape[0], x_misaligned.shape[0], replace=False)],
            noise
        ))

        x_detect = x[:, (mid_point-R):(mid_point+R+1), :]
        y_detect = np.concatenate((y_clean, y_col, y_misaligned, y_noise))
    else:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_misaligned +
            noise[np.random.choice(
                noise.shape[0], x_misaligned.shape[0], replace=False)],
            noise
        ))
        x_detect = x[:, (mid_point-R):(mid_point+R+1), 0]
        y_detect = np.concatenate((y_clean, y_misaligned, y_noise))

    # get training set for triage
    if multi:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_collision +
            noise[np.random.choice(
                noise.shape[0], x_collision.shape[0], replace=False)],
            x_misaligned2 +
            noise[np.random.choice(
                noise.shape[0], x_misaligned2.shape[0], replace=False)],
        ))
        x_triage = x[:, (mid_point-R):(mid_point+R+1), :]
        y_triage = np.concatenate(
            (y_clean, np.zeros((x_collision.shape[0])), y_misaligned2))
    else:
        x = np.concatenate((
            x_clean +
            noise[np.random.choice(
                noise.shape[0], x_clean.shape[0], replace=False)],
            x_collision +
            noise[np.random.choice(
                noise.shape[0], x_collision.shape[0], replace=False)],
        ))
        x_triage = x[:, (mid_point-R):(mid_point+R+1), 0]
        y_triage = np.concatenate((y_clean, np.zeros((x_collision.shape[0]))))

    # ge training set for auto encoder
    # AE is single channel data
    ae_shift_max = 1
    temporal_shifts_ae = np.random.randint(
        ae_shift_max*2+1, size=x_clean.shape[0]) - ae_shift_max
    y_ae = np.zeros((x_clean.shape[0], 2*R+1))   # denoised version
    x_ae = np.zeros((x_clean.shape[0], 2*R+1))   # noise version
    
    # training goal: get matrix that tries to reduce noise version to 
    # denoised version - which is just the template...
    for j in range(x_ae.shape[0]):
        y_ae[j] = x_clean[j, (mid_point-R+temporal_shifts_ae[j]):
                          (mid_point+R+1+temporal_shifts_ae[j]), 0]
        x_ae[j] = x_clean[j, (mid_point-R+temporal_shifts_ae[j]):
                          (mid_point+R+1+temporal_shifts_ae[j]), 0]+noise[
            j, (mid_point-R+temporal_shifts_ae[j]):
            (mid_point+R+1+temporal_shifts_ae[j]), 0]

    return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae

