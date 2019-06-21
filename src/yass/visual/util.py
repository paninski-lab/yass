import numpy as np
import scipy
import os
from scipy.stats import t
from yass.template import align_get_shifts_with_ref, shift_chans


def two_templates_dist_linear_align(templates1, templates2, max_shift=5, step=0.5):

    K1, R, C = templates1.shape
    K2 = templates2.shape[0]

    shifts = np.arange(-max_shift,max_shift+step,step)
    ptps1 = templates1.ptp(1)
    max_chans1 = np.argmax(ptps1, 1)
    ptps2 = templates2.ptp(1)
    max_chans2 = np.argmax(ptps2, 1)

    shifted_templates = np.zeros((len(shifts), K2, R, C))
    for ii, s in enumerate(shifts):
        shifted_templates[ii] = shift_chans(templates2, np.ones(K2)*s)

    distance = np.ones((K1, K2))*1e4
    for k in range(K1):
        candidates = np.abs(ptps2[:, max_chans1[k]] - ptps1[k, max_chans1[k]])/ptps1[k,max_chans1[k]] < 0.5
        
        dist = np.min(np.sum(np.square(
            templates1[k][np.newaxis, np.newaxis] - shifted_templates[:, candidates]),
                             axis=(2,3)), 0)
        dist = np.sqrt(dist)
        distance[k, candidates] = dist
        
    return distance
    
def compute_neighbours2(templates1, templates2, n_neighbours=3):
    
    dist = two_templates_dist_linear_align(templates1, templates2)

    nearest_units1 = np.zeros((dist.shape[0], n_neighbours), 'int32')
    for k in range(dist.shape[0]):
        nearest_units1[k] = np.argsort(dist[k])[:n_neighbours]
        
    nearest_units2 = np.zeros((dist.shape[1], n_neighbours), 'int32')
    for k in range(dist.shape[1]):
        nearest_units2[k] = np.argsort(dist[:, k])[:n_neighbours]
        
    return nearest_units1, nearest_units2

def compute_neighbours_rf2(STAs1, STAs2, n_neighbours=3):
    STAs_th1 = pre_process_sta(STAs1)
    STAs_th2 = pre_process_sta(STAs2)

    norms1 = np.linalg.norm(STAs_th1.T, axis=0)[:, np.newaxis]
    norms2 = np.linalg.norm(STAs_th2.T, axis=0)[:, np.newaxis]
    cos = np.matmul(STAs_th1, STAs_th2.T)/np.matmul(norms1, norms2.T)
    cos[np.isnan(cos)] = 0

    nearest_units_rf1 = np.zeros((cos.shape[0], n_neighbours), 'int32')
    for k in range(cos.shape[0]):
        nearest_units_rf1[k] = np.argsort(cos[k])[-n_neighbours:][::-1]
        
    nearest_units_rf2 = np.zeros((cos.shape[1], n_neighbours), 'int32')
    for k in range(cos.shape[1]):
        nearest_units_rf2[k] = np.argsort(cos[:, k])[-n_neighbours:][::-1]

    return nearest_units_rf1, nearest_units_rf2
    
def pre_process_sta(STAs, th=0.002):

    STAs_th = np.copy(STAs)
    STAs_th[np.abs(STAs_th) < th] = 0
    STAs_th = STAs_th.reshape(STAs.shape[0], -1)

    return STAs_th

def combine_two_spike_train(templates1, templates2, spike_train1, spike_train2):
    
    K1 = templates1.shape[2]
    K2 = templates2.shape[2]
    
    templates = np.concatenate((templates1, templates2), axis=2)
    

    spike_train2_new_id = np.copy(spike_train2)    
    new_id2 = np.arange(K2) + K1
    for j in range(spike_train2.shape[0]):
        spike_train2_new_id[j,1] = new_id2[spike_train2[j,1]]
    
    spike_train = np.concatenate((spike_train1, spike_train2_new_id))
    spike_train = spike_train[np.argsort(spike_train[:,0])]
    
    return templates, spike_train


def combine_two_rf(STAs1, STAs2, Gaussian_params1, Gaussian_params2):

    STAs = np.concatenate((STAs1, STAs2), axis=0)
    Gaussian_params = np.concatenate((Gaussian_params1, Gaussian_params2), axis=0)
    
    return STAs, Gaussian_params

def template_spike_dist_linear_align(templates, spikes, vis_ptp=2.):
    """compares the templates and spikes.
    parameters:
    -----------
    templates: numpy.array shape (K, T, C)
    spikes: numpy.array shape (M, T, C)
    jitter: int
        Align jitter amount between the templates and the spikes.
    upsample int
        Upsample rate of the templates and spikes.
    """

    #print ("templates: ", templates.shape)
    #print ("spikes: ", spikes.shape)
    # new way using alignment only on max channel
    # maek reference template based on templates
    max_idx = templates.ptp(1).max(1).argmax(0)
    ref_template = templates[max_idx]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]

    # stack template max chan waveforms only
    #max_chans = templates.ptp(2).argmax(1)
    #temps = []
    #for k in range(max_chans.shape[0]):
    #    temps.append(templates[k,max_chans[k]])
    #temps = np.vstack(temps)
    #print ("tempsl stacked: ", temps.shape)
    
    #upsample_factor=5
    best_shifts = align_get_shifts_with_ref(
        templates[:, :, max_chan],
        ref_template, nshifts = 21)
    #print (" best shifts: ", best_shifts.shape)
    templates = shift_chans(templates, best_shifts)
    #print ("  new aligned templates: ", templates_aligned.shape)

    # find spike shifts
    #max_chans = spikes.ptp(2).argmax(1)
    #print ("max chans: ", max_chans.shape)
    #spikes_aligned = []
    #for k in range(max_chans.shape[0]):
    #    spikes_aligned.append(spikes[k,max_chans[k]])
    #spikes_aligned = np.vstack(spikes_aligned)
    #print ("spikes aligned max chan: ", spikes_aligned.shape)
    best_shifts = align_get_shifts_with_ref(
        spikes[:,:,max_chan], ref_template, nshifts = 21)
    spikes = shift_chans(spikes, best_shifts)
    #print ("  new aligned spikes: ", spikes_aligned.shape)
    
    n_unit = templates.shape[0]
    n_spikes = spikes.shape[0]

    vis_chan = np.where(templates.ptp(1).max(0) >= vis_ptp)[0]
    templates = templates[:, :, vis_chan].reshape(n_unit, -1)
    spikes = spikes[:, :, vis_chan].reshape(n_spikes, -1)

    if templates.shape[0] == 1:
        idx = np.arange(templates.shape[1])
    elif templates.shape[0] == 2:
        diffs = np.abs(np.diff(templates, axis=0)[0])
        
        idx = np.where(diffs > 1.5)[0]
        min_diff_points = 5
        if len(idx) < 5:
            idx = np.argsort(diffs)[-min_diff_points:]
    else:
        diffs = np.mean(np.abs(
            templates-templates[max_idx][None]), axis=0)
        
        idx = np.where(diffs > 1.5)[0]
        min_diff_points = 5
        if len(idx) < 5:
            idx = np.argsort(diffs)[-min_diff_points:]
                       
    templates = templates[:, idx]
    spikes = spikes[:, idx]

    dist = scipy.spatial.distance.cdist(templates, spikes)

    return dist

def get_l2_features(filename_residual, spike_train, spike_train_upsampled,
                    templates, templates_upsampled, unit1, unit2, n_samples=2000):
    
    _, spike_size, n_channels = templates.shape
    
    # get n_sample of cleaned spikes per template.
    spt1_idx = np.where(spike_train[:, 1] == unit1)[0]
    spt2_idx = np.where(spike_train[:, 1] == unit2)[0]
    
    # subsample
    if len(spt1_idx) + len(spt2_idx) > n_samples:
        ratio = len(spt1_idx)/(len(spt1_idx)+len(spt2_idx))
        
        n_samples1 = int(n_samples*ratio)
        n_samples2 = n_samples - n_samples1

        spt1_idx = np.random.choice(
            spt1_idx, n_samples1, False)
        spt2_idx = np.random.choice(
            spt2_idx, n_samples2, False)
    
    spt1 = spike_train[spt1_idx, 0] - spike_size//2
    spt2 = spike_train[spt2_idx, 0] - spike_size//2

    units1 = spike_train_upsampled[spt1_idx, 1]
    units2 = spike_train_upsampled[spt2_idx, 1]

    spikes_1, _ = read_spikes(
        filename_residual, spt1, n_channels, spike_size,
        units1, templates_upsampled, residual_flag=True)
    spikes_2, _ = read_spikes(
        filename_residual, spt2, n_channels, spike_size,
        units2, templates_upsampled, residual_flag=True)

    spike_ids = np.append(
        np.zeros(len(spikes_1), 'int32'),
        np.ones(len(spikes_2), 'int32'),
        axis=0)
    
    l2_features = template_spike_dist_linear_align(
        templates[[unit1, unit2], :, :],
        np.append(spikes_1, spikes_2, axis=0))
    
    return l2_features.T, spike_ids

def binary_reader_waveforms(standardized_filename, n_channels, n_times, spikes, channels=None):

    # ***** LOAD RAW RECORDING *****
    if channels is None:
        wfs = np.zeros((spikes.shape[0], n_times, n_channels), 'float32')
        channels = np.arange(n_channels)
    else:
        wfs = np.zeros((spikes.shape[0], n_times, channels.shape[0]), 'float32')

    skipped_idx = []
    with open(standardized_filename, "rb") as fin:
        ctr_wfs=0
        ctr_skipped=0
        for spike in spikes:
            # index into binary file: time steps * 4  4byte floats * n_channels
            try:
                fin.seek(spike * 4 * n_channels, os.SEEK_SET)
                wfs[ctr_wfs] = np.fromfile(
                    fin,
                    dtype='float32',
                    count=(n_times * n_channels)).reshape(
                                            n_times, n_channels)[:,channels]
                ctr_wfs+=1
            except:
                # skip loading of spike and decrease wfs array size by 1
                # print ("  spike to close to end, skipping and deleting array")
                wfs=np.delete(wfs, wfs.shape[0]-1,axis=0)
                skipped_idx.append(ctr_skipped)

            ctr_skipped+=1
    fin.close()

    return wfs, skipped_idx

def read_spikes(filename, spikes, n_channels, spike_size, units=None, templates=None,
                channels=None, residual_flag=False):
    ''' Function to read spikes from raw binaries
        
        filename: name of raw binary to be loaded
        spikes:  [times,] array holding all spike times
        units: [times,] unit id of each spike
        templates:  [n_templates, n_times, n_chans] array holding all templates
    '''
        
    # always load all channels and then index into subset otherwise
    # order won't be correct
    #n_channels = CONFIG.recordings.n_channels

    # load default spike_size unless otherwise inidcated
    # PETER: turned off. Let me know if you need this..
    #if spike_size==None:
    #    spike_size = int(CONFIG.recordings.spike_size_ms*CONFIG.recordings.sampling_rate//1000*2+1)

    if channels is None:
        channels = np.arange(n_channels)

    spike_waveforms, skipped_idx = binary_reader_waveforms(filename,
                                             n_channels,
                                             spike_size,
                                             spikes, #- spike_size//2,  # can use this for centering
                                             channels)

    if len(skipped_idx) > 0:
        units = np.delete(units, skipped_idx)

    # if loading residual need to add template back into 
    # Cat: TODO: this is bit messy; loading extrawide noise, but only adding
    #           narrower templates
    if residual_flag:
        #if spike_size is None:
        #    spike_waveforms+=templates[:,:,channels][units]
        # need to add templates in middle of noise wfs which are wider
        #else:
        #    spike_size_default = int(CONFIG.recordings.spike_size_ms*
        #                              CONFIG.recordings.sampling_rate//1000*2+1)
        #    offset = spike_size - spike_size_default
        #    spike_waveforms[:,offset//2:offset//2+spike_size_default]+=templates[:,:,channels][units]
        
        #print ("templates added: ", templates.shape)
        #print ("units: ", units)
        offset = spike_size - templates.shape[1]
        spike_waveforms[:,offset//2:offset//2+templates.shape[1]]+=templates[:,:,channels][units]

    return spike_waveforms, skipped_idx
