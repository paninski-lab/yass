import numpy as np
import scipy
import os
from scipy.stats import t
from yass.template import align_get_shifts_with_ref, shift_chans


def connected_components(img, x, y, cc):
    pixel = [x, y]
    if (pixel in cc) or (x < 0) or (x >= img.shape[0]) or (y < 0) or (y >= img.shape[1]) or (img[x,y] == 0):
        return cc
    else:
        cc.append(pixel)
        cc = connected_components(img, x-1, y, cc)
        cc = connected_components(img, x+1, y, cc)
        cc = connected_components(img, x, y-1, cc)
        cc = connected_components(img, x, y+1, cc)
        
        return cc

def zoom_in_window(sta, unit, th):

    img = sta[unit]

    #th = np.std(sta)*1.5
    sign = np.sign(img.ravel()[np.abs(img).argmax()])
    th = th*sign

    img_2 = np.copy(img)
    if sign == 1:
        img_2[img_2 < th] = 0
    elif sign == -1:
        img_2[img_2 > th] = 0

    img_2 = np.abs(img_2)
    x_start, y_start = np.where(img_2 == np.max(img_2))
    x_start, y_start = x_start[0], y_start[0]
    cc = []

    cc = connected_components(img_2, x_start, y_start, cc)
    buff = 5
    if len(cc) > 0:
        cc = np.vstack(cc)
        windows = [[np.min(cc[:,0])-buff, np.max(cc[:,0])+buff],
                   [np.min(cc[:,1])-buff, np.max(cc[:,1])+buff]]
        if windows[0][0] < 0:
            windows[0][0] = 0
        if windows[0][1] >= img.shape[0]:
            windows[0][1] = img.shape[0]-1
        if windows[1][0] < 0:
            windows[1][0] = 0
        if windows[1][1] >= img.shape[1]:
            windows[1][1] = img.shape[1]-1   
    else:
        windows = None
    
    return windows

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

    # get ref template
    max_idx = templates.ptp(1).max(1).argmax(0)
    ref_template = templates[max_idx]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]

    # align templates on max channel
    best_shifts = align_get_shifts_with_ref(
        templates[:, :, max_chan],
        ref_template, nshifts = 7)
    templates = shift_chans(templates, best_shifts)

    # align all spikes on max channel
    best_shifts = align_get_shifts_with_ref(
        spikes[:,:,max_chan],
        ref_template, nshifts = 7)
    spikes = shift_chans(spikes, best_shifts)
    
    # if shifted, cut shifted parts
    # because it is contaminated by roll function
    cut = int(np.ceil(np.max(np.abs(best_shifts))))
    if cut > 0:
        templates = templates[:,cut:-cut]
        spikes = spikes[:,cut:-cut]

    # get visible channels
    vis_ptp = np.min((vis_ptp, np.max(templates.ptp(1))))
    vis_chan = np.where(templates.ptp(1).max(0) >= vis_ptp)[0]
    templates = templates[:, :, vis_chan].reshape(
        templates.shape[0], -1)
    spikes = spikes[:, :, vis_chan].reshape(
        spikes.shape[0], -1)

    # get a subset of locations with maximal difference
    if templates.shape[0] == 1:
        # single unit: all timepoints
        idx = np.arange(templates.shape[1])
    elif templates.shape[0] == 2:
        # two units:
        # get difference
        diffs = np.abs(np.diff(templates, axis=0)[0])
        # points with large difference
        idx = np.where(diffs > 1.5)[0]
        min_diff_points = 5
        if len(idx) < 5:
            idx = np.argsort(diffs)[-min_diff_points:]
    else:
        # more than two units:
        # diff is mean diff to the largest unit
        diffs = np.mean(np.abs(
            templates-templates[max_idx][None]), axis=0)
        
        idx = np.where(diffs > 1.5)[0]
        min_diff_points = 5
        if len(idx) < 5:
            idx = np.argsort(diffs)[-min_diff_points:]
                       
    templates = templates[:, idx]
    spikes = spikes[:, idx]

    dist = scipy.spatial.distance.cdist(templates, spikes)

    return dist.T


def get_l2_features(reader_residual, spike_train, templates,
                    unit1, unit2, n_samples=2000):

    n_units, n_times, n_channels = templates.shape

    # get spike times
    spt1 = spike_train[spike_train[:, 1] == unit1, 0]
    spt2 = spike_train[spike_train[:, 1] == unit2, 0]
    
    if (len(spt1) < 10) or (len(spt2) < 10):
        return None, None

    # templates
    template1 = templates[unit1]
    template2 = templates[unit2]

    # subsample
    if len(spt1) + len(spt2) > n_samples:
        ratio = len(spt1)/float(len(spt1)+len(spt2))

        n_samples1 = int(n_samples*ratio)

        # at least one sample per grounp
        if n_samples1 == n_samples:
            n_samples1 = n_samples - 1
        elif n_samples1 == 0:
            n_samples1 = 1

        n_samples2 = n_samples - n_samples1

        spt1_idx = np.random.choice(
            np.arange(len(spt1)),
            n_samples1, False)
        spt2_idx = np.random.choice(
            np.arange(len(spt2)),
            n_samples2, False)

    else:
        spt1_idx = np.arange(len(spt1))
        spt2_idx = np.arange(len(spt2))

    spt1 = spt1[spt1_idx]
    spt2 = spt2[spt2_idx]

    # find shifts
    temps = np.concatenate((template1[None], template2[None]),
                           axis=0)
    mc = temps.ptp(1).max(0).argmax()
    shift = np.diff(temps[:, :, mc].argmin(1))[0]

    # residuals
    wf_res = reader_residual.read_waveforms(spt1)[0]
    wfs1 = wf_res + template1

    wf_res = reader_residual.read_waveforms(spt2)[0]
    wfs2 = wf_res + template2

    # if two templates are not aligned, get bigger window
    # and cut oneside to get shifted waveforms
    if shift < 0:
        wfs1 = wfs1[:, -shift:]
        wfs2 = wfs2[:, :shift]
    elif shift > 0:
        wfs1 = wfs1[:, :-shift]
        wfs2 = wfs2[:, shift:]

    # assignment
    spike_ids = np.append(
        np.zeros(len(wfs1), 'int32'),
        np.ones(len(wfs2), 'int32'),
        axis=0)

    # recompute templates using deconvolved spikes
    template1 = np.median(wfs1, axis=0)
    template2 = np.median(wfs2, axis=0)
    l2_features = template_spike_dist_linear_align(
        np.concatenate((template1[None], template2[None]), axis=0),
        np.concatenate((wfs1, wfs2), axis=0))
    
    return l2_features, spike_ids

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
