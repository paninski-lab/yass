import numpy as np
import scipy
import os
from scipy.stats import t
from yass.template import align_get_shifts_with_ref, shift_chans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    buff = 2
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


def get_l2_features(reader_residual, spike_train,
                    templates, soft_assignment,
                    unit1, unit2, n_samples=2000):

    spike_size = templates.shape[1]

    idx1 = spike_train[:, 1] == unit1
    spt1 = spike_train[idx1, 0]
    prob1 = soft_assignment[idx1]

    idx2 = spike_train[:, 1] == unit2
    spt2 = spike_train[idx2, 0]
    prob2 = soft_assignment[idx2]

    if (np.sum(prob1) < 5) or (np.sum(prob2) < 5):
        return None, None

    if np.sum(prob1)+np.sum(prob2) > n_samples:
        ratio1 = np.sum(prob1)/float(np.sum(prob1)+np.sum(prob2))
        n_samples1 = np.min((int(n_samples*ratio1), int(np.sum(prob1))))
        n_samples2 = n_samples - n_samples1

    else:
        n_samples1 = int(np.sum(prob1))
        n_samples2 = int(np.sum(prob2))

    spt1 = np.random.choice(spt1, n_samples1, replace=False, p=prob1/np.sum(prob1))
    spt2 = np.random.choice(spt2, n_samples2, replace=False, p=prob2/np.sum(prob2))

    wfs1 = reader_residual.read_waveforms(spt1, spike_size)[0] + templates[unit1]
    wfs2 = reader_residual.read_waveforms(spt2, spike_size)[0] + templates[unit2]

    # if two templates are not aligned, get bigger window
    # and cut oneside to get shifted waveforms
    mc = templates[[unit1, unit2]].ptp(1).max(0).argmax()
    shift = wfs2[:,:,mc].mean(0).argmin() - wfs1[:,:,mc].mean(0).argmin()
    if np.abs(shift) < spike_size//4:
        if shift < 0:
            wfs1 = wfs1[:, -shift:]
            wfs2 = wfs2[:, :shift]
        elif shift > 0:
            wfs1 = wfs1[:, :-shift]
            wfs2 = wfs2[:, shift:]

    # assignment
    spike_ids = np.hstack((np.zeros(len(wfs1), 'int32'),
        np.ones(len(wfs2), 'int32')))

    # recompute templates using deconvolved spikes
    template1 = np.mean(wfs1, axis=0, keepdims=True)
    template2 = np.mean(wfs2, axis=0, keepdims=True)
    l2_features = template_spike_dist_linear_align(
        np.concatenate((template1, template2), axis=0),
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

# get candidates from template space
def get_candidates(templates1, templates2, spike_train1, spike_train2):
    
    ptps1 = templates1.ptp(1)
    ptps2 = templates2.ptp(1)
    
    n_spikes1 = np.zeros(templates1.shape[0])
    a, b = np.unique(spike_train1[:, 1], return_counts=True)
    n_spikes1[a] = b
    n_spikes2 = np.zeros(templates2.shape[0])
    a, b = np.unique(spike_train2[:, 1], return_counts=True)
    n_spikes2[a] = b
    
    # mask out small ptp
    ptps1[ptps1 < 1] = 0
    ptps2[ptps2 < 1] = 0
    # distances of ptps
    dist_mat = np.sum(np.square(ptps1[:, None] - ptps2[None]), 2)
    
    # compute distance relative to the norm of ptps
    norms1 = np.square(np.linalg.norm(ptps1, axis=1))
    norms2 = np.square(np.linalg.norm(ptps2, axis=1))
    dist_norm_ratio = dist_mat / np.maximum(
        norms1[:, None], norms2[None])
    
    # units need to be close to each other
    idx1 = dist_norm_ratio < 0.5
    
    # ptp of both units need to be bigger than 1
    ptp_max1 = ptps1.max(1)
    ptp_max2 = ptps2.max(1)
    smaller_ptps = np.minimum(ptp_max1[:, None],
                              ptp_max2[None])
    idx2 = smaller_ptps > 1
    
    # expect to have at least 30 spikes
    smaller_n_spikes = np.minimum(n_spikes1[:, None],
                                  n_spikes2[None])
    idx3 = smaller_n_spikes > 30
    units_1, units_2 = np.where(np.logical_and(
        np.logical_and(idx1, idx2), idx3))
    
    pairs = np.vstack((units_1, units_2)).T
    
    return pairs

# match each spike from spt1 to spikes in spt2 (one to one matching)
def matching(spt1, spt2, threshold=5):
    match1 = np.zeros(len(spt1), 'bool')
    match2 = np.zeros(len(spt2), 'bool')
    for ii, s in enumerate(spt1):
        j = np.argmin(np.abs(spt2 - s))
        if np.abs(spt2 - s).min() <= threshold:
            match1[ii] = True
            match2[j]  = True
            
    return match1, match2

# Find best cos similarity between the two templates (from unit1 and unit2, rolling one of the templates until you find best alignment)
def best_match_(unit1, unit2, templates1, templates2, max_shift):
    
    data1 = templates1[unit1].T.ravel()
    data2 = templates2[unit2]
    best_result = -2
    for k in range(-max_shift, max_shift, 1):
        data2_unrolled = np.roll(data2, k, axis = 0).T.ravel()
        result = 1 - scipy.spatial.distance.cosine(data1, data2_unrolled)
        if result > best_result:
            best_result = result
    return best_result

# Find candidate pairs using the above two routines
def match_two_sorts(templates1, templates2, spike_train1, spike_train2, overlap_threshold=0.5):
    
    print('Template 1 Shape:', templates1.shape)
    print('Template 2 Shape:', templates2.shape)
    
    pairs = get_candidates(templates1,templates2, spike_train1, spike_train2)
    best_matches = np.zeros(pairs.shape[0])
        
    print('{} candidate pairs'.format(len(pairs)))
    
    
    
    with tqdm(total=len(pairs)) as pbar:
        for i, (unit1, unit2) in enumerate(pairs):
            best_matches[i] = best_match_(unit1, unit2, templates1, templates2, 12)
            pbar.update()
            
    for i, k1 in enumerate(np.unique(pairs[:,0])):
        idx = pairs[:, 0] == k1
        pairs[idx, 1] = pairs[idx][np.argsort(best_matches[idx])[::-1], 1]
        best_matches[idx] = np.sort(best_matches[idx])[::-1]

    pairs = pairs[best_matches > 0.80]
    matched = np.zeros(len(pairs), 'bool')
    best_matches = best_matches[best_matches > 0.80]
    matched_spikes = []
    missed_spikes1 = []
    missed_spikes2 = []
    
    
    with tqdm(total=np.unique(pairs[:,0]).size) as pbar:
        
        for i, k1 in enumerate(np.unique(pairs[:,0])):
            idx = np.where(pairs[:, 0] == k1)[0]
            spt1 = spike_train1[spike_train1[:, 1]==k1, 0]
            temp1 = templates1[k1]
            flag = False
            
            for j, k2 in enumerate(pairs[idx,1][:1]):
                
                temp2 = templates2[k2]
                spt2 = spike_train2[spike_train2[:, 1]==k2, 0]
                max_ptp = np.vstack((temp1.ptp(0), temp2.ptp(0))).max(0)
                mc = max_ptp.argmax()
                min_point1 = temp1[:, mc].argmin()
                min_point2 = temp2[:, mc].argmin()
                shift = min_point1 - min_point2 - templates1.shape[1]//2 + templates2.shape[1]//2
                spt1 += shift
                match1, match2 = matching(spt1, spt2)
                
                if np.mean(match1) > 0.90:
                    matched[idx[j]] = True
                    missed_spikes1, matched_spikes, missed_spikes2, flag = do_something(missed_spikes1, 
                                                                                  matched_spikes, 
                                                                                  missed_spikes2, 
                                                                                  spt1, 
                                                                                  spt2, 
                                                                                  match1, 
                                                                                  match2, flag)
                    break
                elif np.mean(match1) > overlap_threshold or match1.sum()/ spt2.size > overlap_threshold:
                    matched[idx[j]] = True
                    missed_spikes1, matched_spikes, missed_spikes2, flag = do_something(missed_spikes1, 
                                                                                  matched_spikes, 
                                                                                  missed_spikes2, 
                                                                                  spt1, 
                                                                                  spt2, 
                                                                                  match1, 
                                                                                  match2, flag)
                    
                    spt1 = spt1[~match1]
                
                spt1 -= shift
            pbar.update()
    matched_pairs = pairs[matched]

    sort1_matched = np.unique(pairs[matched, 0])
    sort2_matched = np.unique(pairs[matched, 1])

    sort1_only = np.arange(templates1.shape[0])
    sort1_only = sort1_only[~np.in1d(sort1_only, sort1_matched)]

    sort2_only = np.arange(templates2.shape[0])
    sort2_only = sort2_only[~np.in1d(sort2_only, sort2_matched)]


    return sort1_only, sort2_only, sort1_matched, sort2_matched, matched_pairs, matched_spikes, missed_spikes1, missed_spikes2


def do_something(ls10, ls11, ls01, spt1, spt2, match1, match2, flag):
    
    
    if not flag:
        ls11.append(spt1[match1])
        ls10.append(spt1[~match1])
        ls01.append(spt2[~match2])
    else:
        ls11[-1] = np.concatenate([ls11[-1], spt1[match1]], axis = 0)
        ls10[-1] = np.concatenate([ls10[-1], spt1[~match1]], axis = 0)
        ls01[-1] = np.concatenate([ls01[-1], spt2[~match2]], axis = 0)
    
    flag = True
    return ls10, ls11, ls01, flag



def plot_waveforms(misses1, misses2, matched, ax, mc, reader):
    
    idx = np.random.choice(matched.size, min(200, matched.size))
    wfs = reader.read_waveforms(matched)[0]
    ax.plot(wfs[:,:, mc].T, c = 'goldenrod', alpha = 0.05)
    ax.plot(wfs[:,:,mc].mean(0), c = 'goldenrod')
    temp = wfs.mean(0) 
    
    
    idx1 = np.random.choice(misses1.size, min(200, misses1.size))
    wfs1 = reader.read_waveforms(misses1 + 20)[0]
    
    if idx1.shape[0] > 20:
        
        ax.plot(wfs1[:,:, mc].T, c = 'r', alpha = 0.05)
        ax.plot(wfs1[:,:,mc].mean(0), c = 'r')
        temp1 = wfs1.mean(0)
    else:
        temp1 = np.roll(temp, -20, axis = 0)
        
    idx2 = np.random.choice(misses2.size, min(200, misses2.size))
    wfs2 = reader.read_waveforms(misses2 - 20)[0]
    
    if idx2.shape[0] > 20:
        ax.plot(wfs2[:,:, mc].T, c = 'b', alpha = 0.05)
        ax.plot(wfs2[:,:,mc].mean(0), c = 'b')
        temp2 = wfs2.mean(0)
    else:
        temp2 = np.roll(temp, 20, axis = 0)
    
    
    
    return temp1, temp2, temp

def plot_templates(temp, chunk, color, channels, ax, CONFIG, unit = None, mc = None, alpha = 1.0):
    if temp is not None:
        scale = 5
        for i in channels:
            ax.plot(CONFIG.geom[i, 0]*50 + np.arange(temp.shape[0]) * 10 + chunk * 100, 
                    CONFIG.geom[i,1]*2.5 + temp[:, i]*scale, c = color,lw = 1, alpha = alpha)
    ax.scatter(CONFIG.geom[channels,0]*50, CONFIG.geom[channels,1]*2.5, c = 'k', s = 50)

def isi(spt, template, ax, reader_resid, alpha):
    mc = template.ptp(0).argmax(0)
    spt = spt[np.logical_and(spt>61, spt< 18000000-61)]
    
    spt = np.unique(spt)
    isi = spt[1:]/30 - spt[:-1]/30
    min_loc = template[:,mc].argmin()
    max_loc = template[:,mc].argmax()
    wf, skipped_idx = reader_resid.read_waveforms(spt[1:])
    spt = np.delete(spt[1:], skipped_idx)
    isi = np.delete(isi, skipped_idx)
    wf = wf+template
    ptps = np.absolute(wf[:,max_loc, mc]-wf[:, min_loc,mc])
    ax.scatter(isi, ptps, alpha = alpha, c = 'b')
    ax.set_xlim([1.5, 250])
    ax.set_xscale('log')
    ax.plot([1.5, 250],[template.ptp(0).max(0), template.ptp(0).max(0)])
    ax.set_xlim([0,250])
    ax.set_xlabel('log ISI (ms)')
    ax.set_ylabel('PTP (SU)')
    
    
def ptp_all_chans(spt, wfs, template, unit, other_templates, other_units, mc, channels, grid, row_orig, col_orig, CONFIG, alpha = 0.5):
    
    chan_argsort = np.argsort(template[:,channels].ptp(0))[::-1]
    min_loc = template.argmin(0)
    max_loc = template.argmax(0)
    wfs = wfs + template
    col = col_orig
    row = row_orig
    ctr = 0
    max_ptp = -1
    spt = np.sort(spt)
    for chan in channels[chan_argsort]:
        ax = plt.subplot(grid[row, col])
        ptps = np.absolute(wfs[:,max_loc[chan], chan]-wfs[:, min_loc[chan], chan])
        ax.scatter(spt//30000/60, ptps, alpha = alpha, c = 'k')
        
        ax.plot(spt//30000/60, ptps.cumsum()/(np.arange(ptps.size)+1), 'red', lw = 2)
        if other_units is not None:
            for i, unit2 in enumerate(other_units):
                ax.plot([0, 10], [other_templates[:, chan, unit2].ptp(0), other_templates[:, chan, unit2].ptp(0)], colors[i+3], lw = 2)
        if max_ptp < ptps.mean(0):
            max_ptp = ptps.mean(0)
        ax.set_ylim([0, max_ptp +5])
        ax.set_xlim([0, 10])
        ax.set_xlabel('time')
        ax.set_ylabel('PTP (SU)')
        ax.set_title(str(unit) + ' ' + str(chan)) 
        ctr += 1
        col = col_orig + ctr % 2
        row = row_orig + ctr // 2
        if ctr == 10:
            break
            
def featurize(wfs, mc, CONFIG):
    n_data, n_times, n_chans = wfs.shape
    channels = np.where(CONFIG.neigh_channels[mc])[0]
#     denoiser = denoiser.to(CONFIG.torch_devices[0])
    wfs_copy = wfs.copy()
    best_shifts = align_get_shifts_with_ref(
            wfs_copy[:, :, mc])
    wfs_copy = shift_chans(wfs_copy, best_shifts)
    pca = PCA(n_components= 5)
    try:
        feat = pca.fit_transform(wfs_copy[:,:,channels].reshape([n_data, -1]))
    except:
        feat = None
    return feat


def get_sort(mc1, mc2, temp_templates1, temp_templates2, templates1, templates2):
    other_units = []
    comp_=np.zeros([temp_templates1.shape[0], temp_templates1.shape[0]]) + 1000
    for unit1 in tqdm(range(temp_templates1.shape[0])):
        channels = np.where(CONFIG.neigh_channels[mc1[unit1]])[0]
        unit2s = np.where(np.logical_and(np.in1d(mc2, channels), mc2 != mc1[unit1]))[0]
        channels_to_compare = np.where(templates1[unit1].ptp(0) > 5.0)[0]
        mc = mc1[unit1]
        for k in range(chunks):
            best_shifts = align_get_shifts_with_ref(
                    np.concatenate([temp_templates2[unit2s,:,mc,k], temp_templates1[unit1,:,mc,k][np.newaxis]], axis = 0))
            temp_templates2[unit2s,:,:,k] = shift_chans(temp_templates2[unit2s,:,:,k], best_shifts[:-1])
            temp_templates1[unit1, :, :, k] = shift_chans(temp_templates1[unit1:unit1+1,:,:,k], best_shifts[-1:])[0]
        try:
            comp = np.abs(temp_templates2[unit2s][:, :, channels_to_compare] - 
                          temp_templates1[unit1][:, channels_to_compare]).max(1).max(1).min(-1)
        except:
            continue
        comp_[unit2s, unit1] = comp/np.abs(templates1[unit1]).max(0).max(0)

    other_units = np.where(comp_.T < 0.2)
    return other_units

def get_ptp_firing_rates(fname_templates, fname_spike_train, reader):
    
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)

    ptps = templates.ptp(1).max(1)
    n_spikes = np.zeros(templates.shape[0])

    a, b = np.unique(spike_train[:, 1], return_counts=True)
    n_spikes[a] = b

    recording_length = reader.rec_len/reader.sampling_rate
    f_rates = n_spikes/recording_length

    return ptps, f_rates
        
            

