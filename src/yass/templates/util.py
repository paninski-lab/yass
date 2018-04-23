"""
[Description of file content]
"""
import numpy as np
from scipy import sparse
import logging
import parmap
import os

from yass.batch import BatchProcessor

logger = logging.getLogger(__name__)


# TODO: remove this function and use the explorer directly
def get_templates(spike_train, path_to_recordings,
                  max_memory, spike_size, n_max=5000):

    logger.info('Computing templates...')

    # number of templates
    n_templates = int(np.max(spike_train[:, 1]) + 1)
    spike_train_small = random_sample_spike_train(spike_train, n_max)

    # read recording
    bp = BatchProcessor(
        path_to_recordings, max_memory=max_memory, buffer_size=spike_size)

    # run nn preprocess batch-wsie
    mc = bp.multi_channel_apply
    res = mc(
        compute_weighted_templates,
        mode='memory',
        pass_batch_info=True,
        pass_batch_results=True,
        spike_train=spike_train_small,
        spike_size=spike_size,
        n_templates=n_templates)

    templates = res[0]
    weights = res[1]
    weights[weights == 0] = 1
    templates = templates / weights[np.newaxis, np.newaxis, :]

    return templates, weights
    
    

# TODO: remove this function and use the explorer directly
def get_templates_parallel(spike_train, path_to_recordings,
                  CONFIG, n_max=5000):
                  #CONFIG, n_max=100):

    spike_size =  2 * (CONFIG.spike_size + CONFIG.templates.max_shift) 
    max_memory = CONFIG.resources.max_memory
    n_processors = CONFIG.resources.n_processors
    n_channels = CONFIG.recordings.n_channels  
    sampling_rate = CONFIG.recordings.sampling_rate
    #n_sec_chunk = CONFIG.resources.n_sec_chunk
    n_sec_chunk = 100

    # number of templates
    #print("...subsampling templates")
    n_templates = int(np.max(spike_train[:, 1]) + 1)
    spike_train_small = random_sample_spike_train(spike_train, n_max, CONFIG)

    # determine length of processing chunk based on lenght of rec
    standardized_filename = CONFIG.data.root_folder+ '/tmp/standarized.bin'
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]
    
    buffer_size = 200

    indexes = np.arange(0,fp_len/n_channels,sampling_rate*n_sec_chunk)
    if indexes[-1] != fp_len/n_channels:
        indexes = np.hstack((indexes, fp_len/n_channels))

    idx_list = []
    for k in range(len(indexes)-1):
        idx_list.append([indexes[k],indexes[k+1],buffer_size, indexes[k+1]-indexes[k]+buffer_size])

    idx_list = np.int64(np.vstack(idx_list))

    proc_indexes = np.arange(len(idx_list))
    
    print("...computing templates (fixed chunk size - 100sec for efficiency and memory)")
    if CONFIG.resources.multi_processing:
        res = parmap.map(compute_weighted_templates_parallel, zip(idx_list,proc_indexes), spike_train_small, spike_size, n_templates, n_channels, buffer_size, standardized_filename, processes=n_processors, pm_pbar=True)
    else:
        res=[]
        for k in range(len(idx_list)):
            #print "chunk: ", k
            temp = compute_weighted_templates_parallel([idx_list[k],k], spike_train_small, spike_size, n_templates, n_channels, buffer_size, standardized_filename)
            res.append(temp)
    
    #Reconstruct templates from parallel proecessing
    print("... reconstructing templates")
    res0=np.zeros(res[0][0].shape)
    res1=np.zeros(res[0][1].shape)
    for k in range(len(res)):
        res0+=res[k][0]
        res1+=res[k][1]
    
    print("... dividing templates by weights")
    templates = res0
    weights = res1
    weights[weights == 0] = 1
    templates = templates/weights[np.newaxis, np.newaxis, :]
    
    return templates, weights


def compute_weighted_templates(recording, idx_local, idx, previous_batch,
                               spike_train, spike_size, n_templates):

    n_channels = recording.shape[1]

    # batch info
    data_start = idx[0].start
    data_end = idx[0].stop

    # get offset that will be applied
    offset = idx_local[0].start

    # shift location of spikes according to the batch info
    spike_time = spike_train[:, 0]
    spike_train = spike_train[np.logical_and(spike_time >= data_start,
                                             spike_time < data_end)]
    spike_train[:, 0] = spike_train[:, 0] - data_start + offset

    # calculate weight templates
    weighted_templates = np.zeros(
        (n_templates, 2 * spike_size + 1, n_channels), dtype=np.float32)
    weights = np.zeros(n_templates)

    for k in range(n_templates):
        spt = spike_train[spike_train[:, 1] == k]
        n_spikes = spt.shape[0]
        if n_spikes > 0:
            weighted_templates[k] = np.average(
                recording[spt[:, [0]].astype('int32')
                          + np.arange(-spike_size, spike_size + 1)],
                axis=0,
                weights=spt[:, 2])
            weights[k] = np.sum(spt[:, 2])
            weighted_templates[k] *= weights[k]

    weighted_templates = np.transpose(weighted_templates, (2, 1, 0))

    if previous_batch is not None:
        weighted_templates += previous_batch[0]
        weights += previous_batch[1]

    return weighted_templates, weights


def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))

def compute_weighted_templates_parallel(data_in, spike_train, spike_size, 
                         n_templates, n_channels, buffer_size, 
                         standardized_filename):

    idx_list, chunk_idx = data_in[0], data_in[1]
    
    #prPurple("Processing chunk: "+str(chunk_idx))

    #New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]
    idx_local_end = idx_list[3]

    data_start = idx_start  #idx[0].start
    data_end = idx_stop    #idx[0].stop
    offset = idx_local   #idx_local[0].start
    

    #***** LOAD RAW RECORDING *****
    with open(standardized_filename, "rb") as fin:
        if data_start==0:
            # Seek position and read N bytes
            recordings_1D = np.fromfile(fin, dtype='float32', 
                                count=(data_end+buffer_size)*n_channels)
            recordings_1D = np.hstack((np.zeros(
                 buffer_size*n_channels,dtype='float32'),recordings_1D))
        else:
            fin.seek((data_start-buffer_size)*4*n_channels, os.SEEK_SET)         
            recordings_1D =  np.fromfile(fin, dtype='float32', 
                 count=((data_end-data_start+buffer_size*2)*n_channels))	

        if len(recordings_1D)!=(
                        (data_end-data_start+buffer_size*2)*n_channels):
            recordings_1D = np.hstack((recordings_1D,
                      np.zeros(buffer_size*n_channels,dtype='float32')))

    fin.close()

    #Convert to 2D array
    recording = recordings_1D.reshape(-1,n_channels)

    #Compute search time
    spike_time = spike_train[:, 0]
    spike_train = spike_train[np.logical_and(spike_time >= data_start,
                                             spike_time < data_end)]
    spike_train[:, 0] = spike_train[:, 0] - data_start + offset

    # calculate weight templates
    weighted_templates = np.zeros(
        (n_templates, 2 * spike_size + 1, n_channels), dtype=np.float32)
    weights = np.zeros(n_templates)

    for k in range(n_templates):
        spt = spike_train[spike_train[:, 1] == k]
        n_spikes = spt.shape[0]
        if n_spikes > 0:
            weighted_templates[k] = np.average(
                recording[spt[:, [0]].astype('int32')
                          + np.arange(-spike_size, spike_size + 1)],
                axis=0,
                weights=spt[:, 2])
            weights[k] = np.sum(spt[:, 2])
            weighted_templates[k] *= weights[k]

    weighted_templates = np.transpose(weighted_templates, (2, 1, 0))

    #print (weighted_templates.shape, weights.shape)

    return weighted_templates, weights


def random_sample_spike_train_parallel(data_in, chunk_len, n_templates, 
                                                      n_spikes_sampled):

    proc_idx = data_in[0]
    spike_train  = data_in[1]
    #print("Processing chunk: ", proc_idx)

    idx_keep = np.zeros(spike_train.shape[0], 'bool')

    for k in range(n_templates):
        #if k%100==0: print(k,'/', n_templates)
        idx_data = np.where(spike_train[:, 1] == k)[0]
        n_data = idx_data.shape[0]

        if n_data > n_spikes_sampled:
            idx_sample = np.random.choice(n_data,
                                          n_spikes_sampled,
                                          replace=False)
            idx_keep[idx_data[idx_sample]] = 1
        else:
            idx_keep[idx_data] = 1

    spike_train_small = spike_train[idx_keep]

    return spike_train_small
        
def random_sample_spike_train(spike_train, n_max, CONFIG):

    n_templates = int(np.max(spike_train[:, 1]) + 1)

    multi_processing = CONFIG.resources.multi_processing
    n_processors = CONFIG.resources.n_processors
    
    print("...subsampling templates...")
    if multi_processing: 
        
        #Split spikearray into 100 chunks;
        #Cat: TODO: split array based on mem and other config parameters 
        n_chunks = 100
        spike_train_chunks = np.array_split(spike_train,n_chunks)  

        #Compute chunk len to offset incoming indexes
        chunk_len = len(spike_train[0])

        #Compute processor index
        proc_idx = np.arange(len(spike_train_chunks))

        # No. of spikes in each chunk of data
        n_spikes_sampled = n_max/n_chunks
        
        res = parmap.map(random_sample_spike_train_parallel, 
                         zip(proc_idx,spike_train_chunks), chunk_len, 
                         n_templates, n_spikes_sampled, 
                         processes=n_processors, pm_pbar=True)

        spike_train_small = np.vstack(res)
       
    else: 
        idx_keep = np.zeros(spike_train.shape[0], 'bool')

        for k in range(n_templates):
            if k%100==0: print(k,'/', n_templates)
            idx_data = np.where(spike_train[:, 1] == k)[0]
            n_data = idx_data.shape[0]

            if n_data > n_max:
                idx_sample = np.random.choice(n_data,
                                              n_max,
                                              replace=False)
                idx_keep[idx_data[idx_sample]] = 1
            else:
                idx_keep[idx_data] = 1

        spike_train_small = spike_train[idx_keep]

    return spike_train_small


def align_templates(templates, spike_train, max_shift):
    C, R, K = templates.shape
    spike_size = int((R-1)/2 - max_shift)

    # get main channel for each template
    mainc = np.argmax(
        np.max(templates[:, max_shift:(
            max_shift+2*spike_size+1)], axis=1), axis=0)

    # get templates on their main channel only
    templates_mainc = np.zeros((R, K))
    for k in range(K):
        templates_mainc[:, k] = templates[mainc[k], :, k]

    # reference template
    biggest_template_k = np.argmax(np.max(templates_mainc, axis=0))
    biggest_template = templates_mainc[
        max_shift:(max_shift + 2*spike_size+1), biggest_template_k]

    # find best shift
    fit_per_shift = np.zeros((2*max_shift+1, K))
    for s in range(2*max_shift+1):
        fit_per_shift[s] = np.matmul(
            biggest_template[
                np.newaxis, :], templates_mainc[s:(s+2*spike_size+1)])
    best_shift = np.argmax(fit_per_shift, axis=0)

    templates_final = np.zeros((C, 2*spike_size+1, K))
    for k in range(K):
        s = best_shift[k]
        templates_final[:, :, k] = templates[:, s:(s+2*spike_size+1), k]
        spike_train[spike_train[:, 1] == k, 0] += (s - max_shift)

    return templates_final, spike_train


# TODO: documentation
# TODO: comment code, it's not clear what it does
def merge_templates(templates, weights, spike_train, neighbors,
                    template_max_shift, t_merge_th):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    C, R, K = templates.shape
    th = t_merge_th
    W = template_max_shift

    energy = np.ptp(templates, 1)
    visible_channels = energy > 0.5
    main_channels = np.argmax(energy, 0)

    sparseConnection = sparse.lil_matrix((K, K), dtype='bool')
    for k1 in range(K):
        for k2 in range(k1, K):
            if neighbors[main_channels[k1], main_channels[k2]]:
                ch_idx = np.logical_or(visible_channels[:, k1],
                                       visible_channels[:, k2])
                t1 = templates[ch_idx, :, k1]
                t2 = templates[ch_idx, :, k2]
                if TemplatesSimilarity(t1, t2, th, W):
                    sparseConnection[k1, k2] = 1
                    sparseConnection[k2, k1] = 1

    edges = {x: sparse.find(sparseConnection[x])[1] for x in range(K)}

    groups = list()
    for scc in strongly_connected_components_iterative(np.arange(K), edges):
        groups.append(np.array(list(scc)))

    Knew = len(groups)
    templatesNew = np.zeros((C, R, Knew))
    weightNew = np.zeros(Knew)
    spt_new = np.zeros(spike_train.shape[0], 'int32')
    id_new = np.zeros(spike_train.shape[0], 'int32')
    for k in range(Knew):
        temp = groups[k]
        templatesNew_temp = np.zeros((C, R, temp.shape[0]))
        weight_temp = np.zeros(temp.shape[0])
        if temp.shape[0] > 1:
            ch_idx = np.unique(main_channels[temp])
            shift_temp = determine_shift(
                templates[[ch_idx]][:, :, temp], W)
            for j2 in range(temp.shape[0]):
                weight_temp[j2] = weights[temp[j2]]
                s = shift_temp[j2]
                if s > 0:
                    templatesNew_temp[
                        :, :(R-s), j2] = templates[:, s:, temp[j2]]
                elif s < 0:
                    templatesNew_temp[
                        :, (-s):, j2] = templates[:, :(R+s), temp[j2]]
                elif s == 0:
                    templatesNew_temp[:, :, j2] = templates[:, :, temp[j2]]

                idx_old_id = spike_train[:, 1] == temp[j2]
                spt_new[idx_old_id] = spike_train[idx_old_id, 0] + s
                id_new[idx_old_id] = k

            weightNew[k] = np.sum(weight_temp)
            templatesNew[:, :, k] = np.average(
                templatesNew_temp, axis=2, weights=weight_temp)

        else:
            weightNew[k] = weights[temp[0]]
            templatesNew[:, :, k] = templates[:, :, temp[0]]

            idx_old_id = spike_train[:, 1] == temp[0]
            spt_new[idx_old_id] = spike_train[idx_old_id, 0]
            id_new[idx_old_id] = k

    spike_train_clear_new = np.hstack((
        spt_new[:, np.newaxis], id_new[:, np.newaxis]))

    return templatesNew, spike_train_clear_new, groups


# TODO: documentation
# TODO: comment code, it's not clear what it does
def determine_shift(tt, W):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    C, RW, K = tt.shape
    R = RW - 2*W
    t1 = tt[:, W:(W+R), 0]
    norm1 = np.linalg.norm(t1)

    shift = np.zeros(K, 'int16')
    for k in range(1, K):
        cos = np.zeros(2*W+1)
        for j in range(2*W+1):
            t2 = tt[:, j:(j+R), k]
            norm2 = np.linalg.norm(t2)
            cos[j] = np.sum(t1*t2)/norm1/norm2
        ii = np.argmax(cos)
        shift[k] = ii - W

    amps = np.max(np.abs(tt), axis=(0, 1))
    k_max = np.argmax(amps)
    shift = shift - shift[k_max]

    return shift


# TODO: documentation
# TODO: comment code, it's not clear what it does
def strongly_connected_components_iterative(vertices, edges):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    identified = set()
    stack = []
    index = {}
    boundaries = []

    for v in vertices:
        if v not in index:
            to_do = [('VISIT', v)]
            while to_do:
                operation_type, v = to_do.pop()
                if operation_type == 'VISIT':
                    index[v] = len(stack)
                    stack.append(v)
                    boundaries.append(index[v])
                    to_do.append(('POSTVISIT', v))
                    # We reverse to keep the search order identical to that of
                    # the recursive code;  the reversal is not necessary for
                    # correctness, and can be omitted.
                    to_do.extend(
                        reversed([('VISITEDGE', w) for w in edges[v]]))
                elif operation_type == 'VISITEDGE':
                    if v not in index:
                        to_do.append(('VISIT', v))
                    elif v not in identified:
                        while index[v] < boundaries[-1]:
                            boundaries.pop()
                else:
                    # operation_type == 'POSTVISIT'
                    if boundaries[-1] == index[v]:
                        boundaries.pop()
                        scc = set(stack[index[v]:])
                        del stack[index[v]:]
                        identified.update(scc)
                        yield scc


# TODO: documentation
# TODO: comment code, it's not clear what it does
def TemplatesSimilarity(t1, t2, th, W):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    C, RW = t1.shape
    R = RW - 2*W

    t1 = np.reshape(t1[:, W:(W+R)], R*C)
    t2_shifted = np.zeros((2*W+1, R*C))
    for j in range(2*W+1):
        t2_shifted[j] = np.reshape(t2[:, j:(j+R)], R*C)

    norm1 = np.sqrt(np.sum(np.square(t1), axis=0))
    norm2 = np.sqrt(np.sum(np.square(t2_shifted), axis=1))
    cos = np.matmul(t2_shifted, t1)/(norm1*norm2)

    ii = np.argmax(cos)
    cos = cos[ii]

    similar = 0
    if cos > th[0]:
        t1 = np.reshape(t1, [C, R])
        t2 = np.reshape(t2_shifted[ii], [C, R])

        scale = np.sum(t1*t2)/np.sum(np.square(t1))
        if scale > th[1] and scale < 2 - th[1]:
            similar = 1

    return similar
