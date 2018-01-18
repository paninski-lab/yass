import numpy as np
from scipy import sparse
import logging

from ..geometry import n_steps_neigh_channels


def get_templates(spike_train_clear, batch_size, buff, n_batches, n_channels,
                  spike_size, template_max_shift, scale_to_save, neighbors,
                  path_to_wrec, t_merge_th):
    """

    Returns
    -------
    spike_train_clear
    templates
        n channels x (2*(spike_size+template_max_shift)+1) x
    """
    logger = logging.getLogger(__name__)

    wfile = open(path_to_wrec, 'rb')

    flattenedLength = 2*(batch_size + 2*buff)*n_channels

    K = np.max(spike_train_clear[:, 1])+1
    big_spike_size = 2*spike_size + template_max_shift
    
    templates = np.zeros((n_channels, 2*big_spike_size+1, K))
    weights = np.zeros(K)

    for i in range(n_batches):
        logger.info("extracting waveforms from batch {} out of {} batches"
            .format(i+1, n_batches))
        wfile.seek(flattenedLength*i)

        wrec = wfile.read(flattenedLength)
        wrec = np.fromstring(wrec, dtype='int16')
        wrec = np.reshape(wrec, (-1, n_channels))
        wrec = wrec.astype('float32')/scale_to_save

        idx_batch = np.logical_and(spike_train_clear[:,0] > i*batch_size, spike_train_clear[:,0] < (i+1)*batch_size)

        if np.sum(idx_batch) > 0:

            spike_train_batch = spike_train_clear[idx_batch]
            spt_batch = spike_train_batch[:,0] - i*batch_size + buff
            L_batch = spike_train_batch[:,1]

            for j in range(spt_batch.shape[0]):
                tt = spt_batch[j]
                k = L_batch[j]
                templates[:, :, k] += wrec[(tt-big_spike_size):(tt+big_spike_size+1)].T
                weights[k] += 1

    logger.info("Merging templates.")
    templates = templates/weights[np.newaxis, np.newaxis, :]
    
    merge_shift = 3
    templates = align_templates(templates, template_max_shift-merge_shift)
    
    spike_train_clear, templates = mergeTemplates(templates, weights,
                                                  spike_train_clear,
                                                  neighbors,
                                                  merge_shift,
                                                  t_merge_th)
    templates = templates[:, merge_shift:(merge_shift+(4*spike_size+1))]

    wfile.close()

    return spike_train_clear, templates

def align_templates(templates, max_shift):
    C, R, K = templates.shape
    spike_size = int((R-1)/2 - max_shift)
    
    # get main channel for each template
    mainc = np.argmax(np.max(templates[:,max_shift:(max_shift+2*spike_size+1)], axis=1), axis=0)
    
    # get templates on their main channel only
    templates_mainc = np.zeros((R,K))
    for k in range(K):
        templates_mainc[:,k] = templates[mainc[k], :, k]
    
    # reference template
    biggest_template_k = np.argmax(np.max(templates_mainc, axis=0))
    biggest_template = templates_mainc[max_shift:(max_shift+2*spike_size+1), biggest_template_k]
    
    # find best shift
    fit_per_shift = np.zeros((2*max_shift+1,K))
    for s in range(2*max_shift+1):
        fit_per_shift[s] = np.matmul(
            biggest_template[np.newaxis,:],templates_mainc[s:(s+2*spike_size+1)])
    best_shift = np.argmax(fit_per_shift, axis=0)
    
    templates_final = np.zeros((C,2*spike_size+1,K))
    for k in range(K):
        s = best_shift[k]
        templates_final[:,:,k] = templates[:, s:(s+2*spike_size+1), k]
        
    return templates_final

def mergeTemplates(templates, weights, spike_train, neighbors,
                   template_max_shift, t_merge_th):
    C, R, K = templates.shape
    th = t_merge_th
    W = template_max_shift

    mainC = np.zeros(K)
    visible_channels = sparse.lil_matrix((K, C), dtype='bool')
    for k in range(K):
        amps = np.max(np.abs(templates[:, :, k]), axis=1)
        mainC[k] = np.argmax(amps)
        visible_channels[k, amps > 0.5*np.amax(amps)] = 1
        visible_channels[k,np.argmax(amps)] = 1

    mainC = sparse.csc_matrix(
        (np.ones(K), (np.arange(K), mainC)), shape=(K, C), dtype='bool')

    sparseConnection = sparse.lil_matrix((K, K), dtype='bool')
    for k1 in range(K):
        cc = sparse.find(mainC[k1])[1][0]
        k2s = sparse.find(
            mainC[(k1+1):, neighbors[cc]])[0] + k1 + 1
        for k2 in k2s:
            ch_idx = np.array(np.sum(visible_channels[[k1, k2]], 0) > 0)[0]
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
    spt_new = np.zeros(spike_train.shape[0],'int32')
    id_new = np.zeros(spike_train.shape[0],'int32')
    for k in range(Knew):
        temp = groups[k]
        templatesNew_temp = np.zeros((C, R, temp.shape[0]))
        weight_temp = np.zeros(temp.shape[0])
        if temp.shape[0] > 1:
            ch_idx = np.where(
                np.array(np.sum(visible_channels[temp], 0) > 0)[0])[0]
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

                idx_old_id = spike_train[:,1] == temp[j2]
                spt_new[idx_old_id] = spike_train[idx_old_id,0] + s
                id_new[idx_old_id] = k

            weightNew[k] = np.sum(weight_temp)
            templatesNew[:, :, k] = np.average(
                templatesNew_temp, axis=2, weights=weight_temp)

        else:
            weightNew[k] = weights[temp[0]]
            templatesNew[:, :, k] = templates[:, :, temp[0]]

            idx_old_id = spike_train[:,1] == temp[0]
            spt_new[idx_old_id] = spike_train[idx_old_id,0]
            id_new[idx_old_id] = k

    spike_train_clear_new = np.hstack((
        spt_new[:,np.newaxis],id_new[:,np.newaxis]))

    return spike_train_clear_new, templatesNew


def determine_shift(tt, W):
    C, RW, K = tt.shape
    R = RW - 2*W
    t1 = np.reshape(tt[:, W:(W+R), 0], R*C)
    shift = np.zeros(K, 'int16')
    for k in range(1, K):
        t2 = np.zeros((2*W+1, R*C))
        for j in range(2*W+1):
            t2[j] = np.reshape(tt[:, j:(j+R), k], R*C)

        norm1 = np.sqrt(np.sum(np.square(t1), axis=0))
        norm2 = np.sqrt(np.sum(np.square(t2), axis=1))
        cos = np.matmul(t2, t1)/(norm1*norm2)
        ii = np.argmax(cos)
        shift[k] = ii - W - 1

    amps = np.amax(np.abs(tt), axis=(0, 1))
    k_max = np.argmax(amps)
    shift = shift - shift[k_max]

    return shift


def strongly_connected_components_iterative(vertices, edges):

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


def TemplatesSimilarity(t1, t2, th, W):
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

        cos_per_channel = np.zeros(C)
        for c in range(C):
            t1_temp = t1[c]
            t2_temp = t2[c]
            norm1 = np.sqrt(np.sum(np.square(t1_temp)))
            norm2 = np.sqrt(np.sum(np.square(t2_temp)))
            cos_per_channel[c] = np.dot(t1_temp,t2_temp)/(norm1*norm2)

        if np.min(cos_per_channel) > th[0]:
            diff = np.max(np.abs(t2), axis=1)/np.max(np.abs(t1), axis=1)
            if 1/np.max(diff) > th[1] and np.min(diff) > th[1] and np.min(diff)/np.max(diff) > th[1]:
                similar = 1

    return similar
