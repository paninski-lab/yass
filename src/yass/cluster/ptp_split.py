import logging
import os
import numpy as np
import parmap
import networkx as nx

from yass.cluster.getptp import GETPTP, GETCLEANPTP
from yass import mfm

# from yass import read_config
# CONFIG = read_config()

def run_split_on_ptp(savedir,
                     fname_spike_index,
                     CONFIG,
                     raw_data=True,
                     fname_labels=None,
                     fname_templates=None,
                     fname_shifts=None,
                     fname_scales=None,
                     reader_raw=None, 
                     reader_residual=None,
                     denoiser=None):

    logger = logging.getLogger(__name__)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)

    # save results
    fname_spike_index_new = os.path.join(savedir, 'spike_index.npy')
    fname_labels_new = os.path.join(savedir, 'labels.npy')
    fname_ptp = os.path.join(savedir, 'ptp.npy')


    if os.path.exists(fname_labels_new):
        if fname_labels is None:
            fname_labels_input = None
        else:
            fname_labels_input = os.path.join(savedir, 'input_labels.npy')
        
        return fname_spike_index_new, fname_labels_new,fname_labels_input

    if not os.path.exists(savedir):
            os.makedirs(savedir)

    # get ptp
    logger.info("Get Spike PTP")
    if raw_data:
        getptp = GETPTP(fname_spike_index, reader_raw, CONFIG, denoiser)
        ptp_raw, ptp_deno = getptp.compute_ptps()
    else:
        getcleanptp = GETCLEANPTP(fname_spike_index,
                                  fname_labels,
                                  fname_templates,
                                  fname_shifts,
                                  fname_scales,
                                  reader_residual,
                                  denoiser)
        ptp_raw, ptp_deno = getcleanptp.compute_ptps()

    np.savez(os.path.join(savedir, 'ptps_input.npz'),
             ptp_raw=ptp_raw,
             ptp_deno=ptp_deno)
    
    # if there is an input label, load it
    # otherwise, max channel becomes the input label
    spike_index = np.load(fname_spike_index)
    if fname_labels is None:
        labels = spike_index[:, 1]
    else:
        labels = np.load(fname_labels)

    # triage if denoiser is in
    # Triage out badly collided ones
    if denoiser is not None:
        idx = np.where(ptp_raw < ptp_deno*1.5)[0]
        #idx = np.arange(len(ptp_deno))
        ptps = ptp_deno[idx]
        labels = labels[idx]
        spike_index = spike_index[idx]
    else:
        idx = np.arange(len(ptp_raw))
        ptps = ptp_raw
        
    np.save(os.path.join(savedir, 'idx_keep.npy'), idx)

    logger.info("Run Split")
    new_labels = run_split_parallel(ptps, labels, CONFIG, ptp_cut=5)

    np.save(fname_spike_index_new, spike_index)
    np.save(fname_labels_new, new_labels)
    np.save(fname_ptp, ptps)
    
    # if there is an input labels, update it
    if fname_labels is None:
        fname_labels_input = None
    else:
        fname_labels_input = os.path.join(savedir, 'input_labels.npy')
        np.save(fname_labels_input, labels)

    return fname_spike_index_new, fname_labels_new, fname_labels_input
    
def run_split_parallel(ptps, labels, CONFIG, ptp_cut=5):
    
    all_units = np.unique(labels)

    new_labels = np.ones(len(ptps), 'int32')*-1

    n_processors = CONFIG.resources.n_processors
    if CONFIG.resources.multi_processing:
        units_in = []
        for j in range(n_processors):
            units_in.append(all_units[slice(j, len(all_units), n_processors)])
        results = parmap.map(run_split,
                             units_in,
                             ptps,
                             labels,
                             CONFIG,
                             ptp_cut,
                             processes=n_processors)
        n_labels= 0
        for rr in results:
            for rr2 in rr:
                ii_ = rr2[:, 0]
                lab_ = rr2[:, 1]
                new_labels[ii_] = lab_ + n_labels
                n_labels += len(np.unique(lab_))
    else:
        results = run_split(all_units, ptps, labels, CONFIG, ptp_cut)
        n_labels= 0
        for rr in results:
            ii_ = rr[:, 0]
            lab_ = rr[:, 1]
            new_labels[ii_] = lab_ + n_labels
            n_labels += len(np.unique(lab_))
            
    return new_labels


def run_split(units_in, ptps, labels, CONFIG, ptp_cut=5):

    spike_index_list = []
    for unit in units_in:
        idx_ = np.where(labels == unit)[0]
        ptps_ = ptps[idx_]

        new_assignment = np.zeros(len(idx_), 'int32')        
        idx_big= np.where(ptps_ > ptp_cut)[0]
        if len(idx_big) > 10:
            mask = np.ones((len(idx_big), 1))
            group = np.arange(len(idx_big))
            vbParam = mfm.spikesort(ptps_[idx_big,None,None],
                                    mask,
                                    group,
                                    CONFIG)
            cc_assignment, stability, cc = anneal_clusters(vbParam)

            # get ptp per cc
            mean_ptp_cc = np.zeros(len(cc))
            for k in range(len(cc)):
                mean_ptp_cc[k] = np.mean(ptps_[idx_big][cc_assignment == k])

            # reorder cc label by mean ptp
            cc_assignment_ordered = np.zeros_like(cc_assignment)
            for ii, k in enumerate(np.argsort(mean_ptp_cc)):
                cc_assignment_ordered[cc_assignment == k] = ii
                
            # cc with the smallest mean ptp will have the same assignment as ptps < ptp cut
            new_assignment[idx_big] = cc_assignment_ordered

        spike_index_list.append(np.vstack((idx_, new_assignment)).T)
        
    return spike_index_list


def anneal_clusters(vbParam):
    
    N, K = vbParam.rhat.shape

    stability = calculate_stability(vbParam.rhat)
    if np.all(stability > 0.8):
        cc = [[k] for k in range(K)]
        return vbParam.rhat.argmax(1), stability, cc

    maha = mfm.calc_mahalonobis(vbParam, vbParam.muhat.transpose((1,0,2)))
    maha = np.maximum(maha, maha.T)

    maha_thresh_min = 0
    for k_target in range(K-1, 0, -1):
        # get connected components with k_target number of them
        cc, maha_thresh_min = get_k_cc(maha, maha_thresh_min, k_target)
        # calculate soft assignment for each cc
        rhat_cc = np.zeros([N,len(cc)])
        for i, units in enumerate(cc):
            rhat_cc[:, i] = np.sum(vbParam.rhat[:, units], axis=1)
        rhat_cc[rhat_cc<0.001] = 0.0
        rhat_cc = rhat_cc/np.sum(rhat_cc,axis =1 ,keepdims = True)

        # calculate stability for each component
        # and make decision            
        stability = calculate_stability(rhat_cc)
        if np.all(stability>0.8) or k_target == 1:
            return rhat_cc.argmax(1), stability, cc


def calculate_stability(rhat):
    K = rhat.shape[1]
    mask = rhat > 0.05
    stability = np.zeros(K)
    for clust in range(stability.size):
        if mask[:,clust].sum() == 0.0:
            continue
        stability[clust] = np.average(mask[:,clust] * rhat[:,clust], axis=0, weights = mask[:,clust])

    return stability


def get_cc(maha, maha_thresh):
    row, column = np.where(maha<maha_thresh)
    G = nx.DiGraph()
    for i in range(maha.shape[0]):
        G.add_node(i)
    for i, j in zip(row,column):
        G.add_edge(i, j)
    cc = [list(units) for units in nx.strongly_connected_components(G)]
    return cc


def get_k_cc(maha, maha_thresh_min, k_target):

    # it assumes that maha_thresh_min gives 
    # at least k+1 number of connected components
    k_now = k_target + 1
    if len(get_cc(maha, maha_thresh_min)) != k_now:
        raise ValueError("something is not right")

    maha_thresh = maha_thresh_min
    while k_now > k_target:
        maha_thresh += 1
        cc = get_cc(maha, maha_thresh)
        k_now = len(cc)

    if k_now == k_target:
        return cc, maha_thresh

    else:
        maha_thresh_max = maha_thresh
        maha_thresh_min = maha_thresh - 1
        if len(get_cc(maha, maha_thresh_min)) <= k_target:
            raise ValueError("something is not right")

        ctr = 0
        maha_thresh_max_init = maha_thresh_max
        while True:
            ctr += 1
            maha_thresh = (maha_thresh_max + maha_thresh_min)/2.0
            cc = get_cc(maha, maha_thresh)
            k_now = len(cc)
            if k_now == k_target:
                return cc, maha_thresh
            elif k_now > k_target:
                maha_thresh_min = maha_thresh
            elif k_now < k_target:
                maha_thresh_max = maha_thresh

            if ctr > 1000:
                print(k_now, k_target, maha_thresh, maha_thresh_max_init)
                print(cc)
                print(len(get_cc(maha, maha_thresh+0.001)))
                print(len(get_cc(maha, maha_thresh-0.001)))
                raise ValueError("something is not right")
    
