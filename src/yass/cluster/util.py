import numpy as np
import logging
import os
import tqdm
import parmap
from scipy import signal
from scipy import stats
from scipy.signal import argrelmax
from scipy.spatial import cKDTree
import scipy.sparse

from copy import deepcopy
import math
from sklearn.cluster import AgglomerativeClustering

from yass.explore.explorers import RecordingExplorer
from yass.templates.util import strongly_connected_components_iterative
from yass.geometry import n_steps_neigh_channels
from yass import mfm
from yass.empty import empty
from yass.cluster.cluster import (shift_chans, align_get_shifts_with_ref, binary_reader_waveforms)
from yass.util import absolute_path_to_asset
from yass.detect.run import deduplicate

from scipy.sparse import lil_matrix
from statsmodels import robust
from scipy.signal import argrelmin
import matplotlib
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import networkx as nx
import multiprocessing as mp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from diptest import diptest as dp

from sklearn.decomposition import PCA as PCA_original


#from matplotlib import colors as mcolors
#colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                 #for name, color in colors.items())
#sorted_colors = [name for hsv, name in by_hsv]

colors = [
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink']

sorted_colors=colors

        
def calculate_sparse_rhat(vbParam, tmp_loc, scores,
                          spike_index, neighbors):

    # vbParam.rhat calculation
    n_channels = np.max(spike_index[:, 1]) + 1
    n_templates = tmp_loc.shape[0]

    rhat = lil_matrix((scores.shape[0], n_templates))
    rhat = None
    for channel in range(n_channels):

        idx_data = np.where(spike_index[:, 1] == channel)[0]
        score = scores[idx_data]
        n_data = score.shape[0]

        ch_idx = np.where(neighbors[channel])[0]
        cluster_idx = np.zeros(n_templates, 'bool')
        for c in ch_idx:
            cluster_idx[tmp_loc == c] = 1
        cluster_idx = np.where(cluster_idx)[0]

        if n_data > 0 and cluster_idx.shape[0] > 0:

            local_vbParam = mfm.vbPar(None)
            local_vbParam.muhat = vbParam.muhat[:, cluster_idx]
            local_vbParam.Vhat = vbParam.Vhat[:, :, cluster_idx]
            local_vbParam.invVhat = vbParam.invVhat[:, :, cluster_idx]
            local_vbParam.nuhat = vbParam.nuhat[cluster_idx]
            local_vbParam.lambdahat = vbParam.lambdahat[cluster_idx]
            local_vbParam.ahat = vbParam.ahat[cluster_idx]

            mask = np.ones([n_data, 1])
            group = np.arange(n_data)
            masked_data = mfm.maskData(score, mask, group)

            local_vbParam.update_local(masked_data)
            local_vbParam.rhat[local_vbParam.rhat < 0.1] = 0
            local_vbParam.rhat = local_vbParam.rhat / \
                np.sum(local_vbParam.rhat, axis=1, keepdims=True)

            row_idx, col_idx = np.where(local_vbParam.rhat > 0)
            val = local_vbParam.rhat[row_idx, col_idx]
            row_idx = idx_data[row_idx]
            col_idx = cluster_idx[col_idx]
            rhat_local = np.hstack((row_idx[:, np.newaxis],
                                    col_idx[:, np.newaxis],
                                    val[:, np.newaxis]))
            if rhat is None:
                rhat = rhat_local
            else:
                rhat = np.vstack((rhat, rhat_local))

    return rhat


def calc_prob(data, means, precision): 

    #print data.shape
    #print means.shape
    #print precision.shape

    diff = data[:,np.newaxis] - means
    #print diff.shape
    maha = np.squeeze(np.matmul(diff[:, :, np.newaxis],
                                np.matmul(precision,
                                          diff[..., np.newaxis])),
                      axis=[2, 3])
    #print maha.shape
    
    log_prec = np.linalg.slogdet(precision)[1]/2
    #print (log_prec)
    constant = -np.log(2*np.pi)*precision.shape[1]/2
    prob = -maha/2 + log_prec + constant
    return prob

def calculate_maha_clusters(vbParam):
    diff = np.transpose(vbParam.muhat, [1, 2, 0]) - \
        vbParam.muhat[..., 0].T
    clustered_prec = np.transpose(vbParam.Vhat[:, :, :, 0] *
                                  vbParam.nuhat, [2, 0, 1])
    maha = np.squeeze(np.matmul(diff[:, :, np.newaxis],
                                np.matmul(clustered_prec[:, np.newaxis],
                                          diff[..., np.newaxis])),
                      axis=[2, 3])
    return maha


def merge_move_patches(cluster, neigh_clusters, scores, vbParam, maha, cfg):

    while len(neigh_clusters) > 0:
        i = neigh_clusters[-1]
        # indices = np.logical_or(clusterid == cluster, clusterid == i)
        indices, temp = vbParam.rhat[:, [cluster, i]].nonzero()
        indices = np.unique(indices)
        ka, kb = min(cluster, i), max(cluster, i)
        local_scores = scores[indices]
        local_vbParam = mfm.vbPar(
            vbParam.rhat[:, [cluster, i]].toarray()[indices])
        local_vbParam.muhat = vbParam.muhat[:, [cluster, i]]
        local_vbParam.Vhat = vbParam.Vhat[:, :, [cluster, i]]
        local_vbParam.invVhat = vbParam.invVhat[:, :, [cluster, i]]
        local_vbParam.nuhat = vbParam.nuhat[[cluster, i]]
        local_vbParam.lambdahat = vbParam.lambdahat[[cluster, i]]
        local_vbParam.ahat = vbParam.ahat[[cluster, i]]
        mask = np.ones([local_scores.shape[0], 1])
        group = np.arange(local_scores.shape[0])
        local_maskedData = mfm.maskData(local_scores, mask, group)
        # local_vbParam.update_local(local_maskedData)
        local_suffStat = mfm.suffStatistics(local_maskedData, local_vbParam)

        ELBO = mfm.ELBO_Class(local_maskedData, local_suffStat,
                              local_vbParam, cfg)
        L = np.ones(2)
        (local_vbParam, local_suffStat,
         merged, _, _) = mfm.check_merge(local_maskedData,
                                         local_vbParam,
                                         local_suffStat, 0, 1,
                                         cfg, L, ELBO)
        if merged:
            print("merging {}, {}".format(cluster, i))
            vbParam.muhat = np.delete(vbParam.muhat, kb, 1)
            vbParam.muhat[:, ka] = local_vbParam.muhat[:, 0]

            vbParam.Vhat = np.delete(vbParam.Vhat, kb, 2)
            vbParam.Vhat[:, :, ka] = local_vbParam.Vhat[:, :, 0]

            vbParam.invVhat = np.delete(vbParam.invVhat, kb, 2)
            vbParam.invVhat[:, :, ka] = local_vbParam.invVhat[:, :, 0]

            vbParam.nuhat = np.delete(vbParam.nuhat, kb, 0)
            vbParam.nuhat[ka] = local_vbParam.nuhat[0]

            vbParam.lambdahat = np.delete(vbParam.lambdahat, kb, 0)
            vbParam.lambdahat[ka] = local_vbParam.lambdahat[0]

            vbParam.ahat = np.delete(vbParam.ahat, kb, 0)
            vbParam.ahat[ka] = local_vbParam.ahat[0]

            vbParam.rhat[:, ka] = vbParam.rhat[:, ka] + vbParam.rhat[:, kb]
            n_data_all, n_templates_all = vbParam.rhat.shape
            to_keep = list(set(np.arange(n_templates_all))-set([kb]))
            vbParam.rhat = vbParam.rhat[:, to_keep]

            # clusterid[indices] = ka
            # clusterid[clusterid > kb] = clusterid[clusterid > kb] - 1
            neigh_clusters.pop()

            maha = np.delete(maha, kb, 1)
            maha = np.delete(maha, kb, 0)

            diff = vbParam.muhat[:, :, 0] - local_vbParam.muhat[:, :, 0]

            prec = local_vbParam.Vhat[..., 0] * local_vbParam.nuhat[0]
            maha[ka] = np.squeeze(
                np.matmul(
                    diff.T[:, np.newaxis, :],
                    np.matmul(prec[:, :, 0],
                              diff.T[..., np.newaxis])))

            prec = np.transpose(vbParam.Vhat[..., 0] * vbParam.nuhat,
                                [2, 0, 1])
            maha[:, ka] = np.squeeze(np.matmul(
                diff.T[:, np.newaxis, :],
                np.matmul(prec, diff.T[..., np.newaxis])))

            maha[ka, ka] = np.inf
            neigh_clusters = list(np.where(
                np.logical_or(maha[ka] < 15, maha.T[ka] < 15))[0])
            cluster = ka

        if not merged:
            maha[ka, kb] = maha[kb, ka] = np.inf
            neigh_clusters.pop()

    return vbParam, maha


def try_merge(k1, k2, scores, vbParam, maha, cfg):

    ka, kb = min(k1, k2), max(k1, k2)

    assignment = vbParam.rhat[:, :2].astype('int32')

    idx_ka = assignment[:, 1] == ka
    idx_kb = assignment[:, 1] == kb

    indices = np.unique(assignment[
        np.logical_or(idx_ka, idx_kb), 0])

    rhat = np.zeros((scores.shape[0], 2))
    rhat[assignment[idx_ka, 0], 0] = vbParam.rhat[idx_ka, 2]
    rhat[assignment[idx_kb, 0], 1] = vbParam.rhat[idx_kb, 2]
    rhat = rhat[indices]

    local_scores = scores[indices]
    local_vbParam = mfm.vbPar(rhat)
    local_vbParam.muhat = vbParam.muhat[:, [ka, kb]]
    local_vbParam.Vhat = vbParam.Vhat[:, :, [ka, kb]]
    local_vbParam.invVhat = vbParam.invVhat[:, :, [ka, kb]]
    local_vbParam.nuhat = vbParam.nuhat[[ka, kb]]
    local_vbParam.lambdahat = vbParam.lambdahat[[ka, kb]]
    local_vbParam.ahat = vbParam.ahat[[ka, kb]]

    mask = np.ones([local_scores.shape[0], 1])
    group = np.arange(local_scores.shape[0])
    local_maskedData = mfm.maskData(local_scores, mask, group)
    # local_vbParam.update_local(local_maskedData)
    local_suffStat = mfm.suffStatistics(local_maskedData, local_vbParam)

    ELBO = mfm.ELBO_Class(local_maskedData, local_suffStat, local_vbParam, cfg)
    L = np.ones(2)
    (local_vbParam, local_suffStat,
     merged, _, _) = mfm.check_merge(local_maskedData,
                                     local_vbParam,
                                     local_suffStat, 0, 1,
                                     cfg, L, ELBO)
    if merged:
        print("merging {}, {}".format(ka, kb))

        vbParam.muhat = np.delete(vbParam.muhat, kb, 1)
        vbParam.muhat[:, ka] = local_vbParam.muhat[:, 0]

        vbParam.Vhat = np.delete(vbParam.Vhat, kb, 2)
        vbParam.Vhat[:, :, ka] = local_vbParam.Vhat[:, :, 0]

        vbParam.invVhat = np.delete(vbParam.invVhat, kb, 2)
        vbParam.invVhat[:, :, ka] = local_vbParam.invVhat[:, :, 0]

        vbParam.nuhat = np.delete(vbParam.nuhat, kb, 0)
        vbParam.nuhat[ka] = local_vbParam.nuhat[0]

        vbParam.lambdahat = np.delete(vbParam.lambdahat, kb, 0)
        vbParam.lambdahat[ka] = local_vbParam.lambdahat[0]

        vbParam.ahat = np.delete(vbParam.ahat, kb, 0)
        vbParam.ahat[ka] = local_vbParam.ahat[0]

        idx_delete = np.where(np.logical_or(idx_ka, idx_kb))[0]
        vbParam.rhat = np.delete(vbParam.rhat, idx_delete, 0)
        vbParam.rhat[vbParam.rhat[:, 1] > kb, 1] -= 1

        rhat_temp = np.hstack((indices[:, np.newaxis],
                               np.ones((indices.size, 1))*ka,
                               np.sum(rhat, 1, keepdims=True)))
        vbParam.rhat = np.vstack((vbParam.rhat, rhat_temp))

        maha = np.delete(maha, kb, 1)
        maha = np.delete(maha, kb, 0)

        diff = vbParam.muhat[:, :, 0] - local_vbParam.muhat[:, :, 0]

        prec = local_vbParam.Vhat[..., 0] * local_vbParam.nuhat[0]
        maha[ka] = np.squeeze(
            np.matmul(
                diff.T[:, np.newaxis, :],
                np.matmul(prec[:, :, 0],
                          diff.T[..., np.newaxis])))

        prec = np.transpose(vbParam.Vhat[..., 0] * vbParam.nuhat,
                            [2, 0, 1])
        maha[:, ka] = np.squeeze(
            np.matmul(diff.T[:, np.newaxis, :],
                      np.matmul(prec, diff.T[..., np.newaxis])))

        maha[ka, ka] = np.inf

    if not merged:
        maha[ka, kb] = maha[kb, ka] = np.inf

    return vbParam, maha


def global_cluster_info(vbParam, main_channel,
                        score, spike_index,
                        global_vbParam, global_tmp_loc,
                        global_score, global_spike_index):
    """
    Gather clustering information from each run
    Parameters
    ----------
    vbParam, maskedData: class
        cluster information output from MFM
    score: np.array (n_data, n_features, 1)
        score used for each clustering
    spike_time: np.array (n_data, 1)
        spike time that matches with each score
    global_vbParam, global_maskedData: class
        a class that contains cluster information from all
        previous run,
    global_score: np.array (n_data_all, n_features, 1)
        all scores from previous runs
    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score
    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    Returns
    -------
    global_vbParam, global_maskedData: class
        a class that contains cluster information after
        adding the current one
    global_score: np.array (n_data_all, n_features, 1)
        all scores after adding the current one
    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score
    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    """

    n_idx, k_idx = np.where(vbParam.rhat > 0)
    prob_val = vbParam.rhat[n_idx, k_idx]
    vbParam.rhat = np.hstack((n_idx[:, np.newaxis],
                              k_idx[:, np.newaxis],
                              prob_val[:, np.newaxis]))

    if global_vbParam is None:
        global_vbParam = vbParam
        global_tmp_loc = np.ones(
            vbParam.muhat.shape[1], 'int16')*main_channel
        global_score = score
        global_spike_index = spike_index

    else:

        # append global_vbParam
        global_vbParam.muhat = np.concatenate(
            [global_vbParam.muhat, vbParam.muhat], axis=1)
        global_vbParam.Vhat = np.concatenate(
            [global_vbParam.Vhat, vbParam.Vhat], axis=2)
        global_vbParam.invVhat = np.concatenate(
            [global_vbParam.invVhat, vbParam.invVhat],
            axis=2)
        global_vbParam.lambdahat = np.concatenate(
            [global_vbParam.lambdahat, vbParam.lambdahat],
            axis=0)
        global_vbParam.nuhat = np.concatenate(
            [global_vbParam.nuhat, vbParam.nuhat],
            axis=0)
        global_vbParam.ahat = np.concatenate(
            [global_vbParam.ahat, vbParam.ahat],
            axis=0)

        n_max, k_max = np.max(global_vbParam.rhat[:, :2], axis=0)
        vbParam.rhat[:, 0] += n_max + 1
        vbParam.rhat[:, 1] += k_max + 1
        global_vbParam.rhat = np.concatenate(
            [global_vbParam.rhat, vbParam.rhat],
            axis=0)

        global_tmp_loc = np.hstack((global_tmp_loc,
                                    np.ones(vbParam.muhat.shape[1],
                                            'int16')*main_channel))

        # append score
        global_score = np.concatenate([global_score,
                                       score], axis=0)

        # append spike_index
        global_spike_index = np.concatenate([global_spike_index,
                                             spike_index], axis=0)

    return (global_vbParam, global_tmp_loc,
            global_score, global_spike_index)


def clean_empty_cluster(vbParam, min_spikes=20):

    n_hat = np.sum(vbParam.rhat, 0)
    Ks = n_hat > min_spikes

    vbParam.muhat = vbParam.muhat[:, Ks]
    vbParam.Vhat = vbParam.Vhat[:, :, Ks]
    vbParam.invVhat = vbParam.invVhat[:, :, Ks]
    vbParam.lambdahat = vbParam.lambdahat[Ks]
    vbParam.nuhat = vbParam.nuhat[Ks]
    vbParam.ahat = vbParam.ahat[Ks]
    vbParam.rhat = vbParam.rhat[:, Ks]

    return vbParam


def usample_resample2(wf, upsample_factor):
    n_spikes = wf.shape[1]
    traces=[]
    for j in range(n_spikes):
        traces.append(signal.resample(wf[:,j],91))
        
    return wfs_upsampled
    
    
def upsample_template(wf,upsample_factor,n_steps):

    # reduce waveform to 5 time_steps
    wf = wf[wf.shape[0]//2-n_steps: wf.shape[0]//2+n_steps+1]
    # get shapes
    waveform_size, n_spikes = wf.shape

    # upsample using cubic interpolation
    x = np.linspace(0, waveform_size - 1, num=waveform_size, endpoint=True)
    shifts = np.linspace(0, 1, upsample_factor, endpoint=False)
    xnew = np.sort(np.reshape(x[:, np.newaxis] + shifts, -1))
    wfs_upsampled = np.zeros((waveform_size * upsample_factor, n_spikes))
    
    # compute template and interpolate it
    template = np.mean(wf,axis=1)
    ff = interp1d(x, template, kind='cubic')
    idx_good = np.logical_and(xnew >= 0, xnew <= waveform_size - 1)
    template_upsampled = ff(xnew[idx_good])
        
    return template_upsampled


def shift_template(template_upsampled, n_shifts, window):
    ''' Select n_shifts version of the tempalte shifting from
        -n_shifts/2 to + n_shifts/2 in the original waveform
        
        Cat TODO: this should be done pythonically
    '''
    
    temp_array = []
    for s in range(-n_shifts//2, n_shifts//2, 1):
        temp_array.append(template_upsampled[template_upsampled.shape[0]//2-window+s:
                                             template_upsampled.shape[0]//2+window+s])
    return np.array(temp_array)


def return_shifts(wfs_upsampled, template_shifted, window):
    
    shift_array = []
    out_array = []
    waveform_len = wfs_upsampled.shape[0]

    for k in range(wfs_upsampled.shape[1]):
        temp = np.matmul(wfs_upsampled[waveform_len//2-window:waveform_len//2+window,k],
                        template_shifted.T)
        
        shift_array.append(np.argmax(temp))
    
    return np.array(shift_array) #, out_array


# PCA function return PCA and reconstructed data
def PCA(X, n_components):
    from sklearn import decomposition

    pca = decomposition.PCA(n_components)
    pca.fit(X)
    X = pca.transform(X)
    Y = pca.inverse_transform(X)
    return X, Y, pca


def align_channelwise3(wf, upsample_factor = 20, nshifts = 7):

    wf_up = upsample_resample(wf.T, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
   
    #if type(ref) == 'ndarray':
    #    ref_upsampled = upsample_resample(ref,20)
    #else:
    ref_upsampled = wf_up.mean(0)
    
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]
    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    
    wf_final = np.zeros([wf.shape[0], (wlen-1)//2 +1])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end]
    return wf_final[:,::upsample_factor]


def align_mc_templates(wf, mc, spike_padding, upsample_factor = 5,
                       nshifts = 15):

    ''' Align all waveforms to the master channel
        wf = selected waveform matrix (# spikes, # samples, # featchans)
        mc = maximum channel from featchans; usually first channle, i.e. 0
    '''
    
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    wf_up = []
    for k in range(wf.shape[2]): 
        #print ("aligning : ", k)
        wf_up.append(upsample_resample_parallel_channel(wf[:,:,k], upsample_factor))
    wf_up = np.array(wf_up).swapaxes(1,2).swapaxes(0,2)

    wlen = wf_up.shape[1]
    wf_start = spike_padding*upsample_factor
    wf_end = -spike_padding*upsample_factor
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    ref_upsampled = wf_up[:,:,mc].mean(0)
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:, mc], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[0],wlen_trunc, wf.shape[2]])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end][:, np.arange(wf.shape[2])]

    # plot original waveforms
    #print ("Plotting align_mc")
    #ax = plt.subplot(131)
    #plt.plot(wf[:1000,:,mc].T,alpha=0.1)

    ## plot aligned waveforms
    #ax = plt.subplot(132)
    #plt.plot(wf_final[:1000,::upsample_factor,mc].T, alpha=0.1)

    #plt.savefig('/media/cat/1TB/liam/49channels/data1_allset/tmp/cluster/chunk_000000/channel_31_aligning.png')

    #quit()
    return wf_final[:,::upsample_factor]


def align_mc_cumulative(wf, mc, CONFIG, upsample_factor = 20, nshifts = 7, 
             ref = None):
    ''' Align all waveforms to the master channel
    
        wf = selected waveform matrix (# spikes, # samples, # featchans)
        mc = maximum channel from featchans; usually first channle, i.e. 0
        
    '''
    
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1
    
    # or loop over every channel and parallelize each channel:
    wf_up = []
    for k in range(wf.shape[2]): 
        #print (" upsampling chan (parallel): ", k)
        wf_up.append(upsample_resample_parallel_channel(wf[:,:,k], upsample_factor))
    wf_up = np.array(wf_up).swapaxes(1,2).swapaxes(0,2)
    #print ('wf_upsampled: ', wf_up.shape)

    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    if ref is not None:
        ref_upsampled = upsample_resample(ref[:,np.newaxis],upsample_factor)[0]
    else:
        ref_upsampled = wf_up[:,:,mc].mean(0)
    
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    #print (ref_shifted.shape)
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]
    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:, mc], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[0],wlen_trunc, wf.shape[2]])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end][:, np.arange(wf.shape[2])]
    
    return wf_final[:,::upsample_factor]
    
def upsample_resample(wf, upsample_factor):
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(n_spikes):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces

def upsample_resample_parallel(wf, upsample_factor):
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(n_spikes):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces
    
def upsample_resample_parallel_channel(wf, upsample_factor):
    n_spikes, _ = wf.shape
    
    # dont' parallize alignment - unless seems ok otherwise
    # Cat: TODO: can we parallize recursively and to this?

    wf_up = upsample_parallel(wf, upsample_factor)
        
    return wf_up
 
    
def align_get_shifts(wf, CONFIG, upsample_factor = 5, nshifts = 15):

    ''' Align all waveforms on a single channel
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample_parallel_channel(wf, upsample_factor)

    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = wf_up.mean(0)
    
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts
    
    
    
def upsample_parallel(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces
    

#def RRR3_noregress_recovery_dynamic_features(channel, current_indexes, gen, fig, 
         #grid, x, ax_t, triageflag, alignflag, plotting, n_feat_chans, 
         #n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
         #upsample_factor, nshifts, assignment_global, spike_index, 
         #scale, knn_triage_threshold, deconv_flag, templates, 
         #min_spikes_local, active_chans=None):
    
    #''' Recursive clusteringn function
        #channel: current channel being clusterd
        #wf = wf_PCA: denoised waveforms (# spikes, # time points, # chans)
        #sic = spike_indexes of spikes on current channel
        #gen = generation of cluster; increases with each clustering step        
    #'''


    ## Cat: TODO read from CONFIG File
    #verbose=True
    
    ## load correct waveforms from disk
    ##wf = np.load(wf_fname)[current_indexes]
    #wf = wf_global[current_indexes]
    
    ## ************* CHECK SMALL CLUSTERS *************
    ## Exit clusters that are too small
    #if wf.shape[0] < CONFIG.cluster.min_spikes:
        #return
    #if verbose:
        #print("chan/unit "+str(channel)+' gen: '+str(gen)+' # spikes: '+
              #str(wf.shape[0]))
    
    #''' *************************************************       
        #** ALIGN ALL CHANS TO MAX CHAN - LINEAR INTERP **
        #*************************************************
    #'''
    ## align, note: aligning all channels to max chan which is appended to the end
    ## note: max chan is first from feat_chans above, ensure order is preserved
    
    #if alignflag:
        #if verbose:
            #print ("chan "+str(channel)+' gen: '+str(gen)+" - aligning")

        #mc = wf.mean(0).ptp(0).argmax(0)
        #best_shifts = align_get_shifts(wf[:,:,mc], CONFIG) 
        #wf_align = shift_chans(wf, best_shifts, CONFIG)

    #else:
        #wf_align = wf
    
    #''' *************************************************       
        #************** ACTIVE CHAN SELECTION ************
        #*************************************************
    #''' 
    #active_chans_flag = False
    #if active_chans_flag and gen == 0:
        #stds = np.median(np.abs(wf - np.median(wf_align, axis=0, keepdims=True)), axis=0)*1.4826
        #active_chans = np.where(stds.max(0) > 1.05)[0]

        #neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 1)
        #active_chans = np.hstack((active_chans, np.where(neighbors[channel])[0]))
        #active_chans = np.where(connected_channels(active_chans, channel, neighbors))[0]
        
        ##def plot_with_geom(data, geom, time_scale=0.5, scale=10, color='k', mark_channels=None):
        ##    t, c = data.shape
        ##    plt.plot(geom[:,0]+np.arange(-data.shape[0],0)[:,np.newaxis]/time_scale, 
        ##                 geom[:,1] + data*scale, color=color, alpha=.8)
        ##    if mark_channels is not None:
        ##        plt.scatter(geom[mark_channels,0], geom[mark_channels,1], s=scale*10, color='green')
        ##    for j in range(c):
        ##        plt.text(geom[j,0], geom[j,1], str(j))
        
        ##chunk_dir = CONFIG.data.root_folder+"/tmp/cluster/chunk_000000"
        ##plt.figure(figsize=(30,20))
        ##plot_with_geom(stds, CONFIG.geom, time_scale=2, scale=10, color='k', mark_channels=active_chans)
        ##plt.savefig(chunk_dir+"/channel_{}_active_chans.png".format(channel))
    
    #else:
        #active_chans = np.arange(wf_align.shape[2])
        
    ## Cat: TODO: so we force all subsequent generations to use gen0 alignment
    ##wf=wf_align
        
    #''' ************************************************        
        #****** FIND FEATURE CHANNELS & FEATURIZE *******
        #************************************************
    #'''
    
    #if verbose:
        #print("chan/unit "+str(channel)+' gen: '+str(gen)+' getting feat chans')
    
    ## Cat: TODO: is 10k spikes enough? 
    ## Cat: TODO: what do these metrics look like for 100 spikes!?; should we simplify for low spike count?
    #feat_chans, mc, robust_stds = get_feat_channels_mad_cat(
                                            #wf_align[:10000][:, :, active_chans], n_feat_chans)

    ## featurize using latest alg
    #idx_keep_feature, pca_wf = featurize_residual_triage_cat(wf_align[:, :, active_chans], robust_stds, 
                                                  #feat_chans, mc, n_feat_chans)
    
    #if verbose:
        #print("chan "+str(channel)+' gen: '+str(gen)+", feat chans: "+
                  #str(feat_chans[:n_feat_chans]) + ", max_chan: "+ str(active_chans[mc]))

    #pca_wf = pca_wf[idx_keep_feature][:,:5]
 
    #''' ************************************************        
        #******** KNN TRIAGE & PCA #2 *******************
        #************************************************
    #'''
    ## Cat: TODO: this is hardwired now
    ### knn triage outliars; e.g. remove 2%
    #if True:
        #idx_keep = knn_triage(95, pca_wf)
        #idx_keep = np.where(idx_keep==1)[0]
        #if verbose:
            #print("chan "+str(channel)+' gen: '+str(gen) + 
                  #" triaged, remaining spikes "+ 
                  #str(idx_keep.shape[0]))

        #pca_wf = pca_wf[idx_keep]
        
        ##idx_keep = idx_keep[idx_keep]
    
    #else:
        #idx_keep = np.arange(pca_wf.shape[0])

    ## do another min spike check in case triage killed too many spikes
    #if wf.shape[0] < CONFIG.cluster.min_spikes:
        #return
        
    #pca_wf_all = pca_wf.copy() 
    
    
    #''' ************************************************        
        #************** SUBSAMPLE STEP ****************** 
        #************************************************
    #'''
    ## subsmaple 10,000 spikes if not in deconv and have > 10k spikes
    #if not deconv_flag and (pca_wf.shape[0]> CONFIG.cluster.max_n_spikes):
        #idx_subsampled = np.random.choice(np.arange(pca_wf.shape[0]),
                         #size=CONFIG.cluster.max_n_spikes,
                         #replace=False)
    
        #pca_wf = pca_wf[idx_subsampled]


    #''' ************************************************        
        #************ CLUSTERING STEP *******************
        #************************************************        
    #'''
    ## clustering
    #if verbose:
        #print("chan "+ str(channel)+' gen: '+str(gen)+" - clustering ", 
                                                          #pca_wf.shape)
    #vbParam, assignment = run_mfm3(pca_wf, CONFIG)
    
    
    #''' *************************************************        
        #************* RECOVER SPIKES ********************
        #*************************************************        
    #'''
    ## if we subsampled then recover soft-assignments using above:
    ## Note: for post-deconv reclustering, we can safely cluster only 10k spikes or less
    #if not deconv_flag and (pca_wf.shape[0] <= CONFIG.cluster.max_n_spikes):
        #vbParam2 = deepcopy(vbParam)
        #vbParam2, assignment2 = recover_spikes(vbParam2, 
                                               #pca_wf_all, 
                                               #CONFIG)
    #else:
        #vbParam2, assignment2 = vbParam, assignment

    #idx_recovered = np.where(assignment2!=-1)[0]
    #if verbose:
        #print ("chan "+ str(channel)+' gen: '+str(gen)+" - recovered ",
                                            #str(idx_recovered.shape[0]))
    
    ## make a template that in case the cluster is saved, can be used below
    ## Cat: TODO: can take template to be just core of cluster 
    #template = wf_align[idx_keep][idx_recovered].mean(0)
    #mc = np.argmax(template.ptp(0))
    #sic_current = sic_global[current_indexes][idx_keep][idx_recovered]
    
    ## zero out all wf arrays; not required below here any longer
    #wf = None
    #wf_align = None

    #'''*************************************************        
       #*********** REVIEW AND SAVE RESULTS *************
       #*************************************************        
    #'''
                                
    ## First, check again that triage steps above didn't drop below min_spikes
    ## Cat: TODO: 
    #if (pca_wf.shape[0] < CONFIG.cluster.min_spikes):
        #return
        
    ## Case #1: single cluster found
    #if vbParam.rhat.shape[1] == 1:
        ##print ("chan "+str(channel)+' gen: '+str(gen)+ " CASE #1: converged cluster")
        ## exclude units whose maximum channel is not on the current 
        ## clustered channel; but only during clustering, not during deconv
        #if mc != channel and (deconv_flag==False): 
            #print ("  channel: ", channel, " template has maxchan: ", mc, 
                    #" skipping ...")
            
            ## always plot scatter distributions
            #if gen<20:
                #split_type = 'mfm non_max-chan'
                #end_flag = 'cyan'
                #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment2[idx_recovered],
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type,
                            #end_flag)
                            
            #return 
        #else:         
            #N= len(assignment_global)
            #if verbose:
                #print("chan "+str(channel)+' gen: '+str(gen)+" >>> cluster "+
                    #str(N)+" saved, size: "+str(idx_recovered.shape)+"<<<")
                #print ("")
            
            #assignment_global.append(N * np.ones(assignment2[idx_recovered].shape[0]))
            #spike_index.append(sic_current)
            #templates.append(template)
            
            ### Save only core of distribution
            ##maha=2
            ##vbParam_temp, assignment_core = recover_spikes(vbParam2, 
                                                           ##pca_wf_all, 
                                                           ##CONFIG,
                                                           ##maha)
            ##idx_core = np.where(assignment_core!=-1)[0]
            ##print ("***************  saving core", idx_core.shape, idx_recovered.shape)

            ## plot template if done
            #if plotting:
                ##plot_clustering_template(fig, grid, ax_t, gen, N, wf_align, idx_recovered, 
                ##                        CONFIG, colors, feat_chans, scale)
                #plot_clustering_template(fig, grid, ax_t, gen, N, template, 
                                         #idx_recovered, 
                                         #CONFIG, colors, feat_chans, scale)

                ## always plot scatter distributions
                #if gen<20:
                    #split_type = 'mfm'
                    #end_flag = 'red'
                    #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment2[idx_recovered],
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type,
                            #end_flag)
                            
    ## Case #2: multiple clusters
    #else:
        #mask = vbParam2.rhat[idx_recovered]>0
        #stability = np.average(mask * vbParam2.rhat[idx_recovered], axis = 0, weights = mask)
        #clusters, sizes = np.unique(assignment2[idx_recovered], return_counts = True)
        
        #if verbose:
            #print("chan "+str(channel)+' gen: '+str(gen) + 
                  #" multiple clusters, stability " + str(np.round(stability,2)) + 
                  #" size: "+str(sizes))

        ## if at least one stable cluster
        #if np.any(stability>mfm_threshold):      

            ## always plot scatter distributions
            #if plotting and gen<20:
                #split_type = 'mfm multi split'
                #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment2[idx_recovered],
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type)
                            

            ## remove stable clusters 
            #for clust in np.where(stability>mfm_threshold)[0]:

                #idx = np.where(assignment2[idx_recovered]==clust)[0]
                
                #if idx.shape[0]<CONFIG.cluster.min_spikes: 
                    #continue    # cluster too small
                
                #if verbose:
                    #print("chan "+str(channel)+' gen: '+str(gen)+
                        #" reclustering stable cluster"+ 
                        #str(idx.shape))
                
                #triageflag = False
                #alignflag = True
                #RRR3_noregress_recovery_dynamic_features(channel, 
                     #current_indexes[idx_keep][idx_recovered][idx], 
                     #gen+1, fig, grid, x, ax_t, triageflag, alignflag, 
                     #plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                     #mfm_threshold,  CONFIG, upsample_factor, nshifts, 
                     #assignment_global, spike_index, scale, knn_triage_threshold, 
                     #deconv_flag, templates, min_spikes_local, active_chans)

            ## run mfm on remaining data
            #idx = np.in1d(assignment2[idx_recovered], np.where(stability<=mfm_threshold)[0])
            #if idx.sum()>CONFIG.cluster.min_spikes:
                ##print ("chan "+str(channel)+' gen: '+str(gen)+ " CASE #4: clustering residuals")
                #if verbose:
                    #print("chan "+str(channel)+" reclustering residuals "+
                                            #str(idx.shape))
                #triageflag = False
                #alignflag = True
                
                ## overwrite wf with current index to remove data from memory
                #RRR3_noregress_recovery_dynamic_features(channel, 
                    #current_indexes[idx_keep][idx_recovered][idx],
                    #gen+1, fig, grid, x, ax_t, triageflag, alignflag, 
                    #plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                    #mfm_threshold, CONFIG, upsample_factor, nshifts, 
                    #assignment_global, spike_index, scale, knn_triage_threshold,
                    #deconv_flag, templates, min_spikes_local, active_chans)



        ## ********** if all clusters are unstable ****************
        ## ********** check diptest 
        #else:

            #ctr=0 
            #dp = 1.0
            #idx_temp_keep = np.arange(idx_recovered.shape[0])
            #cluster_idx_keep = np.arange(vbParam2.muhat.shape[0])
            ## loop over cluster until at least 3 loops and take lowest dp value
            #while True:    
                ## use EM aglorithm to get labels for 2 clusters
                #if True: 
                    ## split using EM
                    #gmm = GaussianMixture(n_components=2)
                    #gmm.fit(pca_wf_all[idx_recovered])
                    #labels = gmm.predict_proba(pca_wf_all[idx_recovered])

                    #temp_rhat = labels
                    #temp_assignment = np.zeros(labels.shape[0], 'int32')
                    #idx = np.where(labels[:,1]>0.5)[0]
                    #temp_assignment[idx]=1
                
                #else:
                   ## split using mfm 
                    #temp_assignment = mfm_binary_split2(
                                        #vbParam2.muhat[cluster_idx_keep], 
                                        #assignment2[idx_recovered],
                                        #cluster_idx_keep)
                                        
                    
                ## check if any clusters smaller than min spikes
                #counts = np.unique(temp_assignment, return_counts=True)[1]

                ## update indexes if some clusters too small
                #if min(counts)<CONFIG.cluster.min_spikes:
                    #bigger_cluster_id = np.argmax(counts)
                    #idx_temp_keep = np.where(temp_assignment==bigger_cluster_id)[0]
                    #idx_recovered = idx_recovered[idx_temp_keep]

                    ## this is only requried for MFM?
                    #cluster_idx_keep = np.unique(assignment2[idx_recovered])
                    
                    ## exit if cluster gets decimated below threshld
                    #if idx_recovered.shape[0]<CONFIG.cluster.min_spikes:
                        #return
                
                ## else run the unimodality test
                ## Cat: TODO: this is not perfect, still misses some bimodal distributions
                #else:
                    ## test EM for unimodality
                    #dp_new = test_unimodality(pca_wf_all[idx_recovered], temp_assignment)
                    
                    ## set initial values
                    #if ctr==0:
                        #assignment3 = temp_assignment
                    
                    ## reset values for lower scores
                    #if dp_new <dp:
                        #dp= dp_new
                        #assignment3 = temp_assignment
                    
                    #if ctr>2:
                        ## need to also ensure that we've not deleted any spikes after we
                        ##  saved the last lowest-dp avlue assignment
                        #if assignment3.shape[0] != temp_assignment.shape[0]:
                            #assignment3 = temp_assignment
                        #break
                    
                    #ctr+=1
            
            ## cluster is unimodal, save it: 
            ## Cat: TODO : read these values from file
            #diptest_thresh = 0.995
            #norm_thresh = 1E-4
            ## don' exit on gen0 split ever
            #if (dp> diptest_thresh and gen!=0):
            ##if (dp> diptest_thresh) and (norm>norm_thresh):
                
                ## make sure cluster is on max chan, otherwise omit it
                #if mc != channel and (deconv_flag==False): 
                    #print ("  channel: ", channel, " template has maxchan: ", mc, 
                            #" skipping ...")
                    
                    ## always plot scatter distributions
                    #if gen<20:
                        #split_type = 'mfm-binary - non max chan'
                        #end_flag = 'cyan'                       
                        #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment2[idx_recovered],
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type,
                            #end_flag)
                                
                    #return 
                
                #N= len(assignment_global)
                #if verbose:
                    #print("chan "+str(channel)+' gen: '+str(gen)+" >>> cluster "+
                        #str(N)+" saved, size: "+str(idx_recovered.shape)+"<<<")
                
                #assignment_global.append(N * np.ones(assignment3.shape[0]))
                #spike_index.append(sic_current)
                #templates.append(template)

                ## plot template if done
                #if plotting:
                    #plot_clustering_template(fig, grid, ax_t, gen, N, 
                                         #template, 
                                         #idx_recovered, 
                                         #CONFIG, colors, feat_chans, scale)
                                         
                    ## always plot scatter distributions
                    #if gen<20:
                        ## hack to expand the assignments back out to size of original
                        ## data stream
                        #assignment3 = np.zeros(pca_wf_all[idx_recovered].shape[0],'int32')
                        #split_type = 'mfm-binary, dp: '+ str(round(dp,5))
                        #end_flag = 'green'
                        #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment3,
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type,
                            #end_flag)       
                                        
            ## cluster is not unimodal, split it along binary split and recluster 
            #else:
                ## plot EM labeled data
                #if gen<20 and plotting:
                    #split_type = 'mfm-binary, dp: '+ str(round(dp,5))
                    #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment3,
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type)

                #if verbose:
                    #print("chan "+str(channel)+' gen: '+str(gen)+ 
                                    #" no stable clusters, binary split "+
                                    #str(idx_recovered.shape))

                ## loop over binary split
                #for clust in np.unique(assignment3): #np.where(stability>mfm_threshold)[0]:
                    #idx = np.where(assignment3==clust)[0]
                    
                    #if idx.shape[0]<CONFIG.cluster.min_spikes: 
                        #continue    # cluster too small
                    
                    #if verbose:
                        #print("chan "+str(channel)+' gen: '+str(gen)+
                            #" reclustering cluster"+ 
                            #str(idx.shape))
                    
                    #triageflag = False
                    #alignflag = True
                    #RRR3_noregress_recovery_dynamic_features(channel, 
                         #current_indexes[idx_keep][idx_recovered][idx], 
                         #gen+1, fig, grid, x, ax_t, triageflag, alignflag, 
                         #plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                         #mfm_threshold,  CONFIG, upsample_factor, nshifts, 
                         #assignment_global, spike_index, scale, knn_triage_threshold, 
                         #deconv_flag, templates, min_spikes_local, active_chans)

        
        
#def cluster_iterative_dynamic_features_connected_components(channel, wf, sic, gen, fig, 
         #grid, x, ax_t, triageflag, alignflag, plotting, n_feat_chans, 
         #n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
         #upsample_factor, nshifts, assignment_global, spike_index, 
         #scale, knn_triage_threshold, deconv_flag, templates, 
         #min_spikes_local):
    
    #''' Recursive clusteringn function
        #channel: current channel being clusterd
        #wf = wf_PCA: denoised waveforms (# spikes, # time points, # chans)
        #sic = spike_indexes of spikes on current channel
        #gen = generation of cluster; increases with each clustering step        
    #'''

   ## wf = wf[:15000]

    ## Cat: TODO read from CONFIG File
    #verbose=True
    
    ##if gen>0: return
    ## ************* CHECK SMALL CLUSTERS *************
    ## Exit clusters that are too small
    #if wf.shape[0] < CONFIG.cluster.min_spikes:
        #return
    #if verbose:
        #print("chan/unit "+str(channel)+' gen: '+str(gen)+' # spikes: '+
              #str(wf.shape[0]))
    
    #''' *************************************************       
        #** ALIGN ALL CHANS TO MAX CHAN - LINEAR INTERP **
        #*************************************************
    #'''
    ## align, note: aligning all channels to max chan which is appended to the end
    ## note: max chan is first from feat_chans above, ensure order is preserved
    
    ##if alignflag:
    #if True:
        #if verbose:
            #print ("chan "+str(channel)+' gen: '+str(gen)+" - aligning")

        #mc = wf.mean(0).ptp(0).argmax(0)
        #best_shifts = align_get_shifts(wf[:,:,mc], CONFIG) 
        #wf_align = shift_chans(wf, best_shifts, CONFIG)
        ##wf_align = shift_chans(wf, best_shifts, CONFIG)
    #else:
        #wf_align = wf
    

    #''' ************************************************        
        #****** FIND FEATURE CHANNELS & FEATURIZE *******
        #************************************************
    #'''
    
    #if verbose:
        #print("chan/unit "+str(channel)+' gen: '+str(gen)+' getting feat chans')
    
    ## try to keep more std information    
    ##feat_chans, mc, robust_stds = get_feat_channels_mad4(wf_align, 
    ##                                                     n_feat_chans)
    ## featurize using latest alg
    ##idx_keep, pca_wf = featurize_residual_triage4(wf_align[:,:,feat_chans], 
    ##                                              robust_stds[:,feat_chans])
                                                         
    #feat_chans, mc, robust_stds = get_feat_channels_mad_cat(wf_align, 
                                                            #n_feat_chans)

    #idx_keep, pca_wf = featurize_residual_triage_cat(wf_align, 
                                                     #robust_stds, 
                                                     #feat_chans, 
                                                     #mc, 
                                                     #n_feat_chans)

    #if verbose:
        #print("chan "+str(channel)+' gen: '+str(gen)+", feat chans: "+
                  #str(feat_chans[:n_feat_chans]) + ", max_chan: "+ str(mc))

    #idx_keep = np.arange(pca_wf.shape[0])
    #pca_wf = pca_wf[idx_keep]
 
    #''' ************************************************        
        #******** KNN TRIAGE & PCA #2 *******************
        #************************************************
    #'''
    ## Cat: TODO: this is hardwired now
    ### knn triage outliars; e.g. remove 2%
    #if True:
        #idx_keep_triage = knn_triage(95, pca_wf)
        #idx_keep_triage = np.where(idx_keep_triage==1)[0]
        #if verbose:
            #print("chan "+str(channel)+' gen: '+str(gen) + 
                  #" triaged, remaining spikes "+ 
                  #str(idx_keep[idx_keep_triage].shape[0]))

        #pca_wf = pca_wf[idx_keep_triage]
        
        #idx_keep = idx_keep[idx_keep_triage]

    ## do another min spike check in case triage killed too many spikes
    #if wf.shape[0] < CONFIG.cluster.min_spikes:
        #return
        
    #pca_wf_all = pca_wf.copy()
        
    #''' ************************************************        
        #************** SUBSAMPLE STEP ****************** 
        #************************************************
    #'''
    ## subsmaple 10,000 spikes 
    #if not deconv_flag and (pca_wf.shape[0]> CONFIG.cluster.max_n_spikes):
        #idx_subsampled = np.random.choice(np.arange(pca_wf.shape[0]),
                         #size=CONFIG.cluster.max_n_spikes,
                         #replace=False)
    
        #pca_wf = pca_wf[idx_subsampled]


    #''' ************************************************        
        #************ CLUSTERING STEP *******************
        #************************************************        
    #'''
    ## clustering
    #if verbose:
        #print("chan "+ str(channel)+' gen: '+str(gen)+" - clustering ", 
                                                          #pca_wf.shape)
    #vbParam, assignment = run_mfm3(pca_wf, CONFIG)
    
    
    #''' *************************************************        
        #************* RECOVER SPIKES ********************
        #*************************************************        
    #'''
    ## if we subsampled then recover soft-assignments using above:
    ## Note: for post-deconv reclustering, we can safely cluster only 10k spikes or less
    #if not deconv_flag and (pca_wf.shape[0] <= CONFIG.cluster.max_n_spikes):
        #vbParam2 = deepcopy(vbParam)
        #vbParam2, assignment2 = recover_spikes(vbParam2, 
                                               #pca_wf_all, 
                                               #CONFIG)
    #else:
        #vbParam2, assignment2 = vbParam, assignment

    #idx_recovered = np.where(assignment2!=-1)[0]
    
    #if verbose:
        #print ("chan "+ str(channel)+' gen: '+str(gen)+" - recovered ",
                                            #str(idx_recovered.shape[0]))

    
    #'''*************************************************        
       #*********** SINGLE CLUSTER CASE *****************
       #*************************************************        
    ##'''
                               
    ## First, check again that triage steps above didn't drop below min_spikes
    #if (pca_wf.shape[0] < CONFIG.cluster.min_spikes): # or
                                ##(idx_recovered.shape[0]<min_spikes_local)):
        #print ("EXITING EARLY:  pca_wf.shape: ", pca_wf.shape, 
               #"  idx_recovered.shape: ", idx_recovered.shape)
        #return
        
    ## Case #1: single cluster found
    #if vbParam.rhat.shape[1] == 1:
        ## exclude units whose maximum channel is not on the current 
        ## clustered channel; but only during clustering, not during deconv
        #if mc != channel and (deconv_flag==False): 
            #print ("  channel: ", channel, 
                   #" template has maxchan: ", mc, 
                   #" skipping ...")
            
            ## always plot scatter distributions
            #if gen<20:
                #split_type = 'mfm non_max-chan'
                #end_point = 'cyan'
                #plot_clustering_scatter(fig, 
                            #grid, x, gen,  
                            #assignment2[idx_recovered],
                            #pca_wf_all[idx_recovered], 
                            #vbParam2.rhat[idx_recovered],
                            #channel,
                            #split_type,
                            #end_point)                        
            #return 
        #else:         
            #N= len(assignment_global)
            #if verbose:
                #print("chan "+str(channel)+' gen: '+str(gen)+" >>> cluster "+
                    #str(N)+" saved, size: "+str(wf[idx_recovered].shape)+"<<<")
                #print ("")
            
            #assignment_global.append(N * np.ones(assignment2[idx_recovered].shape[0]))
            #spike_index.append(sic[idx_recovered])
            #templates.append(wf_align[idx_recovered].mean(0))
            
            #print ("   TODO: save cores") 
            ### Save only core of distribution
            ##maha=1
            ##vbParam_temp, assignment_core = recover_spikes(vbParam2, 
                                                           ##pca_wf_all[idx_recovered?!?!], 
                                                           ##CONFIG,
                                                           ##maha)
            ##idx_core = np.where(assignment_core!=-1)[0]
            ##print ("***************  saving core", idx_core.shape, idx_recovered.shape)

            ## plot template if done
            #if plotting:
                #plot_clustering_template(fig, grid, ax_t, gen, N, wf_align, idx_recovered, 
                                         #CONFIG, colors, feat_chans, scale)

                ## always plot scatter distributions
                #if gen<20:
                    #split_type = 'mfm'
                    #end_point = 'red'
                    #plot_clustering_scatter(fig, grid, x, gen,  
                                            #assignment2[idx_recovered],
                                            #pca_wf_all[idx_recovered], 
                                            #vbParam2.rhat[idx_recovered],
                                            #channel,
                                            #split_type,
                                            #end_point)

    ##'''*************************************************        
    ##   *********** MULTIPLE CLUSTERS *******************
    ##   *************************************************        
    ##'''
    ## Case #2: multiple clusters
    #else:
        #ccomps, rhat_cc = get_connected_components(vbParam2.rhat[idx_recovered], 
                                          #assignment2[idx_recovered])
        
        #clusters, sizes = np.unique(ccomps, return_counts = True)
        
        #if verbose:
            #print("chan "+str(channel)+' gen: '+str(gen) + " multiple clusters")

        #if clusters.shape[0]>1:

            ## always plot scatter distributions
            #if plotting and gen<20:
                #split_type = 'mfm split'
                #end_point = 'false'
                #plot_clustering_scatter(fig, grid, x, gen,                             
                            #ccomps, 
                            #pca_wf_all[idx_recovered], 
                            #rhat_cc,
                            #channel,
                            #split_type,
                            #end_point)   
                            
            ## select stable clusters and recluster
            #for clust in clusters:

                #idx = np.where(ccomps==clust)[0]
                
                #if wf[idx_keep][idx].shape[0]<CONFIG.cluster.min_spikes: 
                    #continue    # cluster too small
                
                #if verbose:
                    #print("chan "+str(channel)+' gen: '+str(gen)+
                        #" reclustering connected componenet cluster"+ 
                        #str(wf[idx_keep][idx].shape))
                
                #triageflag = False
                #alignflag = True
                #cluster_iterative_dynamic_features_connected_components(channel, 
                     #wf[idx_keep][idx_recovered][idx], 
                     #sic[idx_keep][idx_recovered][idx], 
                     #gen+1, fig, grid, x, ax_t, triageflag, alignflag, 
                     #plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                     #mfm_threshold,  CONFIG, upsample_factor, nshifts, 
                     #assignment_global, spike_index, scale, knn_triage_threshold, 
                     #deconv_flag, templates, min_spikes_local)


    ##'''*************************************************        
    ##   *********** MULTIPLE CLUSTERS UNSTABLE **********
    ##   *************************************************        
    ##'''
    
        ## connected components returst single cluster - run EM for now
        #else: 

            ## split using EM
            #gmm = GaussianMixture(n_components=2)
            #gmm.fit(pca_wf_all[idx_recovered])
            #labels = gmm.predict_proba(pca_wf_all[idx_recovered])

            #vbParam2.rhat = labels
            #assignment3 = np.zeros(labels.shape[0], 'int32')
            #idx = np.where(labels[:,1]>0.5)[0]
            #assignment3[idx]=1
            
            ## test EM for unimodality
            #dp = test_unimodality(pca_wf_all[idx_recovered], assignment3)
            
            ## Cat: TODO : read this from file
            #diptest_thresh = 0.995
            
            ## save cluster if unimodal
            #if (dp > diptest_thresh):
            ##if (dp> diptest_thresh) and (norm>norm_thresh):
                #assignment3[:]=0

                ## make sure cluster on max chan            
                #if mc != channel and (deconv_flag==False): 
                    #print ("  channel: ", channel, " template has maxchan: ", mc, 
                            #" skipping ...")
                    
                    ## always plot scatter distributions
                    #if gen<20:
                        #split_type = 'EM non_max-chan'
                        #end_point = 'cyan'
                        #plot_clustering_scatter(fig, grid, x, gen, 
                                    #assignment2[idx_recovered], 
                                    #pca_wf_all[idx_recovered],
                                    #vbParam2.rhat,  
                                    #channel,
                                    #split_type,
                                    #end_point)
                                    
                    #return 
                
                #N= len(assignment_global)
                #if verbose:
                    #print("chan "+str(channel)+' gen: '+str(gen)+" >>> EM saved cluster "+
                        #str(N)+" saved, size: "+str(wf[idx_recovered].shape)+"<<<")
                
                #assignment_global.append(N * np.ones(assignment3.shape[0]))
                #spike_index.append(sic[idx_recovered])
                #templates.append(wf_align[idx_recovered].mean(0))

                ## plot template if done
                #if plotting:
                    #plot_clustering_template(fig, grid, ax_t, gen, N, wf_align, idx_recovered, 
                                            #CONFIG, colors, feat_chans, scale)

                    ## always plot scatter distributions
                    #if gen<20:
                        ## hack to expand the assignments back out to size of original
                        ## data stream
                        #split_type = 'EM saved cluster'
                        #end_point = 'red'
                        #plot_clustering_scatter(fig, grid, x, gen, 
                                    #assignment3, 
                                    #pca_wf_all[idx_recovered],
                                    #vbParam2.rhat,  
                                    #channel,
                                    #split_type,
                                    #end_point)

            ## split based on EM binary split; TODO update this function
            #else:
                ## plot EM labeled data
                #if gen<20 and plotting:
                    #split_type = 'EM'
                    #end_point = 'false'
                    #plot_clustering_scatter(fig, grid, x, gen, 
                                    #assignment3, 
                                    #pca_wf_all[idx_recovered],
                                    #vbParam2.rhat,  
                                    #channel,
                                    #split_type,
                                    #end_point)
                                    
                #if verbose:
                    #print("chan "+str(channel)+' gen: '+str(gen)+ 
                                    #" no stable clusters, binary split "+
                                    #str(wf[idx_keep][idx_recovered].shape))

                ## loop over dual split
                #for clust in np.unique(assignment3): #np.where(stability>mfm_threshold)[0]:
                    #idx = np.where(assignment3==clust)[0]
                    
                    #if idx.shape[0]<CONFIG.cluster.min_spikes: 
                        #continue    # cluster too small
                    
                    #if verbose:
                        #print("chan "+str(channel)+' gen: '+str(gen)+
                            #" reclustering cluster"+ 
                            #str(wf[idx_keep][idx].shape))
                    
                    #triageflag = False
                    #alignflag = True
                    #cluster_iterative_dynamic_features_connected_components(channel, 
                         #wf[idx_keep][idx_recovered][idx], 
                         #sic[idx_keep][idx_recovered][idx], 
                         #gen+1, fig, grid, x, ax_t, triageflag, alignflag, 
                         #plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                         #mfm_threshold,  CONFIG, upsample_factor, nshifts, 
                         #assignment_global, spike_index, scale, knn_triage_threshold, 
                         #deconv_flag, templates, min_spikes_local)

        
                    
                    

# def test_unimodality(pca_wf, assignment, max_spikes = 10000):
    
    # n_samples = np.max(np.unique(assignment, return_counts=True)[1])

    # # compute diptest metric on current assignment+LDA
    # #lda = LDA(n_components = 1)
    # #trans = lda.fit_transform(pca_wf[:max_spikes], assignment[:max_spikes])
    # #diptest = dp(trans.ravel())
    
    # ## find indexes of data
    # idx1 = np.where(assignment==0)[0]
    # idx2 = np.where(assignment==1)[0]
    # min_spikes = min(idx1.shape, idx2.shape)[0]

    # # limit size difference between clusters to maximum of 5 times
    # ratio = 1
    # idx1=idx1[:min_spikes*ratio][:max_spikes]
    # idx2=idx2[:min_spikes*ratio][:max_spikes]

    # idx_total = np.concatenate((idx1,idx2))

    # ## run LDA on remaining data
    # lda = LDA(n_components = 1)
    # #print (pca_wf[idx_total].shape, assignment[idx_total].shape) 
    # trans = lda.fit_transform(pca_wf[idx_total], assignment[idx_total])
    # diptest = dp(trans.ravel())

    # ## also compute gaussanity of distributions
    # ## first pick the number of bins; this metric is somewhat sensitive to this
    # # Cat: TODO number of bins is dynamically set; need to work on this
    # #n_bins = int(np.log(n_samples)*3)
    # #y1 = np.histogram(trans, bins = n_bins)
    # #normtest = stats.normaltest(y1[0])

    # return diptest[1] #, normtest[1]


def KMEANS(data, n_clusters):
   from sklearn import cluster, datasets
   clusters = cluster.KMeans(n_clusters, max_iter=1000, n_jobs=-1, random_state = 121)
   clusters.fit(data)
   return clusters.labels_


# def plot_clustering_template(fig, grid, ax_t, gen, N, wf_mean, idx_recovered, CONFIG, 
                             # colors, feat_chans, scale):
        # # plot templates 
        # #wf_mean = wf[idx_recovered].mean(0)
        
        # # plot template
        # temp_clrs = []
        # ax_t.plot(CONFIG.geom[:,0]+
                  # np.arange(-wf_mean.shape[0]//2,wf_mean.shape[0]//2,1)[:,np.newaxis]/1.5, 
                  # CONFIG.geom[:,1] + wf_mean[:,:]*scale, c=colors[N%100], 
                  # alpha=min(max(0.4, idx_recovered.shape[0]/1000.),1))

        # # plot feature channels as scatter dots
        # for i in feat_chans:
             # ax_t.scatter(CONFIG.geom[i,0]+gen, CONFIG.geom[i,1]+N, 
                                                    # s = 30, 
                                                    # color = colors[N%50],
                                                    # alpha=1)


def get_connected_components(rhat, assignment):
    
   KK = rhat.shape[1]
   mask = rhat > 0.1
   connection = np.zeros((KK, KK))
   for k in range(KK):
       temp = rhat[mask[:, k]]
       connection[k] = np.mean(temp[:, [k]]/(temp + temp[:, [k]]), 0) < 0.9
   connection = (connection + connection.T) > 0

   edges = {x: np.where(connection[x])[0] for x in range(KK)}
   groups = list()
   for scc in strongly_connected_components_iterative(np.arange(KK), edges):
       groups.append(np.array(list(scc)))

   ccomps = np.zeros(len(assignment), 'int16')
   rhat_new = np.zeros((rhat.shape[0], len(groups)))
   for k in range(len(groups)):
       for j in groups[k]:
           ccomps[assignment==j] = k

       rhat_new[:, k] = np.sum(rhat[:, groups[k]], axis=1)

   return ccomps, rhat_new
   
   
                            
# def plot_clustering_scatter(fig, grid, x, gen, 
                            # assignment2, 
                            # pca_wf, 
                            # rhat,
                            # channel,
                            # split_type,
                            # end_point='false'):
                                                       
                            
    # if np.all(x[gen]<20) and (gen <20):

        # # add generation index
        # ax = fig.add_subplot(grid[gen, x[gen]])
        # x[gen] += 1

        # # compute cluster memberships
        # mask = rhat>0
        # stability = np.average(mask * rhat, axis = 0, weights = mask)

        # clusters, sizes = np.unique(assignment2, return_counts=True)
        # # make legend
        # labels = []
        # for clust in clusters:
            # patch_j = mpatches.Patch(color = sorted_colors[clust%100], 
                                    # label = "size = "+str(int(sizes[clust])))
            
            # labels.append(patch_j)
        
        # # make list of colors; this could be done simpler
        # temp_clrs = []
        # for k in assignment2:
            # temp_clrs.append(sorted_colors[k])

        # # make scater plots
        # if pca_wf.shape[1]>1:
            # print (pca_wf.shape, len(temp_clrs))
            # ax.scatter(pca_wf[:,0], pca_wf[:,1], 
                # c = temp_clrs, edgecolor = 'k',alpha=0.1)
            
            # # add red dot for converged clusters; cyan to off-channel
            # if end_point!='false':
                # ax.scatter(pca_wf[:,0].mean(), pca_wf[:,1].mean(), c= end_point, s = 2000,alpha=.5)
        # else:
            # for clust in clusters:
                # ax.hist(pca_wf[np.where(assignment2==clust)[0]], 100)

        # # finish plotting
        # ax.legend(handles = labels, fontsize=5)
        # ax.set_title(str(sizes.sum())+", "+split_type+'\nmfm stability '+
                     # str(np.round(stability,2)))
       
    
    
# def get_feat_channels_mad_cat(wf, n_feat_chans):
    # '''  Function that uses MAD statistic like robust variance estimator to select channels
    # '''

    # # compute robust stds over units
    # stds = np.median(np.abs(wf - np.median(wf, axis=0, keepdims=True)), axis=0)*1.4826
   
    # # max per channel
    # std_max = stds.max(0)
    
    # # order channels by largest diptest value
    # feat_chans = np.argsort(std_max)[::-1]
    # #feat_chans = feat_chans[std_max[feat_chans] > 1.2]

    # max_chan = wf.mean(0).ptp(0).argmax(0)
    # #if len(feat_chans) == 0:
    # #    feat_chans = np.array([std_max.argmax()])

    # return feat_chans, max_chan, stds
    

# def robust_stds(data):
    # return np.median(np.abs(data - np.median(data, axis=0, keepdims=True)), axis=0)*1.4826
    
    
# def get_feat_channels_mad4(wf, n_feat_chans):
    # '''  Function that uses MAD statistic like robust variance estimator to select channels
    # '''

    # # compute robust stds over units
    # stds = robust_stds(wf)
   
    # # max per channel
    # std_max = stds.max(0)
    
    # # order channels 
    # feat_chans = np.argsort(std_max)[-n_feat_chans:][::-1]
    # feat_chans = feat_chans[std_max[feat_chans] > 1.05]
    
    # max_chan = wf.mean(0).ptp(0).argmax(0)
    
    # if max_chan not in feat_chans: 
        # feat_chans = np.concatenate((feat_chans, [max_chan]))

    # #if len(feat_chans) == 0:
    # #    feat_chans = np.array([std_max.argmax()])

    # return feat_chans, max_chan, stds
            

# def featurize_residual_triage4(wf, stds, std_th=1.05, resid_th=4, min_rank=2, max_rank=5):
                              
    # print (stds.shape, std_th, min_rank)
    # if np.sum(stds > std_th) >= min_rank:
        # wf_temp = wf[:, stds > std_th]
    # else:
        # temp_threshold = np.sort(stds.reshape(-1))[-(min_rank+1)]
        # wf_temp = wf[:, stds > temp_threshold]

    # n, m = wf_temp.shape
    # rank = np.min([min_rank, n])
    # max_rank = np.min([max_rank, n])
    # pca = PCA_original(n_components=rank)
    # pca.fit(wf_temp)
    # feature_data = pca.transform(wf_temp)
    # resid = pca.inverse_transform(pca.transform(wf_temp)) - wf_temp
    # stds = robust_stds(resid)
    # keep = np.abs(resid).max(1) < resid_th
    # while np.any(stds > std_th):
        # rank += 1
        # pca = PCA_original(n_components=rank)
        # pca.fit(wf_temp)
        # feature_data = pca.transform(wf_temp)
        # resid = pca.inverse_transform(feature_data) - wf_temp
        # stds = robust_stds(resid)
        # keep = np.abs(resid).max(1) < resid_th

    # if rank > max_rank:
        # feature_data = feature_data[:, :max_rank]

    # # convert index back to integers
    # keep = np.where(keep)[0]

    # return keep, feature_data
    
# def featurize_residual_triage_cat(wf, robust_stds, feat_chans, max_chan, 
                                  # n_feat_chans,
                                  # triage_th=0.1, noise_th=4, 
                                  # min_rank=2, max_rank=5):
    
    # # select argrelmax of mad metric greater than trehsold
    # #n_feat_chans = 5

    # n_features_per_channel = 2
    # wf_final = []
    # # select up to 2 features from max amplitude chan;
    # trace = robust_stds[:,max_chan]
    # idx = argrelmax(trace, axis=0, mode='clip')[0]
    # if idx.shape[0]>0:
        # idx_sorted = np.argsort(trace[idx])[::-1]
        # idx_thresh = idx[idx_sorted[:n_features_per_channel]]
        # wf_final.append(wf[:,idx_thresh,max_chan])
        
    # ## loop over all feat chans and select max 2 argrelmax time points as features
    # n_feat_chans = np.min((n_feat_chans, wf.shape[2]))
    # for k in range(n_feat_chans):
        # # don't pick max channel again
        # if feat_chans[k]==max_chan: continue
        
        # trace = robust_stds[:,feat_chans[k]]
        # idx = argrelmax(trace, axis=0, mode='clip')[0]
        # if idx.shape[0]>0:
            # idx_sorted = np.argsort(trace[idx])[::-1]
            # idx_thresh = idx[idx_sorted[:n_features_per_channel]]
            # wf_final.append(wf[:,idx_thresh,feat_chans[k]])
    
    # # Cat: TODO: this may crash if weird data goes in
    # wf_final = np.array(wf_final).swapaxes(0,1).reshape(wf.shape[0],-1)

    # # run PCA on argrelmax points    
    # selected_rank = 5
    # pca = PCA_original(n_components=min(selected_rank, wf_final.shape[1]))
    # pca.fit(wf_final)
    # feature_data = pca.transform(wf_final)
    
    # # convert boolean to integer indexes
    # keep = np.arange(wf_final.shape[0])

    # return keep, feature_data
    
    
    
# def featurize_residual_triage(wf, robust_stds, triage_th=0.1, noise_th=4, 
                    # min_rank=2, max_rank=5):
    
    # # Cat: some suggestions for this step
    # #   1. fix the number of features to some value, e.g. 5-10, not arbitrary number
    # #   2. try to pick single time points from multipole channels rather 
    # #       than multiple time points from the same channel
    # #      - of course within some threshold
    # #      - so we could write an argrelmax-based algorithm step for this
    # #   3. this time point selection should not be independent from
    # #       the pca reconstruction error below;
    # #      - maybe we want to set the 1.2 threshold dynamically based on            
    # #       PCA reconstruction error
    # #   4. A bigger issue is that it's not clear to me why PCA rank is the 
    # #       correct way to find feautrization dimension; 
    # #      - so here it's possible that we might underpresent small clusters
    # #       that are different than 90% of the data
    # #      - does this triage step remove good spikes? (e.g. somas if data is 90% axons)
    # #   5. PCA dimensionality issue: we currently limit to max of 3D for mfm and downstream
    # #       clustering; 
    # #
    
    # # select only time points where rstds are over fixed threshold
    # if np.sum(robust_stds > 1.2) >= min_rank:
        # wf_temp = wf[:, robust_stds > 1.2]
    # # if there aren't at least min # timepoints, find the largest ones
    # else:
        # temp_threshold = np.sort(robust_stds.reshape(-1))[-(min_rank+1)]
        # wf_temp = wf[:, robust_stds > temp_threshold]

    # # - compute difference between waveforms and pca-reconstruction
    # # - compute index of spikes that have max single-time point reconstruction error
    # #   over some treshold (where does this threshold come from?)
    # n, m = wf_temp.shape
    # rank = np.min([min_rank, n])
    # max_rank = np.min([max_rank, n])
    # pca = PCA_original(n_components=rank)
    # pca.fit(wf_temp)
    # feature_data = pca.transform(wf_temp)
    # resid = np.abs(pca.inverse_transform(pca.transform(wf_temp)) - wf_temp)
    # keep = resid.max(1) < noise_th
    
    # # loop over reconstruction error from PCA reconstruction while increasing rank
    # #  until minimum number of spikes are removed
    # while (np.mean(keep) < (1 - triage_th)) and (rank <= max_rank):
        # rank += 1
        # pca = PCA_original(n_components=rank)
        # pca.fit(wf_temp)
        # feature_data = pca.transform(wf_temp)
        # resid = np.abs(pca.inverse_transform(feature_data) - wf_temp)
        # keep =  resid.max(1) < noise_th
    
    # # convert boolean indexes to integer indexes
    # keep = np.where(keep)[0]
    
    # return keep, feature_data
    

# def recover_spikes(vbParam, pca, CONFIG, maha_dist = 1):
    
    # N, D = pca.shape
    # C = 1
    # maskedData = mfm.maskData(pca[:,:,np.newaxis], np.ones([N, C]), np.arange(N))
    
    # vbParam.update_local(maskedData)
    # assignment = mfm.cluster_triage(vbParam, pca[:,:,np.newaxis], D*maha_dist)
    
    # # zero out low assignment vals
    # if True:
        # vbParam.rhat[vbParam.rhat < 0.001] = 0
        # vbParam.rhat = vbParam.rhat/np.sum(vbParam.rhat,
                                         # 1, keepdims=True)
        # assignment = np.argmax(vbParam.rhat, axis=1)
    
    # return vbParam, assignment

# def clustering_annealing():
    # ''' annealing function
        # currently not used; might be needed later
    # ''' 
    
    # pass
    # #print("annealing")
    # #annealing_thresholds = np.arange(0.0, 1.01, 0.01)  # set threshold for annealing clusters together
    # #network_threshold = 0.0 # set very low threshold for assigning spikes to a particular unit
    
    # ##for network_threshold in network_thresholds:
    # #network = np.zeros((vbParam.rhat.shape[1],vbParam.rhat.shape[1]),'float32')

    # ## use compute 
    # #for k in range(vbParam.rhat.shape[1]):
        # #indexes = np.where(vbParam.rhat[:,k]>network_threshold)[0]

        # #temp = vbParam.rhat.copy()
        # #matches = np.average(temp[indexes], axis=0)

        # #network[k]=matches
    
    # ## add 1 to ensure 
    # #network+=1  
    # #network_pd = pd.DataFrame(network)

    # ## search for minimum annealing threshold that genrates 2 groups
    # #for ctr, threshold in enumerate(annealing_thresholds):
        # ## Cat: add 1.0 to correlation matrix ranging -1.0 .. +1.0
        # ## TODO: this may be problematic; find more robust metric of overlap
        # #corr = network_pd.corr() 
        # #links = corr.stack().reset_index()
        # #links.columns = ['var1', 'var2','value']

        # #links_filtered=links.loc[ (links['value'] >= threshold) & (links['var1'] != links['var2']) ]

        # #G=nx.from_pandas_dataframe(links_filtered, 'var1', 'var2')

        # ## find groups
        # #l = list(G.edges())
        # #taken=[False]*len(l)
        # #l=map(set,l)

        # #groups = merge_all(l, taken)
        
        # ## compute # of groups by considering total grouped units vs. all units
        # ## try to find pairing where all units are in 2 gruops, and no isolated clusters
        # #flattened = np.int32([val for sublist in groups for val in sublist])
        # #n_groups = len(groups)+np.max(assignment)+1-len(flattened)
    
        # #if n_groups==2:
            # #print ("found 2 groups, exiting")
            # #break

    # #if n_groups==2:
        # ## loop over 2 groups and recluster
        # #lengths = []
        # #for group in groups: 
            # #lengths.append(np.where(np.in1d(assignment, np.int32(group)))[0].shape[0])
        # #print ("groups: ", lengths)
        
        # #for group in groups: 
            # #idx = np.where(np.in1d(assignment, np.int32(group)))[0]
        
            # #print("reclustering annealed group ", wf[idx_keep][idx].shape, group)
            # #RRR3_noregress(channel, wf[idx_keep][idx], sic[idx_keep][idx], gen+1, fig, grid, 
                 # #False, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
                 # #n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor,
                 # #nshifts, assignment_global, spike_index)
                 

# def regress(wf, mc):
    # template = wf.mean(0)
    # channels = np.where(template.ptp(0)>0)[0]
    # wf_x = wf[:,:,mc]
    # wf_y = wf[:,:, channels]
    # A = np.matmul(np.matmul(wf_y.transpose([2,1,0]), wf_x)/2+np.matmul(wf_x.T,wf_y.transpose([2,0,1]))/2\
                  # ,np.linalg.inv(np.matmul(wf_x.T, wf_x)))
    # residual = wf_y - np.matmul(A, wf_x.T).transpose([2,1,0])
    # return residual



def make_CONFIG2(CONFIG):
    ''' Makes a copy of several attributes of original config parameters
        to be sent into parmap function; original CONFIG can't be pickled;
    '''
    
    # make a copy of the original CONFIG object;
    # multiprocessing doesn't like the methods in original CONFIG        
    CONFIG2 = empty()
    CONFIG2.recordings=empty()
    CONFIG2.resources=empty()
    CONFIG2.deconvolution=empty()
    CONFIG2.data=empty()
    CONFIG2.cluster=empty()
    CONFIG2.cluster.prior=empty()
    CONFIG2.cluster.min_spikes = CONFIG.cluster.min_spikes

    CONFIG2.recordings.spike_size_ms = CONFIG.recordings.spike_size_ms
    CONFIG2.recordings.sampling_rate = CONFIG.recordings.sampling_rate
    CONFIG2.recordings.n_channels = CONFIG.recordings.n_channels
    
    CONFIG2.resources.n_processors = CONFIG.resources.n_processors
    CONFIG2.resources.multi_processing = CONFIG.resources.multi_processing
    CONFIG2.resources.n_sec_chunk = CONFIG.resources.n_sec_chunk

    CONFIG2.data.root_folder = CONFIG.data.root_folder
    CONFIG2.data.geometry = CONFIG.data.geometry
    CONFIG2.geom = CONFIG.geom
    
    CONFIG2.cluster.prior.a = CONFIG.cluster.prior.a
    CONFIG2.cluster.prior.beta = CONFIG.cluster.prior.beta
    CONFIG2.cluster.prior.lambda0 = CONFIG.cluster.prior.lambda0
    CONFIG2.cluster.prior.nu = CONFIG.cluster.prior.nu
    CONFIG2.cluster.prior.V = CONFIG.cluster.prior.V

    CONFIG2.recordings.n_channels = CONFIG.recordings.n_channels
    CONFIG2.neigh_channels = CONFIG.neigh_channels
    CONFIG2.cluster.max_n_spikes = CONFIG.cluster.max_n_spikes
    CONFIG2.merge_threshold = CONFIG.cluster.merge_threshold

    CONFIG2.path_to_output_directory = CONFIG.path_to_output_directory

    CONFIG2.deconvolution.max_spikes = CONFIG.deconvolution.max_spikes

    return CONFIG2




def binary_reader(idx_list, buffer_size, standardized_filename,
                  n_channels):

    # New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]

    data_start = idx_start
    data_end = idx_stop
    offset = idx_local

    # ***** LOAD RAW RECORDING *****
    with open(standardized_filename, "rb") as fin:
        if data_start == 0:
            # Seek position and read N bytes
            recordings_1D = np.fromfile(
                fin,
                dtype='float32',
                count=(data_end + buffer_size) * n_channels)
            recordings_1D = np.hstack((np.zeros(
                buffer_size * n_channels, dtype='float32'), recordings_1D))
        else:
            fin.seek((data_start - buffer_size) * 4 * n_channels, os.SEEK_SET)
            recordings_1D = np.fromfile(
                fin,
                dtype='float32',
                count=((data_end - data_start + buffer_size * 2) * n_channels))

        if len(recordings_1D) != (
              (data_end - data_start + buffer_size * 2) * n_channels):
            recordings_1D = np.hstack((recordings_1D,
                                       np.zeros(
                                           buffer_size * n_channels,
                                           dtype='float32')))
    fin.close()

    # Convert to 2D array
    recording = recordings_1D.reshape(-1, n_channels)
    
    return recording
    

#def mfm_binary_split(muhat, assignment_orig):
    #centers = muhat[:,:,0].T
    #label = AgglomerativeClustering(n_clusters=2).fit(centers).labels_
    #assignment = np.zeros(len(assignment_orig), 'int16')
    #for j in range(2):
        #clusters = np.where(label==j)[0]
        #for k in clusters:
            #assignment[assignment_orig==k] = j
        
    #return assignment
    
    
def mfm_binary_split2(muhat, assignment_orig, cluster_index=None):
    
    centers = muhat[:,:,0].T

    K, D = cetners.reshape
    if cluster_index is None:
        cluster_index = np.arange(K)

    label = AgglomerativeClustering(n_clusters=2).fit(centers).labels_
    assignment = np.zeros(len(assignment_orig), 'int16')
    for j in range(2):
        clusters = cluster_index[np.where(label==j)[0]]
        for k in clusters:
            assignment[assignment_orig==k] = j

    return assignment

def connected_channels(channel_list, ref_channel, neighbors, keep=None):
    if keep is None:
        keep = np.zeros(len(neighbors), 'bool')
    if keep[ref_channel] == 1:
        return keep
    else:
        keep[ref_channel] = 1
        chans = channel_list[neighbors[ref_channel][channel_list]]
        for c in chans:
            keep = connected_channels(channel_list, c, neighbors, keep=keep)
        return keep


# def pca_space_trans(X, pca):
#     pca_len = pca.mean_.shape[0]
#     wf_start = int(0.2 * (pca_len-1))
#     wf_end = -int(0.3 * (pca_len-1))
#     X = X - pca.mean_[wf_start:wf_end]
#     X_transformed = np.dot(X, pca.components_[:,wf_start:wf_end].T)
#     return X_transformed

# def temp_space_trans(X_transformed, pca):
#     pca_len = pca.mean_.shape[0]
#     wf_start = int(0.2 * (pca_len-1))
#     wf_end = -int(0.3 * (pca_len-1))
#     X = np.dot(X_transformed, pca.components_[:,wf_start:wf_end])
#     return X + pca.mean_[wf_start:wf_end]
                           
                           
# def inverse_alignment(den_wf, orig_wf, best_shifts, upsample_factor):
#     den_nspikes, den_waveform_len = den_wf.shape
#     orig_nspikes, orig_waveform_len = orig_wf.shape
        
#     den_wf_up = np.zeros([den_wf.shape[0], (den_waveform_len-1)*upsample_factor+1])
#     orig_wf_up = np.zeros([orig_wf.shape[0], (orig_waveform_len-1)*upsample_factor+1])
#     for i in range(den_wf.shape[0]):
#         den_wf_up[i] = signal.resample(den_wf[i], (den_waveform_len-1)*upsample_factor+1)
#         orig_wf_up[i] = signal.resample(orig_wf[i], (orig_waveform_len-1)*upsample_factor+1)
        
#     wlen = orig_wf_up.shape[1]
#     wf_start = int(.2 * (wlen-1))
#     wf_end = -int(.3 * (wlen-1))
    
#     wf_final = orig_wf_up.copy()
#     for i, s in enumerate(best_shifts):
#         wf_final[i, wf_start-s:wf_end-s] = den_wf_up[i]
        
#     return wf_final[:,::upsample_factor]
    

# def align_temp(wf, upsample_factor, nshifts, CONFIG, ref = None):
    
#     nspikes, waveform_len = wf.shape
    
#     nshifts = (nshifts*upsample_factor)
#     if nshifts%2==0:
#         nshifts+=1
    
#     wf_up = np.zeros([wf.shape[0], (waveform_len-1)*upsample_factor+1])
#     for i in range(wf.shape[0]):
#         wf_up[i] = signal.resample(wf[i], (waveform_len-1)*upsample_factor+1)
    
#     wlen = wf_up.shape[1]
#     wf_start = int(.2 * (wlen-1))
#     wf_end = -int(.3 * (wlen-1))
    
#     wf_trunc = wf_up[:,wf_start:wf_end]
#     wlen_trunc = wf_trunc.shape[1]
    
    
#     if ref is None:
#         ref_upsampled = wf_up.mean(0)
#     else:
#         ref_upsampled = signal.resample(wf[i], (waveform_len-1)*upsample_factor+1)
        
#     ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
#     for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
#         ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]
        
#     bs_indices = np.matmul(wf_trunc[:, np.newaxis, :], ref_shifted).squeeze(1).argmax(1)
#     best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    
#     wf_final = np.zeros([wf.shape[0], wlen_trunc])
#     for i, s in enumerate(best_shifts):
#         wf_final[i] = wf_up[i,-s + wf_start: -s + wf_end]
    
#     return wf_final[:,::upsample_factor], best_shifts, ref_upsampled[::upsample_factor]


# def align_temp_roll(wf_mc, upsample_factor, nshifts, ref = None):

#     nspikes, waveform_len = wf_mc.shape

#     nshifts = (nshifts*upsample_factor)
#     if nshifts%2==0:
#         nshifts+=1

#     wf_mc_up = np.zeros([wf_mc.shape[0], (waveform_len-1)*upsample_factor+1])
#     for i in range(wf_mc.shape[0]):
#         wf_mc_up[i] = signal.resample(wf_mc[i], (waveform_len-1)*upsample_factor+1)

#     wlen = wf_mc_up.shape[1]
    
#     if ref is None:
#         ref_upsampled = wf_mc_up.mean(0)
#     else:
#         ref_upsampled = signal.resample(ref, wlen)
        
#     ref_shifted = np.zeros([wlen, nshifts])

#     for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
#         ref_shifted[:,i] = np.roll(ref_upsampled, -s)

#     bs_indices = np.matmul(wf_mc_up[:, np.newaxis, :], ref_shifted).squeeze(1).argmax(1)
#     best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

#     wf_final_mc = np.zeros([wf_mc.shape[0], wlen])
#     for i, s in enumerate(best_shifts):
#         wf_final_mc[i] = np.roll(wf_mc_up[i], s)

#     return wf_final_mc[:,::upsample_factor], best_shifts, ref_upsampled[::upsample_factor]

# def inverse_roll(wf_mc, best_shifts, upsample_factor):
    
#     nspikes, waveform_len = wf_mc.shape

    
#     wf_mc_up = np.zeros([wf_mc.shape[0], (waveform_len-1)*upsample_factor+1])
#     for i in range(wf_mc.shape[0]):
#         wf_mc_up[i] = signal.resample(wf_mc[i], (waveform_len-1)*upsample_factor+1)
#     wlen = wf_mc_up.shape[1]
#     wf_final_mc = np.zeros([wf_mc.shape[0], wlen])
#     for i, s in enumerate(best_shifts):
#         wf_final_mc[i] = np.roll(wf_mc_up[i], -s)
        
#     return wf_final_mc[:,::upsample_factor]


def gather_clustering_result(chunk_dir, out_dir, units):

    '''load clustering results
    '''

    print ("  gathering clustering results")

    # convert clusters to templates; keep track of weights
    templates = []
    spike_indexes = []

    # this list tracks the post-clustering ids before the cleaning step
    unit_recluster_list=[]

    # this step loads either cluster or unit data so that both clustering 
    # and post-deconv unit-based reclusteirng can work with same routine
    if out_dir == 'cluster':
        fname = chunk_dir+'/channel_'
    else:
        fname = chunk_dir+'/unit_'

    for unit in units:
        data = np.load(fname+str(unit).zfill(6)+'.npz')
        temp_temp = data['templates']
        if (temp_temp.shape[0]) !=0:
            templates.append(temp_temp)
            temp = data['spiketime']
            unit_recluster_list.append(len(temp))
            for s in range(len(temp)):
                spike_times = temp[s]
                spike_indexes.append(spike_times)
        else:
            unit_recluster_list.append(0)

    spike_indexes = np.array(spike_indexes)    
    templates = np.vstack(templates)

    # Clip tempaltes and only work with default spike_size templates going forward
    # spike_size_default = int(CONFIG.recordings.spike_size_ms*
                                  # CONFIG.recordings.sampling_rate//1000*2+1)
    # offset = templates.shape[1]-spike_size_default
    # templates = templates[:, offset//2:offset//2+spike_size_default]
            
    # rearange spike indees from id 0..N
    spike_train = np.zeros((0,2),'int32')
    for k in range(spike_indexes.shape[0]):    
        temp = np.zeros((spike_indexes[k].shape[0],2),'int32')
        temp[:,0] = spike_indexes[k]
        temp[:,1] = k
        spike_train = np.vstack((spike_train, temp))

    print ("  templates loaded: ", templates.shape)
    np.save(chunk_dir  + '/templates_post_'+out_dir+'.npy', templates)
    np.save(chunk_dir  + '/spike_train_post_'+out_dir+'.npy', spike_train)
    np.save(chunk_dir + '/unit_recluster_list.npy', unit_recluster_list)

    return templates, spike_train

def recompute_templates(templates, spike_train, CONFIG, chunk_dir, out_dir):

    print('Recomputing templates..')

    # get shape
    n_units, n_timepoints, n_channels = templates.shape

    # raw recording
    fname_recording = os.path.join(CONFIG.path_to_output_directory,
                               'preprocess', 'standardized.bin')

    # make a list of spike time
    spike_times = {k: [] for k in range(n_units)}
    for j in range(spike_train.shape[0]):
        tt, ii = spike_train[j]
        spike_times[ii].append(tt)

    # gather input arguments
    args_in = []
    for unit in range(n_units):
        fname_out = os.path.join(
            chunk_dir,
            "template_unit_{}.npy".format(str(unit).zfill(6)))
        if not os.path.exists(fname_out):
            args_in.append([
                fname_recording,
                spike_times[unit],
                n_channels,
                n_timepoints,
                fname_out])

    # run computing function
    if CONFIG.resources.multi_processing:
        #p = mp.Pool(processes = self.CONFIG.resources.n_processors)
        #p.map_async(Cluster, args_in).get(988895)
        #p.close()
        parmap.map(recompute_templates_parallel,
                   args_in,
                   processes=CONFIG.resources.n_processors,
                   pm_pbar=True)
    else:
        for unit in range(len(args_in)):
            recompute_templates_parallel(args_in[unit])

    # gather all info
    templates_new = np.zeros((n_units, n_timepoints, n_channels))
    for unit in range(n_units):
        fname_out = os.path.join(
            chunk_dir,
            "template_unit_{}.npy".format(str(unit).zfill(6)))
        templates_new[unit] = np.load(fname_out)

    np.save(chunk_dir  + '/templates_post_'+out_dir+'_recomputed.npy', templates_new)

    return templates_new

def recompute_templates_parallel(arg_in):
    fname_recording = arg_in[0]
    spike_times = np.array(arg_in[1])
    n_channels = arg_in[2]
    spike_size = arg_in[3]
    fname_out = arg_in[4]
    
    max_spikes = 1000
    if len(spike_times) > max_spikes:
        spike_times = np.random.choice(a=spike_times,
                                       size=max_spikes,
                                       replace=False)

    wf, _ = binary_reader_waveforms(
        fname_recording, n_channels,
        spike_size, spike_times - spike_size//2)

    ref_template = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'ref_template.npy')))

    cut_edge = (spike_size - len(ref_template))//2

    mc = np.mean(wf, axis=0).ptp(0).argmax()
    best_shifts = align_get_shifts_with_ref(
        wf[:, cut_edge:-cut_edge, mc],
        ref_template)
    wf = shift_chans(wf, best_shifts)
    template = np.median(wf, 0)

    np.save(fname_out,template)

def global_merge_max_dist(templates, spike_indexes, CONFIG, chunk_dir, out_dir):

    ''' Function that cleans low spike count templates and merges the rest
    '''

    weights = np.zeros(templates.shape[0], 'int32')
    unique_ids, unique_weights = np.unique(spike_indexes[:,1], return_counts=True)
    weights[unique_ids] = unique_weights

    ''' ************************************************
        **************** CLEAN TEMPLATES  **************
        ************************************************
    '''
    # Clean templates; return index of templates that were deleted
    print ("Cleaning templates ")
    temp_fname = os.path.join(chunk_dir,'templates_post_'+out_dir+'_post_cleaning.npy')
    if os.path.exists(temp_fname)==False:
        templates, spike_train, weights, idx_kept = clean_templates(templates,
                                                        spike_indexes,
                                                        weights,
                                                        CONFIG)

        np.save(os.path.join(chunk_dir,'templates_post_'+out_dir+
                            '_post_cleaning.npy'), templates)
        np.save(os.path.join(chunk_dir,'spike_train_post_'+out_dir+
                            '_post_cleaning.npy'), spike_train)
        np.save(os.path.join(chunk_dir,'idx_kept.npy'), idx_kept)

    else:
        templates = np.load(os.path.join(chunk_dir,'templates_post_'+
                                out_dir+'_post_cleaning.npy'))
        spike_train = np.load(os.path.join(chunk_dir,'spike_train_post_'+out_dir+
                                '_post_cleaning.npy'))
        idx_kept = np.load(os.path.join(chunk_dir,'idx_kept.npy'))

    print("  "+out_dir+ " templates/spiketrain before merge: ",
                                        templates.shape, spike_train.shape)

    print ("\nPost-clustering merge... ")

    merge_dir = os.path.join(chunk_dir,'merge')
    if not os.path.isdir(merge_dir):
        os.makedirs(merge_dir)

    ''' ************************************************
        ********** COMPUTE SIMILARITY METRICS **********
        ************************************************
    '''
    # ************** GET SIM_MAT ****************
    abs_max_file = os.path.join(merge_dir,'abs_max_vector_post_cluster.npy')
    if os.path.exists(abs_max_file)==False:

        # run merge algorithm
        sim_mat = abs_max_dist(templates, CONFIG, merge_dir)
        np.save(abs_max_file, sim_mat)

    else:
        sim_mat = np.load(abs_max_file)
   
    temp_fname = os.path.join(merge_dir,'templates_post_'+out_dir+
                                '_post_merge.npy')
                                
    if os.path.exists(temp_fname)==False:

        merge_pairs = np.vstack(np.where(sim_mat))
        unit_killed = np.zeros(templates.shape[0], 'bool')
        for j in range(merge_pairs.shape[1]):
            k1, k2 = merge_pairs[:,j]
            if weights[k1] > weights[k2]:
                unit_killed[k2] = True
            else:
                unit_killed[k1] = True
        unit_keep = np.where(~unit_killed)[0]

        templates = templates[unit_keep]
        weights = weights[unit_keep]

        spike_train_new = np.copy(spike_train)
        spike_train_new = spike_train_new[
            np.in1d(spike_train[:,1], unit_keep)]
        dics = {unit: ii for ii, unit in enumerate(unit_keep)}
        for j in range(spike_train_new.shape[0]):
            spike_train_new[j,1] = dics[spike_train_new[j,1]]
        spike_train = spike_train_new

        np.save(merge_dir+'/templates_post_'+out_dir+'_post_merge.npy', templates)
        np.save(merge_dir+'/spike_train_post_'+out_dir+'_post_merge.npy', spike_train)


        # Cat: TODO: why are we saving these twice?
        np.save(chunk_dir+'/templates_post_'+out_dir+'_post_merge.npy', templates)
        np.save(chunk_dir+'/spike_train_post_'+out_dir+'_post_merge.npy', spike_train)

    else:
        
        templates = np.load(os.path.join(merge_dir,'templates_post_'+out_dir+
                            '_post_merge.npy'))
        spike_train = np.load(os.path.join(merge_dir,'spike_train_post_'+out_dir+
                                '_post_merge.npy'))

    print("  "+out_dir+" templates/spike train after merge : ", 
                                templates.shape, spike_train.shape)
    
    ''' ************************************************
        ************ REMOVE COLLISION UNITS ************
        ************************************************
    '''
    temp_deconv_dir = os.path.join(chunk_dir,'temp_deconv')
    if not os.path.isdir(temp_deconv_dir):
        os.makedirs(temp_deconv_dir)
        
    temp_fname = os.path.join(chunk_dir,'templates_post_'+out_dir+
                        '_post_merge_post_collision_kill.npy')
    if os.path.exists(temp_fname)==False:
        templates, spike_train, idx_kept = deconvolve_template(
            templates, spike_train, CONFIG, temp_deconv_dir)
        weights = weights[idx_kept]

        mad_collision_dir = os.path.join(chunk_dir,'mad_collision')
        if not os.path.isdir(mad_collision_dir):
            os.makedirs(mad_collision_dir)
            
        templates, spike_train, weights = mad_based_unit_kill(
            templates, spike_train, weights, CONFIG, mad_collision_dir)

        np.save(chunk_dir+'/templates_post_'+out_dir+'_post_merge_post_collision_kill.npy',
                templates)
        np.save(chunk_dir+'/spike_train_post_'+out_dir+'_post_merge_post_collision_kill.npy',
                spike_train)
    else:
        templates = np.load(os.path.join(chunk_dir,'templates_post_'+out_dir+
                            '_post_merge_post_collision_kill.npy'))
        spike_train = np.load(os.path.join(chunk_dir,'spike_train_post_'+
                            out_dir+'_post_merge_post_collision_kill.npy'))

    print("  "+out_dir+ " templates/spiketrain after removing collisions: ",
                                        templates.shape, spike_train.shape)
    # clip templates
    # Cat: TODO: This value needs to be read from CONFIG
    shift_allowance = 10
    templates = templates[:, shift_allowance:-shift_allowance]

    # save data for clustering step
    if out_dir=='cluster':
        fname = os.path.join(CONFIG.path_to_output_directory,
                    'spike_train_cluster.npy')
        np.save(fname, spike_train)

        fname = os.path.join(CONFIG.path_to_output_directory,
                            'templates_cluster.npy')
        np.save(fname, templates)

    return spike_train, templates
    

def mad_based_unit_kill(templates, spike_train, weights, CONFIG, save_dir):

    print('\nMad based Collision Detection...')

    ids=[]
    for k in range(templates.shape[0]):
        fname = os.path.join(save_dir,'unit_{}.npz'.format(k))
        if os.path.exists(fname)==False:
            ids.append(k)        
 
    standardized_filename = os.path.join(CONFIG.path_to_output_directory,
                                         'preprocess', 
                                         'standardized.bin')
    if CONFIG.resources.multi_processing:
        parmap.map(mad_based_unit_kill_parallel, ids, templates, spike_train,
                   standardized_filename, save_dir,
                   processes=CONFIG.resources.n_processors,
                   pm_pbar=True)                
    else:
        for id_ in ids:
            mad_based_unit_kill_parallel(id_, templates, spike_train,
                                     standardized_filename,
                                     save_dir)

    # reload all saved data
    pairs = []
    for k in range(templates.shape[0]):
        fname = os.path.join(save_dir,'unit_{}.npz'.format(k))
        tmp = np.load(fname)
        if tmp['kill']:
            pairs.append([k, tmp['unit_matched']])

    unique_pairs = []
    non_unique_pairs = []    
    for x, y in pairs:
        if [y, x] not in pairs:
            unique_pairs.append([x,y])
        elif [y, x] not in non_unique_pairs:
            non_unique_pairs.append([x,y])

    collision_units = [k for k, k_ in unique_pairs]

    for x, y in non_unique_pairs:
        if weights[x] > weights[y]:
            collision_units.append(x)
        else:
            collision_units.append(y)
    collision_units = np.array(collision_units)

    print(' MAD collision detector: {} units removed'.format(len(collision_units)))

    # which ones are beign kept
    idx_kept = np.arange(templates.shape[0])
    idx_kept = idx_kept[~np.in1d(idx_kept, collision_units)]
    np.save(os.path.join(save_dir,'idx_kept.npy'), idx_kept)

    # update templates
    templates = templates[idx_kept]
    weights = weights[idx_kept]

    # update spike train
    spike_train_new = np.copy(spike_train)
    spike_train_new = spike_train_new[
        np.in1d(spike_train[:,1], idx_kept)]
    dics = {unit: ii for ii, unit in enumerate(idx_kept)}
    for j in range(spike_train_new.shape[0]):
        spike_train_new[j,1] = dics[spike_train_new[j,1]]
    spike_train = spike_train_new
    
    return templates, spike_train, weights

def mad_based_unit_kill_parallel(unit, templates, spike_train, standardized_filename, save_dir,
                        mad_gap=0.8, mad_gap_breach=4, up_factor=8, residual_max_norm=1.2):
    jitter = 1
    n_units, n_times, n_channels = templates.shape
    visch = np.where(templates[unit].ptp(axis=0) > 2.)[0]
    if len(visch) == 0:
        visch = np.array([templates[unit].ptp(axis=0).argmax()])

    spt = spike_train[spike_train[:, 1] == unit, 0] - n_times//2
    wf =  binary_reader_waveforms(
        standardized_filename=standardized_filename,
        n_channels=n_channels,
        n_times=n_times, spikes=spt,
        channels=None)[0]
    wf = wf[:, :, visch]

    ref_template = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'ref_template.npy')))

    mc = np.mean(wf, axis=0).ptp(0).argmax()
    cut_edge = (n_times - len(ref_template))//2
    best_shifts = align_get_shifts_with_ref(
        wf[:, cut_edge:-cut_edge, mc],
        ref_template, nshifts=jitter)
    wf = shift_chans(wf, best_shifts)

    t_mad = np.median(
            np.abs(np.median(wf, axis=0)[None, :, :] - wf), axis=0)

    col_loc = t_mad > mad_gap
    #mean_clipped = mean[:, jitter:-jitter]
    #delta_minus = np.abs(mean_clipped - mean[:, :-2])
    #delta_plus = np.abs(mean_clipped - mean[:, 2:])
    #emad = MAD_bernoulli_two_uniforms(delta_minus, delta_plus)

    t_mad = np.median(
            np.abs(np.median(wf, axis=0)[None, :, :] - wf), axis=0)

    col_loc = t_mad > mad_gap
    
    kill = False
    max_diff = None
    unit_matched = None
    if np.sum(col_loc) > mad_gap_breach:
        idx_no_target = np.arange(n_units)
        idx_no_target = np.delete(idx_no_target, unit)
        residual, unit_matched = run_deconv(templates[unit],
                                            templates[idx_no_target], up_factor)

        if unit_matched is not None:
            unit_matched = idx_no_target[unit_matched]
            visch_matched = np.where(templates[unit_matched].ptp(axis=0) > 2.)[0]
            visch_all = np.union1d(visch, visch_matched)

            ii = np.where(col_loc.sum(0)>mad_gap_breach)[0]
            cc = np.unique(visch[ii])
            visch_all = visch_all[~np.in1d(visch_all, cc)]

            if len(visch_all) > 0:
                max_diff = np.max(np.abs(residual[:, visch_all]))
            else:
                max_diff = 0

            if max_diff < residual_max_norm:
                kill = True
    
    fname = os.path.join(save_dir,'unit_{}.npz'.format(unit))
    np.savez(fname,
             kill=kill,
             unit_matched=unit_matched,
             max_diff=max_diff)

def MAD_bernoulli_two_uniforms(a, b):
    """Analytical MAD of a addition of uniform drawn according to Bernoulli rv.

    i ~ Bernoulli(0.5)
    x ~ uniform(0., a)
    y ~ uniform(0., b)
    z = ix + (1-i)y
    This functions computes the analytical median absolute deviation
    of the resulting z random variable.

    params:
    -------
    a: int or np.ndarray
    b: int or np.ndarray

    returns:
    --------
    float or np.ndarray
    """
    same_sign_indicator = np.array(a * b > 0.).astype(np.float)
    a, b = np.abs(a), np.abs(b)
    # Make sure that a , b on all elements of the arrays
    concat = np.array([a, b])
    a, b = concat.min(axis=0), concat.max(axis=0)
    # Median of the random variable
    a_inv, b_inv = 1. / a, 1. / b
    a_inv_plus_b_inv = a_inv + b_inv
    m = 1. / a_inv_plus_b_inv
    t1 = (a - m) * a_inv_plus_b_inv
    t2 = (2 * m - a) * (a_inv / 2. + b_inv)
    # remainder is t3 = (b - 2*m) * 1/b / 2. where t1 + t2 + t3 = 1

    # handle two where t1 less than or greater than 0.5
    indicator = np.array(t1 > 0.5).astype(np.float)
    opt1 = (.5 / t1) * (a - m)
    opt2 = (a - m) + (0.5 - t1) / t2 * (2 * m - a)
    out = indicator * opt1 + (1 - indicator) * opt2
    # handle two cases where a and b have different signs
    return same_sign_indicator * out + (1 - same_sign_indicator) * m

def deconvolve_template(templates, spike_train, CONFIG, save_dir):

    print('\nDeconvolve on Templates...')

    ids=[]
    for k in range(templates.shape[0]):
        fname = (save_dir+'/unit_{}.npz'.format(k))
        if os.path.exists(fname)==False:
            ids.append(k)        
 
    up_factor=8
    residual_max_norm=1.2
    if CONFIG.resources.multi_processing:
        parmap.map(deconv_template_parallel, ids, templates, up_factor,
                   residual_max_norm, save_dir,
                   processes=CONFIG.resources.n_processors,
                   pm_pbar=True)                
    else:
        for id_ in ids:
            deconv_template_parallel(id_, templates,
                                     up_factor, residual_max_norm,
                                     save_dir)

    # reload all saved data
    res=[]
    for k in range(templates.shape[0]):
        fname = (save_dir+'/unit_{}.npz'.format(k))
        res.append(np.load(fname)['collision'])
    collision_units = np.hstack(res)

    print(' Templates Deconvolution: {} units removed'.format(sum(collision_units)))

    # which ones are beign kept
    idx_kept = np.where(~collision_units)[0]
    np.save(save_dir+'/idx_kept.npy', idx_kept)

    # update templates
    templates = templates[idx_kept]

    # update spike train
    spike_train_new = np.copy(spike_train)
    spike_train_new = spike_train_new[
        np.in1d(spike_train[:,1], idx_kept)]
    dics = {unit: ii for ii, unit in enumerate(idx_kept)}
    for j in range(spike_train_new.shape[0]):
        spike_train_new[j,1] = dics[spike_train_new[j,1]]
    spike_train = spike_train_new
    
    return templates, spike_train, idx_kept

def deconv_template_parallel(unit, templates, up_factor, residual_max_norm, save_dir):

    n_units = templates.shape[0]

    it = 0
    max_it = 3
    run = True
    collision = False
    data = templates[unit]
    idx_no_target = np.arange(n_units)
    idx_no_target = np.delete(idx_no_target, unit)
    deconv_units = []
    while it < max_it and run:
        residual, best_fit_unit = run_deconv(data, templates[idx_no_target], up_factor)
        if best_fit_unit is None:
            run = False
        elif np.max(np.abs(residual)) < residual_max_norm:
            best_fit_unit = idx_no_target[best_fit_unit]
            deconv_units.append(best_fit_unit)
            collision = True
            run = False
        else:
            data = residual
            best_fit_unit = idx_no_target[best_fit_unit]
            deconv_units.append(best_fit_unit)
            idx_no_target = np.arange(n_units)
            idx_no_target = np.delete(idx_no_target,
                                      np.hstack((deconv_units, unit)))

            it += 1
    
    fname = (save_dir+'/unit_{}.npz'.format(unit))
    np.savez(fname,
             collision=collision,
             residual=residual,
             deconv_units = deconv_units)


def run_deconv(data, templates, up_factor):
    n_units, n_times, n_chans = templates.shape

    # norm of templates
    norm_temps = np.square(templates).sum(axis=(1,2))

    # calculate objective function in deconv
    temps = np.flip(templates, axis=1)
    obj = np.zeros((n_units, n_times))
    for j in range(n_units):
        for c in range(n_chans):
            obj[j] += np.convolve(temps[j,:,c], data[:,c], 'same')
    obj = 2*obj - norm_temps[:, np.newaxis]

    if np.max(obj) > 0:
        best_fit_unit = np.max(obj, axis=1).argmax()
        best_fit_time = obj[best_fit_unit].argmax()
        shift = best_fit_time - n_times//2
        shifted_temp = np.roll(templates[best_fit_unit], shift, axis=0)

        up_temp = scipy.signal.resample(
            x=shifted_temp,
            num=n_times * up_factor,
            axis=0)

        up_shifted_temps = up_temp[(np.arange(0,n_times)[:,None]*up_factor + np.arange(up_factor))]
        up_shifted_temps = np.concatenate((up_shifted_temps, np.roll(up_shifted_temps, shift=1, axis=0)), 1)
        if shift > 0:
            up_shifted_temps[:shift+1] = 0
        elif shift < 0:
            up_shifted_temps[shift-1:] = 0
        elif shift == 0:
            up_shifted_temps[[0,-1]] = 0

        idx_best_fit = np.max(np.abs(data[:,None] - up_shifted_temps), (0,2)).argmin()
        residual = data - up_shifted_temps[:,idx_best_fit]
    else:
        residual = data
        best_fit_unit = None

    return residual, best_fit_unit


def get_denoised_templates(templates, CONFIG):

    ref_template = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'ref_template.npy')))
    pca_main_mean = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_main_mean.npy')))
    pca_sec_mean = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_sec_mean.npy')))
    pca_main_components = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_main_components.npy')))
    pca_sec_components = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_sec_components.npy')))

    n_templates, n_times, n_channels = templates.shape
    n_times_ref = len(ref_template)
    shift_allowance = int((n_times - n_times_ref)//2)


    # step 1. get fits with ref_templates
    fits = get_fits(templates, ref_template)


    # step 2. get shifts
    max_chans = templates.ptp(1).argmax(1)
    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 1)
    shifts = np.zeros((n_templates, n_channels), 'int32')
    for k in range(n_templates):
        shifts[k] = get_shifts(fits[k], max_chans[k], neighbors)
    shifts -= int(fits.shape[2]//2)
    shifts = -shifts


    # step 3. shift templates
    reshaped_templates = np.reshape(np.transpose(templates, (0, 2, 1)), (-1, n_times))
    reshaped_shifts = np.reshape(shifts, -1)
    aligned_reshaped_templates = shift_chans(reshaped_templates, reshaped_shifts)
    aligned_templates = np.reshape(
        aligned_reshaped_templates, [n_templates, n_channels, n_times]).transpose((0, 2, 1))


    # step 4. denoise templates
    aligned_denoised_templates = denoise(aligned_templates,
        pca_main_mean, pca_main_components,
        pca_sec_mean, pca_sec_components)


    # step 5. shift aligned templates back to its original location
    reshaped_aligned_denoised_templates = np.reshape(
        aligned_denoised_templates.transpose((0, 2, 1)), [-1, n_times])
    reshaped_denoised_templates = shift_chans(reshaped_aligned_denoised_templates, -reshaped_shifts)
    denoised_templates = np.reshape(
        reshaped_denoised_templates, [n_templates, n_channels, n_times]).transpose((0, 2, 1))


    # step 6. undo denoising on channels with small ptp
    ptp_threshold = 0.5
    denoised_templates = undo_denoise(
        templates, denoised_templates, shifts, ptp_threshold)

    return denoised_templates, shift_allowance


def remove_collision_templates(templates, neigh_channels):

    collision = np.zeros(templates.shape[0], 'bool')
    for k in range(templates.shape[0]):

        th = np.min((templates[k].min()*0.1, -0.5))
        th_big = -2

        local_mins = argrelmin(templates[k], axis=0, order=2)
        val = templates[k][local_mins[0], local_mins[1]]
        times = local_mins[0][val < th]
        chans = local_mins[1][val < th]
        val = val[val < th]

        no_dendrites = templates[k][:, chans].max(0) < -templates[k][:, chans].min(0)
        if np.any(no_dendrites == True):
            times = times[no_dendrites]
            chans = chans[no_dendrites]
            val = val[no_dendrites]

            data_in = (np.hstack((times[:, np.newaxis], chans[:, np.newaxis])), np.abs(val), None, None)
            idx_survive, _ = deduplicate(data_in, neigh_channels)

            if np.sum(val[idx_survive] < th_big) > 1:
                collision[k] = 1

    return templates[~collision], collision
    

def get_fits(templates, ref_template):

    n_templates, n_times, n_channels = templates.shape
    n_times_ref = len(ref_template)

    shift_allowance = int((n_times - n_times_ref)//2)

    reshaped_templates = np.reshape(templates.transpose(0, 2, 1), (-1, n_times))
    
    shifts = np.arange(-shift_allowance, shift_allowance+1)
    fits = np.zeros((reshaped_templates.shape[0], 2*shift_allowance+1))
    for j in range(len(shifts)):
        fits[:, j] =np.sum(
            reshaped_templates[:, j:(j+n_times_ref)]*ref_template[np.newaxis], axis=1)

    #fits = np.abs(fits)
    return np.reshape(fits, (n_templates, n_channels, -1))


def get_shifts(fits, max_channel, neighbors, window=3):

    n_channels, n_shifts = fits.shape

    shifts = np.ones(n_channels, 'int32')*n_shifts
    mid_point = int(n_shifts//2)
    shifts[max_channel] = get_shift_per_channel(fits[max_channel], 
        mid_point, window)

    current_chans = np.array([max_channel])
    while np.any(shifts == n_shifts):
        neigh_chans = np.any(neighbors[current_chans], axis=0)
        next_chans = np.where(np.logical_and(neigh_chans, shifts==n_shifts))[0]

        for c in next_chans:
            shifted_nearby_chans = np.where(np.logical_and(neighbors[c], shifts<n_shifts))[0]
            mid_point = int(np.mean(shifts[shifted_nearby_chans]))
            shifts[c] = get_shift_per_channel(fits[c], mid_point, window)

        current_chans = next_chans

    return shifts


def get_shift_per_channel(fits, mid_point, window):

    start_index = np.max((mid_point - window, 0))
    end_index = np.min((mid_point+window+1, len(fits)))

    return start_index + np.argmax(fits[start_index:end_index])
    


def denoise(templates, pca_main_mean, pca_main_components, pca_sec_mean, pca_sec_components):

    n_templates, n_times, n_channels = templates.shape
    shift_allowance = (n_times - len(pca_main_mean))//2
    max_channels = templates.ptp(1).argmax(1)

    norms = np.linalg.norm(templates, axis=1)
    normalized_templates = templates/norms[:, np.newaxis]
    denoised_templates = np.copy(normalized_templates)

    for k in range(n_templates):

        # denoise max channels
        main_chan_template = normalized_templates[k][
        shift_allowance:-shift_allowance][:, max_channels[k]][np.newaxis]
        denoised_temp = pca_denoise(main_chan_template, pca_main_mean, pca_main_components)[0]
        denoised_templates[k][shift_allowance:-shift_allowance][:, max_channels[k]] = denoised_temp

        # denoise non max channels
        off_chans_idx = np.arange(n_channels) != max_channels[k]
        sec_chan_templates = normalized_templates[k][shift_allowance:-shift_allowance][:, off_chans_idx].T
        denoised_temp = pca_denoise(sec_chan_templates, pca_sec_mean, pca_sec_components)
        denoised_templates[k][shift_allowance:-shift_allowance][:, off_chans_idx] = denoised_temp.T

    return denoised_templates*norms[:, np.newaxis]

def pca_denoise(data, pca_mean, pca_components):
    data_pca = np.matmul(data-pca_mean, pca_components.T)
    return np.matmul(data_pca, pca_components)+pca_mean

def undo_denoise(templates, denoised_templates, shifts, ptp_threshold):

    n_templates, n_times, n_channels = templates.shape

    ptps = templates.ptp(1)
    for k in range(n_templates):
        undo_channels = np.where(ptps[k] < ptp_threshold)[0]
        denoised_templates[k][:, undo_channels] = np.copy(templates[k][:, undo_channels])

    return denoised_templates


def merge_templates(templates, weights):

    largest_unit = np.argmax(weights)
    mc = templates[largest_unit].ptp(0).argmax()

    wf_out = align_mc_templates(templates, mc, spike_padding=15,
                                upsample_factor = 5, nshifts = 15)

    weights = np.float32(weights) 
    weights /= np.sum(weights)
    idx_remove = weights < np.max(weights)*0.05
    weights[idx_remove] = 0

    return np.average(templates, axis=0, weights=weights)


# def centre_templates(templates, spike_train_cluster, CONFIG, spike_padding, spike_width):
#     ''' Function centres templates to the mean of all tempaltes on max channel;
#         - deals with duplicate detected templates
#     '''
#
#     print ("  centering and clipping templates ")
#     print (templates.shape)
#     # grab max channel templates and make mean
#     max_chan_templates = []
#     #print (templates.shape)
#     templates = templates.swapaxes(0,2)
#     for k in range(templates.shape[2]):
#         temp = templates[:,:,k]
#         max_chan = temp.ptp(1).argmax(0)
#         trace = temp[max_chan]
#         max_chan_templates.append(trace/trace.ptp(0))
#
#     mean = np.array(max_chan_templates).mean(0)
#     min_loc_all = np.argmin(mean)
#
#     # compute shifts for all templates based on max channel trough location
#     shifts = []
#     indexes = []
#     templates_centred = []
#     for k in range(len(max_chan_templates)):
#         min_loc = np.argmin(max_chan_templates[k])
#         shift = min_loc_all-min_loc
#         if abs(shift)<=spike_padding:
#             indexes.append(k)
#             trace = templates[:,
#                 max_chan_templates[k].shape[0]//2-spike_width//2-shift:
#                 max_chan_templates[k].shape[0]//2+spike_width//2+1-shift,k]
#             templates_centred.append(trace)
#
#     templates_centred = np.array(templates_centred).swapaxes(1,2)
#     #np.save('/media/cat/1TB/liam/49channels/data1_allset/tmp/cluster/chunk_000000/templates_centred2.npy'  , templates_centred)
#
#
#     # delete spike indexes for templates that are shifted outside the window
#     # Note: this could possibly delete good templates; but very unlikely
#     spike_train_cluster_new = []
#     for ctr,k in enumerate(indexes):
#         temp = np.where(spike_train_cluster[:,1]==k)[0]
#         temp_train = spike_train_cluster[temp]
#         temp_train[:,1]=ctr
#         spike_train_cluster_new.append(temp_train)
#
#     spike_train_cluster_new = np.vstack(spike_train_cluster_new)
#
#     return templates_centred, spike_train_cluster_new


def abs_max_dist(temp, CONFIG, merge_dir):
        
    ''' Compute absolute max distance using denoised templates
        Distances are computed between absolute value templates, but
        errors are normalized
    '''

    # Cat: TODO: don't compare all pair-wise templates, but just those
    #           with main chan + next 3-6 largest shared channels
    #      - not sure if this is necessary though, may be already fast enough
    print ("  Computing merge matrix")
    print ("  temp shape (temps, times, chans):" , temp.shape)
    
    #dist_max = np.zeros((temp.shape[0],temp.shape[0]), 'float32')
    
    # make ids to be loaded
    ids=[]
    for k in range(temp.shape[0]):
        fname = (merge_dir+'/unit_'+str(k)+'.npy')
        if os.path.exists(fname)==False:
            ids.append(k)        
    '''
    if CONFIG.resources.multi_processing:
        parmap.map(parallel_abs_max_dist, ids, temp, merge_dir,
                         processes=CONFIG.resources.n_processors,
                         pm_pbar=True)                
    else:
        for id_ in ids:
            parallel_abs_max_dist(id_, temp, merge_dir)
    '''
    up_factor = 16
    if CONFIG.resources.multi_processing:
        parmap.map(parallel_abs_max_dist2, ids, temp, up_factor, merge_dir,
                         processes=CONFIG.resources.n_processors,
                         pm_pbar=True)                
    else:
        for id_ in ids:
            parallel_abs_max_dist2(id_, temp, up_factor, merge_dir)


    # reload all saved data
    res=[]
    for k in range(temp.shape[0]):
        fname = (merge_dir+'/unit_'+str(k)+'.npy')
        res.append(np.load(fname))

    dist_max=np.vstack(res)
    return dist_max

def parallel_abs_max_dist(id1, temp, merge_dir):
    
    fname = merge_dir+'/unit_'+str(id1)+'.npy'
    if os.path.exists(fname):
        return
    
    # Cat: TODO: read spike_padding from CONFIG
    spike_padding = 15
    
    dist_max = np.zeros(temp.shape[0],'float32')

    # Cat: TODO read the overlap # of chans from CONFIG
    # number of largest amplitude channels to search for overlap
    threshold = 0.3*temp[id1].ptp(0).max()
    vis_chans = temp.ptp(1) > threshold
    max_chans_id1 = np.where(vis_chans[id1])[0]
    for id2 in range(id1+1,temp.shape[0],1):
        max_chans_id2 = np.where(vis_chans[id2])[0]
        if np.intersect1d(max_chans_id1, max_chans_id2).shape[0]==0: 
            continue

        # load both templates into an array and align them to the max
        temps = []
        template1 = temp[id1]
        temps.append(template1)
        ptp_1=template1.ptp(0).max(0)

        template2 = temp[id2]
        ptp_2=template2.ptp(0).max(0)
        temps.append(template2)
                    
        if ptp_1>=ptp_2:
            mc = template1.ptp(0).argmax(0)
        else:
            mc = template2.ptp(0).argmax(0)

        temps = np.array(temps)

        # ref template
        ref_template = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'ref_template.npy')))
            
        # get shifts
        upsample_factor = 16
        nshifts = 21
        best_shifts = align_get_shifts_with_ref(
            temps[:, 10:-10, mc],
            ref_template, upsample_factor, nshifts)

        wf_out = shift_chans(temps, best_shifts)
        wf_out = wf_out[:,10:-10]

        # compute distances and noramlize by largest template ptp
        # note this makes sense for max distance, but not sum;
        #  for sum should be dividing by total area
        diff = np.max(np.abs(np.diff(wf_out, axis=0)))
        diff_rel = diff/(max(ptp_1,ptp_2))

        # compute max distance
        if diff<1 or diff_rel<0.15:
            dist_max[id2] = 1.0
    
    np.save(fname, dist_max)
    #return dist_max
    
def parallel_abs_max_dist2(unit, templates, up_factor, merge_dir, residual_max_norm=1.2):
    
    fname = merge_dir+'/unit_'+str(unit)+'.npy'
    if os.path.exists(fname):
        return

    n_units, n_times, n_chans = templates.shape
    
    max_ptp = templates[unit].ptp(0).max()
    threshold = 0.5*max_ptp
    vis_chans = templates.ptp(1) > threshold
    vis_chan_unit1 = np.where(vis_chans[unit])[0]
    candidates = np.where(np.any(vis_chans[:, vis_chan_unit1], axis=1))[0]
    candidates = candidates[candidates > unit]

    temps = np.flip(templates[candidates], axis=1)
    data = templates[unit]
    max_diff = np.zeros(len(candidates))
    for j in range(len(candidates)):
        obj = np.zeros(n_times)
        for c in range(n_chans):
            obj += np.convolve(temps[j,:,c], data[:,c], 'same')

        best_fit_time = obj.argmax()
        shift = best_fit_time - n_times//2
        shifted_temp = np.roll(templates[candidates[j]], shift, axis=0)

        up_temp = scipy.signal.resample(
            x=shifted_temp,
            num=n_times * up_factor,
            axis=0)

        up_shifted_temps = up_temp[(np.arange(0,n_times)[:,None]*up_factor + np.arange(up_factor))]
        up_shifted_temps = np.concatenate((up_shifted_temps, np.roll(up_shifted_temps, shift=1, axis=0)), 1)
        if shift > 0:
            up_shifted_temps[:shift+1] = 0
        elif shift < 0:
            up_shifted_temps[shift-1:] = 0
        elif shift == 0:
            up_shifted_temps[[0,-1]] = 0

        idx_best_fit = np.max(np.abs(data[:,None] - up_shifted_temps), (0,2)).argmin()
        residual = data - up_shifted_temps[:,idx_best_fit]

        max_diff[j] = np.max(np.abs(residual))

    ptp_candidates = templates[candidates].ptp(1).max(1)
    ptp_candidates[ptp_candidates < max_ptp] = max_ptp
    max_diff_rel = max_diff/ptp_candidates

    dist_max = np.zeros(n_units)
    dist_max[candidates[np.logical_or(max_diff < residual_max_norm, max_diff_rel < 0.15)]] = 1

    np.save(fname, dist_max)

def global_merge_all_ks(chunk_dir, recording_chunk, CONFIG, min_spikes,
                         out_dir, units):

    ''' Function that cleans low spike count templates and merges the rest 
    '''
   
    n_channels = CONFIG.recordings.n_channels

    # convert clusters to templates; keep track of weights
    templates = []
    templates_std = []
    weights = []
    spike_ids = []
    spike_indexes = []
    channels = np.arange(n_channels)
    tmp_loc = []
    
    # Cat: TODO: load from config
    n_pca = 3
    
    #ctr_id = 0 
    # Cat: TODO: make sure this step is correct
    if out_dir == 'cluster':
        
        for channel in range(CONFIG.recordings.n_channels):
#         for channel in [0]:
            data = np.load(chunk_dir+'/channel_{}.npz'.format(channel))
            templates.append(data['templates'])
            templates_std.append(data['templates_std'])
            weights.append(data['weights'])
           
            #spike_index = data['spike_train_merged']
            temp = data['spike_index']
            for s in range(len(temp)):
                spike_times = temp[s][:,0]
                spike_indexes.append(spike_times)
    else: 
        for unit in units:
            data = np.load(chunk_dir+'/unit_'+str(unit).zfill(6)+'.npz')
            templates.append(data['templates'])
            templates_std.append(data['templates_std'])
            weights.append(data['weights'])
           
            #spike_index = data['spike_train_merged']
            temp = data['spike_index']
            for s in range(len(temp)):
                spike_times = temp[s][:,0]
                spike_indexes.append(spike_times)

    spike_indexes = np.array(spike_indexes)
    templates = np.vstack(templates)
    templates_std = np.vstack(templates_std)
    weights = np.hstack(weights)
    
    # rearange spike indees from id 0..N
    spike_train = np.zeros((0,2),'int32')
    for k in range(spike_indexes.shape[0]):    
        temp = np.zeros((spike_indexes[k].shape[0],2),'int32')
        temp[:,0] = spike_indexes[k]
        temp[:,1] = k
        spike_train = np.vstack((spike_train, temp))
    spike_indexes = spike_train

    print("  "+out_dir+ " templates/spiketrain before merge/cutoff: ", templates.shape, spike_indexes.shape)

    np.save(CONFIG.path_to_output_directory+'/'+out_dir+'/templates_post_'+out_dir+'_before_merge_before_cutoff.npy', templates)
    np.save(CONFIG.path_to_output_directoryr+'/'+out_dir+'/spike_train_post_'+out_dir+'_before_merge_before_cutoff.npy', spike_indexes)
    #print (np.unique(spike_indexes[:,1]))


    ''' ************************************************
        ********** COMPUTE SIMILARITY MATRIX ***********
        ************************************************
    '''

    # first denoise/smooth all templates in global template space;
    # then use temps_PCA for cos_sim_test; this removes many bumps
    spike_width = templates.shape[1]
    temps_stack = np.swapaxes(templates, 1,0)
    temps_stack = np.reshape(temps_stack, (temps_stack.shape[0], 
                             temps_stack.shape[1]*temps_stack.shape[2])).T
    _, temps_PCA, pca_object = PCA(temps_stack, n_pca)

    temps_PCA = temps_PCA.reshape(templates.shape[0],templates.shape[2],templates.shape[1])
    temps_PCA = np.swapaxes(temps_PCA, 1,2)

    # ************** GET SIM_MAT ****************
    # initialize cos-similarty matrix and boolean version 
    # Cat: TODO: it seems most safe for this matrix to be recomputed 
    #            everytime whenever templates.npy file doesn't exist
    # Cat: TODO: this should be parallized, takes several mintues for 49chans, 
    #            should take much less
    
    cos_sim_file = (CONFIG.path_to_output_directory+'/'+out_dir+'/cos_sim_vector_post_'+out_dir+'.npy')
    if os.path.exists(cos_sim_file)==False:
        sim_temp = calc_cos_sim_vector(temps_PCA, CONFIG)
        sim_temp[np.diag_indices(sim_temp.shape[0])] = 0
        sim_temp[np.tril_indices(sim_temp.shape[0])] = 0
        
        np.save(cos_sim_file, sim_temp)

    else:
        sim_temp = np.load(cos_sim_file)        
        sim_temp[np.diag_indices(sim_temp.shape[0])] = 0
        sim_temp[np.tril_indices(sim_temp.shape[0])] = 0
        
    # ************** RUN KS TEST ****************
    global_merge_file = (CONFIG.path_to_output_directory+'/'+out_dir+'/merge_matrix_post_'+out_dir+'.npz')
    if os.path.exists(global_merge_file)==False:
        sim_mat = run_KS_test_new(sim_temp, spike_indexes, recording_chunk, 
                                        CONFIG, chunk_dir, pca_object,
                                        spike_width, plotting = False)
        
        np.savez(global_merge_file, sim_mat = sim_mat)
        sim_mat = sim_mat

    else:
        data = np.load(global_merge_file)
        sim_mat = data['sim_mat']
                    

    ''' ************************************************
        ************* MERGE SELECTED UNITS *************
        ************************************************
    '''

    # compute connected nodes and sum spikes over them
    #G = nx.from_numpy_array(sim_mat_sum)
    G = nx.from_numpy_array(sim_mat)
    final_spike_indexes = []
    final_template_indexes = []
    for i, cc in enumerate(nx.connected_components(G)):
        final_template_indexes.append(list(cc))
        sic = np.zeros(0, dtype = int)
        for j in cc:
            idx = np.where(spike_indexes[:,1]==j)[0]
            sic = np.concatenate([sic, spike_indexes[:,0][idx]])
        temp = np.concatenate([sic[:,np.newaxis], i*np.ones([sic.size,1],dtype = int)],axis = 1)
        final_spike_indexes.append(temp)

    final_spike_train = np.vstack(final_spike_indexes)
    
    # recompute tmp_loc from weighted templates
    tmp_loc = []
    templates_final = []
    weights_final = []
    for t in final_template_indexes:
        # compute average weighted template and find peak
        idx = np.int32(t)

        # compute weighted template
        weighted_average = np.average(templates[idx],axis=0,weights=weights[idx])
        templates_final.append(weighted_average)

    # convert templates to : (n_channels, waveform_size, n_templates)
    templates = np.float32(templates_final)
    templates = np.swapaxes(templates, 2,0)
     
    np.save(CONFIG.path_to_output_directory+'/'+out_dir+'/templates_post_'+out_dir+'_post_merge_pre_cutoff.npy', templates)
    np.save(CONFIG.path_to_output_directory+'/'+out_dir+'/spike_train_post_'+out_dir+'_post_merge_pre_cutoff.npy', final_spike_train)

    print("  "+out_dir+" templates/spike train after merge, pre-spike cutoff: ", templates.shape, final_spike_train.shape)

    ''' ************************************************
        ************* DELETE TOO FEW SPIKES ************
        ************************************************
    '''

    # Cat: TODO: Read this threshold from CONFIG
    #            Maybe needs to be in fire rate (Hz) not absolute spike #s
    final_spike_train_cutoff = []
    del_ctr = []
    tmp_loc = []
    final_templates_cutoff = []
    print ("units: ", np.unique(final_spike_train[:,1]).shape[0])
    ctr = 0
    for unit in np.unique(final_spike_train[:,1]):
        idx_temp = np.where(final_spike_train[:,1]==unit)[0]
        if idx_temp.shape[0]>=min_spikes:
            temp_train = final_spike_train[idx_temp]
            temp_train[:,1]=ctr
            final_spike_train_cutoff.append(temp_train)
            final_templates_cutoff.append(templates[:,:,unit])

            max_chan = templates[:,:,unit].ptp(1).argmax(0)
            tmp_loc.append(max_chan)

            ctr+=1
                
    final_spike_train_cutoff = np.vstack(final_spike_train_cutoff)
    final_templates_cutoff = np.array(final_templates_cutoff).T

    # these are saved outside
    np.save(CONFIG.path_to_output_directory+'/'+out_dir+'/templates_post_'+out_dir+'_post_merge_post_cutoff.npy', final_templates_cutoff)
    np.save(CONFIG.path_to_output_directoryr+'/'+out_dir+'/spike_train_post_'+out_dir+'_post_merge_post_cutoff.npy', final_spike_train_cutoff)

    print("  "+out_dir+" templates & spike train post merge, post-spike cutoff: ", final_templates_cutoff.shape, final_spike_train_cutoff.shape)
    
    return final_spike_train_cutoff, tmp_loc, final_templates_cutoff

   
   
def calc_cos_sim_vector(temp, CONFIG):
    """
        Calculates cosine similarity for close enough templates
        input:
        ------
        temp: list of all templates
        
        output:
        -------
        sim_temp: cosine similarity for groups of close enough templates
    """
    print ("  calculating cos sim vector (todo: parallelize)...")
    
    ## find the top 5 channels for all templates
    top_chan = temp.ptp(1).argsort(1)[:,-5:][:,::-1]
    
    ## find the main channel for all templates
    mc = temp.ptp(1).argmax(1)
    
    #initialize cosine similarity matrix
    sim_temp = np.zeros([temp.shape[0], temp.shape[0]])
    
    #for temp_id in tqdm_notebook(range(temp.shape[0])):
    for temp_id in tqdm.tqdm(range(temp.shape[0])):
        
        ## select all templates whose main channel is in the 
        ## top channels for unit temp_id and for pairs which 
        ## haven't been compared
        idx = np.where(np.logical_and(np.in1d(mc, top_chan[temp_id]), sim_temp[temp_id] == 0.0))[0]
        idx = np.union1d(idx, np.asarray([temp_id]))
        if idx.size == 1:
            #print ("skipping cos sim")
            continue
            
        ## find max chan for alignment
        chan = temp[idx].mean(0).ptp(0).argmax(0)
        ## align templates
        aligned_temp = align_mc(temp[idx], chan, CONFIG, 5, 15)
        
        ## find feature channels (wherever the ptp of the template of template is >1SU)
        feat_channels = np.where(aligned_temp.mean(0).ptp(0)>1)[0]
        sim = (aligned_temp[:, np.newaxis, :, feat_channels] * aligned_temp[:, :, feat_channels]).sum((2,3))
        norm = np.sqrt((aligned_temp[:, :, feat_channels]**2).sum((1,2)))
        sim = sim/(norm[:,np.newaxis] * norm)
        sim_temp[idx[:,np.newaxis],idx] = sim
        
    return sim_temp

def calc_ks_dist(A, B, lin): 
    cdf2 = np.zeros(lin.shape[0]) 
    cdf = np.zeros(lin.shape[0])
    for i,j in enumerate(lin):
        cdf[i] = (A<j).sum()/A.size
        cdf2[i] = (B<j).sum()/B.size
    print(cdf.size)
    return np.abs(cdf - cdf2).max()
    

def run_KS_test_new(sim_temp, spike_train, recording_chunk, CONFIG, 
                    path_dir, pca_object, spike_width, plotting = False):
    """
        input:
        ------
        sim_temp: cosine similarity matrix for all templates (n_units x n_units np.ndarray)
        spike_train: Spike train post clustering
        CONFIG: config file
        plotting: binary flag to enable plotting
        
        output:
        -------
        sim_mat: adjacency matrix joining all pairs which are unimodal according to the dip test
        dip: dip test values for all pairwise comparisons
    
    """
    
    ## initialize parameters
    # Cat: TODO must export these params to CONFIG
    cos_sim_threshold = 0.85
    
    # Cat: TODO this must be linked to min_spikes threshold cutoff in previous
    #       step;
    N = 100
    n_chan = CONFIG.recordings.n_channels
    wf_start = 0
    wf_end = 31
    upsample_factor = 5
    nshifts = 15
    num_max_channels = 0
    num_mad_channels = 2
    num_pca_components = 3
    knn_triage_threshold = 90
    dip_test_threshold = 0.95
    
    #spike_train[:,0]+=200
    ## find every pair for which cos similarity is > some threshold
    row, column = np.where(sim_temp > cos_sim_threshold)
    
    ## initialize dip statistic and similarity matrix
    dip = np.zeros(row.size)
    sim_mat = np.zeros(sim_temp.shape, dtype = bool)
    print (sim_temp.shape)
    
    #for j in tqdm_notebook(range(row.size)):
    
    row_list = []
    for k in range(row.size):
        row_list.append([row[k], column[k]])
    
    args_in = []
    for k in range(len(row_list)):
        if os.path.exists(path_dir+'/pair_'+
                        str(row_list[k][0])+"_"+str(row_list[k][1])+'.npz')==False:
            args_in.append([
                [row_list[k], k],
                spike_train,
                N,
                sim_mat,
                dip,
                CONFIG,
                upsample_factor,
                nshifts,
                num_max_channels,
                num_mad_channels,
                wf_start, wf_end,
                num_pca_components,
                knn_triage_threshold,
                dip_test_threshold,
                path_dir,
                pca_object,
                spike_width])

    # run merge in parallel
    if len(args_in)>0:
        if CONFIG.resources.multi_processing:
        #if False:
            p = mp.Pool(processes = CONFIG.resources.n_processors)
            with tqdm.tqdm(total=len(args_in)) as pbar:
                for i, _ in tqdm.tqdm(enumerate(p.imap_unordered(parallel_merge, args_in))):
                    pbar.update()
        else:
            for arg_in in args_in:
                parallel_merge(arg_in)
                
    print ("... finished merge step...")
    
    sim_mat_final = np.zeros(sim_temp.shape, dtype = bool)
    for k in range(len(row_list)):
        data = np.load(path_dir+'/pair_'+str(row_list[k][0])+"_"+
                        str(row_list[k][1])+'.npz')
        sim_mat_final+=data['sim_mat']
        
    return sim_mat_final

def parallel_merge(data):
    
    pair = data[0][0]
    j = data[0][1]
    spike_train = data[1]
    N = data[2]
    sim_mat = data[3]
    dip = data[4]
    CONFIG = data[5]
    upsample_factor = data[6]
    nshifts = data[7]
    num_max_channels = data[8]
    num_mad_channels = data[9]
    wf_start = data[10]
    wf_end = data[11]
    num_pca_components = data[12]
    knn_triage_threshold = data[13]
    dip_test_threshold = data[14]
    path_dir = data[15]
    pca_object = data[16]
    spike_width = data[17]
    
    n_sample_spikes = 100
    
    #for j in range(pair_list):
    #pair = data[0]#[j]
    #j = data[1]
    #print(" comparing pair: ", pair)
    
    ## find all events belonging to pair[0] and pair[1]
    idx = np.where(np.in1d(spike_train[:,1], pair))[0]
    
    ## find all events belonging to pair[0]
    idx1 = np.where(spike_train[idx,1] == pair[0])[0]
    
    ## find all events belonging to pair[1]
    idx2 = np.where(spike_train[idx,1] == pair[1])[0]
    
    
    
    ## choose N events from pair; also ensure that we have sufficient
    #   spikes and no edge cases are occuring
    ## read waveforms
    # Cat: TODO: spike_size must be read from CONFIG or will crash
    re = RecordingExplorer(os.path.join(CONFIG.path_to_output_directory, 'preprocess',
                                        'standardized.bin'),
                   path_to_geom = CONFIG.data.root_folder+CONFIG.data.geometry, 
                   spike_size = spike_width//2, 
                   neighbor_radius = 100, 
                   dtype = 'float32',
                   n_channels = CONFIG.recordings.n_channels, 
                   data_order = 'samples')
                   
    ctr=0
    while True:
        idx1 = np.random.choice(idx1, min(N, idx1.size))
        idx2 = np.random.choice(idx2, min(N, idx2.size))
        
        idx = np.concatenate([idx[idx1], idx[idx2]])
        idx1 = np.where(spike_train[idx,1] == pair[0])[0]
        idx2 = np.where(spike_train[idx,1] == pair[1])[0]
        
        wf = re.read_waveforms(spike_train[idx,0])
        
        # Cat: TODO: this will eventually crash; need to save blank sim_mat for pair
        if wf.shape[0]==(N*2):
            break
        else:
            ctr+=1
        
        if ctr>5:
            print ("pair: ",pair, " --->>> too few spikes in cluster... need to fix this...<<<---")
            quit()


    # Cat: TODO: this routine can crash often whenever there are less than 2 x N spikes
    #       chosen; Need to fix this a bit better;
    #       One option, make N = min_spikes value - 2 or 3 spikes

    # denoise waveforms before clustering 
    template_ = wf.mean(0)
    max_chan = template_.ptp(0).argmax(0)
    
    #temps_max_chans = template_[:,max_chan]
    #_, wf_PCA = PCA(wf_PCA,num_pca_components)
    
    #print (wf.shape)
    wf_temp = np.swapaxes(wf, 1,2)
    wf_temp = np.reshape(wf_temp, (-1, wf_temp.shape[2]))
    #print (wf_temp.shape)
    X = pca_object.transform(wf_temp)
    wf_temp_PCA = pca_object.inverse_transform(X)
    
    wf_temp_PCA = wf_temp_PCA.reshape(wf.shape[0],wf.shape[2],wf.shape[1])
    wf_PCA = np.swapaxes(wf_temp_PCA, 1,2)
            
    # Cat: TODO: recording_chunk is a global variable; 
    #            this might cause problems eventually
    #data_start = 0
    #spike_size = 30
    #print (recording_chunk.shape)
    #print (spike_train)
    #wf = load_waveforms_from_memory_merge(recording_chunk, data_start, 
    #                                offset, spike_train[idx,0], 
    #                                spike_size)
    
    ## align waveforms
    align_wf = align_mc(wf_PCA, wf_PCA.mean(0).ptp(0).argmax(), CONFIG, upsample_factor, nshifts)
    
    ## create data to get feature channels by concatenating the templates of the two clusters
    to_get_feat = np.concatenate([align_wf[idx1].mean(0, keepdims =True), align_wf[idx2].mean(0, keepdims =True)], axis = 0)
    
    ## get feature channels using only the templates
    feat_channels, max_chans, mad_chans = get_feat_channels_union(to_get_feat, num_max_channels, num_mad_channels)
    
    # exit if no overlapping feature channels
    if len(feat_channels)==0:
        np.savez(path_dir+'/pair_'+str(pair[0])+"_"+str(pair[1]), 
             sim_mat=sim_mat, 
             dip= dip)
        return 0
        
    data_in = align_wf[:, wf_start:wf_end][:,:,feat_channels].swapaxes(1,2).reshape(align_wf.shape[0],-1)
    
    ## PCA on concatenated waveforms
    pca_wf, pca_reconstruct, _ = PCA(data_in, num_pca_components)
    
    ##triage to remove outliers to remove compression issues
    idx_keep = knn_triage(knn_triage_threshold, pca_wf)
    data_in = data_in[idx_keep]
    
    ## run PCA on triaged data
    pca_wf, pca_reconstruct, _ = PCA(data_in, num_pca_components)
    idx1 = spike_train[idx,1][idx_keep] == pair[0]
    idx2 = spike_train[idx,1][idx_keep] == pair[1]
    
    ## run LDA on remaining data
    lda = LDA(n_components = 1)
    
    trans = lda.fit_transform(pca_wf, spike_train[idx,1][idx_keep])
    
    ## run diptest
    diptest = dp(trans.ravel())
    dip[j] = diptest[1]
    
    if dip[j] > dip_test_threshold:
        #print(" pair: ", pair, '--> merged <--')
        sim_mat[pair[0], pair[1]] = sim_mat[pair[1], pair[0]]  = True
        
        np.savez(path_dir+'/pair_'+str(pair[0])+"_"+str(pair[1]), 
             sim_mat=sim_mat, 
             dip= dip)
        
        return 1

    else:     
        np.savez(path_dir+'/pair_'+str(pair[0])+"_"+str(pair[1]), 
             sim_mat=sim_mat, 
             dip= dip)

        # return no merge
        return 0

    

def get_feat_channels_union(templates, n_max_chans, n_mad_chans):
    """
        input:
        ------
        
        templates: pair of templates for feature channel calculation (2, R, C)
        n_max_chans: number of max channels (int)
        n_mad_chans: number of mad channels (int)
        
        output:
        -------
        
        feat_chans: feature channels (union of max and mad channels)
        max_chans: max channels
        mad_chans: mad channels
        
        
    """
    temp1 = templates[0]
    temp2 = templates[1]
    ptp1 = temp1.ptp(0)
    ptp2 = temp2.ptp(0)
    
    ## rank channels based on peak to peak values and choose top n_max_chans
    rank_amp_chans = (temp1+temp2).ptp(0).argsort()
    max_chans = rank_amp_chans[::-1][:n_max_chans]
    
    ## select all channels where either of the templates is > 2 peak to peak
    chan_indexes_2SU = np.where(np.logical_or(ptp1>2, ptp2>2))[0]
    
    ## rank channels based on maximum mad values and select top n_mad_chans
    mad_chans = chan_indexes_2SU[robust.mad(templates[:,:,chan_indexes_2SU], axis = 0).max(0).argsort()[::-1]][:n_mad_chans]
    
    feat_chans = np.union1d(max_chans, mad_chans)
    
    return feat_chans, max_chans, mad_chans
    
   
   
    
def chunk_merge(chunk_dir, channels, CONFIG):
    
    ''' Function that cleans noisy templates and merges the rest    
        
        1. local merge step: looks only at tempaltes with same
        max chan and tries to merge
        
        2. neighbour merge step: look at tempaltes on nearby chans
        and tries to merge 
    
    '''
    
    n_channels = CONFIG.recordings.n_channels

    # convert clusters to templates; keep track of weights
    templates = []
    weights = []
    spike_ids = []
    spike_indexes = []
    channels = np.arange(n_channels)
    tmp_loc = []
    for channel in channels:
        data = np.load(chunk_dir+'/channel_{}.npz'.format(channel), encoding='latin1')
        templates.append(data['templates'])
        weights.append(data['weights'])
        
        spike_index = data['spike_index']
        for s in spike_index:
            spike_indexes.append(s[:,0])
    
    spike_indexes = np.array(spike_indexes)        
    templates = np.vstack(templates)
    weights = np.hstack(weights)

    # delete clusters < 300 spikes
    # Cat: TODO: min_spikes should be re-read from CONFIG
    min_spikes = 300
    idx_delete = []
    for s in range(len(spike_indexes)):
        if spike_indexes[s].shape[0]<min_spikes:
            idx_delete.append(s)
    
    templates = np.delete(templates,idx_delete,axis=0)
    weights = np.delete(weights,idx_delete,axis=0)
    spike_indexes = np.delete(spike_indexes,idx_delete,axis=0)

    # compute merge matrix
    global_merge_file = (chunk_dir+ '/merge_matrix.npz')
    if os.path.exists(global_merge_file)==False:
        # initialize cos-similarty matrix and boolean version 
        sim_mat = np.zeros((templates.shape[0], templates.shape[0]), 'bool')    
        sim_temp = np.zeros((templates.shape[0], templates.shape[0]), 'float32')
        
        print ("Computing global merge over templates: ", templates.shape)
        if CONFIG.resources.multi_processing:
            n_processors = CONFIG.resources.n_processors
            indexes = np.array_split(np.arange(templates.shape[0]), 
                                max(n_processors, templates.shape[0]/20))
            res = parmap.map(
                calc_global_merge_parallel,
                indexes,
                templates,
                sim_mat,
                sim_temp,
                CONFIG,            
                processes=n_processors,
                pm_pbar=True)
        else:
            res = []
            for k in range(len(templates)):
                temp = calc_global_merge_parallel(
                    [k], 
                    templates,
                    sim_mat,
                    sim_temp,
                    CONFIG)
                res.append(temp)    
        
        # sum all matrices
        sim_mat_sum = np.zeros(sim_mat.shape, 'bool')
        sim_temp_sum = np.zeros(sim_temp.shape, 'float32')
        for k in range(len(res)):
            sim_mat_sum+=res[k][0]
            sim_temp_sum+=res[k][1]

        np.savez(global_merge_file, sim_mat=sim_mat_sum, sim_temp = sim_temp_sum)

    else:
        data = np.load(global_merge_file, encoding='latin1')
        sim_mat_sum = data['sim_mat']        
        sim_mat_temp = data['sim_temp']        

    # ********************************************************
    # compute connected nodes and sum spikes over them
    G = nx.from_numpy_array(sim_mat_sum)
    final_spike_indexes = []
    final_template_indexes = []
    for i, cc in enumerate(nx.connected_components(G)):
        final_template_indexes.append(list(cc))
        sic = np.zeros(0, dtype = int)
        for j in cc:
            sic = np.concatenate([sic, spike_indexes[j]])
        final_spike_indexes.append(np.concatenate([sic[:,np.newaxis], i*np.ones([sic.size,1],dtype = int)],axis = 1))

    # keep final spike train as stacked list, easier to parse on other end
    final_spike_train = final_spike_indexes
    
    # recompute tmp_loc from weighted templates
    tmp_loc = []
    templates_final= []
    weights_final = []
    for t in final_template_indexes:
        # compute average weighted template and find peak
        idx = np.int32(t)
        weighted_average = np.average(templates[idx],axis=0,weights=weights[idx])
        templates_final.append(weighted_average)
       
        # find max chan and append
        max_chan = weighted_average.ptp(0).argmax(0)
        tmp_loc.append(max_chan)
    
        # add weights of all templates for merged template
        weights_final.append(weights[idx].sum(0))

    # convert templates to : (n_channels, waveform_size, n_templates)
    templates = np.float32(templates_final)
    templates = np.swapaxes(templates, 2,0)

    # save chunk data
    np.savez(chunk_dir+'/channels_merged.npz',
             spike_train_merged = final_spike_train,
             tmp_loc = tmp_loc,
             templates = templates,
             weights = weights_final)        
        
# def local_merge(channel, spike_index, sic, wf, CONFIG):
    # ''' Function that cleans noisy templates and merges the rest    
    # '''
    
    # plotting = True
    
    # n_chans = wf.shape[2]

    # CONFIG.cluster.merge_threshold = 1.0

    # # Cat: TODO: delete noisy/messy/lowSNR/few spike templates
    
    # # make templates on all chans for each cluster from raw waveforms 
    # temp = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
    # for t in range(len(spike_index)): 
        # idx = np.in1d(sic[:,0], spike_index[t][:,0])
        # temp[t] = np.mean(wf[idx],axis=0)

    # # find merge matrix across all clusters
    # sim_mat, sim_mat_floats, templates_aligned = calc_local_merge(channel, temp, temp, CONFIG)

    # # plot matrix
    # if plotting: 
            # plot_merge_matrix(channel, n_chans, temp, CONFIG, sim_mat, 
                      # sim_mat_floats, spike_index, templates_aligned, "premerge")

    # # run merge until convergence
    # # Cat: TODO: this technically should be inside loop below
    
    # spike_index_new = merge_spikes(sim_mat, spike_index)

    # while len(spike_index_new)!=temp.shape[0]:

        # # compute templates again: 
        # temp = np.zeros((len(spike_index_new),wf.shape[1],wf.shape[2]),'float32')
        # for t in range(len(spike_index_new)): 
            # idx = np.in1d(sic[:,0], spike_index_new[t][:,0])
            # temp[t] = np.mean(wf[idx],axis=0)

        # sim_mat, sim_mat_floats, templates_aligned = calc_local_merge(channel, temp, temp, CONFIG)

        # spike_index_new = merge_spikes(sim_mat, spike_index_new)

    # # compute final spike lists: 
    # temp = np.zeros((len(spike_index_new),wf.shape[1],wf.shape[2]),'float32')
    # for t in range(len(spike_index_new)): 
        # idx = np.in1d(sic[:,0], spike_index_new[t][:,0])
        # temp[t] = np.mean(wf[idx],axis=0)

    # # recompute sim_mat_floats
    # sim_mat, sim_mat_floats, templates_aligned = calc_local_merge(channel, temp, temp, CONFIG)

    # # plot final merge table
    # if plotting: 
            # plot_merge_matrix(channel, n_chans, temp, CONFIG, sim_mat, sim_mat_floats, 
                      # spike_index_new, templates_aligned, "postmerge")

    # # save waveforms for merged clusters for each channel
    # wf_array = []
    # for k in range(len(spike_index_new)):
        # idx = np.in1d(sic[:,0], spike_index_new[k][:,0])
        # wf_array.append(wf[idx])
        
    # np.save(CONFIG.data.root_folder+'tmp/cluster/channel_'+
                                            # str(channel)+'_clusters.npy', wf_array)
    
    # np.savez(CONFIG.data.root_folder+'tmp/cluster/channel_'+
                            # str(channel)+'_weighted_templates', 
                            # templates=temp, 
                            # weights=np.asarray([sic.shape[0] for sic in spike_index_new]))
    
        
    # return spike_index_new


# def merge_spikes(sim_mat, spike_index):
    # ''' Merge spikes from units together based on merge matrix sim_mat
    # ''' 
        
    # # find groups to be merged
    # merge_list = []
    # for k in range(sim_mat.shape[0]):
        # merge_list.append(np.where(sim_mat[k]==True)[0].tolist())
    
    # for i in range(2):
        # for k in range(len(merge_list)):
            # for p in range(k+1, len(merge_list), 1):
                # if len(np.intersect1d(merge_list[k],merge_list[p]))>0:
                    # merge_list[k]=np.union1d(merge_list[k],merge_list[p]).tolist()
                    # merge_list[p]=np.union1d(merge_list[k],merge_list[p]).tolist()
    
    # # generate unique groups to be merged, including singles 
    # unique_data = [list(x) for x in set(tuple(x) for x in merge_list)]
    # print ("unique merge lists: ", unique_data)
   
    # # merge 
    # spike_list = []
    # ctr=0
    # for k in range(len(unique_data)):
        # temp_list=[]
        # for p in range(len(unique_data[k])):
            # idx = unique_data[k][p]
            # temp_list.append(spike_index[idx])

        # spike_index_new = np.vstack(temp_list)
        # spike_index_new[:,1]=ctr    
        # spike_list.append(spike_index_new)
        # ctr+=1
        
    # spike_index_new = spike_list
    # return spike_index_new

# def plot_merge_matrix(channel, n_chans, temp, CONFIG, sim_mat, sim_mat_floats, 
                      # spike_index, templates_aligned, plot_string):

    # fig = plt.figure(figsize =(50,50))

    # merge_threshold = 2
    # clrs = ['blue','red']
    # ctr=0
    # scale = 10
    # for i in range(temp.shape[0]):
        # for j in range(i, temp.shape[0]):
            # ctr+=1
            # ax=plt.subplot(sim_mat.shape[0],sim_mat.shape[0],i*temp.shape[0]+j+1)
            
            # for k in range(n_chans):
                # plt.text(CONFIG.geom[k,0], CONFIG.geom[k,1], str(k), fontsize=10)
            
            # plt.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], c = 'red', s = 400, alpha=.6)

            # if i==j: 
                # plt.plot(CONFIG.geom[:,0] + np.arange(-temp.shape[1],0)[:,np.newaxis]/1.5, 
                                # CONFIG.geom[:,1] + temp[i]*scale, c='black', alpha=1)

                # plt.title("Unit: "+str(i)+ " " + str(spike_index[i].shape[0]), fontsize=10)

            # else:                 
                
                # plt.plot(CONFIG.geom[:,0] + np.arange(-templates_aligned[i,j,0].shape[0],0)[:,np.newaxis]/1.5, 
                                    # CONFIG.geom[:,1] + templates_aligned[i,j,0]*scale, c=clrs[0], alpha=0.4)

                # plt.plot(CONFIG.geom[:,0] + np.arange(-templates_aligned[i,j,1].shape[0],0)[:,np.newaxis]/1.5, 
                                    # CONFIG.geom[:,1] + templates_aligned[i,j,1]*scale, c=clrs[1], alpha=0.4)

                # # plot feat channel colours
                # feat_chans = np.where(temp[i].ptp(0)>merge_threshold)[0]
                # plt.scatter(CONFIG.geom[feat_chans,0], CONFIG.geom[feat_chans,1], c =clrs[0], s = 50,alpha=0.5)

                # feat_chans = np.where(temp[j].ptp(0)>merge_threshold)[0]
                # plt.scatter(CONFIG.geom[feat_chans,0]+5, CONFIG.geom[feat_chans,1], c = clrs[1], s = 50,alpha=0.5)
                 

                # if sim_mat[i][j]==True:
                    # plt.title(r"$\bf{TO MERGE}$" + "\n"+str(i) + " blue: "+
                            # str(spike_index[i].shape[0])+"  " + str(j) + " red: "+
                            # str(spike_index[j].shape[0])+" "+str(round(sim_mat_floats[i,j],2)),
                            # fontsize=10)
                # else:
                    # plt.title(str(i) + " blue: "+str(spike_index[i].shape[0])+"  " + str(j) + " red: "+
                            # str(spike_index[j].shape[0])+" "+str(round(sim_mat_floats[i,j],2)),
                            # fontsize=10)

          
            # plt.xticks([])
            # plt.yticks([])
    # plt.savefig(CONFIG.data.root_folder+"tmp/cluster/channel_"+
                # str(channel)+'_'+plot_string+".png")
    # plt.close('all')


def calc_sim_all_chan(channel, base_temp, com_temp, CONFIG, merge_threshold=0.8,
                      size_base=None, size_com=None, clusters_base=None, 
                      clusters_com=None):
    
    # Cat: TO FIX THIS
    n_chans = CONFIG.recordings.n_channels
    plotting = False
   
    sim_mat = np.zeros([base_temp.shape[0], com_temp.shape[0]], dtype = bool)
    
    # only matching units with identical max chans; NOT CORRECT / NOT NECESSARY
    # select feature channels as all chans over threshold
    feat_chan_thresh = 2
    base_ptp = base_temp.mean(0).ptp(0)
    feat_channels = np.where((base_ptp > feat_chan_thresh))[0]
    max_chan = np.argmax(base_ptp)

    print (feat_channels, "MAX CHAN: ", max_chan)
    
    mc = np.where(feat_channels==max_chan)[0][0]
    base_temp_aligned = align_mc(base_temp[:,:,feat_channels], mc,  
                                 CONFIG, upsample_factor=20, nshifts=7)
    com_temp_aligned = base_temp_aligned
    print ("alinged ", base_temp_aligned.shape)
    
    temp = (base_temp_aligned[:, np.newaxis] * com_temp_aligned).sum(2).sum(2)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            if (base_temp_aligned[i]**2).sum() > (com_temp_aligned[j]**2).sum():
                temp[i,j] = temp[i,j]/ (base_temp_aligned[i]**2).sum()
            else:
                temp[i,j] = temp[i,j]/ (com_temp_aligned[j]**2).sum()

    row, column = np.where(temp>merge_threshold)

    for i in range(row.size):
        sim_mat[row[i], column[i]] = True
    
    #if plotting: 
        #for i in range(temp.shape[0]):
            #plt.figure(figsize = (20,15))
            #plt.plot(CONFIG.geom[:,0] + np.arange(51)[:,np.newaxis], base_temp_aligned[where[i]]*5+CONFIG.geom[:,1], c = 'k')
            #plt.scatter(CONFIG.geom[mc,0], CONFIG.geom[mc,1], c = 'orange', s = 500)

            #labels = []
            #patch_j = mpatches.Patch(color = 'k', label = "ks cluster {}, size = {}".format(clusters_base[where[i]], size_base[where[i]]))
            #labels.append(patch_j)
            #for j in np.where(temp[i] > 0.90)[0]:
                #plt.plot(CONFIG.geom[:,0] + np.arange(51)[:,np.newaxis], com_temp_aligned[where2[j]]*5+CONFIG.geom[:,1], c = colors[where2[j]])
                #patch_j = mpatches.Patch(color = colors[where2[j]], label = "yass cluster {}, size = {}".format(clusters_com[where2[j]], size_com[where2[j]]))
                #labels.append(patch_j)
            
            #for j in range(n_chans):
                #plt.text(CONFIG.geom[j,0], CONFIG.geom[j,1], "{}".format(j), fontsize = 20)
            #plt.legend(handles = labels)    
            #plt.savefig("data/tmp3/temp_match_yass_ks_aclean/match_ks_template_{}.png".format(where[i]))
            #plt.close('all')
    return sim_mat, temp, feat_channels
    
    
# functions for finding groups
def dfs(node,index,taken,l):
    taken[index]=True
    ret=node
    for i,item in enumerate(l):
        if not taken[i] and not ret.isdisjoint(item):
            ret.update(dfs(item,i,taken,l))
    return ret

def merge_all(l, taken):
    ret=[]
    for i,node in enumerate(l):
        if not taken[i]:
            ret.append(list(dfs(node,i,taken,l)))
    return ret
   

def get_feat_channels_mad(wf_data, n_feat_chans):
    '''  Function that uses MAD statistic like robust variance estimator to select channels
    '''

    # compute robust stds over units
    stds = np.median(np.square(wf_data - np.median(wf_data, axis=0, keepdims=True)), axis=0)
   
    # max per channel
    std_max = stds.max(0)
    
    # order channels by largest diptest value
    feat_chans = np.argsort(std_max)[-n_feat_chans:][::-1]
    feat_chans = feat_chans[std_max[feat_chans] > 1]
        
    max_chan = wf_data.mean(0).ptp(0).argmax(0)

    return feat_chans, max_chan
    
    
def get_feat_channels_diptest(wf_data, n_feat_chans):
    '''  Function that uses Hartigan diptest to identify bimodal channels
        Steps:  denoise waveforms first, then find highest variance time point,
                then compute diptest and rank in order of max value;
    '''

    # denoise data before computing std
    #wf_data_PCA = []
    #for k in range(

    # compute stds over units
    stds = wf_data.std(0)

    # find location of highest variance time point in each channel (clip edges)
    # Cat: TODO: need to make this more robust to wider windows; ideally only look at 20-30pts in middle of spike
    time_pts = stds[5:-5].argmax(0)+5

    # compute diptest for each channel at the location of highest variance point
    diptests = np.zeros(wf_data.shape[2])
    for k in range(wf_data.shape[2]):
        diptests[k]=dp(wf_data[:,time_pts[k],k])[0]
        #print (k, dp(wf_data[:,time_pts[k],k]))

    # order channels by largest diptest value
    idx = np.argsort(diptests)[::-1]
    max_chan = wf_data.mean(0).ptp(0).argmax(0)
    feat_chans = idx[:n_feat_chans]
        
    max_chan = wf_data.mean(0).ptp(0).argmax(0)

    return feat_chans, max_chan

def get_feat_channels(template, wf_data, n_max_chans, n_mad_chans):

    rank_amp_chans = np.max(np.ptp(template, axis =0),axis=0)
    rank_indexes = np.argsort(rank_amp_chans)[::-1]
    max_chans = rank_indexes[:n_max_chans]      # select top chans
    ptps = np.ptp(template,axis=0)
    chan_indexes_2SU = np.where(ptps>2)[0]
    rank_chans_max = np.max(robust.mad(wf_data[:,:,chan_indexes_2SU],axis=0),axis=0)
    rank_indexes = np.argsort(rank_chans_max,axis=0)[::-1]
    mad_chans = chan_indexes_2SU[rank_indexes][:n_mad_chans]      # select top chans
    feat_chans = np.union1d(max_chans, mad_chans)
    
    return feat_chans, max_chans
    
def get_feat_channels_ks_test(template, wf_data, n_max_chans, n_mad_chans):

    rank_amp_chans = np.ptp(template, axis =0)
    rank_indexes = np.argsort(rank_amp_chans)[::-1] 
    max_chans = rank_indexes[:n_max_chans]      # select top chans
    ptps = np.ptp(template,axis=0)
    chan_indexes_2SU = np.where(ptps>2)[0]
    rank_chans_max = np.max(robust.mad(wf_data[:,:,chan_indexes_2SU],axis=1),axis=1)
    rank_indexes = np.argsort(rank_chans_max,axis=0)[::-1]
    mad_chans = chan_indexes_2SU[rank_indexes][:n_mad_chans]      # select top chans
    feat_chans = np.union1d(max_chans, mad_chans)
    
    return feat_chans, max_chans, mad_chans


# def align_wf(data_in, upsample_factor, n_steps):
    # ''' TODO parallelize this step
    # '''
    # data_aligned = np.zeros(data_in.shape)
    # for k in range(data_in.shape[2]):
        # data_aligned[:,:,k] = align_channelwise(data_in[:,:,k].T, upsample_factor=20, n_steps=7)
    # return data_aligned


# def knn_triage(th, pca_wf):
    # #th = 90
# #     # get distance to nearest neighbors
    # tree = cKDTree(pca_wf)
    # dist, ind = tree.query(pca_wf, k=11)
    # dist = np.sum(dist, 1)
# #     # triage far ones
    # idx_keep1 = dist < np.percentile(dist, th)
    # return idx_keep1

# def run_mfm_2(kk, CONFIG):
    # mask = np.ones((kk.shape[0], 1))
    # group = np.arange(kk.shape[0])
    # vbParam2 = mfm.spikesort(kk[:,:,np.newaxis],
                            # mask,
                            # group, CONFIG)
    
    # vbParam2.rhat[vbParam2.rhat < 0.1] = 0 #Cat todo; look at this
    
    # vbParam2.rhat = vbParam2.rhat/np.sum(vbParam2.rhat,
                                         # 1, keepdims=True)

    # assignment2 = np.argmax(vbParam2.rhat, axis=1)
    # return vbParam2, assignment2



# def run_mfm_3(kk, CONFIG):
    # mask = np.ones((kk.shape[0], 1))
    # group = np.arange(kk.shape[0])
    # vbParam2 = mfm.spikesort(kk[:,:,np.newaxis],
                            # mask,
                            # group, CONFIG)
    
    # return vbParam2


# def run_mfm3(kk, CONFIG):
    # if verbose:
        # print("chan "+ str(self.channel)+' gen: '+str(self.gen)+" - clustering ", 
                                                          # pca_wf.shape)
    # mask = np.ones((kk.shape[0], 1))
    # group = np.arange(kk.shape[0])
    # vbParam2 = mfm.spikesort(kk[:,:,np.newaxis],
                            # mask,
                            # group, CONFIG)
    # vbParam2.rhat[vbParam2.rhat < 0.1] = 0
    # vbParam2.rhat = vbParam2.rhat/np.sum(vbParam2.rhat,
                                         # 1, keepdims=True)

    # assignment2 = np.argmax(vbParam2.rhat, axis=1)
    # return vbParam2, assignment2

def binary_reader_from_memory(idx_list):

    # New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]

    data_start = idx_start
    data_end = idx_stop
    offset = idx_local

    
    return recording

def load_waveforms_from_memory(recording, data_start, offset, spike_train, 
                               spike_size):
                                          
    # offset spike train to t=0 to properly index into data 
    spike_train[:,0] = spike_train[:,0]-data_start
    #print ("  spike_train: ", spike_train)

    # load all waveforms at once - use offset 
    waveforms = recording[spike_train[:, [0]].astype('int32')+offset
                                 + np.arange(-spike_size//2, spike_size//2)]

    return waveforms    



def load_waveforms_from_memory_merge(recording, data_start, offset, spike_train, 
                               spike_size):
                                           
    # offset spike train to t=0 to properly index into data 
    spike_train = spike_train-data_start


    # load all waveforms at once - use offset 
    waveforms = recording[spike_train[:,np.newaxis]+offset
                  + np.arange(-spike_size, spike_size + 1)]

    return waveforms    



def align_singletrace_lastchan(wf, CONFIG, upsample_factor = 5, nshifts = 15, 
             ref = None):

    ''' Align all waveforms to the master channel
    
        wf = selected waveform matrix (# spikes, # samples, # featchans)
        mc = maximum channel from featchans; usually first channle, i.e. 0
    '''
    
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    wf_up = upsample_parallel(wf.T, upsample_factor)
   
    wlen = wf_up.shape[1]
    wf_start = int(.25 * (wlen-1))
    wf_end = -int(.35 * (wlen-1))
    #print (wf_start,wf_end)
    #print (wf_up.shape)
    wf_trunc = wf_up[:,wf_start:wf_end]
    #print ('wf_trunc: ', wf_trunc.shape)
    wlen_trunc = wf_trunc.shape[1]
    
    ref_upsampled = wf_up[-1]
    #print (ref_upsampled.shape)
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    #print (ref_shifted.shape)

    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2)+1)):
        #print (i,s+wf_end, s+ wf_start, ref_upsampled[s+wf_start:s+wf_end].shape)
        ref_shifted[:,i] = ref_upsampled[s+wf_start:s+wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[1],wlen_trunc])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end]

    return np.float32(wf_final[:,::upsample_factor]), best_shifts


def get_template_PCA_rotation(wf_shifted=None, n_pca=3):
    #wf_shifted=None
    if wf_shifted is None:
        print ("... loading templates from disk... FIX THIS")
        templates = np.load('/media/cat/1TB/liam/49channels/data1_allset/tmp_noactive_recovery/cluster/templates_post_cluster_post_merge_post_cutoff.npy')
        templates = templates[:,15:-15,:]
        print (templates.shape)
    else:
        templates=wf_shifted.swapaxes(0,2)
    
    #print (templates.shape)

    max_chans = templates.ptp(1).argmax(0)
    #print (max_chans.shape)

    templates_maxchan_normalized = []
    for k in range(templates.shape[2]):
        templates_maxchan_normalized.append(templates[max_chans[k],:,k]/templates[max_chans[k],:,k].ptp(0))

    templates_maxchan_normalized = np.array(templates_maxchan_normalized)
    #print (templates_maxchan_normalized.shape)

    _, templates_maxchan_normalized_PCA, pca_object1 = PCA(templates_maxchan_normalized,n_pca)

    #print (templates_maxchan_normalized_PCA.shape)

    diffs = np.max(np.abs(templates_maxchan_normalized-templates_maxchan_normalized_PCA),axis=1)
    
    idx = np.where(diffs<0.15)[0]
    
    if idx.shape[0]>3:
        _, _, pca_object2 = PCA(templates_maxchan_normalized[idx],n_pca)
    else:
        print (" insufficient spikes in get_tempalte_PCA_rotation...")
        pca_object2 = pca_object1


    return pca_object1, pca_object2


def clean_templates(templates, spike_train_cluster, weights, CONFIG):

    # find units < 3SU 
    # Cat: TODO: read this threshold and flag from CONFIG
    template_threshold = 3

    print ("  cleaning templates (temps, time, chan): ", templates.shape)
    ptps = templates.ptp(1).max(1)
    idx1 = np.where(ptps>=template_threshold)[0]
    print ("  deleted clusters < {}SU: ".format(template_threshold), templates.shape[0]-idx1.shape[0])

    # find units with large peaks off-centre
    if False:
        idx2 = find_clean_templates2(templates, CONFIG)
        
        # merge all units to be deleted
        idx_all = np.intersect1d(idx1, idx2)
        print ("  deleted collision clusters: ", idx1.shape[0]-idx_all.shape[0])

    else:
        idx_all = np.copy(idx1)
    
    # redundant step to order indexes
    idx = np.argsort(idx_all)
    idx_all = idx_all[idx]
    
    # remerge keep units 
    templates = templates[idx_all]
    weights = weights[idx_all]
    spike_train_cluster_new2 = []
    for ctr,k in enumerate(list(idx_all)):
        temp = np.where(spike_train_cluster[:,1]==k)[0]
        temp_train = spike_train_cluster[temp]
        temp_train[:,1]=ctr
        spike_train_cluster_new2.append(temp_train)
        
    spike_train_cluster_new2 = np.vstack(spike_train_cluster_new2)

    return templates, spike_train_cluster_new2, weights, idx_all


def find_clean_templates(templates, CONFIG):
    
    # normalize templates on max channels:
    templates = templates.transpose(1,2,0)
    print ("  clean_templates collisions shape (time, chan, temps): ", templates.shape)
    max_chans = templates.ptp(0).argmax(0)
    temps_normalized=[]
    for u in range(templates.shape[2]):
        trace = templates[:,max_chans[u],u]
        temps_normalized.append(trace/trace.ptp(0))
    temps_normalized = np.array(temps_normalized)
    
    # first 
    mean_template = temps_normalized.mean(0)
    templates = np.vstack((temps_normalized, mean_template))

    # Cat: TODO: limit # of shifts here, we don't want to correct bad templates using
    #           alignment
    templates_aligned = templates

    # find largest negative peak and the following one
    mean_template = templates_aligned.mean(0)

    #print (mean_template.shape)
    mean_template_centre = np.argmin(mean_template)
    
    template_ids = []
    ctr=0
    max_dist_to_trough = 3
    for k in range(templates_aligned.shape[0]-1):
        # find max trough location
        max_time = np.argmin(templates_aligned[k,mean_template_centre-max_dist_to_trough: 
                                    mean_template_centre+max_dist_to_trough])+mean_template_centre-max_dist_to_trough

        # if trough location too far from centre exclude
        if abs(max_time-mean_template_centre)>max_dist_to_trough:
            continue

        # find next largest trough 
        min_times = argrelmin(templates_aligned[k], axis=0, order=1, mode='clip')[0]
        del_idx = np.where(min_times==max_time)[0]
        min_times_nonlowest = np.delete(min_times, del_idx)

        # find next largest trough location
        if min_times_nonlowest.shape[0]!=0:
            nearest = min_times_nonlowest[np.argmin(templates_aligned[k][min_times_nonlowest])]
        else:
            # this is the case where template has only a single trough
            # it usually indicates a good template. 
            nearest = 0
        
        # if next largest trough is not too large, keep templates
        if templates_aligned[k][nearest]>=-0.15: # or abs(max_time-17)>4:
            template_ids.append(k)
            
    return np.array(template_ids)

def find_clean_templates2(templates, CONFIG):
    
    # normalize templates on max channels:
    print ("  clean_templates collisions shape (temps, time, chan): ", templates.shape)
    
    pca_main_components = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_main_components.npy')))
    pca_sec_components = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_sec_components.npy')))
    pca_main_mean = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_main_mean.npy')))
    pca_sec_mean = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'pca_sec_mean.npy')))

    ref_template = np.load(absolute_path_to_asset(
        os.path.join('template_space', 'ref_template.npy')))
    
    buffer = (templates.shape[1] - pca_main_components.shape[1])//2
    (templates_mc, templates_sec, 
     ptp_mc, ptp_sec, ids_sec) = get_normalized_templates(
        templates[:, buffer:-buffer], 
        CONFIG.neigh_channels, ref_template)

    denoised_templates_mc = pca_denoise(
        templates_mc, pca_main_mean, pca_main_components)
    denoised_templates_sec = pca_denoise(
        templates_sec, pca_sec_mean, pca_sec_components)
    
    diff_mc = np.linalg.norm(templates_mc - denoised_templates_mc, axis=1)
    diff_sec = np.linalg.norm(templates_sec - denoised_templates_sec, axis=1)

    ptp_th = 3
    mc_th = 0.3
    sec_th = 0.5
    idx_remove_mc = np.where(np.logical_and(diff_mc > mc_th, ptp_mc > ptp_th))[0]
    idx_remove_sec = np.unique(ids_sec[np.logical_and(diff_sec > sec_th, ptp_sec > ptp_th)])
    idx_remove = np.unique(np.hstack((idx_remove_mc, idx_remove_sec)))

    idx_keep = np.delete(np.arange(templates.shape[0], dtype='int16'), idx_remove)
    
    return idx_keep

def get_normalized_templates(templates, neigh_channels, ref_template):

    """
    plot normalized templates on their main channels and secondary channels
    templates: number of channels x temporal window x number of units
    geometry: number of channels x 2
    """

    K, R, C = templates.shape
    mc = np.argmax(templates.ptp(1), 1)

    # get main channel templates
    templates_mc = np.zeros((K, R))
    for k in range(K):
        templates_mc[k] = templates[k, :, mc[k]]

    # shift templates_mc
    best_shifts_mc = align_get_shifts_with_ref(
                    templates_mc,
                    ref_template)
    templates_mc = shift_chans(templates_mc, best_shifts_mc)
    ptp_mc = templates_mc.ptp(1)

    # normalize templates
    norm_mc = np.linalg.norm(templates_mc, axis=1, keepdims=True)
    templates_mc /= norm_mc

    # get secdonary channel templates
    templates_sec = np.zeros((0, R))
    best_shifts_sec = np.zeros(0)
    unit_ids_sec = np.zeros((0), 'int32')
    for k in range(K):
        neighs = np.copy(neigh_channels[mc[k]])
        neighs[mc[k]] = False
        neighs = np.where(neighs)[0]
        templates_sec = np.concatenate((templates_sec, templates[k, :, neighs]), axis=0)
        best_shifts_sec = np.hstack((best_shifts_sec, np.repeat(best_shifts_mc[k], len(neighs))))
        unit_ids_sec = np.hstack((unit_ids_sec, np.ones(len(neighs), 'int32')*k))

    # shift templates_sec
    best_shifts_sec = align_get_shifts_with_ref(
                    templates_sec,
                    ref_template)
    templates_sec = shift_chans(templates_sec, best_shifts_sec)
    ptp_sec = templates_sec.ptp(1)

    # normalize templates
    norm_sec = np.linalg.norm(templates_sec, axis=1, keepdims=True)
    templates_sec /= norm_sec

    return templates_mc, templates_sec, ptp_mc, ptp_sec, unit_ids_sec


#def binary_reader_waveforms(filename, n_channels, n_times, spikes, channels=None, data_type='float32'):
    #''' Reader for loading raw binaries
    
        #standardized_filename:  name of file contianing the raw binary
        #n_channels:  number of channels in the raw binary recording 
        #n_times:  length of waveform 
        #spikes: 1D array containing spike times in sample rate of raw data
        #channels: load specific channels only
        #data_type: float32 for standardized data
    
    #'''

    ## ***** LOAD RAW RECORDING *****
    #if channels is None:
        #wfs = np.zeros((spikes.shape[0], n_times, n_channels), data_type)
    #else:
        #wfs = np.zeros((spikes.shape[0], n_times, channels.shape[0]), data_type)

    #with open(filename, "rb") as fin:
        #for ctr,s in enumerate(spikes):
            ## index into binary file: time steps * 4  4byte floats * n_channels
            #fin.seek(s * 4 * n_channels, os.SEEK_SET)
            #wfs[ctr] = np.fromfile(
                #fin,
                #dtype='float32',
                #count=(n_times * n_channels)).reshape(n_times, n_channels)[:,channels]

    #fin.close()
    #return wfs
