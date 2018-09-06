import numpy as np
import logging
import os
import tqdm
import parmap
from scipy import signal
from scipy.spatial import cKDTree
from copy import deepcopy

from yass.explore.explorers import RecordingExplorer
from yass import mfm
from yass.empty import empty
from scipy.sparse import lil_matrix
from statsmodels import robust
from scipy.signal import argrelmin
import matplotlib
import progressbar

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import networkx as nx
import multiprocessing as mp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from diptest.diptest import diptest as dp


colors=np.asarray(["#000000", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"])
        
colors = np.concatenate([colors,colors])
        
        
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
    return X,Y





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

def align_last_chan(wf, CONFIG, upsample_factor = 5, nshifts = 15):

    ''' Align all waveforms to the master channel
    
        wf = selected waveform matrix (# spikes, # samples, # featchans)
        max_channel: is the last channel provided in wf 
    '''
    
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    wf_up = []
    for k in range(wf.shape[2]): 
        #print ("aligning : ", k)
        wf_up.append(upsample_resample_parallel_channel(wf[:,:,k], upsample_factor, CONFIG))
    wf_up = np.array(wf_up).swapaxes(1,2).swapaxes(0,2)

    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = wf_up[:,:,-1].mean(0)
    
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:, -1], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[0],wlen_trunc, wf.shape[2]])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end][:, np.arange(wf.shape[2])]

    return wf_final[:,::upsample_factor]


def align_mc(wf, mc, CONFIG, upsample_factor = 5, nshifts = 15, 
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
        #print ("aligning : ", k)
        wf_up.append(upsample_resample_parallel_channel(wf[:,:,k], upsample_factor, CONFIG))
    wf_up = np.array(wf_up).swapaxes(1,2).swapaxes(0,2)

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
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:, mc], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[0],wlen_trunc, wf.shape[2]])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end][:, np.arange(wf.shape[2])]

    return wf_final[:,::upsample_factor]


def align_mc_templates(wf, mc, CONFIG, spike_padding, upsample_factor = 5, 
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
        wf_up.append(upsample_resample_parallel_channel(wf[:,:,k], upsample_factor, CONFIG))
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
        wf_up.append(upsample_resample_parallel_channel(wf[:,:,k], upsample_factor, CONFIG))
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
    
def align_channelwise3_parallel(wf, upsample_factor = 20, nshifts = 7):
    ''' NOT IN USE
    ''' 
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
    
def upsample_resample_parallel_channel(wf, upsample_factor, CONFIG):
    n_spikes, _ = wf.shape
    
    # dont' parallize alignment - unless seems ok otherwise
    # Cat: TODO: can we parallize recursively and to this?
    if n_spikes<1000000:
        wf_up = upsample_parallel(wf, upsample_factor)
    else: 
        wf_array = np.array_split(wf, CONFIG.resources.n_processors)
        wf_up = parmap.map(upsample_parallel, wf_array, upsample_factor, 
                           processes=CONFIG.resources.n_processors)
        wf_up = np.vstack(wf_up)
        
    return wf_up
    
    
def upsample_parallel(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces


def RRR3_noregress_recovery(channel, wf, sic, gen, fig, grid, x, ax_t, triageflag, 
         alignflag, plotting, n_feat_chans, n_dim_pca, 
         wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor, nshifts, 
         assignment_global, spike_index, scale, knn_triage_threshold):
    
    ''' Recursive clusteringn function
        channel: current channel being clusterd
        wf = wf_PCA: denoised waveforms (# spikes, # time points, # chans)
        sic = spike_indexes of spikes on current channel
        gen = generation of cluster; increases with each clustering step        
    '''

    # Cat: TODO read from CONFIG File
    verbose=False
    
    # ************* CHECK SMALL CLUSTERS *************
    # Exit clusters that are too small
    if wf.shape[0] < CONFIG.cluster.min_spikes:
        return
    if verbose:
        print("chan/unit "+str(channel)+' gen: '+str(gen)+' # spikes: '+
              str(wf.shape[0]))
        
    #*************************************************        
    # ************ FIND FEATURE CHANNELS *************
    #*************************************************        
    # select feature chans 
    feat_chans, max_chan = get_feat_channels_diptest(wf, n_feat_chans)

    if verbose:
        print("chan "+str(channel)+' gen: '+str(gen)+", feat chans: "+
                  str(feat_chans) + ", max_chan: "+ str(max_chan))
    
    # save max_channel relative to feat_chans location
    mc = max_chan
    #np.savez('/media/cat/1TB/liam/49channels/data1_allset/tmp/cluster/wfs.npz',
             #wf = wf,
             #feat_chans = feat_chans,
             #max_chan = max_chan)
    
    #*************************************************        
    # ************ ALIGN FEATURE CHANNELS ************
    #*************************************************        
    # align, note: aligning all channels to max chan which is appended to the end
    # note: max chan is first from feat_chans above, ensure order is preserved
    
    if alignflag:
        if verbose:
            print ("chan "+str(channel)+' gen: '+str(gen)+" - aligning")

        feat_chans_mc = np.append(feat_chans, [mc], axis=0)
        #print ("feat_chans_mc: ", feat_chans_mc)
        wf_align = align_last_chan(wf[:,:,feat_chans_mc], CONFIG, 
                               upsample_factor, nshifts)
    else:
        wf_align = wf[:,:,feat_chans]


    #*************************************************        
    # *************** PCA STEP #1 ********************
    #*************************************************        
    # compress globally all waveforms on feature chans
    # Cat: TODO: is this waveform clipping necessary here? align_ fucntion
    #            already returns clipped waveforms
    
    # old PCA approach using concateanted data over all chans
    if True:
        data_in = wf_align[:,wf_start:wf_end].swapaxes(1,2).reshape(wf.shape[0],-1)
        pca_wf, _ = PCA(data_in, 3)
    
    # new PCA; first step compress on each channel 
    else:
        wf_PCA = np.zeros((wf_align.shape[0], n_dim_pca, n_feat_chans))
        for ch in range(n_feat_chans):
            wf_PCA[:,:,ch],_ = PCA(wf_align[:,:,ch], n_dim_pca)
        
        # second step concatenate all chans together;
        data_in = wf_PCA.reshape(wf_PCA.shape[0], -1)
        
        # third step recompress stacked features
        pca_wf,_ = PCA(data_in, n_dim_pca)
    

    #*************************************************        
    # ******** KNN TRIAGE & PCA #2 *******************
    #*************************************************        
    # knn triage outliars; e.g. remove 10%-20% of outliars
    if triageflag:
        idx_keep = knn_triage(knn_triage_threshold*100, pca_wf)
        idx_keep = np.where(idx_keep==1)[0]
        if verbose:
            print("chan "+str(channel)+' gen: '+str(gen) + 
                " triaged, remaining spikes "+ str(idx_keep.shape[0]))

        # rerun global compression on residual waveforms
        pca_wf,_ = PCA(data_in[idx_keep],3)
    else:
        # keep all spikes
        idx_keep = np.ones(pca_wf.shape[0],dtype = bool)
        
    # select only non-triaged spikes
    pca_wf_all = pca_wf.copy() #[idx_keep]
    
    
    #*************************************************        
    # ************** SUBSAMPLE STEP ******************
    #*************************************************        
    # subsmaple 10,000 spikes 
    idx_subsampled = np.random.choice(np.arange(pca_wf.shape[0]), 
                 size=min(pca_wf.shape[0],CONFIG.cluster.max_n_spikes),
                 replace=False)
    pca_wf = pca_wf[idx_subsampled]


    #*************************************************        
    # ************* CLUSTERING STEP ******************
    #*************************************************        
    # clustering
    if verbose:
        print("chan "+ str(channel)+' gen: '+str(gen)+" - clustering ", 
                                                          pca_wf.shape)
    vbParam, assignment = run_mfm3(pca_wf, CONFIG)
    
    
    #*************************************************        
    # ************* RECOVER SPIKES *******************
    #*************************************************        
    # if we subsampled then recover soft-assignments using above:
    if pca_wf.shape[0] <= CONFIG.cluster.max_n_spikes:
        vbParam2 = deepcopy(vbParam)
        vbParam2, assignment2 = recover_spikes(vbParam2, pca_wf_all, 
                                                            CONFIG)
    else:
        vbParam2, assignment2 = vbParam, assignment

    idx_recovered = np.where(assignment2!=-1)[0]
    if verbose:
        print ("chan "+ str(channel)+' gen: '+str(gen)+" - recovered ",
                                            str(idx_recovered.shape[0]))


    #*************************************************        
    # *********** REVIEW AND SAVE RESULTS ************
    #*************************************************        
    # always plot distribution
    if plotting:
        plot_clustering_scatter(fig, grid, x, gen, vbParam,  
                                assignment2, colors, pca_wf_all, channel,
                                idx_recovered)
                
    # Case #1: single cluster found
    if vbParam.rhat.shape[1] == 1:
        
        # check if template has > 2 peaks > 0.5SU on any channel and reject
        if False: 
            template = np.mean(wf[idx_recovered],axis=0)
            max_ptp = template.ptp(0).max(0)
            n_troughs = 0
            for k in range(template.shape[1]):
                idx = np.where(template[argrelmin(template[:,k],axis=0),k]<-0.5)[0]
                temp = len(idx)
                # don't exclude templates > 10SU 
                if (temp>n_troughs) and (max_ptp<10):
                    n_troughs=temp
        else:
            n_troughs=1

        if n_troughs> 1:
            if verbose:
                print ("chan ", str(channel), ' gen: ', str(gen), 
                       "N_TROUGHS: ", n_troughs, "SKIPPING TEMPLATE")
        else:
            N= len(assignment_global)
            if verbose:
                print("chan "+str(channel)+' gen: '+str(gen)+" >>> cluster "+
                    str(N)+" saved, size: "+str(wf[idx_recovered].shape)+"<<<")
            
            assignment_global.append(N * np.ones(assignment2[idx_recovered].shape[0]))
            spike_index.append(sic[idx_recovered])

            # plot template if done
            if plotting:
                plot_clustering_template(fig, grid, ax_t, gen, N, wf, idx_recovered, 
                                         CONFIG, colors, feat_chans, scale)
        
    # Case #2: multiple clusters
    else:
        mask = vbParam.rhat>0
        stability = np.average(mask * vbParam.rhat, axis = 0, weights = mask)
        clusters, sizes = np.unique(assignment2[idx_recovered], return_counts = True)
        
        if verbose:
            print("chan "+str(channel)+' gen: '+str(gen) + 
              " multiple clusters, stability " + str(np.round(stability,2)) + 
              " size: "+str(sizes))

        # remove stable clusters 
        for clust in np.where(stability>mfm_threshold)[0]:
            idx = np.where(assignment2==clust)[0]
            
            if wf[idx_keep][idx].shape[0]<CONFIG.cluster.min_spikes: 
                continue    # cluster too small
            
            if verbose:
                print("chan "+str(channel)+' gen: '+str(gen)+
                    " reclustering stable cluster"+ 
                    str(wf[idx_keep][idx].shape))
            RRR3_noregress_recovery(channel, wf[idx_keep][idx], 
                 sic[idx_keep][idx], gen+1, fig, grid, x, ax_t, False, alignflag, 
                 plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                 mfm_threshold,  CONFIG, upsample_factor, nshifts, 
                 assignment_global, spike_index, scale, knn_triage_threshold)

        # if all clusters are unstable: triage (also annealing is an option)
        if np.all(stability<=mfm_threshold):

            if verbose:
                print("chan "+str(channel)+' gen: '+str(gen)+ 
                                " no stable clusters, triaging "+
                                str(wf[idx_keep][idx_recovered].shape))

            RRR3_noregress_recovery(channel, wf[idx_keep][idx_recovered], 
                 sic[idx_keep][idx_recovered], gen+1, fig, grid, x, ax_t, True, 
                 alignflag, plotting, n_feat_chans, n_dim_pca, wf_start, 
                 wf_end, mfm_threshold, CONFIG, upsample_factor, nshifts, 
                 assignment_global, spike_index, scale, knn_triage_threshold)
        
        else:
            # run mfm on remaining data
            idx = np.in1d(assignment2, np.where(stability<=mfm_threshold)[0])
            if idx.sum()>CONFIG.cluster.min_spikes:
                if verbose:
                    print("chan "+str(channel)+" reclustering residuals "+
                                            str(wf[idx_keep][idx].shape))
                RRR3_noregress_recovery(channel, wf[idx_keep][idx],
                    sic[idx_keep][idx], gen+1, fig, grid, x, ax_t, False, alignflag, 
                    plotting, n_feat_chans, n_dim_pca, wf_start, wf_end, 
                    mfm_threshold, CONFIG, upsample_factor, nshifts, 
                    assignment_global, spike_index, scale, knn_triage_threshold)



#def RRR3_noregress_recovery_deconv(unit, wf, sic, gen, fig, grid, 
         #triageflag, alignflag, plotting, n_feat_chans, n_dim_pca, 
         #wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor, nshifts, 
         #assignment_global, spike_index, scale):
          

    #RRR3_noregress_recovery_deconv(unit, wf_PCA, unit_sp, gen, fig, grid, 
            #triageflag, alignflag, plotting, n_feat_chans, 
            #n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
            #upsample_factor, nshifts, assignment_global, spike_index, scale)


    #verbose=True
    #print ("TODO:  IMPLEMENT DIPTEST FEAT CHANS ")
    
    ## ************* CHECK SMALL CLUSTERS *************
    ## Exit clusters that are too small
    #if wf.shape[0] < CONFIG.cluster.min_spikes:
        ##print ("exiting too few spikes<<<FIX THIS")
        #return

    #if verbose:
        #print("unit "+str(unit)+' gen: '+str(gen)+' # spikes: '+
              #str(wf.shape[0]))
        
        
    ## ************ FIND FEATURE CHANNELS *************
    ## select feature chans 
    #feat_chans, max_chans = get_feat_channels(wf.mean(0), wf, n_max_chans, 
                                                              #n_mad_chans)  
    #if verbose:
        #print("unit "+str(unit)+' gen: '+str(gen)+", feat chans: "+
                  #str(feat_chans) + ", max_chan: "+ str(max_chans[0]))
    
    ## save max_channel relative to feat_chans location
    ##mc = np.where(feat_chans==max_chans[0])[0][0]
    ##mc = max_chans[0]
    #mc = wf[:,:,feat_chans].mean(0).ptp(0).argmax()
    ##print ("Max chan: ", mc)
    
    
    ## ************ ALIGN FEATURE CHANNELS ************
    ## align, note: aligning all channels to max chan; 
    ## note: max chan is first from feat_chans above, ensure order is preserved
    #if alignflag:
        #if verbose:
            #print ("unit "+str(unit)+' gen: '+str(gen)+" - aligning")
        #wf_align = align_mc(wf[:,:,feat_chans], mc, CONFIG, 
                            #upsample_factor, nshifts, ref = None)
    #else:
        #wf_align = wf[:,:,feat_chans]


    ## *************** PCA STEP #1 ********************
    ## compress globally all waveforms on feature chans
    #data_in = wf_align[:,wf_start:wf_end].swapaxes(1,2).reshape(wf.shape[0],-1)
    #pca_wf, _ = PCA(data_in, 3)


    ## ******** KNN TRIAGE & PCA #2 *******************
    ## knn triage outliars; e.g. remove 10%-20% of outliars
    #if triageflag:
        #idx_keep = knn_triage(mfm_threshold*100, pca_wf)
        #idx_keep = np.where(idx_keep==1)[0]
        #if verbose:
            #print("unit "+str(unit)+' gen: '+str(gen) + 
                #" triaged, remaining spikes "+ str(idx_keep.shape[0]))

        ## rerun global compression on residual waveforms
        #pca_wf,_ = PCA(data_in[idx_keep],3)
    #else:
        ## keep all spikes
        #idx_keep = np.ones(pca_wf.shape[0],dtype = bool)
        
    ## select only non-triaged spikes
    #pca_wf_all = pca_wf.copy() #[idx_keep]
    
    
    ## ************** SUBSAMPLE STEP ******************
    ## subsmaple 10,000 spikes 
    #idx_subsampled = np.random.choice(np.arange(pca_wf.shape[0]), 
                 #size=min(pca_wf.shape[0],CONFIG.cluster.max_n_spikes),
                 #replace=False)
    #pca_wf = pca_wf[idx_subsampled]


    ## ************* CLUSTERING STEP ******************
    ## clustering
    #if verbose:
        #print("unit "+ str(unit)+' gen: '+str(gen)+" - clustering ", 
                                                          #pca_wf.shape)
    #vbParam, assignment = run_mfm3(pca_wf, CONFIG)
    
    
    ## ************* RECOVER SPIKES *******************
    ## if we subsampled then recover soft-assignments using above:
    #if pca_wf.shape[0] <= CONFIG.cluster.max_n_spikes:
        #vbParam2 = deepcopy(vbParam)
        #vbParam2, assignment2 = recover_spikes(vbParam2, pca_wf_all, 
                                                            #CONFIG)
    #else:
        #vbParam2, assignment2 = vbParam, assignment

    #idx_recovered = np.where(assignment2!=-1)[0]
    #if verbose:
        #print ("unit "+ str(unit)+' gen: '+str(gen)+" - recovered ",
                                            #str(idx_recovered.shape[0]))


    ## ************* REVIEW AND SAVE RESULTS ******************
    ## if # clusters > 1 plot scatter plot
    ## plot distribution first
    ##if plotting:
        ##x = np.zeros(100, dtype = int)
        ##plot_clustering_scatter(fig, grid, x, gen, vbParam,  
                                ##assignment2, colors, pca_wf_all, unit,
                                ##idx_recovered)
                
    ## if single cluster found
    #if vbParam.rhat.shape[1] == 1:
        
        ## Cat TODO: heuristic; may want to remove
        ## check if template has > 2 peaks > 0.5SU on any channel and reject
        #if True: 
            #template = np.mean(wf[idx_recovered],axis=0)
            #max_ptp = template.ptp(0).max(0)
            #n_troughs = 0
            #for k in range(template.shape[1]):
                #idx = np.where(template[argrelmin(template[:,k],axis=0),k]<-0.5)[0]
                #temp = len(idx)
                #if (temp>n_troughs) and (max_ptp<10):
                    #n_troughs=temp
        #else:
            #n_troughs=1

        #if n_troughs> 1:
            #if verbose:
                #print ("unit ", str(unit), ' gen: ', str(gen), 
                   #"N_TROUGHS: ", n_troughs, "SKIPPING TEMPLATE")
                #return

        #N= len(assignment_global)
        #if verbose:
            #print("unit "+str(unit)+' gen: '+str(gen)+" >>> cluster "+
                #str(N)+" saved, size: "+str(wf[idx_recovered].shape)+"<<<")
        
        #assignment_global.append(N * np.ones(assignment2[idx_recovered].shape[0]))
        #spike_index.append(sic[idx_recovered])

        #if plotting:
            #plot_clustering_template(fig, grid, ax_t, gen, N, wf, idx_recovered, 
                                     #CONFIG, colors, feat_chans, scale)
        
    ## if multiple clusters
    #else:
        #mask = vbParam.rhat>0
        #stability = np.average(mask * vbParam.rhat, axis = 0, weights = mask)
        #clusters, sizes = np.unique(assignment2[idx_recovered], return_counts = True)
        
        #if verbose:
            #print("unit "+str(unit)+' gen: '+str(gen) + 
              #" multiple clusters, stability " + str(np.round(stability,2)) + 
              #" size: "+str(sizes))

        ## remove stable clusters 
        #for clust in np.where(stability>mfm_threshold)[0]:
            #idx = np.where(assignment2==clust)[0]
            
            #if wf[idx_keep][idx].shape[0]<CONFIG.cluster.min_spikes: 
                #continue    # cluster too small
            
            #if verbose:
                #print("unit "+str(unit)+' gen: '+str(gen)+
                    #" reclustering stable cluster"+ 
                    #str(wf[idx_keep][idx].shape))
                    
            #RRR3_noregress_recovery_deconv(unit, wf[idx_keep][idx], 
                 #sic[idx_keep][idx], gen+1, 
                 #fig, grid, ax_t, False, alignflag, plotting, n_mad_chans, n_max_chans, 
                 #n_dim_pca, wf_start, wf_end, mfm_threshold,  CONFIG, 
                 #upsample_factor, nshifts, assignment_global, spike_index, scale)

        ## if all clusters are unstable: try annealing, or triaging
        #if np.all(stability<=mfm_threshold):
            
            #if verbose:
                #print("unit "+str(unit)+' gen: '+str(gen)+ 
                                #" no stable clusters, triaging "+
                                #str(wf[idx_keep][idx_recovered].shape))

            #RRR3_noregress_recovery_deconv(unit, wf[idx_keep][idx_recovered], 
                 #sic[idx_keep][idx_recovered], gen+1, fig, grid, ax_t,
                 #True, alignflag, plotting, n_mad_chans, n_max_chans, 
                 #n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor,
                 #nshifts, assignment_global, spike_index, scale)
         
        #else:
            ## run mfm on remaining data
            #idx = np.in1d(assignment2, np.where(stability<=mfm_threshold)[0])
            #if idx.sum()>CONFIG.cluster.min_spikes:
                #if verbose:
                    #print("unit "+str(unit)+" reclustering residuals "+
                                            #str(wf[idx_keep][idx].shape))
                #RRR3_noregress_recovery_deconv(unit, wf[idx_keep][idx],
                    #sic[idx_keep][idx], 
                    #gen+1, fig, grid, ax_t, False, alignflag, plotting, 
                    #n_mad_chans, n_max_chans, n_dim_pca, wf_start, wf_end, 
                    #mfm_threshold, CONFIG, upsample_factor, nshifts, 
                    #assignment_global, spike_index, scale)


def plot_clustering_template(fig, grid, ax_t, gen, N, wf, idx_recovered, CONFIG, 
                             colors, feat_chans, scale):
        # plot templates 
        #ax_t = fig.add_subplot(grid[13:, 6:])
        wf_mean = wf[idx_recovered].mean(0)
        
        # plot template
        ax_t.plot(CONFIG.geom[:,0]+
                  np.arange(-wf_mean.shape[0],0)[:,np.newaxis]/3., 
                  CONFIG.geom[:,1] + wf_mean[:,:]*scale, c=colors[N%100])

        # plot feature channels
        for i in feat_chans:
             ax_t.scatter(CONFIG.geom[i,0]+gen, CONFIG.geom[i,1]+N, 
                                                    s = 30, 
                                                    color = colors[N%100],
                                                    alpha=1)

def plot_clustering_scatter(fig, grid, x, gen, vbParam,  
                            assignment2, colors, pca_wf, channel,
                            idx_recovered):
    if np.all(x[gen]<20) and gen <20:
        mask = vbParam.rhat>0
        stability = np.average(mask * vbParam.rhat, axis = 0, weights = mask)
        labels = []
        clusters, sizes = np.unique(assignment2[idx_recovered], 
                                            return_counts=True)

        ax = fig.add_subplot(grid[gen, x[gen]])
        x[gen] += 1
        for clust in clusters:
            patch_j = mpatches.Patch(color = colors[clust%100], 
            label = "size = {}, stability = {}".format(sizes[clust], 
            stability[clust]))
            labels.append(patch_j)
        ax.scatter(pca_wf[idx_recovered,0], pca_wf[idx_recovered,1], 
            c = colors[assignment2[idx_recovered]%100], edgecolor = 'k')
        
        if clusters.size == 1:
            ax.scatter(pca_wf[:,0].mean(), pca_wf[:,1].mean(), c= 'r', s = 2000)
        ax.legend(handles = labels)
        ax.set_title(str(sizes.sum()))

def recover_spikes(vbParam, pca, CONFIG):
    
    N, D = pca.shape
    C = 1
    maskedData = mfm.maskData(pca[:,:,np.newaxis], np.ones([N, C]), np.arange(N))
    
#     for i in range(10):
    vbParam.update_local(maskedData)
#         suffStat = suffStatistics(maskedData, vbParam)
#         vbParam.update_global(suffStat, CONFIG)
    assignment = mfm.cluster_triage(vbParam, pca[:,:,np.newaxis], 3)
    
    return vbParam, assignment

def clustering_annealing():
    ''' annealing function
        currently not used; might be needed later
    ''' 
    
    pass
    #print("annealing")
    #annealing_thresholds = np.arange(0.0, 1.01, 0.01)  # set threshold for annealing clusters together
    #network_threshold = 0.0 # set very low threshold for assigning spikes to a particular unit
    
    ##for network_threshold in network_thresholds:
    #network = np.zeros((vbParam.rhat.shape[1],vbParam.rhat.shape[1]),'float32')

    ## use compute 
    #for k in range(vbParam.rhat.shape[1]):
        #indexes = np.where(vbParam.rhat[:,k]>network_threshold)[0]

        #temp = vbParam.rhat.copy()
        #matches = np.average(temp[indexes], axis=0)

        #network[k]=matches
    
    ## add 1 to ensure 
    #network+=1  
    #network_pd = pd.DataFrame(network)

    ## search for minimum annealing threshold that genrates 2 groups
    #for ctr, threshold in enumerate(annealing_thresholds):
        ## Cat: add 1.0 to correlation matrix ranging -1.0 .. +1.0
        ## TODO: this may be problematic; find more robust metric of overlap
        #corr = network_pd.corr() 
        #links = corr.stack().reset_index()
        #links.columns = ['var1', 'var2','value']

        #links_filtered=links.loc[ (links['value'] >= threshold) & (links['var1'] != links['var2']) ]

        #G=nx.from_pandas_dataframe(links_filtered, 'var1', 'var2')

        ## find groups
        #l = list(G.edges())
        #taken=[False]*len(l)
        #l=map(set,l)

        #groups = merge_all(l, taken)
        
        ## compute # of groups by considering total grouped units vs. all units
        ## try to find pairing where all units are in 2 gruops, and no isolated clusters
        #flattened = np.int32([val for sublist in groups for val in sublist])
        #n_groups = len(groups)+np.max(assignment)+1-len(flattened)
    
        #if n_groups==2:
            #print ("found 2 groups, exiting")
            #break

    #if n_groups==2:
        ## loop over 2 groups and recluster
        #lengths = []
        #for group in groups: 
            #lengths.append(np.where(np.in1d(assignment, np.int32(group)))[0].shape[0])
        #print ("groups: ", lengths)
        
        #for group in groups: 
            #idx = np.where(np.in1d(assignment, np.int32(group)))[0]
        
            #print("reclustering annealed group ", wf[idx_keep][idx].shape, group)
            #RRR3_noregress(channel, wf[idx_keep][idx], sic[idx_keep][idx], gen+1, fig, grid, 
                 #False, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
                 #n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor,
                 #nshifts, assignment_global, spike_index)
                 

def regress(wf, mc):
    template = wf.mean(0)
    channels = np.where(template.ptp(0)>0)[0]
    wf_x = wf[:,:,mc]
    wf_y = wf[:,:, channels]
    A = np.matmul(np.matmul(wf_y.transpose([2,1,0]), wf_x)/2+np.matmul(wf_x.T,wf_y.transpose([2,0,1]))/2\
                  ,np.linalg.inv(np.matmul(wf_x.T, wf_x)))
    residual = wf_y - np.matmul(A, wf_x.T).transpose([2,1,0])
    return residual



def make_CONFIG2(CONFIG):
    ''' Makes a copy of several attributes of original config parameters
        to be sent into parmap function; original CONFIG can't be pickled;
    '''
    
    # make a copy of the original CONFIG object;
    # multiprocessing doesn't like the methods in original CONFIG        
    CONFIG2 = empty()
    CONFIG2.recordings=empty()
    CONFIG2.resources=empty()
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


    return CONFIG2

def run_cluster_features_chunks(spike_index_clear, n_dim_pca_compression, 
                         n_dim_pca, wf_start, wf_end, 
                         n_feat_chans, CONFIG, out_dir,
                         mfm_threshold, upsample_factor, nshifts):
    
    ''' New voltage feature based clustering; parallel version
    ''' 
    
    # Cat: TODO: Edu said the CONFIG file can be passed as a dictionary
    CONFIG2 = make_CONFIG2(CONFIG)
    
    # loop over chunks 
    gen = 0     #Set default generation for starting clustering stpe
    assignment_global = []
    spike_index = []
    
    # parallelize over chunks of data
    #res_file = CONFIG.data.root_folder+'tmp/spike_train_cluster.npy'
    #if os.path.exists(res_file)==False: 
    
    # Cat: TODO: link spike_size in CONFIG param
    spike_size = int(CONFIG.recordings.spike_size_ms*
                     CONFIG.recordings.sampling_rate//1000)
    n_processors = CONFIG.resources.n_processors
    n_channels = CONFIG.recordings.n_channels
    sampling_rate = CONFIG.recordings.sampling_rate
    root_folder = CONFIG.data.root_folder

    # select length of recording to chunk data for processing;
    # Cat: TODO: read this value from CONFIG; use initial_batch_size
    n_sec_chunk = 1200
    #min_spikes = int(n_sec_chunk * 2/3.)
    min_spikes = int(max(n_sec_chunk*0.25, 310))
    
    # determine length of processing chunk based on lenght of rec
    standardized_filename = os.path.join(CONFIG.data.root_folder, out_dir,
                                         'standarized.bin')
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # make index list for chunk/parallel processing
    # Cat: TODO: read buffer size from CONFIG file
    # Cat: TODO: ensure all data read including residuals (there are multiple
    #                places to fix this)
    buffer_size = 200
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list))
    proc_indexes = np.arange(len(idx_list))

    #if CONFIG.resources.multi_processing:
    # Cat: TODO: the logic below is hardcoded for clustering a single chunk
        
    idx = idx_list[0]
    proc_index = proc_indexes[0]
    # read chunk of data
    print ("Clustering initial chunk: ", 
            idx[0]/CONFIG.recordings.sampling_rate, "(sec)  to  ", 
            idx[1]/CONFIG.recordings.sampling_rate, "(sec)")

    # make chunk directory if not available:
    # save chunk in own directory to enable cumulative recovery 
    chunk_dir = CONFIG.data.root_folder+"/tmp/cluster/chunk_"+ \
                                                str(proc_index).zfill(6)
    if not os.path.isdir(chunk_dir):
        os.makedirs(chunk_dir)
       
    # check to see if chunk is done
    global recording_chunk
    recording_chunk = None
    if os.path.exists(chunk_dir+'/complete.npy')==False:
   
        # select only spike_index_clear that is in the chunk
        indexes_chunk = np.where(
                    np.logical_and(spike_index_clear[:,0]>=idx[0], 
                    spike_index_clear[:,0]<idx[1]))[0]
        
        spike_index_chunk = spike_index_clear[indexes_chunk]      
        
        # read recording chunk and share as global variable
        # Cat: TODO: recording_chunk should be a shared variable in 
        #            multiprocessing module;
        
        buffer_size = 200
        standardized_filename = os.path.join(CONFIG.data.root_folder,
                                            'tmp', 'standarized.bin')
        n_channels = CONFIG.recordings.n_channels
        root_folder = CONFIG.data.root_folder
        
        recording_chunk = binary_reader(idx, buffer_size, 
                    standardized_filename, n_channels)


        # Cat: TODO: this parallelization may not be optimally asynchronous
        # make arg list first
        channels = np.arange(CONFIG.recordings.n_channels)
        args_in = []
        for channel in channels:
        #for channel in [6]:
            args_in.append([channel, idx, proc_index,CONFIG2, 
                spike_index_chunk, n_dim_pca, n_dim_pca_compression,
                wf_start, wf_end, n_feat_chans, out_dir, 
                mfm_threshold, upsample_factor, nshifts])

        # Cat: TODO: have single-core option also here     
        if CONFIG.resources.multi_processing:       
            p = mp.Pool(CONFIG.resources.n_processors)
            res = p.map_async(cluster_channels_chunks_args, args_in).get(988895)
            p.close()

        else:
            res = []
            for arg_in in args_in:
                res.append(cluster_channels_chunks_args(arg_in))

        ## save simple flag that chunk is done
        ## Cat: TODO: fix this; or run chunk wise-global merge
        np.save(chunk_dir+'/complete.npy',np.arange(10))
    
    else:
        print ("... clustering previously completed...")

    
    # Cat: TODO: this logic isn't quite correct; should merge with above
    if os.path.exists(os.path.join(CONFIG.data.root_folder, 
                          'tmp/templates.npy'))==False: 

        # reload recording chunk if not already in memory
        if recording_chunk is None: 
            buffer_size = 200
            standardized_filename = os.path.join(CONFIG.data.root_folder,
                                                'tmp', 'standarized.bin')
            n_channels = CONFIG.recordings.n_channels
            root_folder = CONFIG.data.root_folder
            
            recording_chunk = binary_reader(idx, buffer_size, 
                        standardized_filename, n_channels)
                    
        # run global merge function
        spike_train, tmp_loc, templates = global_merge_all_ks(chunk_dir, 
                                        recording_chunk, CONFIG2, min_spikes)
    
        # sort by time
        indexes = np.argsort(spike_train[:,0])
        final_spike_train = spike_train[indexes]
        
        tmp_loc = np.int32(tmp_loc)

        np.save(os.path.join(CONFIG.data.root_folder, 
                          'tmp/cluster/spike_train_post_clustering_post_merge_post_cutoff.npy'), final_spike_train)
        np.save(os.path.join(CONFIG.data.root_folder, 
                          'tmp/tmp_loc.npy'), tmp_loc)
        np.save(os.path.join(CONFIG.data.root_folder, 
                          'tmp/cluster/templates_post_clustering_post_merge_post_cutoff.npy'), templates)
                          
    else:
        
        final_spike_train = np.load(os.path.join(CONFIG.data.root_folder, 
                          'tmp/cluster/spike_train_post_clustering_post_merge_post_cutoff.npy'))
        tmp_loc = np.load(os.path.join(CONFIG.data.root_folder, 
                          'tmp/cluster/tmp_loc.npy'))
        templates = np.load(os.path.join(CONFIG.data.root_folder, 
                          'tmp/cluster/templates_post_clustering_post_merge_post_cutoff.npy'))


    return final_spike_train, tmp_loc, templates


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
    
    
def cluster_channels_chunks_args(data_in):

    ''' Clustering wrapper function run chunks in parallel for each channel
    
        spike_indexes_chunk:  only spikes in the current chunk
                         Note; this is not offest to zero yet
    '''
    #data_in = data_in.get() 
    channel = data_in[0]
    idx_list = data_in[1]
    proc_index = data_in[2]
    CONFIG = data_in[3]
    spike_indexes_chunk = data_in[4]
    n_dim_pca = data_in[5]
    n_dim_pca_compression = data_in[6]
    wf_start = data_in[7]
    wf_end = data_in[8]
    n_feat_chans = data_in[9]
    out_dir = data_in[10]
    mfm_threshold = data_in[11]
    upsample_factor = data_in[12]
    nshifts = data_in[13]

    data_start = idx_list[0]
    data_end = idx_list[1]
    offset = idx_list[2]
    
    # save chunk in own directory to enable cumulative recovery 
    chunk_dir = CONFIG.data.root_folder+"/tmp/cluster/chunk_"+ \
                                                str(proc_index).zfill(6)
    if not os.path.isdir(chunk_dir):
        os.makedirs(chunk_dir)

    # check to see if chunk + channel already completed
    filename_postclustering = (chunk_dir + "/channel_"+
                                                    str(channel)+".npz")
    if os.path.exists(filename_postclustering)==False: 
        
        # starting params
        chans = [] 
        scale = 10 
        spike_size = int(CONFIG.recordings.spike_size_ms*
                         CONFIG.recordings.sampling_rate//1000)
        triageflag = False
        alignflag = True
        plotting = True
        subsample_nspikes = CONFIG.cluster.max_n_spikes

        gen = 0     #Set default generation for starting clustering stpe
        assignment_global = []
        spike_index = []
        feat_chans_cumulative = []
        shifts = []
        aligned_wfs_cumulative = []
        
        # Cat: TODO: Is this index search expensive for hundreds of chans and many 
        #       millions of spikes?  Might want to do once rather than repeat
        indexes = np.where(spike_indexes_chunk[:,1]==channel)[0]
        spike_train = spike_indexes_chunk[indexes]
        print ('Starting channel: '+str(channel)+ ', events: '+
                                                    str(spike_train.shape[0]))

        # read waveforms from recording chunk in memory
        # load waveforms with some padding then clip them
        # Cat: TODO: spike_padding to be read/fixed in CONFIG
        spike_padding = 15
        spike_train = spike_indexes_chunk[indexes]
        knn_triage_threshold = 0.90
        
        # Cat: TODO: recording_chunk is a global variable; 
        #            this might cause problems eventually
        wf = load_waveforms_from_memory(recording_chunk, data_start, 
                                        offset, spike_train, 
                                        spike_size + spike_padding)
        
        # PCA denoise waveforms before processing
        if False:
            wf_PCA = np.zeros(wf.shape)
            for ch in range(wf.shape[2]):
                _, wf_PCA[:,:,ch] = PCA(wf[:,:,ch], n_dim_pca_compression)
        else:
            wf_PCA = wf

        # plotting parameters
        if plotting:
            # Cat: TODO: this global x is not necessary, should make it local
            #global x, ax_t
            x = np.zeros(100, dtype = int)
            fig = plt.figure(figsize =(100,100))
            grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
            ax_t = fig.add_subplot(grid[13:, 6:])
        else:
            fig = []
            grid = []
            x = []
            ax_t = []
        
        # Cat: TODO: legacy code fix; don't really need indexes subsampled here
        #       Subsammpling is done inside clustering function.
        indexes_subsampled = np.arange(indexes.shape[0])
        
        # cluster
        #print (wf.shape, wf[indexes_subsampled][:,spike_padding:-spike_padding].shape)
        RRR3_noregress_recovery(channel, wf_PCA[indexes_subsampled][:,spike_padding:-spike_padding], 
             spike_train[indexes_subsampled], gen, fig, grid, x, ax_t, 
             triageflag, alignflag, plotting, n_feat_chans, 
             n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
             upsample_factor, nshifts, assignment_global, spike_index, scale,
             knn_triage_threshold)

        # finish plotting 
        if plotting: 
            #ax_t = fig.add_subplot(grid[13:, 6:])
            for i in range(CONFIG.recordings.n_channels):
                ax_t.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), alpha=0.4, 
                                                                fontsize=30)
                # fill bewteen 2SUs on each channel
                ax_t.fill_between(CONFIG.geom[i,0] + np.arange(-61,0,1)/3.,
                    -scale + CONFIG.geom[i,1], scale + CONFIG.geom[i,1], 
                    color='black', alpha=0.05)
                
            # plot max chan with big red dot                
            ax_t.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], s = 2000, 
                                                    color = 'red')
            # if at least 1 cluster is found:
            if len(spike_index)>0: 
                sic_temp = np.concatenate(spike_index, axis = 0)
                assignment_temp = np.concatenate(assignment_global, axis = 0)
                idx = sic_temp[:,1] == channel
                clusters, sizes = np.unique(assignment_temp[idx], return_counts= True)
                clusters = clusters.astype(int)

                chans.extend(channel*np.ones(clusters.size))

                labels=[]
                for i, clust in enumerate(clusters):
                    patch_j = mpatches.Patch(color = colors[clust%100], label = "size = {}".format(sizes[i]))
                    labels.append(patch_j)
                ax_t.legend(handles = labels, fontsize=30)

            # plto title
            fig.suptitle("Channel: "+str(channel), fontsize=25)
            fig.savefig(chunk_dir+"/channel_{}.png".format(channel))
            plt.close(fig)

        # Save weighted templates also:
        # Cat: TODO: note clustering is done on PCA denoised waveforms but
        #            templates are computed on original raw signal
        temp = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
        temp_std = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
        for k in range(len(spike_index)):
            idx = np.in1d(spike_train[:,0], spike_index[k][:,0])
            
            # align waveforms using original lengths; use first 1000 spikes
            wf_temp = wf[idx][:1000]
            mc = wf_temp.mean(0).ptp(0).argmax(0)
            
            # Cat: TODO: This align function returns 61 time points
            wf_temp_aligned =  align_mc_templates(wf_temp, mc, CONFIG, 
                                        spike_padding,
                                        upsample_factor, nshifts)

            temp[k] = np.mean(wf[idx],axis=0)
            temp_std[k] = robust.mad(wf[idx],axis=0)

        # save all clustered data
        np.savez(filename_postclustering, spike_index=spike_index, 
                        indexes_subsampled=indexes_subsampled,
                        templates=temp,
                        templates_std=temp_std,
                        weights=np.asarray([sic.shape[0] for sic in spike_index]))

    else: 
        # 
        data = np.load(filename_postclustering, encoding='latin1')
        spike_index = data['spike_index']

    print ("")
    print ("**********************************************************")
    print ("**** Channel ", str(channel), " # clusters: ", len(spike_index))
    print ("**********************************************************")
    print ("")
    
    # overwrite this variable just in case multiprocessing doesn't destroy it
    wf = None
    
    return channel
   
    
def cluster_channels_chunks(channel, idx_list, proc_index, CONFIG, 
                spike_indexes_chunk, n_dim_pca, wf_start, wf_end, 
                n_mad_chans, n_max_chans, out_dir, mfm_threshold, 
                upsample_factor, nshifts):

    ''' Clustering wrapper function run chunks in parallel for each channel
        spike_indexes_chunk:  only spikes in the current chunk
                         Note; this is not offest to zero yet
    '''

    data_start = idx_list[0]
    data_end = idx_list[1]
    offset = idx_list[2]
    
    # Cat: TODO: read from CONFIG
    knn_triage_threshold = 0.90
   
    # save chunk in own directory to enable cumulative recovery 
    chunk_dir = CONFIG.data.root_folder+"tmp/cluster/chunk_"+ \
                                                str(proc_index).zfill(6)
    if not os.path.isdir(chunk_dir):
        os.makedirs(chunk_dir)

    # check to see if chunk + channel already completed
    filename_postclustering = (chunk_dir + "/channel_"+
                                                    str(channel)+".npz")
    if os.path.exists(filename_postclustering)==False: 
        
        # starting params
        chans = [] 
        scale = 10 
        spike_size = int(CONFIG.recordings.spike_size_ms*
                         CONFIG.recordings.sampling_rate//1000)
        triageflag = False
        alignflag = True
        plotting = True
        subsample_nspikes = CONFIG.cluster.max_n_spikes

        gen = 0     #Set default generation for starting clustering stpe
        assignment_global = []
        spike_index = []
        feat_chans_cumulative = []
        shifts = []
        aligned_wfs_cumulative = []
        
        # Cat: TODO: Is this index search expensive for hundreds of chans and many 
        #       millions of spikes?  Might want to do once rather than repeat
        indexes = np.where(spike_indexes_chunk[:,1]==channel)[0]
        spike_train = spike_indexes_chunk[indexes]
        print ('Starting channel: '+str(channel)+ ', events: '+str(spike_train.shape[0]))

        # read waveforms from recording chunk in memory
        spike_train = spike_indexes_chunk[indexes]
        wf = load_waveforms_from_memory(recording_chunk, 
                                       data_start, offset, 
                                       spike_train, spike_size)
                                        
        # plotting parameters
        if plotting:
            # Cat: TODO: this global x is not necessary, should make it local
            global x, ax_t
            x = np.zeros(100, dtype = int)
            fig = plt.figure(figsize =(100,100))
            grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
        
        # legacy code fix
        indexes_subsampled = np.arange(indexes.shape[0])
        
        # initialize plots:
        if plotting: 
            ax_t = fig.add_subplot(grid[13:, 6:])

        # cluster
        RRR3_noregress_recovery(channel, wf[indexes_subsampled], 
             spike_train[indexes_subsampled], gen, fig, grid, 
             triageflag, alignflag, plotting, n_mad_chans, n_max_chans, 
             n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
             upsample_factor, nshifts, assignment_global, spike_index, scale,
             knn_triage_threshold)

        # finish plotting 
        if plotting: 
            #ax_t = fig.add_subplot(grid[13:, 6:])
            for i in range(CONFIG.recordings.n_channels):
                ax_t.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), alpha=0.4, 
                                                                fontsize=30)
                # fill bewteen 2SUs on each channel
                ax_t.fill_between(CONFIG.geom[i,0] + np.arange(-61,0,1)/3.,
                    -scale + CONFIG.geom[i,1], scale + CONFIG.geom[i,1], 
                    color='black', alpha=0.05)
                
            # plot max chan with big red dot                
            ax_t.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], s = 2000, 
                                                    color = 'red')
            # if at least 1 cluster is found:
            if len(spike_index)>0: 
                sic_temp = np.concatenate(spike_index, axis = 0)
                assignment_temp = np.concatenate(assignment_global, axis = 0)
                idx = sic_temp[:,1] == channel
                clusters, sizes = np.unique(assignment_temp[idx], return_counts= True)
                clusters = clusters.astype(int)

                chans.extend(channel*np.ones(clusters.size))

                labels=[]
                for i, clust in enumerate(clusters):
                    patch_j = mpatches.Patch(color = colors[clust%100], label = "size = {}".format(sizes[i]))
                    labels.append(patch_j)
                ax_t.legend(handles = labels, fontsize=30)

            # plto title
            fig.suptitle("Channel: "+str(channel), fontsize=25)
            fig.savefig(chunk_dir+"/channel_{}.png".format(channel))
            plt.close(fig)

        # Save weighted templates also:
        temp = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
        for k in range(len(spike_index)):
            idx = np.in1d(spike_train[:,0], spike_index[k][:,0])
            temp[k] = np.mean(wf[idx],axis=0)

        # save all clustered data
        np.savez(filename_postclustering, spike_index=spike_index, 
                        indexes_subsampled=indexes_subsampled,
                        templates=temp,
                        weights=np.asarray([sic.shape[0] for sic in spike_index]))

    else: 
        # 
        data = np.load(filename_postclustering, encoding='latin1')
        spike_index = data['spike_index']

    print ("")
    print ("**********************************************************")
    print ("**** Channel ", str(channel), " # clusters: ", len(spike_index))
    print ("**********************************************************")
    print ("")
    
   
    
def global_merge_all_ks(chunk_dir, recording_chunk, CONFIG, min_spikes):

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
    
    #ctr_id = 0 
    # Cat: TODO: make sure this step is correct
    for channel in range(CONFIG.recordings.n_channels):
        data = np.load(chunk_dir+'/channel_{}.npz'.format(channel))
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

    print("  cluster templates/spiketrain before merge/cutoff: ", templates.shape, spike_indexes.shape)

    np.save(CONFIG.data.root_folder+'/tmp/cluster/templates_post_clustering_before_merge_before_cutoff.npy', templates)
    np.save(CONFIG.data.root_folder+'/tmp/cluster/spike_train_post_clustering_before_merge_before_cutoff.npy', spike_indexes)
    #print (np.unique(spike_indexes[:,1]))

    ''' ************************************************
        ********** COMPUTE SIMILARITY MATRIX ***********
        ************************************************
    '''

    # ************** GET SIM_MAT ****************
    # initialize cos-similarty matrix and boolean version 
    # Cat: TODO: it seems most safe for this matrix to be recomputed 
    #            everytime wheneve templates.npy file doesn't exist
    # Cat: TODO: this should be parallized, takes several mintues for 49chans, 
    #            should take much less
    cos_sim_file = (CONFIG.data.root_folder+ '/tmp/cluster/cos_sim_vector_post_cluster.npy')
    if os.path.exists(cos_sim_file)==False:
        sim_temp = calc_cos_sim_vector(templates, CONFIG)
        sim_temp[np.diag_indices(sim_temp.shape[0])] = 0
        sim_temp[np.tril_indices(sim_temp.shape[0])] = 0
        
        np.save(cos_sim_file, sim_temp)

    else:
        sim_temp = np.load(cos_sim_file)        
        sim_temp[np.diag_indices(sim_temp.shape[0])] = 0
        sim_temp[np.tril_indices(sim_temp.shape[0])] = 0
        
    global_merge_file = (CONFIG.data.root_folder+ '/tmp/cluster/merge_matrix_post_clustering.npz')
    if os.path.exists(global_merge_file)==False:
        sim_mat = run_KS_test_new(sim_temp, spike_indexes, recording_chunk, 
                                        CONFIG, chunk_dir, plotting = False)
        
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

        max_chan = weighted_average.ptp(0).argmax(0)

        tmp_loc.append(max_chan)
    
    # convert templates to : (n_channels, waveform_size, n_templates)
    templates = np.float32(templates_final)
    templates = np.swapaxes(templates, 2,0)
     
    np.save(CONFIG.data.root_folder+'/tmp/cluster/templates_post_clustering_post_merge_pre_cutoff.npy', templates)
    np.save(CONFIG.data.root_folder+'/tmp/cluster/spike_train_post_clustering_post_merge_pre_cutoff.npy', final_spike_train)

    print("  cluster templates/spike train after merge, pre-spike cutoff: ", templates.shape, final_spike_train.shape)

    ''' ************************************************
        ************* DELETE TOO FEW SPIKES ************
        ************************************************
    '''

    # Cat: TODO: Read this threshold from CONFIG
    #            Maybe needs to be in fire rate (Hz) not absolute spike #s
    final_spike_train_cutoff = []
    ctr = 0
    del_ctr = []
    for unit in np.unique(final_spike_train[:,1]):
        idx_temp = np.where(final_spike_train[:,1]==unit)[0]
        if idx_temp.shape[0]>=min_spikes:
            temp_train = final_spike_train[idx_temp]
            temp_train[:,1]=ctr
            final_spike_train_cutoff.append(temp_train)
            ctr+=1
        else:
            del_ctr.append(unit)
                
    final_spike_train_cutoff = np.vstack(final_spike_train_cutoff)
    templates = np.delete(templates,del_ctr,axis=2)

    # these are saved outside
    #np.save(CONFIG.data.root_folder+'/tmp/deconv/templates_post_deconv_post_merge_post_cutoff.npy', templates)
    #np.save(CONFIG.data.root_folder+'/tmp/deconv/spike_train_post_deconv_post_merge_post_cutoff.npy', final_spike_train_cutoff)
      
    print("  cluster templates/spike train after merge, post-spike cutoff: ", templates.shape, final_spike_train_cutoff.shape)
    
    return final_spike_train_cutoff, tmp_loc, templates

    
def global_merge_all_ks_deconv(deconv_chunk_dir, recording_chunk, units, 
                                CONFIG, min_spikes):

    ''' Function that cleans noisy templates and merges the rest    
    '''
    
    # Cat: TODO: this fucntion largely copies the clustering-step merge function
    #             Would be good to merge these 2 functions somehow
    
    n_channels = CONFIG.recordings.n_channels

    # convert clusters to templates; keep track of weights
    templates = []
    templates_std = []
    weights = []
    spike_ids = []
    spike_indexes = []
    channels = np.arange(n_channels)
    tmp_loc = []
    
    for unit in units:
        data = np.load(deconv_chunk_dir+'/unit_'+str(unit).zfill(6)+'.npz')
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

    print("  deconv templates/spiketrain before merge/cutoff: ", templates.shape, spike_indexes.shape)
    np.save(CONFIG.data.root_folder+'/tmp/deconv/templates_post_deconv_before_merge_before_cutoff.npy', templates)
    np.save(CONFIG.data.root_folder+'/tmp/deconv/spike_train_post_deconv_before_merge_before_cutoff.npy', spike_indexes)

    ''' ************************************************
        ********** COMPUTE SIMILARITY MATRIX ***********
        ************************************************
    '''

    # ************** GET SIM_MAT ****************
    # initialize cos-similarty matrix and boolean version 
    # Cat: TODO: it seems most safe for this matrix to be recomputed 
    #            everytime wheneve templates.npy file doesn't exist
    # Cat: TODO: this should be parallized, takes several mintues for 49chans, 
    #            should take much less
    cos_sim_file = (CONFIG.data.root_folder+ '/tmp/deconv/cos_sim_vector_post_deconv.npy')
    if os.path.exists(cos_sim_file)==False:
        sim_temp = calc_cos_sim_vector(templates, CONFIG)
        sim_temp[np.diag_indices(sim_temp.shape[0])] = 0
        sim_temp[np.tril_indices(sim_temp.shape[0])] = 0
        
        np.save(cos_sim_file, sim_temp)

    else:
        sim_temp = np.load(cos_sim_file)        
        sim_temp[np.diag_indices(sim_temp.shape[0])] = 0
        sim_temp[np.tril_indices(sim_temp.shape[0])] = 0
        
    global_merge_file = (CONFIG.data.root_folder+ '/tmp/deconv/merge_matrix_post_deconv.npz')
    if os.path.exists(global_merge_file)==False:
        sim_mat = run_KS_test_new(sim_temp, spike_indexes, recording_chunk, 
                                        CONFIG, deconv_chunk_dir, plotting = False)
        
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

        max_chan = weighted_average.ptp(0).argmax(0)

        tmp_loc.append(max_chan)
    
    # convert templates to : (n_channels, waveform_size, n_templates)
    templates = np.float32(templates_final)
    templates = np.swapaxes(templates, 2,0)
     
    np.save(CONFIG.data.root_folder+'/tmp/deconv/templates_post_deconv_post_merge_pre_cutoff.npy', templates)
    np.save(CONFIG.data.root_folder+'/tmp/deconv/spike_train_post_deconv_post_merge_pre_cutoff.npy', final_spike_train)

    print("  templates/spike train after merge, pre-spike cutoff: ", templates.shape, final_spike_train.shape)

    ''' ************************************************
        ************* DELETE TOO FEW SPIKES ************
        ************************************************
    '''

    # Cat: TODO: Read this threshold from CONFIG
    #            Maybe needs to be in fire rate (Hz) not absolute spike #s
    #min_spikes = 300
    final_spike_train_cutoff = []
    ctr = 0
    del_ctr = []
    for unit in np.unique(final_spike_train[:,1]):
        idx_temp = np.where(final_spike_train[:,1]==unit)[0]
        #print (idx_temp.shape)
        if idx_temp.shape[0]>=min_spikes:
            temp_train = final_spike_train[idx_temp]
            temp_train[:,1]=ctr
            final_spike_train_cutoff.append(temp_train)
            
            ctr+=1
        else:
            del_ctr.append(unit)
                
    final_spike_train_cutoff = np.vstack(final_spike_train_cutoff)
    templates = np.delete(templates,del_ctr,axis=2)

    # these are saved outside
    np.save(CONFIG.data.root_folder+'/tmp/deconv/templates_post_deconv_post_merge_post_cutoff.npy', templates)
    np.save(CONFIG.data.root_folder+'/tmp/deconv/spike_train_post_deconv_post_merge_post_cutoff.npy', final_spike_train_cutoff)
   
    print("  deconv templates/spike train after merge, post-spike cutoff: ", templates.shape, final_spike_train_cutoff.shape)
    
    return final_spike_train_cutoff, tmp_loc, templates
   
   
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
    

def run_KS_test_new(sim_temp, spike_train, recording_chunk, CONFIG, path_dir, 
                     plotting = False):
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
    N = 300
    n_chan = CONFIG.recordings.n_channels
    wf_start = 0
    wf_end = 31
    upsample_factor = 5
    nshifts = 15
    num_max_channels = 0
    num_mad_channels = 2
    num_pca_components = 3
    knn_triage_threshold = 80
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
                path_dir])

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
    idx1 = np.random.choice(idx1, min(N, idx1.size))
    
    ## choose N events from pair[1]
    idx2 = np.random.choice(idx2, min(N, idx2.size))
    idx = np.concatenate([idx[idx1], idx[idx2]])
    idx1 = np.where(spike_train[idx,1] == pair[0])[0]
    idx2 = np.where(spike_train[idx,1] == pair[1])[0]
    
    ## read waveforms
    #offset = 200

    re = RecordingExplorer(CONFIG.data.root_folder+'/tmp/standarized.bin', 
                   path_to_geom = CONFIG.data.root_folder+CONFIG.data.geometry, 
                   spike_size = 30, 
                   neighbor_radius = 100, 
                   dtype = 'float32',
                   n_channels = CONFIG.recordings.n_channels, 
                   data_order = 'samples')
    
    wf = re.read_waveforms(spike_train[idx,0])
    
    if wf.shape[0]<n_sample_spikes:
        print (" insufficient spikes, skipping...")
        return 
        #continue
        
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
    align_wf = align_mc(wf, wf.mean(0).ptp(0).argmax(), CONFIG, upsample_factor, nshifts)
    
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
    pca_wf, pca_reconstruct = PCA(data_in, num_pca_components)
    
    ##triage to remove outliers to remove compression issues
    idx_keep = knn_triage(knn_triage_threshold, pca_wf)
    data_in = data_in[idx_keep]
    
    ## run PCA on triaged data
    pca_wf, pca_reconstruct = PCA(data_in, num_pca_components)
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
    

#def run_KS_test(sim_temp, mean, std, CONFIG, resampling = True, plotting = False):
    
    #cfg = CONFIG
    #path_fig = '/media/cat/1TB/liam/49channels/data1_allset/tmp/cluster/chunk_000000/'
    #size_resample = 500
    #n_chan = CONFIG.recordings.n_channels

    #row,column = np.where(sim_temp > 0.90)
    #print(row.size)
    #sim_mat = np.zeros(sim_temp.shape, dtype = bool)
    #Y = np.random.normal(size = 1500)
    #for i, pair in enumerate(zip(row, column)):
        #print('*********************', pair, "**************************")
        #if resampling:
            #to_align = np.concatenate([mean[[pair]], std[[pair]]], axis = 2)
            
            ## align both waveforms and std in one step
            #aligned = align_mc(to_align, to_align[:,:,:49].mean(0).ptp(0).argmax(0), cfg, 5, 15)
    
            #aligned_means = aligned[:,:,:n_chan]
            #aligned_stds = aligned[:,:,n_chan:]
            #wf1 = np.random.normal(loc = aligned_means[0:1].repeat(size_resample, axis = 0).ravel(),
                         #scale = aligned_stds[0:1].repeat(size_resample, axis = 0).ravel()).reshape([size_resample,aligned.shape[1],n_chan])
            #wf2 = np.random.normal(loc = aligned_means[1:2].repeat(size_resample, axis = 0).ravel(),
                         #scale = aligned_stds[1:2].repeat(size_resample, axis = 0).ravel()).reshape([size_resample,aligned.shape[1],n_chan])
            #align_wf = np.concatenate([wf1,wf2], axis = 0)
            #y_act = np.concatenate([np.zeros(size_resample), np.ones(size_resample)])
            
            #feat_channels, max_chans, mad_chans = get_feat_channels_ks_test(aligned_means.mean(0), aligned_means, 3, 3)
            #print(feat_channels, max_chans, mad_chans)
            
            ## 
            #wf_start = 0
            #wf_end = 31
            #data_in = align_wf[:, wf_start:wf_end][:,:,feat_channels].swapaxes(1,2).reshape(align_wf.shape[0],-1)
            #pca_wf, pca_reconstruct = PCA(data_in, 3)
            #lda = LDA(n_components = 1)
            #trans = lda.fit_transform(pca_wf, y_act)
            #trans = (trans - trans.mean())/ trans.std()
            #lin = np.arange(trans.min(), trans.max(), 0.01)
            #ks_dist = calc_ks_dist(trans, Y, lin)
            #if ks_dist < 0.10:
                #print('^^^^^^^^^^^^^^ merged ^^^^^^^^^^^^^^^^^^')
                #sim_mat[pair[0], pair[1]] = sim_mat[pair[1], pair[0]]  = True
        
        #else:
            #idx = np.where(np.in1d(spike_train[:,1], pair))[0]
            #idx1 = np.where(spike_train[idx,1] == pair[0])[0]
            #idx2 = np.where(spike_train[idx,1] == pair[1])[0]
            #idx1 = np.random.choice(idx1, min(500, idx1.size))
            #idx2 = np.random.choice(idx2, min(500, idx2.size))
            #idx = np.concatenate([idx[idx1], idx[idx2]])
            #idx1 = np.where(spike_train[idx,1] == pair[0])[0]
            #idx2 = np.where(spike_train[idx,1] == pair[1])[0]
            
            ## Cat: TODO: readwaveforms
            #wf = re.read_waveforms(spike_train[idx,0])


            #align_wf, _ = align_mc(wf, wf.mean(0).ptp(0).argmax(), cfg, 5, 15)
            #to_get_feat = np.concatenate([align_wf[idx1].mean(0, keepdims =True), align_wf[idx2].mean(0, keepdims =True)], axis = 0)
            #feat_channels, max_chans, mad_chans = get_feat_channels(to_get_feat.mean(0), to_get_feat, 3, 3)
            #wf_start = 0
            #wf_end = 31
            #data_in = align_wf[:, wf_start:wf_end][:,:,feat_channels].swapaxes(1,2).reshape(wf.shape[0],-1)
            #pca_wf, pca_reconstruct, _ = PCA(data_in, 3)
            #idx_keep = triage(90, pca_wf)
            #data_in = data_in[idx_keep]
            #pca_wf, pca_reconstruct, _ = PCA(data_in, 3)
            #idx1 = spike_train[idx,1][idx_keep] == pair[0]
            #idx2 = spike_train[idx,1][idx_keep] == pair[1]
        

            #lda = LDA(n_components = 1)
            #trans = lda.fit_transform(pca_wf, spike_train[idx,1][idx_keep])
            #trans = (trans - trans.mean())/ trans.std()
            #lin = np.arange(trans.min(), trans.max(), 0.01)
            #ks_dist = calc_ks_dist(trans, Y, lin)
            #if ks_dist < 0.10:
                #print('^^^^^^^^^^^^^^ merged ^^^^^^^^^^^^^^^^^^')
                #sim_mat[pair[0], pair[1]] = sim_mat[pair[1], pair[0]] = True
            
        ##if plotting:
            ##if resampling:
                ##plt.figure(figsize = (40,50))    
                ##grid = plt.GridSpec(10,8, hspace =0.2, wspace = 0.2)
                ##plt.subplot(grid[:3,:8])
                ##plt.plot(data_in.T*10+30, 'k',alpha = 0.2)
                ##plt.plot(data_in[:500].mean(0).T*10+30, 'b',alpha = 0.2)
                ##plt.plot(data_in[500:].mean(0).T*10+30, 'r',alpha = 0.2)
                ##plt.plot(pca_reconstruct.T*10, 'y', alpha = 0.2)
                ##plt.plot(pca_reconstruct[:500].mean(0).T*10, 'b', alpha = 0.2)
                ##plt.plot(pca_reconstruct[500:].mean(0).T*10, 'r', alpha = 0.2)
                ##plt.subplot(grid[3:5,:2])
                ##plt.scatter(pca_wf[:500,0], pca_wf[:500,1], c = colors[pair[0]], alpha = 0.5)
                ##plt.scatter(pca_wf[500:,0], pca_wf[500:,1], c = colors[pair[1]], alpha = 0.5)
                ##plt.xlabel('PCA 0')
                ##plt.ylabel('PCA 1')
                ##plt.subplot(grid[3:5,2:4])
                ##plt.scatter(pca_wf[:500,0], pca_wf[:500,2], c = colors[pair[0]], alpha = 0.5)
                ##plt.scatter(pca_wf[500:,0], pca_wf[500:,2], c = colors[pair[1]], alpha = 0.5)
                ##plt.xlabel('PCA 0')
                ##plt.ylabel('PCA 2')
                ##plt.subplot(grid[3:5,4:6])
                ##plt.scatter(pca_wf[:500,2], pca_wf[:500,1], c = colors[pair[0]], alpha = 0.5)
                ##plt.scatter(pca_wf[500:,2], pca_wf[500:,1], c = colors[pair[1]], alpha = 0.5)
                ##plt.xlabel('PCA 2')
                ##plt.ylabel('PCA 1')
                ##plt.subplot(grid[3:5,6:8])
                ##plt.hist(trans[:500], bins = 100, color = colors[pair[0]])
                ##plt.hist(trans[500:], bins = 100, color = colors[pair[1]])
                ##plt.subplot(grid[5:,:8])

                ##plt.plot(CONFIG.geom[:,0] + np.arange(31)[:,np.newaxis], align_wf[:500].mean(0)*10+CONFIG.geom[:,1], c = 'b')
                ##plt.plot(CONFIG.geom[:,0] + np.arange(31)[:,np.newaxis], align_wf[500:].mean(0)*10+CONFIG.geom[:,1], c = 'r')
                ##plt.scatter(CONFIG.geom[max_chans,0],CONFIG.geom[max_chans,1]+5,s = 200,c= 'green')
                ##plt.scatter(CONFIG.geom[mad_chans,0]+5,CONFIG.geom[mad_chans,1]+5,s = 200,c= 'k')
                ##if ks_dist > 0.1:
                    ##plt.scatter(-800, 450, s = 500,c= 'r')
                ##else:
                    ##plt.scatter(-800, 450, s = 500,c= 'g')
                ##for chan in range(49):
                    ##plt.text(CONFIG.geom[chan,0], CONFIG.geom[chan,1]-5, '{}'.format(chan))
                ##plt.title('Sampled, Cosine_distance = {}, ks_distance = {}'.format(sim_temp[pair[0], pair[1]], ks_dist), fontsize = 25)
                ##plt.savefig(path_fig + 'cluster_{}_{}.png'.format(pair[0], pair[1]))
            ##else:
                ##plt.figure(figsize = (40,50))    
                ##grid = plt.GridSpec(10,8, hspace =0.2, wspace = 0.2)
                ##plt.subplot(grid[:3,:])
                ##plt.plot(data_in.T*10+30, 'k',alpha = 0.2)
                ##plt.plot(data_in[idx1].mean(0).T*10+30, 'b',alpha = 0.2)
                ##plt.plot(data_in[idx2].mean(0).T*10+30, 'r',alpha = 0.2)
                ##plt.plot(pca_reconstruct.T*10, 'y', alpha = 0.2)
                ##plt.plot(pca_reconstruct[idx1].mean(0).T*10, 'b', alpha = 0.2)
                ##plt.plot(pca_reconstruct[idx2].mean(0).T*10, 'r', alpha = 0.2)
                ##plt.subplot(grid[3:5,:2])
                ##plt.scatter(pca_wf[:,0], pca_wf[:,1], c = colors[spike_train[idx,1][idx_keep].astype(int)], alpha = 0.5)
                ##plt.xlabel('PCA 0')
                ##plt.ylabel('PCA 1')
                ##plt.subplot(grid[3:5,2:4])
                ##plt.scatter(pca_wf[:,0], pca_wf[:,2], c = colors[spike_train[idx,1][idx_keep].astype(int)], alpha = 0.5)
                ##plt.xlabel('PCA 0')
                ##plt.ylabel('PCA 2')
                ##plt.subplot(grid[3:5,4:6])
                ##plt.scatter(pca_wf[:,2], pca_wf[:,1], c = colors[spike_train[idx,1][idx_keep].astype(int)], alpha = 0.5)
                ##plt.xlabel('PCA 2')
                ##plt.ylabel('PCA 1')
                ##plt.subplot(grid[3:5,6:8])
                ##plt.hist(trans[idx1], bins = 100, color = colors[pair[0]])
                ##plt.hist(trans[idx2], bins = 100, color = colors[pair[1]])
                ##plt.subplot(grid[5:,:])
                ##plt.plot(CONFIG.geom[:,0] + np.arange(51)[:,np.newaxis], align_wf[idx_keep][idx1].mean(0)*10+CONFIG.geom[:,1], c = 'b')
                ##plt.plot(CONFIG.geom[:,0] + np.arange(51)[:,np.newaxis], align_wf[idx_keep][idx2].mean(0)*10+CONFIG.geom[:,1], c = 'r')
                ##plt.scatter(CONFIG.geom[max_chans,0],CONFIG.geom[max_chans,1]+5,s = 200,c= 'green')
                ##plt.scatter(CONFIG.geom[mad_chans,0]+5,CONFIG.geom[mad_chans,1]+5,s = 200,c= 'k')
                ##if ks_dist > 0.1:
                    ##plt.scatter(-800, 450, s = 500,c= 'r')
                ##else:
                    ##plt.scatter(-800, 450, s = 500,c= 'g')
                ##for chan in range(49):
                    ##plt.text(CONFIG.geom[chan,0], CONFIG.geom[chan,1]-5, '{}'.format(chan))
                ##plt.title('Actual, Cosine_distance = {}, ks_distance = {}'.format(sim_temp[pair[0], pair[1]], ks_dist), fontsize = 25)
                ##plt.show()
    
    #return sim_mat
    
    
def global_merge_all_old(chunk_dirs, CONFIG):

    ''' Function that cleans noisy templates and merges the rest    
    '''
    
    n_channels = CONFIG.recordings.n_channels

    # convert clusters to templates; keep track of weights
    templates = []
    weights = []
    spike_ids = []
    spike_indexes = []
    channels = np.arange(n_channels)
    tmp_loc = []
    
    for chunk_dir in chunk_dirs:
        data = np.load(chunk_dir+'/channels_merged.npz')
        templates.append(data['templates'].T)
        #print (data['templates'].shape)
        weights.append(data['weights'])
       
        spike_index = data['spike_train_merged']
        for s in spike_index:
            spike_indexes.append(s[:,0])
    
    spike_indexes = np.array(spike_indexes)        
    templates = np.vstack(templates)
    weights = np.hstack(weights)
    
    # delete clusters < 300 spikes
    min_spikes = 300
    idx_delete = []
    for s in range(len(spike_indexes)):
        if spike_indexes[s].shape[0]<min_spikes:
            idx_delete.append(s)

    templates = np.delete(templates,idx_delete,axis=0)
    weights = np.delete(weights,idx_delete,axis=0)
    spike_indexes = np.delete(spike_indexes,idx_delete,axis=0)

    
    # initialize cos-similarty matrix and boolean version 
    sim_mat = np.zeros((templates.shape[0], templates.shape[0]), 'bool')    
    sim_temp = np.zeros((templates.shape[0], templates.shape[0]), 'float32')

    global_merge_file = (CONFIG.data.root_folder+ '/tmp/global_merge_matrix.npz')
    if os.path.exists(global_merge_file)==False:
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

        np.savez(global_merge_file, 
                 sim_mat_sum = sim_mat_sum, 
                 sim_temp_sum = sim_temp_sum)

    else:
        data = np.load(global_merge_file)
        sim_mat_sum= data['sim_mat']
        sim_temp_sum = data['sim_temp']
                
    # ********************************************************
    # compute connected nodes and sum spikes over them
    G = nx.from_numpy_array(sim_mat_sum)
    final_spike_indexes = []
    final_template_indexes = []
    #print (nx.number_connected_components(G))
    for i, cc in enumerate(nx.connected_components(G)):
        final_template_indexes.append(list(cc))
        sic = np.zeros(0, dtype = int)
        for j in cc:
            sic = np.concatenate([sic, spike_indexes[j]])
        final_spike_indexes.append(np.concatenate([sic[:,np.newaxis], i*np.ones([sic.size,1],dtype = int)],axis = 1))

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

        max_chan = weighted_average.ptp(0).argmax(0)

        tmp_loc.append(max_chan)
    
    # convert templates to : (n_channels, waveform_size, n_templates)
    templates = np.float32(templates_final)
    templates = np.swapaxes(templates, 2,0)
     
    return final_spike_train, tmp_loc, templates
   
    
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
    
    
def global_merge(chunk_dir, channels, CONFIG):
    ''' Function that cleans noisy templates and merges the rest    
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


    # compute local merge
    
    # make list of templates by channel
    templates_list_channels = []
    
    # parallelize over channels
    parmap(calc_local_merge_parallel)

    # compute neighbour merge 
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
    
    
def local_merge(channel, spike_index, sic, wf, CONFIG):
    ''' Function that cleans noisy templates and merges the rest    
    '''
    
    plotting = True
    
    n_chans = wf.shape[2]

    CONFIG.cluster.merge_threshold = 1.0

    # Cat: TODO: delete noisy/messy/lowSNR/few spike templates
    
    # make templates on all chans for each cluster from raw waveforms 
    temp = np.zeros((len(spike_index),wf.shape[1],wf.shape[2]),'float32')
    for t in range(len(spike_index)): 
        idx = np.in1d(sic[:,0], spike_index[t][:,0])
        temp[t] = np.mean(wf[idx],axis=0)

    # find merge matrix across all clusters
    sim_mat, sim_mat_floats, templates_aligned = calc_local_merge(channel, temp, temp, CONFIG)

    # plot matrix
    if plotting: 
            plot_merge_matrix(channel, n_chans, temp, CONFIG, sim_mat, 
                      sim_mat_floats, spike_index, templates_aligned, "premerge")

    # run merge until convergence
    # Cat: TODO: this technically should be inside loop below
    
    spike_index_new = merge_spikes(sim_mat, spike_index)

    while len(spike_index_new)!=temp.shape[0]:

        # compute templates again: 
        temp = np.zeros((len(spike_index_new),wf.shape[1],wf.shape[2]),'float32')
        for t in range(len(spike_index_new)): 
            idx = np.in1d(sic[:,0], spike_index_new[t][:,0])
            temp[t] = np.mean(wf[idx],axis=0)

        sim_mat, sim_mat_floats, templates_aligned = calc_local_merge(channel, temp, temp, CONFIG)

        spike_index_new = merge_spikes(sim_mat, spike_index_new)

    # compute final spike lists: 
    temp = np.zeros((len(spike_index_new),wf.shape[1],wf.shape[2]),'float32')
    for t in range(len(spike_index_new)): 
        idx = np.in1d(sic[:,0], spike_index_new[t][:,0])
        temp[t] = np.mean(wf[idx],axis=0)

    # recompute sim_mat_floats
    sim_mat, sim_mat_floats, templates_aligned = calc_local_merge(channel, temp, temp, CONFIG)

    # plot final merge table
    if plotting: 
            plot_merge_matrix(channel, n_chans, temp, CONFIG, sim_mat, sim_mat_floats, 
                      spike_index_new, templates_aligned, "postmerge")

    # save waveforms for merged clusters for each channel
    wf_array = []
    for k in range(len(spike_index_new)):
        idx = np.in1d(sic[:,0], spike_index_new[k][:,0])
        wf_array.append(wf[idx])
        
    np.save(CONFIG.data.root_folder+'tmp/cluster/channel_'+
                                            str(channel)+'_clusters.npy', wf_array)
    
    np.savez(CONFIG.data.root_folder+'tmp/cluster/channel_'+
                            str(channel)+'_weighted_templates', 
                            templates=temp, 
                            weights=np.asarray([sic.shape[0] for sic in spike_index_new]))
    
        
    return spike_index_new



def inter_channel_merge(base_temp, com_temp, size_base, size_com, clusters_base, clusters_com, tmp_loc):
    mc_base = base_temp.ptp(1).argmax(1)
    mc_com = com_temp.ptp(1).argmax(1)
    sim_mat = np.zeros([mc_base.shape[0], mc_com.shape[0]], dtype = bool)
    sim_temp = np.zeros([mc_base.shape[0], mc_com.shape[0]])
    finished_pairs = []
    for i in tqdm_notebook(range(mc_base.shape[0])):
        
        sim_mat[i,i] = True
        for j in np.where(np.in1d (mc_com, np.where(CONFIG.neigh_channels[mc_base[i]])[0]))[0]:
            if j == i or [min(i,j), max(i,j)] in finished_pairs:
                continue
            feat_channels = np.where(np.logical_or(base_temp[i].ptp(0)>2, com_temp[j].ptp(0)>2))[0]
            chan = feat_channels[(base_temp[i,:, feat_channels] + com_temp[j,:, feat_channels]).T.ptp(0).argmax(0)]
            bt_aligned = align_mc(base_temp[i:i+1,:,:], chan, base_temp[i,:,chan], 20, 501)[0]
            ct_aligned = align_mc(com_temp[j:j+1,:,:], chan, base_temp[i,:,chan], 20, 501)[0]
            
            temp = (bt_aligned[:,feat_channels]*ct_aligned[:,feat_channels]).sum()
            
            if (bt_aligned[:,feat_channels]**2).sum() > (ct_aligned[:,feat_channels]**2).sum():
                temp = temp/ (bt_aligned[:,feat_channels]**2).sum()
            else:
                temp = temp/ (ct_aligned[:,feat_channels]**2).sum()
            
            if temp>0.90:
                sim_mat[i, j] = sim_mat[j, i] = True
                sim_temp[i, j] = sim_temp[j, i] = temp
                finished_pairs.append([min(i,j), max(i,j)])
            else:
                finished_pairs.append([min(i,j), max(i,j)])

    return sim_mat, sim_temp


def merge_spikes(sim_mat, spike_index):
    ''' Merge spikes from units together based on merge matrix sim_mat
    ''' 
        
    # find groups to be merged
    merge_list = []
    for k in range(sim_mat.shape[0]):
        merge_list.append(np.where(sim_mat[k]==True)[0].tolist())
    
    for i in range(2):
        for k in range(len(merge_list)):
            for p in range(k+1, len(merge_list), 1):
                if len(np.intersect1d(merge_list[k],merge_list[p]))>0:
                    merge_list[k]=np.union1d(merge_list[k],merge_list[p]).tolist()
                    merge_list[p]=np.union1d(merge_list[k],merge_list[p]).tolist()
    
    # generate unique groups to be merged, including singles 
    unique_data = [list(x) for x in set(tuple(x) for x in merge_list)]
    print ("unique merge lists: ", unique_data)
   
    # merge 
    spike_list = []
    ctr=0
    for k in range(len(unique_data)):
        temp_list=[]
        for p in range(len(unique_data[k])):
            idx = unique_data[k][p]
            temp_list.append(spike_index[idx])

        spike_index_new = np.vstack(temp_list)
        spike_index_new[:,1]=ctr    
        spike_list.append(spike_index_new)
        ctr+=1
        
    spike_index_new = spike_list
    return spike_index_new

def plot_merge_matrix(channel, n_chans, temp, CONFIG, sim_mat, sim_mat_floats, 
                      spike_index, templates_aligned, plot_string):

    fig = plt.figure(figsize =(50,50))

    merge_threshold = 2
    clrs = ['blue','red']
    ctr=0
    scale = 10
    for i in range(temp.shape[0]):
        for j in range(i, temp.shape[0]):
            ctr+=1
            ax=plt.subplot(sim_mat.shape[0],sim_mat.shape[0],i*temp.shape[0]+j+1)
            
            for k in range(n_chans):
                plt.text(CONFIG.geom[k,0], CONFIG.geom[k,1], str(k), fontsize=10)
            
            plt.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], c = 'red', s = 400, alpha=.6)

            if i==j: 
                plt.plot(CONFIG.geom[:,0] + np.arange(-temp.shape[1],0)[:,np.newaxis]/1.5, 
                                CONFIG.geom[:,1] + temp[i]*scale, c='black', alpha=1)

                plt.title("Unit: "+str(i)+ " " + str(spike_index[i].shape[0]), fontsize=10)

            else:                 
                
                plt.plot(CONFIG.geom[:,0] + np.arange(-templates_aligned[i,j,0].shape[0],0)[:,np.newaxis]/1.5, 
                                    CONFIG.geom[:,1] + templates_aligned[i,j,0]*scale, c=clrs[0], alpha=0.4)

                plt.plot(CONFIG.geom[:,0] + np.arange(-templates_aligned[i,j,1].shape[0],0)[:,np.newaxis]/1.5, 
                                    CONFIG.geom[:,1] + templates_aligned[i,j,1]*scale, c=clrs[1], alpha=0.4)

                # plot feat channel colours
                feat_chans = np.where(temp[i].ptp(0)>merge_threshold)[0]
                plt.scatter(CONFIG.geom[feat_chans,0], CONFIG.geom[feat_chans,1], c =clrs[0], s = 50,alpha=0.5)

                feat_chans = np.where(temp[j].ptp(0)>merge_threshold)[0]
                plt.scatter(CONFIG.geom[feat_chans,0]+5, CONFIG.geom[feat_chans,1], c = clrs[1], s = 50,alpha=0.5)
                 

                if sim_mat[i][j]==True:
                    plt.title(r"$\bf{TO MERGE}$" + "\n"+str(i) + " blue: "+
                            str(spike_index[i].shape[0])+"  " + str(j) + " red: "+
                            str(spike_index[j].shape[0])+" "+str(round(sim_mat_floats[i,j],2)),
                            fontsize=10)
                else:
                    plt.title(str(i) + " blue: "+str(spike_index[i].shape[0])+"  " + str(j) + " red: "+
                            str(spike_index[j].shape[0])+" "+str(round(sim_mat_floats[i,j],2)),
                            fontsize=10)

          
            plt.xticks([])
            plt.yticks([])
    plt.savefig(CONFIG.data.root_folder+"tmp/cluster/channel_"+
                str(channel)+'_'+plot_string+".png")
    plt.close('all')


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

def calc_local_merge_parallel(template_indexes, base_temp, sim_mat, sim_temp, CONFIG):
    
    # base and comparison are the same
    com_temp = base_temp
    
    nshifts = 15
    upsample_factor = 20
    merge_threshold = CONFIG.merge_threshold
    feat_chans_threshold = 2
   
    # compute max channel for each template
    mc_base = base_temp.ptp(1).argmax(1)
    mc_com = com_temp.ptp(1).argmax(1)
    
    # keep track of finished pairs and don't compare them
    finished_pairs = []

    # loop only over indexes for the current proc_index
    for i in template_indexes:
        sim_mat[i,i] = True
        
        # compare templates only with those that have max_amplitude on a neighbouring channel
        for j in np.where(np.in1d(mc_com, np.where(CONFIG.neigh_channels[mc_base[i]])[0]))[0]:
            
            if j == i or [min(i,j), max(i,j)] in finished_pairs:
                continue
            feat_channels = np.where(np.logical_or(base_temp[i].ptp(0)>feat_chans_threshold, 
                                            com_temp[j].ptp(0)>feat_chans_threshold))[0]
            
            # find max chan of average of both templates
            max_chan_index = (base_temp[i,:, feat_channels] + com_temp[j,:, feat_channels]).T.ptp(0).argmax(0)
            chan = feat_channels[max_chan_index]
            
            # align together to average template shape max chan         
            wfs_aligned = align_mc(np.vstack((base_temp[i:i+1,:,feat_channels],com_temp[j:j+1,:,feat_channels])), 
                                            max_chan_index, CONFIG, upsample_factor, nshifts)
            bt_aligned = wfs_aligned[0]
            ct_aligned = wfs_aligned[1]
            
            temp = (bt_aligned*ct_aligned).sum()
            
            if (bt_aligned**2).sum() > (ct_aligned**2).sum():
                temp = temp/ (bt_aligned**2).sum()
            else:
                temp = temp/ (ct_aligned**2).sum()
            
            if temp>merge_threshold:
                sim_mat[i, j] = sim_mat[j, i] = True
            
            finished_pairs.append([min(i,j), max(i,j)])
            sim_temp[i, j] = sim_temp[j, i] = temp

    return sim_mat, sim_temp

def calc_global_merge_parallel(template_indexes, base_temp, sim_mat, sim_temp, CONFIG):
    
    # base and comparison are the same
    com_temp = base_temp
    
    nshifts = 15
    upsample_factor = 20
    merge_threshold = CONFIG.merge_threshold
    feat_chans_threshold = 2
   
    # compute max channel for each template
    mc_base = base_temp.ptp(1).argmax(1)
    mc_com = com_temp.ptp(1).argmax(1)
    
    # keep track of finished pairs and don't compare them
    finished_pairs = []

    # loop only over indexes for the current proc_index
    for i in template_indexes:
        sim_mat[i,i] = True
        
        # compare templates only with those that have max_amplitude on a neighbouring channel
        for j in np.where(np.in1d(mc_com, np.where(CONFIG.neigh_channels[mc_base[i]])[0]))[0]:
            
            if j == i or [min(i,j), max(i,j)] in finished_pairs:
                continue
            feat_channels = np.where(np.logical_or(base_temp[i].ptp(0)>feat_chans_threshold, 
                                            com_temp[j].ptp(0)>feat_chans_threshold))[0]
            
            # find max chan of average of both templates
            max_chan_index = (base_temp[i,:, feat_channels] + com_temp[j,:, feat_channels]).T.ptp(0).argmax(0)
            chan = feat_channels[max_chan_index]
            
            # align together to average template shape max chan         
            wfs_aligned = align_mc(np.vstack((base_temp[i:i+1,:,feat_channels],com_temp[j:j+1,:,feat_channels])), 
                                            max_chan_index, CONFIG, upsample_factor, nshifts)
            bt_aligned = wfs_aligned[0]
            ct_aligned = wfs_aligned[1]
            
            temp = (bt_aligned*ct_aligned).sum()
            
            if (bt_aligned**2).sum() > (ct_aligned**2).sum():
                temp = temp/ (bt_aligned**2).sum()
            else:
                temp = temp/ (ct_aligned**2).sum()
            
            if temp>merge_threshold:
                sim_mat[i, j] = sim_mat[j, i] = True
            
            finished_pairs.append([min(i,j), max(i,j)])
            sim_temp[i, j] = sim_temp[j, i] = temp

    return sim_mat, sim_temp
    
    

def calc_global_merge(base_temp, com_temp, CONFIG):
    
    nshifts = 15
    upsample_factor = 20
    merge_threshold = CONFIG.cluster.merge_threshold
    feat_chans_threshold = 2
   
    # compute max channel for each template
    mc_base = base_temp.ptp(1).argmax(1)
    mc_com = com_temp.ptp(1).argmax(1)
    
    # initialize cos-similarty matrix and boolean version 
    sim_mat = np.zeros([mc_base.shape[0], mc_com.shape[0]], dtype = bool)    
    sim_temp = np.zeros([mc_base.shape[0], mc_com.shape[0]])
    
    # keep track of finished pairs and don't compare them
    finished_pairs = []
    for i in range(mc_base.shape[0]):
        print (i)
        sim_mat[i,i] = True
        # compare templates only with those that have max_amplitude on a neighbouring channel
        for j in np.where(np.in1d(mc_com, np.where(CONFIG.neigh_channels[mc_base[i]])[0]))[0]:
               
            if j == i or [min(i,j), max(i,j)] in finished_pairs:
                continue
            feat_channels = np.where(np.logical_or(base_temp[i].ptp(0)>feat_chans_threshold, 
                                            com_temp[j].ptp(0)>feat_chans_threshold))[0]
            
            # find max chan of average of both templates
            max_chan_index = (base_temp[i,:, feat_channels] + com_temp[j,:, feat_channels]).T.ptp(0).argmax(0)
            chan = feat_channels[max_chan_index]
            
            # align together to average template shape max chan         
            wfs_aligned = align_mc(np.vstack((base_temp[i:i+1,:,feat_channels],com_temp[j:j+1,:,feat_channels])), 
                                            max_chan_index, CONFIG, upsample_factor, nshifts)
            bt_aligned = wfs_aligned[0]
            ct_aligned = wfs_aligned[1]
            
            temp = (bt_aligned*ct_aligned).sum()
            
            if (bt_aligned**2).sum() > (ct_aligned**2).sum():
                temp = temp/ (bt_aligned**2).sum()
            else:
                temp = temp/ (ct_aligned**2).sum()
            
            if temp>merge_threshold:
                sim_mat[i, j] = sim_mat[j, i] = True
            
            finished_pairs.append([min(i,j), max(i,j)])
            sim_temp[i, j] = sim_temp[j, i] = temp

    return sim_mat, sim_temp
    


    

#def calc_local_merge(channel, base_temp, com_temp, CONFIG):
def calc_local_merge(channel, templates, template_stds, CONFIG):
    
    nshifts = 15
    upsample_factor = 20
    merge_threshold = CONFIG.cluster.merge_threshold
    feat_chans_threshold = 2
   
    # initialize cos-similarty matrix and boolean version 
    sim_mat = np.zeros([mc_base.shape[0], mc_com.shape[0]], dtype = bool)    
    sim_temp = np.zeros([mc_base.shape[0], mc_com.shape[0]])
    
    
    #Concatenate templates and std and align together using the existing function
    templates_and_std = np.concatenate([templates, templates_std], axis = 2)
    
    templates_and_std_aligned = align_mc(templates_and_std, channel, CONFIG, upsample_factor, nshifts)
    templates_aligned = templates_and_std[:,:,:template.shape[2]]
    std_aligned = templates_and_std[:,:,template.shape[2]:]
        
    for i in range(templates_aligned.shape[0]):
        for j in range(i+1,templates_aligned.shape[0]):
            feat_chan_2SU = np.where(templates_aligned[[i,j]].mean(0).ptp(0).argmax())[0]
            temp = (templates_aligned[i, :, feat_chan_2SU] * templates_aligned[j, :, feat_chan_2SU]).sum(0)
            
            temp_i = (templates_aligned[i, :, feat_chan_2SU] * templates_aligned[i, :, feat_chan_2SU]).sum(0)
            temp_j = (templates_aligned[j, :, feat_chan_2SU] * templates_aligned[j, :, feat_chan_2SU]).sum(0)
                  
            
            for k,chan in enumerate(feat_chan_2SU):
                if temp_i[k] > temp_j[k]:
                    temp[k] = temp[k]/temp_i[k]
                else:
                    temp[k] = temp[k]/temp_j[k]
                    
            min_cos = temp.min()
            sim_temp[i,j] = min_cos
            if min_cos > merge_threshold_upper:
                sim_mat[i,j] = True
 
            elif min_cos>merge_threshold_lower and min_cos<merge_threshold_upper:
                ks_dist = calc_ks_dist(templates_aligned[[i,j]], std_aligned[[i,j]], 500)
                if ks_dist < 0.1:
                    sim_mat[i,j] = True
                
    return sim_temp, sim_mat, templates_aligned

    
    
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
   
    
def get_feat_channels_diptest(wf_data, n_feat_chans):
    '''  Function that uses Hartigan diptest to identify bimodal channels
        Steps:  denoise waveforms first, then find highest variance time point,
                then compute diptest and rank in order of max value;
    '''

    # compute stds over units
    stds = wf_data.std(0)

    # find location of highest variance time point in each channel (clip edges)
    # Cat: TODO: need to make this more robust to wider windows; ideally only look at 20-30pts in middle of spike
    time_pts = stds[5:-5].argmax(0)+5

    # compute diptest for each channel at the location of highest variance point
    diptests = np.zeros(wf_data.shape[2])
    for k in range(wf_data.shape[2]):
        diptests[k]=dp(wf_data[:,time_pts[k],k])[0]

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


def align_wf(data_in, upsample_factor, n_steps):
    ''' TODO parallelize this step
    '''
    data_aligned = np.zeros(data_in.shape)
    for k in range(data_in.shape[2]):
        data_aligned[:,:,k] = align_channelwise(data_in[:,:,k].T, upsample_factor=20, n_steps=7)
    return data_aligned


def knn_triage(th, pca_wf):
    #th = 90
#     # get distance to nearest neighbors
    tree = cKDTree(pca_wf)
    dist, ind = tree.query(pca_wf, k=11)
    dist = np.sum(dist, 1)
#     # triage far ones
    idx_keep1 = dist < np.percentile(dist, th)
    return idx_keep1

def run_mfm_2(kk, CONFIG):
    mask = np.ones((kk.shape[0], 1))
    group = np.arange(kk.shape[0])
    vbParam2 = mfm.spikesort(kk[:,:,np.newaxis],
                            mask,
                            group, CONFIG)
    
    vbParam2.rhat[vbParam2.rhat < 0.1] = 0 #Cat todo; look at this
    
    vbParam2.rhat = vbParam2.rhat/np.sum(vbParam2.rhat,
                                         1, keepdims=True)

    assignment2 = np.argmax(vbParam2.rhat, axis=1)
    return vbParam2, assignment2



def run_mfm_3(kk, CONFIG):
    mask = np.ones((kk.shape[0], 1))
    group = np.arange(kk.shape[0])
    vbParam2 = mfm.spikesort(kk[:,:,np.newaxis],
                            mask,
                            group, CONFIG)
    
    return vbParam2


def run_mfm3(kk, CONFIG):
    mask = np.ones((kk.shape[0], 1))
    group = np.arange(kk.shape[0])
    vbParam2 = mfm.spikesort(kk[:,:,np.newaxis],
                            mask,
                            group, CONFIG)
    vbParam2.rhat[vbParam2.rhat < 0.1] = 0
    vbParam2.rhat = vbParam2.rhat/np.sum(vbParam2.rhat,
                                         1, keepdims=True)

    assignment2 = np.argmax(vbParam2.rhat, axis=1)
    return vbParam2, assignment2

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
    spike_train = spike_train-data_start


    # load all waveforms at once - use offset 
    waveforms = recording[spike_train[:, [0]].astype('int32')+offset
                  + np.arange(-spike_size, spike_size + 1)]

    return waveforms    



def load_waveforms_from_memory_merge(recording, data_start, offset, spike_train, 
                               spike_size):
                                           
    # offset spike train to t=0 to properly index into data 
    spike_train = spike_train-data_start


    # load all waveforms at once - use offset 
    waveforms = recording[spike_train[:,np.newaxis]+offset
                  + np.arange(-spike_size, spike_size + 1)]

    return waveforms    























