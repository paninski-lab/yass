import numpy as np
import logging
import os
import parmap
from scipy import signal
from scipy.spatial import cKDTree

from yass import mfm
from yass.empty import empty
from scipy.sparse import lil_matrix
from statsmodels import robust
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import networkx as nx

colors = np.asarray(["#000000", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
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


#def align_channelwise(wf, upsample_factor=20, n_steps=7):
    
    #n_shifts=n_steps*upsample_factor
    #window=n_steps*upsample_factor
    #waveform_len = wf.shape[0]
    
    ## upsample waveforms
    #wf_upsampled = upsample_resample(wf, upsample_factor)
    
    ## upsample tempalte
    #template_upsampled = upsample_resample(np.mean(wf,axis=1)[:,np.newaxis], upsample_factor).reshape(upsample_factor*waveform_len)
    
    ## shift template
    #template_shifted = shift_template(template_upsampled, n_shifts, window)
    
    ## find optimal fits between template and waveforms and return shifts
    #shift_array = return_shifts(wf_upsampled.T, template_shifted, window)

    #aligned_chunks = np.zeros((len(shift_array), waveform_len))
    #for ctr, shift in enumerate(shift_array):
        #chunk = wf_upsampled[ctr,n_shifts-shift:][::upsample_factor][:waveform_len]

        ## conditional required in case shift leads to short waveforms
        #if len(chunk) < waveform_len: 
            #chunk = np.concatenate((chunk, np.zeros(waveform_len-len(chunk))))
        #aligned_chunks[ctr] = chunk

    #return aligned_chunks


# PCA function return PCA and reconstructed data
def PCA(X, n_components):
    from sklearn import decomposition

    pca = decomposition.PCA(n_components)
    pca.fit(X)
    X = pca.transform(X)
    Y = pca.inverse_transform(X)
    return X,Y



def run_cluster_features(spike_index_clear, n_dim_pca, wf_start, wf_end, 
                         n_mad_chans, n_max_chans, CONFIG, out_dir):
    
    ''' New voltage feature based clustering
    ''' 
    
    # loop over channels 
    # Cat: TODO: Parallelize over channels
    cluster_ctr=0
    spike_list = []
    tmp_loc = []
    channels = np.arange(49)
    for channel in channels: 
        
        # **** grab spike waveforms ****
        indexes = np.where(spike_index_clear[:,1]==channel)[0]
        wf_data = load_waveforms_parallel(spike_index_clear[indexes], 
                                          CONFIG, out_dir)
        wf_data = np.swapaxes(wf_data,2,0)

        # **** find feature channels ****
        # find max amplitude chans 
        template = np.mean(wf_data,axis=2)
        rank_amp_chans = np.max(np.abs(template),axis=1)
        rank_indexes = np.argsort(rank_amp_chans,axis=0)[::-1] 
        max_chans = rank_indexes[:n_max_chans]      # select top chans
              
        # find top 3 mad chans out of chans with template > 2SU
        ptps = np.ptp(template,axis=1)
        chan_indexes_2SU = np.where(ptps>2)[0]
        rank_chans_max = np.max(robust.mad(wf_data[chan_indexes_2SU,:,:],axis=2),axis=1)

        # rank channels by max mad value
        rank_indexes = np.argsort(rank_chans_max,axis=0)[::-1]
        mad_chans = chan_indexes_2SU[rank_indexes][:n_mad_chans]      # select top chans

        # make feature chans from union 
        feat_chans = np.union1d(max_chans, mad_chans)


        # **** cluster ****
        wf_data = wf_data.T
        data_in = wf_data[:,:,feat_chans]
                
        data_aligned = []
        for k in range(data_in.shape[2]):
            #print ("aligning ch: ",k)
            data_aligned.append(align_channelwise(data_in[:,:,k].T, upsample_factor=20, n_steps=7))

        data_in = np.array(data_aligned)
        #print ("aligned data: ", data_in.shape)
        data_in = data_in[:,:,wf_start:wf_end]
        
        # reshape data for PCA
        data_in = data_in.swapaxes(0,1).reshape(data_in.shape[1],-1)
        #print ("reshaped aligned data_in: ", data_in.shape)

        # norm = np.max(pca_wf,axis=1,keepdims=True)
        pca_wf,pca_wf_reconstruct = PCA(data_in,n_dim_pca)
        #print pca_wf.shape
        
        # triage percentile
        th = 90
        # get distance to nearest neighbors
        tree = cKDTree(pca_wf)
        dist, ind = tree.query(pca_wf, k=11)
        dist = np.sum(dist, 1)
       
        # triage far ones
        idx_keep1 = dist < np.percentile(dist, th)
        pca_wf = pca_wf[idx_keep1]
        wf_data_original = wf_data[idx_keep1].copy()

        # save indexes for mapping back
        indexes=indexes[idx_keep1]

        # run pca second time
        pca_wf_original,pca_wf_reconstruct = PCA(data_in[idx_keep1],n_dim_pca)

        # run mfm iteratively
        spike_train_clustered = run_mfm(wf_data_original, pca_wf_original, 
                                        feat_chans, idx_keep1, wf_start,
                                        wf_end, n_dim_pca, CONFIG)
        
        print ("chan: ", channel, "  feat chans: ", feat_chans, data_in.shape, 
                                ' # clusters: ', len(spike_train_clustered))

        # make 2 column list 
        for c in range(len(spike_train_clustered)):
            temp = np.zeros((spike_train_clustered[c].shape[0],2),'int32')
            temp[:,0]=spike_index_clear[:,0][indexes[spike_train_clustered[c]]]
            temp[:,1]=cluster_ctr
            spike_list.append(temp)
            cluster_ctr+=1
            tmp_loc.append(channel)
            
    # format output in time order
    print ("..formating spike trains ...")
    s = np.vstack(spike_list)
    indexes = np.argsort(s[:,0])
    spike_train_clustered = s[indexes]
    
    return spike_train_clustered, tmp_loc


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

def align_mc(wf, mc, CONFIG, upsample_factor = 20, nshifts = 7, 
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
    
    # don't parallelize < 500 spikes
    #if n_spikes<1000:
    if n_spikes<50000:
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
    

def RRR3_noregress(channel, wf, sic, gen, fig, grid, triageflag, alignflag, plotting, 
         chans, n_mad_chans, n_max_chans, n_dim_pca, wf_start, wf_end, 
         mfm_threshold, CONFIG, upsample_factor, nshifts, assignment_global, 
         spike_index):
    
    min_cluster_spikes = 100
    
    if wf.shape[0] < 30:
        #print ("exiting too few spikes<<<FIX THIS")
        return

    alignflag=True

    print ("")
    print("***Channel "+str(channel) + ' generation: '+str(gen)+ ' # spikes: '+ str(wf.shape[0]))
        
    # select feature chans 
    feat_chans, max_chans = get_feat_channels(wf.mean(0), wf, n_max_chans, n_mad_chans)  
    print("chan "+str(channel) + ", feat chans: "+str(feat_chans) + ", max_chan: "+ str(max_chans[0]))
    
    # save max_channel relative to feat_chans location
    mc = np.where(feat_chans==max_chans[0])[0][0]
    #print ("MAX CHAN: ", mc) 
    
    # align, note: aligning all channels to max chan; 
    # note: max chan is first from feat_chans above, ensure order is preserved
    if alignflag:
        print ("chan "+str(channel)+ " - aligning")
        wf_align = align_mc(wf[:,:,feat_chans], mc, CONFIG, 
                            upsample_factor, nshifts, ref = None)
    else:
        #print ("skip - aligning")
        wf_align = wf[:,:,feat_chans]
        
    # run pca
    data_in = wf_align[:,0:40].swapaxes(1,2).reshape(wf.shape[0],-1)

    pca_wf, _ = PCA(data_in, 3)
    if triageflag:
        idx_keep = triage(mfm_threshold*100, pca_wf)
        print("chan "+str(channel) + " triaged, remaining spikes "+str(idx_keep.sum()))
        pca_wf,_ = PCA(data_in[idx_keep],3)
    else:
        idx_keep = np.ones(pca_wf.shape[0],dtype = bool)

    # clustering
    print("chan "+ str(channel)+" - clustering ")
    vbParam, assignment = run_mfm3(pca_wf, CONFIG)
    
    #np.save(CONFIG.data.root_folder+'chan_'+str(channel)+"_pcadistribution.npy", pca_wf)
    #np.save(CONFIG.data.root_folder+'chan_'+str(channel)+"_vmParam_rhat.npy", vbParam.rhat)
    #np.save(CONFIG.data.root_folder+'chan_'+str(channel)+"_assignment.npy", assignment)
    
    # if > 1 cluster plot the results. 
    if vbParam.rhat.shape[1] != 1:
        if plotting:
            if np.all(x[gen]<=20) and gen <20:
                mask = vbParam.rhat>0
                stability = np.average(mask * vbParam.rhat, axis = 0, weights = mask)
                labels = []
                clusters, sizes = np.unique(assignment, return_counts=True)

                fig.add_subplot(grid[gen, x[gen]])
                x[gen] += 1
                for clust in clusters:
                    patch_j = mpatches.Patch(color = colors[clust%100], 
                      label = "size = {}, stability = {}".format(sizes[clust], 
                      stability[clust]))
                    labels.append(patch_j)
                plt.scatter(pca_wf[:,0], pca_wf[:,1], 
                  c = colors[assignment%100], edgecolor = 'k')
                plt.legend(handles = labels)
                plt.title(str(sizes.sum()))
                plt.suptitle("Channel: "+str(channel), fontsize=25)
    
    # if single cluster found
    if vbParam.rhat.shape[1] == 1:
        N= len(assignment_global)            
        print("chan "+str(channel) + " >>>>>>>>>> cluster "+str(N) + 
        " saved, size: " + str(wf[idx_keep].shape)+ "<<<<<<<<<<<<<")
        
        assignment_global.append(N * np.ones(assignment.shape[0]))
        spike_index.append(sic[idx_keep])

        if plotting:
            # plot distributions as generations
            if np.all(x[gen]<=20) and gen <20:
                mask = vbParam.rhat>0
                stability = np.average(mask * vbParam.rhat, axis = 0, weights = mask)
                labels = []
                clusters, sizes = np.unique(assignment, return_counts=True)

                ax = fig.add_subplot(grid[gen, x[gen]])
                x[gen] += 1
                for clust in clusters:
                    patch_j = mpatches.Patch(color = colors[clust%100], label = "size = {}, stability = {}".format(sizes[clust], stability[clust]))
                    labels.append(patch_j)
                plt.scatter(pca_wf[:,0], pca_wf[:,1], c = colors[assignment%100], edgecolor = 'k')
                if clusters.size == 1:
                    ax.scatter(pca_wf[:,0].mean(), pca_wf[:,1].mean(), c= 'r', s = 2000)
                ax.legend(handles = labels)
                plt.title(str(sizes.sum()))
                plt.suptitle("Channel: "+str(channel), fontsize=25)

            # plot templates 
            ax = fig.add_subplot(grid[15:, 5:])
            wf_mean = wf[idx_keep].mean(0)
            scale = 10
            plt.plot(CONFIG.geom[:,0] + np.arange(-wf_mean.shape[0],0)[:,np.newaxis]/3., CONFIG.geom[:,1] + wf_mean[:,:]*scale, c=colors[N%100])

            for i in feat_chans:
                 plt.scatter(CONFIG.geom[i,0], CONFIG.geom[i,1], s = 750, color = 'orange')
                
            for i in range(49):
                plt.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), fontsize=30)
            
    # if > 1 cluster
    else:
        mask = vbParam.rhat>0
        stability = np.average(mask * vbParam.rhat, axis = 0, weights = mask)
        clusters, sizes = np.unique(assignment, return_counts = True)
        print("chan "+str(channel) + 
              " multiple clusters, stability " + str(stability) + 
              " size: "+str(sizes))

        # remove stable clusters 
        for clust in np.where(stability>mfm_threshold)[0]:
            idx = assignment == clust
            if wf[idx_keep][idx].shape[0]<min_cluster_spikes: 
                #print ("skiping cluster, too small...")
                continue
            print("chan " + str(channel) + " reclustering "+ str(wf[idx_keep][idx].shape))
            RRR3_noregress(channel, wf[idx_keep][idx], sic[idx_keep][idx], gen+1, 
                 fig, grid, False, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
                 n_dim_pca, wf_start, wf_end, mfm_threshold,  CONFIG, 
                 upsample_factor, nshifts, assignment_global, spike_index)

        # if all clusters are unstable: try annealing, or triaging
        if np.all(stability<=mfm_threshold):
            
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
            #else:                                             
            
            print("chan "+str(channel) + " no stable clusters, triaging " + str(wf[idx_keep].shape))
            RRR3_noregress(channel, wf[idx_keep], sic[idx_keep], gen+1, fig, grid, 
                 True, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
                 n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor,
                 nshifts, assignment_global, spike_index)
        
        else:
            # run mfm on remaining data
            idx = np.in1d(assignment, np.where(stability<=mfm_threshold)[0])
            if idx.sum()>min_cluster_spikes:
                print("chan "+str(channel) + " reclustering residuals "+str(wf[idx_keep][idx].shape))
                RRR3_noregress(channel, wf[idx_keep][idx], sic[idx_keep][idx], gen+1, fig, grid, 
                     False, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
                     n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, upsample_factor,
                     nshifts, assignment_global, spike_index)


def regress(wf, mc):
    template = wf.mean(0)
    channels = np.where(template.ptp(0)>0)[0]
    wf_x = wf[:,:,mc]
    wf_y = wf[:,:, channels]
    A = np.matmul(np.matmul(wf_y.transpose([2,1,0]), wf_x)/2+np.matmul(wf_x.T,wf_y.transpose([2,0,1]))/2\
                  ,np.linalg.inv(np.matmul(wf_x.T, wf_x)))
    residual = wf_y - np.matmul(A, wf_x.T).transpose([2,1,0])
    return residual


def run_cluster_features_2(spike_index_clear, n_dim_pca, wf_start, wf_end, 
                         n_mad_chans, n_max_chans, CONFIG, out_dir,
                         mfm_threshold, upsample_factor, nshifts):
    
    ''' New voltage feature based clustering
    ''' 
    
    # loop over channels 
    # hold spike times and chans in two arrays passed in and out
    all_assignments = []
    spike_times = []
    chans = [] 
    channels = np.arange(CONFIG.recordings.n_channels)
    triageflag = True
    alignflag = True
    plotting = True

    gen = 0     #Set default generation for starting clustering stpe
    assignment_global = []
    spike_index = []
    for channel in channels: 
    #for channel in np.arange(30,49,1): 

        print("***********Channel {}**********************".format(channel))
        
        # plotting parameters
        global x
        x = np.zeros(100, dtype = int)
        fig = plt.figure(figsize =(100,100))
        grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
        
        # read waveforms
        indexes = np.where(spike_index_clear[:,1]==channel)[0]
        # Cat: TEMP LIMIT TO 10000 spikes
        indexes = np.random.choice(indexes, size=min(indexes.shape[0],10000),replace=False)
        wf = load_waveforms_parallel(spike_index_clear[indexes], 
                                          CONFIG, out_dir)
  
        # cluster
        RRR3_noregress(channel, wf, spike_index_clear[indexes], gen, fig, grid, 
             triageflag, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
             n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
             upsample_factor, nshifts, assignment_global, spike_index)

        ax = fig.add_subplot(grid[15:, 5:])
        
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
        plt.legend(handles = labels, fontsize=30)
        
        
        plt.savefig("/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/june_30/channel_{}.png".format(channel))
        plt.close('all')
        print("****** Channel {},  total clusters *****".format(channel))
        print ("")
        print ("")
        print ("")
        
                
    # add channels of templates to be saved as tmp_loc
    tmp_loc = np.int32(chans)

    #assignment_global = np.int32(assignment_global)
    #spike_index = np.int32(spike_index)
    #print assignment_global
    #print spike_index
    
    #spike_train_clustered = np.zeros((spike_index.shape[0],2),'int32')
    #spike_train_clustered[:,0]=spike_index
    #spike_train_clustered[:,1]=assignment_global
    
    print (sic_temp.shape)
    print (assignment_temp.shape)
    
    spike_train_clustered = np.zeros((sic_temp.shape[0], 3), 'int32')
    spike_train_clustered[:,0] = sic_temp[:,0]
    spike_train_clustered[:,2] = sic_temp[:,1]
    spike_train_clustered[:,1] = assignment_temp
    
    # sort by time
    indexes = np.argsort(spike_train_clustered[:,0])
    spike_train_clustered = spike_train_clustered[indexes]
    
    return spike_train_clustered, tmp_loc



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

    CONFIG2.recordings.spike_size_ms = CONFIG.recordings.spike_size_ms
    CONFIG2.recordings.sampling_rate = CONFIG.recordings.sampling_rate
    CONFIG2.recordings.n_channels = CONFIG.recordings.n_channels
    
    CONFIG2.resources.n_processors = CONFIG.resources.n_processors
    CONFIG2.resources.multi_processing = CONFIG.resources.multi_processing
    
    CONFIG2.data.root_folder = CONFIG.data.root_folder
    CONFIG2.geom = CONFIG.geom
    
    # Cat: FIX THIS - LOAD FROM CONFIG FILE
    CONFIG2.cluster.prior.a = 1
    CONFIG2.cluster.prior.beta = 1
    CONFIG2.cluster.prior.lambda0 = 0.01
    CONFIG2.cluster.prior.nu = 5
    CONFIG2.cluster.prior.V = 2

    CONFIG2.neigh_channels = CONFIG.neigh_channels
    
    CONFIG2.merge_threshold = CONFIG.cluster.merge_threshold


    return CONFIG2

def run_cluster_features_2_parallel(spike_index_clear, n_dim_pca, wf_start, wf_end, 
                         n_mad_chans, n_max_chans, CONFIG, out_dir,
                         mfm_threshold, upsample_factor, nshifts):
    
    ''' New voltage feature based clustering; parallel version
    ''' 
    
    CONFIG2 = make_CONFIG2(CONFIG)
    
    # loop over channels 
    # hold spike times and chans in two arrays passed in and out
    all_assignments = []
    spike_times = []
    chans = [] 
    triageflag = True
    alignflag = True
    plotting = True

    gen = 0     #Set default generation for starting clustering stpe
    assignment_global = []
    spike_index = []
    
    # parallelize over channels: 
    channels = np.arange(CONFIG.recordings.n_channels)
    #channels = [41]
    res_file = '/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/res.npy'
    if os.path.exists(res_file)==False: 
        if CONFIG.resources.multi_processing:
            res = parmap.map(cluster_channels, channels, CONFIG2, 
                             spike_index_clear, n_dim_pca, wf_start, wf_end, 
                             n_mad_chans, n_max_chans, out_dir, mfm_threshold, 
                             upsample_factor, nshifts, 
                             processes=CONFIG.resources.n_processors, 
                             pm_pbar=False)
        else:
            res = []
            for channel in channels:
                temp = cluster_channels(channel, CONFIG2, spike_index_clear, 
                             n_dim_pca, wf_start, wf_end, n_mad_chans, 
                             n_max_chans, out_dir, mfm_threshold, 
                             upsample_factor, nshifts)                         
                res.append(temp)

        print ("res: ", len(res), " res[0]: ", len(res[0]))
        
        np.save(res_file, res)
    else:
        res = np.load(res_file, encoding='latin1')


    # run inter-channel template merge   
    spike_train_merged, tmp_loc, templates = global_merge(channels, CONFIG2)
   
    # sort by time
    indexes = np.argsort(spike_train_merged[:,0])
    final_spike_train = spike_train_merged[indexes]

    return final_spike_train, np.int32(tmp_loc), templates
    
def cluster_channels(channel, CONFIG, spike_index_clear, n_dim_pca, 
                     wf_start, wf_end, n_mad_chans, n_max_chans, 
                     out_dir, mfm_threshold, upsample_factor, nshifts):

    ''' Clustering wrapper function run in parallel for each channel
    '''


    # starting params
    chans = [] 
    triageflag = False
    alignflag = True
    plotting = True
    subsample_nspikes = 50000

    gen = 0     #Set default generation for starting clustering stpe
    assignment_global = []
    spike_index = []
    
    # plotting parameters
    global x
    x = np.zeros(100, dtype = int)
    fig = plt.figure(figsize =(100,100))
    grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
    
    # load all indexes and waveforms
    indexes = np.where(spike_index_clear[:,1]==channel)[0]
    wf = load_waveforms_parallel(spike_index_clear[indexes], CONFIG, out_dir)
    print ('Starting channel: '+str(channel)+ ', events: '+str(wf.shape[0]))
    #print("***********Channel {}**********************".format(channel))

    # save file temporarily with meta-data for debugging; remove eventually
    spike_index_postclustering = "/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_{}_spike_index_postclustering.npz".format(channel)
    if os.path.exists(spike_index_postclustering)==False: 
        # subsample for clustering step
        indexes_subsampled = np.random.choice(np.arange(indexes.shape[0]), 
                             size=min(indexes.shape[0],subsample_nspikes),replace=False)
        
        #indexes_subsampled = np.arange(indexes.shape[0])
        
        # cluster
        RRR3_noregress(channel, wf[indexes_subsampled], 
             spike_index_clear[indexes][indexes_subsampled], gen, fig, grid, 
             triageflag, alignflag, plotting, chans, n_mad_chans, n_max_chans, 
             n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
             upsample_factor, nshifts, assignment_global, spike_index)

        # finish plotting 
        if plotting: 
            ax = fig.add_subplot(grid[15:, 5:])
            
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
            plt.legend(handles = labels, fontsize=30)
        
            plt.savefig("/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_{}.png".format(channel))
            plt.close('all')

        # temporary save of data fore debugging; 
        np.savez(spike_index_postclustering, spike_index=spike_index, indexes_subsampled=indexes_subsampled)
    else: 

        # temporary load of data for debugging; 
        data = np.load(spike_index_postclustering, encoding='latin1')
        spike_index = data['spike_index']
        indexes_subsampled = data['indexes_subsampled']
    
    # turn off local merge
    if False:
        # merge tempaltes and return spike_index for merged 
        spike_index_postmerge = "/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_{}_spike_index_postmerge.npy".format(channel)
        if os.path.exists(spike_index_postmerge)==False:
            spike_index = local_merge(channel, spike_index, 
                                  spike_index_clear[indexes], wf,
                                  CONFIG)
            np.save(spike_index, spike_index_local_merge)
        else:
            spike_index = np.load(spike_index_postmerge, encoding='latin1')

    #if False:    
        ## recover clear spikes
        #spike_index = recover_clear_spikes_pcaspace(channel, wf, 
                         #spike_index_clear[indexes], indexes_subsampled, 
                         #spike_index, upsample_factor, nshifts, CONFIG)

    print ("****** Channel "+str(channel)+ 
           ", total clusters "+str(len(spike_index))+ 
           " *****")
    print ("")
    print ("")
    print ("")
    
    return spike_index

   
    
def global_merge(channels, CONFIG):
    ''' Function that cleans noisy templates and merges the rest    
    '''
    
    n_channels = 49
    
    # convert clusters to templates; keep track of weights
    templates = []
    weights = []
    spike_ids = []
    spike_indexes = []
    channels = np.arange(n_channels)
    tmp_loc = []
    
    for channel in channels:
        data = np.load(
          '/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_{}_weighted_templates.npz'.format(channel), encoding='latin1')
        templates.append(data['templates'])
        weights.append(data['weights'])
        
        tmp_loc.extend([channel]*len(data['weights']))
        
        spike_index = np.load('/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_{}_spike_index_postmerge.npy'.format(channel), encoding='latin1')
        for s in spike_index:
            spike_indexes.append(s[:,0])
    
    templates = np.vstack(templates)
    weights = np.hstack(weights)
    
    # save pre-merge templates
    np.save('/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/preglobal_merge_templates.npy',
                                     templates)    
        
    # compute global merge in parallel
    #sim_mat, sim_temp = calc_global_merge(templates, templates, CONFIG)
    
    # initialize cos-similarty matrix and boolean version 
    sim_mat = np.zeros((templates.shape[0], templates.shape[0]), 'bool')    
    sim_temp = np.zeros((templates.shape[0], templates.shape[0]), 'float32')

    global_merge_file = '/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/global_merge_matrix.npz'
    if os.path.exists(global_merge_file)==False:
        print ("Computing global merge over templates: ", templates.shape)
        if CONFIG.resources.multi_processing:
            n_processors = CONFIG.resources.n_processors
            indexes = np.array_split(np.arange(templates.shape[0]), n_processors)
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
    print (nx.number_connected_components(G))
    for i, cc in enumerate(nx.connected_components(G)):
        final_template_indexes.append(list(cc))
        sic = np.zeros(0, dtype = int)
        for j in cc:
            sic = np.concatenate([sic, spike_indexes[j]])
        final_spike_indexes.append(np.concatenate([sic[:,np.newaxis], i*np.ones([sic.size,1],dtype = int)],axis = 1))

    final_spike_train = np.vstack(final_spike_indexes)
    
    # recompute tmp_loc from weighted templates
    tmp_loc = []
    templates_final= []
    for t in final_template_indexes:
        # compute average weighted template and find peak
        idx = np.int32(t)
        print (t, idx, templates.shape, weights.shape)
        weighted_average = np.average(templates[idx],axis=0,weights=weights[idx])
        max_chan = weighted_average.ptp(0).argmax(0)
        
        templates_final.append(weighted_average)
        tmp_loc.append(max_chan)
    
    # convert templates to : (n_channels, waveform_size, n_templates)
    templates = np.float32(templates_final)
    print (templates.shape)
    templates = np.swapaxes(templates, 2,0)
    
    # save cluster assignment list
    np.save('/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/global_merge_assignments.npy',
                                     final_template_indexes)
    
    return final_spike_train, tmp_loc, templates

    
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
        
    np.save('/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_'+
                                            str(channel)+'_clusters.npy', wf_array)
    
    np.savez('/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_'+
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
    plt.savefig("/media/cat/12TB/Dropbox/Dropbox/data_temp/liam/clusters/july_1/channel_"+
                str(channel)+'_'+plot_string+".png")
    plt.close('all')


def calc_sim_all_chan(channel, base_temp, com_temp, CONFIG, merge_threshold=0.8,
                      size_base=None, size_com=None, clusters_base=None, 
                      clusters_com=None):
    
    # Cat: TO FIX THIS
    n_chans = 49
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
    
    # don't initialize these, just pass them in
    #sim_mat = np.zeros([mc_base.shape[0], mc_com.shape[0]], dtype = bool)    
    #sim_temp = np.zeros([mc_base.shape[0], mc_com.shape[0]])
    
    # keep track of finished pairs and don't compare them
    finished_pairs = []

    # loop only over indexes for the mutiprocessing core
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
    
    

def calc_local_merge(channel, base_temp, com_temp, CONFIG):
    
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
    templates_aligned = np.zeros((base_temp.shape[0],base_temp.shape[0],2, 
                            (base_temp.shape[1]-1)//2+1, base_temp.shape[2]), 'float32')
    for i in range(mc_base.shape[0]):
        sim_mat[i,i] = True
        # compare templates only with those that have max_amplitude on a neighbouring channel
        #for j in np.where(np.in1d(mc_com, np.where(CONFIG.neigh_channels[mc_base[i]])[0]))[0]:
        for j in range(mc_com.shape[0]):
            if j == i or [min(i,j), max(i,j)] in finished_pairs:
                continue
            feat_channels = np.where(np.logical_or(base_temp[i].ptp(0)>feat_chans_threshold, 
                                            com_temp[j].ptp(0)>feat_chans_threshold))[0]
            
            # find max chan of average of both templates
            max_chan_index = (base_temp[i,:, feat_channels] + com_temp[j,:, feat_channels]).T.ptp(0).argmax(0)
            chan = feat_channels[max_chan_index]
            
            # align together to average template shape max chan         
            wfs_aligned = align_mc(np.vstack((base_temp[i:i+1,:,:],com_temp[j:j+1,:,:])), 
                                            max_chan_index, CONFIG, upsample_factor, nshifts)
            bt_aligned = wfs_aligned[0,:,feat_channels]
            ct_aligned = wfs_aligned[1,:,feat_channels]
            
            temp = (bt_aligned*ct_aligned).sum()
            
            if (bt_aligned**2).sum() > (ct_aligned**2).sum():
                temp = temp/ (bt_aligned**2).sum()
            else:
                temp = temp/ (ct_aligned**2).sum()
            
            if temp>merge_threshold:
                sim_mat[i, j] = sim_mat[j, i] = True
            
            finished_pairs.append([min(i,j), max(i,j)])
            sim_temp[i, j] = sim_temp[j, i] = temp

            templates_aligned[i,j] = wfs_aligned

    return sim_mat, sim_temp, templates_aligned
    
    
    
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
    
def get_feat_channels(template, wf_data, n_max_chans, n_mad_chans):

    rank_amp_chans = np.max(np.abs(template),axis=0)
    rank_indexes = np.argsort(rank_amp_chans)[::-1] 
    max_chans = rank_indexes[:n_max_chans]      # select top chans
    ptps = np.ptp(template,axis=0)
    chan_indexes_2SU = np.where(ptps>2)[0]
    rank_chans_max = np.max(robust.mad(wf_data[:,:,chan_indexes_2SU],axis=0),axis=0)
    rank_indexes = np.argsort(rank_chans_max,axis=0)[::-1]
    mad_chans = chan_indexes_2SU[rank_indexes][:n_mad_chans]      # select top chans
    feat_chans = np.union1d(max_chans, mad_chans)
    
    return feat_chans, max_chans


def align_wf(data_in, upsample_factor, n_steps):
    ''' TODO parallelize this step
    '''
    data_aligned = np.zeros(data_in.shape)
    for k in range(data_in.shape[2]):
        data_aligned[:,:,k] = align_channelwise(data_in[:,:,k].T, upsample_factor=20, n_steps=7)
    return data_aligned


def triage(th, pca_wf):
    th = 90
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

            
def load_waveforms_parallel(spike_train, CONFIG, out_dir): 
    
    # Cat: TODO: link spike_size in CONFIG param
    spike_size = int(CONFIG.recordings.spike_size_ms*
                     CONFIG.recordings.sampling_rate//1000)
    n_processors = CONFIG.resources.n_processors
    n_channels = CONFIG.recordings.n_channels
    sampling_rate = CONFIG.recordings.sampling_rate

    # select length of recording to read at once and grab data
    # currently fixed to 60 sec, but may wish to change
    # n_sec_chunk = CONFIG.resources.n_sec_chunk
    n_sec_chunk = 60

    # determine length of processing chunk based on lenght of rec
    standardized_filename = os.path.join(CONFIG.data.root_folder, out_dir,
                                         'standarized.bin')
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # make index list for chunk/parallel processing
    buffer_size = 400
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

    if CONFIG.resources.multi_processing:
        res = parmap.map(
            load_waveforms_,
            zip(idx_list, proc_indexes),
            spike_train,
            spike_size,
            n_channels,
            buffer_size,
            standardized_filename,
            processes=n_processors,
            pm_pbar=False)
    else:
        res = []
        for k in range(len(idx_list)):
            temp = load_waveforms_(
                [idx_list[k], k], spike_train, spike_size, 
                n_channels, buffer_size, standardized_filename)
            res.append(temp)

    # Reconstruct templates from parallel proecessing
    wfs = np.vstack(res)
    #print wfs.shape
    
    return wfs

def load_waveforms_(data_in, spike_train, spike_size,
                    n_channels, buffer_size,
                    standardized_filename):

    idx_list = data_in[0]

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
            
            # pad data with zeros at the beginning
            recordings_1D = np.hstack((np.zeros(
                buffer_size * n_channels, dtype='float32'), recordings_1D))
        else:
            fin.seek((data_start - buffer_size) * 4 * n_channels, os.SEEK_SET)
            recordings_1D = np.fromfile(
                fin,
                dtype='float32',
                count=((data_end - data_start + buffer_size * 2) * n_channels))

        # pad recording with zeros at end if not expected size
        if len(recordings_1D) != (
              (data_end - data_start + buffer_size * 2) * n_channels):
            recordings_1D = np.hstack((recordings_1D,
                                       np.zeros(
                                           buffer_size * n_channels,
                                           dtype='float32')))

    fin.close()

    # Convert to 2D array
    recording = recordings_1D.reshape(-1, n_channels)
                                    
    # convert spike train back to 0-offset values for indexeing into recordings
    indexes = np.where(np.logical_and(spike_train[:,0]>=data_start, 
                                      spike_train[:,0]<data_end))[0]
    spike_train = spike_train[indexes]-data_start 

    # load all waveforms at once
    waveforms = recording[spike_train[:, [0]].astype('int32')+offset
                  + np.arange(-spike_size, spike_size + 1)]

    return waveforms    
























