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

    CONFIG2.neigh_channels = CONFIG.neigh_channels
    CONFIG2.cluster.max_n_spikes = CONFIG.cluster.max_n_spikes
    
    CONFIG2.spike_size = CONFIG.spike_size

    return CONFIG2

def partition_input(save_dir, max_time,
                    fname_spike_index, fname_up=None):

    # make directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    # load data
    spike_index = np.load(fname_spike_index)
    if fname_up is not None:
        up_data = np.load(fname_up)
        spike_index_up = up_data['spike_train_upsampled']
        templates_up = up_data['templates_upsampled']

    # consider only spikes times less than max_time
    idx_keep = np.where(spike_index[:,0] < max_time)[0]

    # re-organize spike times and templates id
    n_units = np.max(spike_index[:, 1]) + 1
    spike_index_list = [[] for ii in range(n_units)]
    for j in idx_keep:
        tt, ii = spike_index[j]
        spike_index_list[ii].append(tt)

    if fname_up is not None:
        up_id_list = [[] for ii in range(n_units)]
        for j in idx_keep:
            ii = spike_index[j, 1]
            up_id = spike_index_up[j, 1]
            up_id_list[ii].append(up_id)


    # partition upsampled templates also
    # and save them
    fnames = []
    for unit in range(n_units):

        fname = os.path.join(save_dir, 'partition_{}.npz'.format(unit))

        if fname_up is not None:
            unique_up_ids = np.unique(up_id_list[unit])
            up_templates = templates_up[:, :, unique_up_ids]
            new_id_map = {iid: ctr for ctr, iid in enumerate(unique_up_ids)}
            up_id2 = [new_id_map[iid] for iid in up_id_list[unit]]

            np.savez(fname,
                     spike_times = spike_index_list[unit],
                     up_ids = up_id2,
                     up_templates = up_templates)
        else:
            np.savez(fname,
                     spike_times = spike_index_list[unit])
       
        fnames.append(fname)
        
    return np.arange(n_units), fnames

def gather_clustering_result(result_dir, out_dir):

    '''load clustering results
    '''

    logger = logging.getLogger(__name__)
    
    logger.info("gathering clustering results")

    # make output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # convert clusters to templates
    templates = []
    spike_indexes = []
    
    filenames = sorted(os.listdir(result_dir))
    for fname in filenames:
        data = np.load(os.path.join(result_dir, fname))
        temp_temp = data['templates']
        if (temp_temp.shape[0]) != 0:
            templates.append(temp_temp)
            temp = data['spiketime']
            for s in range(len(temp)):
                spike_indexes.append(temp[s])

    spike_indexes = np.array(spike_indexes)    
    templates = np.vstack(templates)

    logger.info("{} units loaded: ".format(len(spike_indexes)))

    fname_templates = os.path.join(out_dir, 'templates.npy')
    np.save(fname_templates, templates)

    # rearange spike indees from id 0..N
    spike_train = np.zeros((0,2), 'int32')
    for k in range(spike_indexes.shape[0]):    
        temp = np.zeros((spike_indexes[k].shape[0],2), 'int32')
        temp[:,0] = spike_indexes[k]
        temp[:,1] = k
        spike_train = np.vstack((spike_train, temp))

    fname_spike_train = os.path.join(out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)

    return fname_templates, fname_spike_train

def recompute_templates(fname_templates, fname_spike_train,
                        reader, out_dir,
                        multi_processing=None,
                        n_processors=1):

    logger = logging.getLogger(__name__)
    
    logger.info("recomputing templates")

    # make output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # make temp folder
    tmp_folder = os.path.join(out_dir, 'recompute')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # load templates
    templates = np.load(fname_templates)
    # get parameters
    n_units, n_timepoints, n_channels = templates.shape
    # resave it with different name
    fname_templates_original = os.path.join(
        out_dir, 'templates_original.npy')
    np.save(fname_templates_original, templates)
    # turn off
    templates = None

    # partition spike train per unit for multiprocessing
    units, fnames_input = partition_input(
        os.path.join(tmp_folder, 'partition'),
        reader.rec_len,
        fname_spike_train)


    # gather input arguments
    fnames_out = []
    for unit in units:
        fnames_out.append(os.path.join(
            tmp_folder,
            "template_unit_{}.npy".format(unit)))

    # run computing function
    if multi_processing:
        parmap.starmap(compute_template,
                   list(zip(fnames_input, fnames_out)),
                   reader,
                   n_timepoints,
                   processes=n_processors,
                   pm_pbar=True)
    else:
        for ctr in units:
            compute_template(
                fnames_input[ctr],
                fnames_out[ctr],
                reader,
                n_timepoints)

    # gather all info
    templates_new = np.zeros((n_units, n_timepoints, n_channels))
    for ctr, unit in enumerate(units):
        if os.path.exists(fnames_out[ctr]):
            templates_new[unit] = np.load(fnames_out[ctr])

    fname_templates = os.path.join(out_dir, 'templates.npy')
    np.save(fname_templates, templates_new)

    return fname_templates


def compute_template(fnames_input, fname_out, reader, spike_size):
    
    # load spike times
    spike_times = np.load(fnames_input)['spike_times']

    # subsample upto 1000
    max_spikes = 1000
    if len(spike_times) > max_spikes:
        spike_times = np.random.choice(a=spike_times,
                                       size=max_spikes,
                                       replace=False)

    # get waveforms
    wf, _ = reader.read_waveforms(spike_times,
                                  spike_size)

    # max channel
    mc = np.mean(wf, axis=0).ptp(0).argmax()

    # load reference template
    ref_template = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'ref_template.npy')))

    # get shift
    cut_edge = (spike_size - len(ref_template))//2
    if cut_edge > 0:
        best_shifts = align_get_shifts_with_ref(
            wf[:, cut_edge:-cut_edge, mc],
            ref_template)
    else:
        best_shifts = align_get_shifts_with_ref(
            wf[:, :, mc],
            ref_template)

    # shift
    wf = shift_chans(wf, best_shifts)

    # save result
    np.save(fname_out,np.median(wf, axis=0))


def post_cluster_process(templates, spike_indexes, full_run, CONFIG, chunk_dir, out_dir):

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
            
        if full_run:
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

    #collision_units = []
    #for k in range(templates.shape[0]):
    #    fname = os.path.join(save_dir,'unit_{}.npz'.format(k))
    #    tmp = np.load(fname)
    #    if tmp['kill']:
    #        collision_units.append(k)

    # logic:
    # 1. if tmp['kill'] is True, it is a collision
    # 2. if tmp['kill'] is Fale and no matched unit, then clean unit
    # 3. if tmp['kill'] is Fale and matched to a clean unit, then collision
    # 4. if tmp['kill'] is False and matched to non-clean, kill smaller of two
    collision_units = []
    clean_units = []
    matched_pairs = []
    for k in range(templates.shape[0]):
        fname = os.path.join(save_dir,'unit_{}.npz'.format(k))
        tmp = np.load(fname)
        # logic #1
        if tmp['kill']:
            collision_units.append(k)
        # logic #2
        elif tmp['unit_matched'] == None:
            clean_units.append(k)
        else:
            matched_pairs.append([k, tmp['unit_matched']])
    collision_units = np.array(collision_units)
    clean_units = np.array(clean_units)
    matched_pairs = np.array(matched_pairs)

    # logic #3
    if len(matched_pairs) > 0:
        matched_to_clean = np.in1d(matched_pairs[:,1], clean_units)
        collision_units = np.hstack((collision_units,
                                     matched_pairs[matched_to_clean, 0]))

        # logic #4
        collision_pairs = matched_pairs[~matched_to_clean]
        if len(collision_pairs) > 0:
            for j in range(collision_pairs.shape[0]):
                k, k_ = collision_pairs[j]
                if weights[k] > weights[k_]:
                    collision_units = np.append(collision_units, k_)
                else:
                    collision_units = np.append(collision_units, k_)
    # reload all saved data
    #pairs = []
    #for k in range(templates.shape[0]):
    #    fname = os.path.join(save_dir,'unit_{}.npz'.format(k))
    #    tmp = np.load(fname)
    #    if tmp['kill']:
    #        pairs.append([k, tmp['unit_matched']])

    #unique_pairs = []
    #non_unique_pairs = []    
    #for x, y in pairs:
    #    if [y, x] not in pairs:
    #        unique_pairs.append([x,y])
    #    elif [y, x] not in non_unique_pairs:
    #        non_unique_pairs.append([x,y])

    #collision_units = [k for k, k_ in unique_pairs]

    #for x, y in non_unique_pairs:
    #    if weights[x] > weights[y]:
    #        collision_units.append(x)
    #    else:
    #        collision_units.append(y)
    #collision_units = np.array(collision_units)


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
                        mad_gap=0.8, mad_gap_breach=3, up_factor=16, residual_max_norm=1.2):
    jitter = 1
    n_units, n_times, n_channels = templates.shape
    visch = np.where(templates[unit].ptp(axis=0) > 2.)[0]
    if len(visch) == 0:
        visch = np.array([templates[unit].ptp(axis=0).argmax()])

    # get spike times and waveforms
    spt = spike_train[spike_train[:, 1] == unit, 0] - n_times//2
    max_spikes = 500
    if len(spt) > max_spikes:
        spt = np.random.choice(a=spt,
                               size=max_spikes,
                               replace=False)
    wf =  binary_reader_waveforms(
        standardized_filename=standardized_filename,
        n_channels=n_channels,
        n_times=n_times, spikes=spt,
        channels=None)[0]
    # limit to visible channels
    wf = wf[:, :, visch]
    # max channel within visible channels
    mc = np.mean(wf, axis=0).ptp(0).argmax()

    # upsample waveforms and aligne them
    wf_up = scipy.signal.resample(wf, n_times * up_factor, axis=1)
    mean_up = np.mean(wf_up[:,:,mc],0)

    n_times_up = wf_up.shape[1]
    t_start = n_times_up//4
    t_end = n_times_up - n_times_up//4

    shifts = np.arange(-up_factor//2, up_factor//2+1)
    fits = np.zeros((wf.shape[0], len(shifts)))
    for j, shift in enumerate(shifts):
        fits[:,j] = np.sum(wf_up[:,t_start+shift:t_end+shift,mc]*mean_up[t_start:t_end], 1)
    best_shifts = shifts[fits.argmax(1)]

    wf_aligned = np.zeros(wf_up.shape, 'float32')
    for j in range(wf.shape[0]):
        wf_aligned[j] = np.roll(wf_up[j], -best_shifts[j], axis=0)

    wf_aligned = wf_aligned[:,t_start:t_end]
    wf_aligned = wf_aligned[:,np.arange(0, wf_aligned.shape[1], up_factor)]

    # mad value for aligned waveforms
    t_mad = np.median(
            np.abs(np.median(wf_aligned, axis=0)[None] - wf_aligned), axis=0)

    mad_loc = t_mad > mad_gap

    # if every visible channels are mad channels, kill it
    if np.all(mad_loc.sum(0) > mad_gap_breach):
        kill = True
        unit_matched = None

    # if no mad chanels, keep it
    elif np.all(mad_loc.sum(0) <= mad_gap_breach):
        kill = False
        unit_matched = None

    # if not, but if there is a unit that matches on
    # non mad channels, hold off
    # if there is no matching unit, keep it
    else:
        kill = False

        mad_channels = visch[mad_loc.sum(0) > mad_gap_breach]
        non_mad_channels = np.setdiff1d(np.arange(n_channels), mad_channels)

        idx_no_target = np.arange(n_units)
        idx_no_target = np.delete(idx_no_target, unit)
        
        # run deconv without channels with high collisions
        residual, unit_matched = run_deconv(templates[unit][:,non_mad_channels],
                                            templates[idx_no_target][:,:,non_mad_channels],
                                            up_factor)
        
        if np.max(np.abs(residual)) > residual_max_norm:
            unit_matched = None
        else:
            unit_matched = idx_no_target[unit_matched]

    fname = os.path.join(save_dir,'unit_{}.npz'.format(unit))
    np.savez(fname,
             kill=kill,
             unit_matched=unit_matched
            )


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


def clean_templates(templates, spike_train_cluster, weights, CONFIG):

    # find units < 3SU 
    # Cat: TODO: read this threshold and flag from CONFIG
    template_threshold = 3

    print ("  cleaning templates (temps, time, chan): ", templates.shape)
    ptps = templates.ptp(1).max(1)
    idx1 = np.where(ptps>=template_threshold)[0]
    print ("  deleted clusters < {}SU: ".format(template_threshold), templates.shape[0]-idx1.shape[0])

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

def pca_denoise(data, pca_mean, pca_components):
    data_pca = np.matmul(data-pca_mean, pca_components.T)
    return np.matmul(data_pca, pca_components)+pca_mean
