import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm
import parmap
import scipy
import networkx as nx
import logging

from yass import read_config
from yass.cluster.cluster import align_get_shifts_with_ref, shift_chans

from statsmodels import robust
from scipy.signal import argrelmin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from yass.deconvolve.match_pursuit import (MatchPursuit_objectiveUpsample, 
                                            Residual)
                                                                                        
from yass.cluster.util import (binary_reader,
                               global_merge_max_dist, PCA, 
                               load_waveforms_from_memory,
                               make_CONFIG2, upsample_parallel, 
                               clean_templates, find_clean_templates,
                               global_merge_max_dist)

from yass.cluster.cluster import Cluster

from yass.deconvolve.merge import TemplateMerge

from diptest import diptest as dp

import multiprocessing as mp

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                 for name, color in colors.items())
sorted_colors = [name for hsv, name in by_hsv]


class Deconv(object):

    def __init__(self, spike_train, templates, output_directory, 
                 recordings_filename, CONFIG):

        self.spike_train = spike_train
        self.templates = templates.transpose(1,2,0)
        self.output_directory = output_directory
        self.recordings_filename = recordings_filename
        self.CONFIG = make_CONFIG2(CONFIG)
        
        # initialize 
        self.initialize()
        
        # run iterative deconv-clustering on initial chunk - OPTIONAL
        # Cat: TODO: maybe try to use a single deconv() function for both
        if self.n_iterations_deconv>0: 
            self.deconv_iterative()
        
        # run deconv on all data
        self.deconv_final()
        
        # save final output in main directory
        np.save(os.path.join(self.CONFIG.path_to_output_directory,
                            'spike_train_post_deconv_post_merge.npy'),
                    self.spike_train)
        np.save(os.path.join(self.CONFIG.path_to_output_directory,
                            'templates_post_deconv_post_merge.npy'),
                    self.templates)
                    
                
    def initialize(self): 
        
        # location of standardized .bin file
        self.standardized_filename = os.path.join(self.CONFIG.path_to_output_directory,
                                             'preprocess', 
                                             self.recordings_filename)
        self.n_channels = self.CONFIG.recordings.n_channels
        self.sampling_rate = self.CONFIG.recordings.sampling_rate
        self.n_processors = self.CONFIG.resources.n_processors
        self.n_sec_chunk = self.CONFIG.resources.n_sec_chunk
        self.buffer_size = 200
        self.spike_size_default = int(self.CONFIG.recordings.spike_size_ms*
                                      self.CONFIG.recordings.sampling_rate//1000*2+1)
                                      
        # compute segment list for parallel processing
        self.compute_idx_list()

        # Cat: TODO: read from CONFIG
        self.threshold = 50.
        self.conv_approx_rank = 10
        self.default_upsample_value=0
        self.upsample_max_val = 32.

        # Cat: TODO: max iteration not currently used
        self.max_iter = 5000
        
        # set the number of cluster iterations to be done
        # Cat: TODO: read from CONFIG
        self.n_iterations_merge = 1

        # iterative deconv parameters
        # Cat: TODO: read from CONFIG
        self.n_iterations_deconv = 0
        self.n_seconds_initial = 1200
        self.initial_chunk = int(self.n_seconds_initial//self.CONFIG.resources.n_sec_chunk) 

        # make deconv directory
        self.deconv_dir = os.path.join(self.CONFIG.path_to_output_directory,
                                  'deconv')
        if not os.path.isdir(self.deconv_dir):
            os.makedirs(self.deconv_dir)
            
        # if self.n_iterations_deconv>0: 
            # if not os.path.isdir(self.deconv_dir+'/initial/'):
                # os.makedirs(self.deconv_dir+'/initial/')
       
    def compute_idx_list(self):
    
        fp_len = np.memmap(self.standardized_filename, 
                   dtype='float32', mode='r').shape[0]

        # Generate indexes in chunks (units of seconds); usually 10-60sec
        indexes = np.arange(0, fp_len / self.n_channels, 
                            self.sampling_rate * self.n_sec_chunk)
        if indexes[-1] != fp_len/self.n_channels:
            indexes = np.hstack((indexes, fp_len/self.n_channels))

        # Make the 4 parameter list to be sent to deconvolution algorithm
        idx_list = []
        for k in range(len(indexes) - 1):
            idx_list.append([
                indexes[k], indexes[k + 1], self.buffer_size,
                indexes[k + 1] - indexes[k] + self.buffer_size
            ])

        self.idx_list = np.int64(np.vstack(idx_list)) #[:2]


    def deconv_iterative(self):

        self.idx_list_local = self.idx_list[:self.initial_chunk]
        deconv_chunk_dir = os.path.join(self.CONFIG.path_to_output_directory,
                                        'deconv',
                                        'initial/')
        if not os.path.isdir(deconv_chunk_dir):
            os.makedirs(deconv_chunk_dir)
        
        # loop over iterations
        for k in range(self.n_iterations_deconv): 

            # initialize recluster directory
            self.deconv_chunk_dir = os.path.join(deconv_chunk_dir,
                                                str(k))

            if not os.path.isdir(self.deconv_chunk_dir):
                os.makedirs(self.deconv_chunk_dir)

            # run match pursuit and return templates and spike_trains
            self.match_pursuit_function()

            # run post-deconv merge; recomputes self.templates and self.spike_train
            #self.merge()

            # reclusters and loads a new self.templates structure
            self.reclustering_function()
            
            # post recluster merge
            _, templates = global_merge_max_dist(
                                        self.deconv_chunk_dir+'/recluster/',
                                        self.CONFIG,
                                        'deconv',
                                        np.arange(self.templates.shape[2]))
            
            # reshape templates
            self.templates = templates.transpose(1,2,0)

            
    def deconv_final(self):

        # Cat: TODO: no need to use chunks anymore, just run all data over single
        #      temporariy fix: run over rest of data in single chunk run:

        # select segments to be processed
        self.idx_list_local = self.idx_list
        self.deconv_chunk_dir = os.path.join(self.CONFIG.path_to_output_directory,
                                        'deconv',
                                        'final/')
        if not os.path.isdir(self.deconv_chunk_dir):
            os.makedirs(self.deconv_chunk_dir)

        # Cat: TODO: don't recomp temp_temp for final step if prev. computed
        self.match_pursuit_function()

        # first compute residual
        print ("  computing residual to generate clean spikes")
        self.compute_residual_function()

        # run post-deconv merge
        self.merge()


    def merge(self): 
                            
        print ("\nPost-deconv merge...")

        # iterate over template-based merge step
        if not os.path.isdir(self.deconv_chunk_dir+'/merge'):
            os.makedirs(self.deconv_chunk_dir+'/merge')
        
        for i in range(self.n_iterations_merge):
            print ("  running merge iteration: ", i+1 , "/", self.n_iterations_merge)

            tm = TemplateMerge(self.templates, self.spike_train,
                               self.sparse_upsampled_templates,
                               self.spike_train_upsampled,
                               self.CONFIG, 
                               self.deconv_chunk_dir+'/merge',
                               iteration=i, 
                               recompute=False)
           
            merge_list = tm.get_merge_pairs()

            (self.spike_train, 
             self.templates,
             self.spike_train_upsampled,
             self.sparse_upsampled_templates,
             self.connected_components) = merge_pairs(
                self.templates, self.spike_train,
                self.sparse_upsampled_templates, self.spike_train_upsampled,
                merge_list, self.CONFIG)

            # save spike train
            np.savez(os.path.join(self.deconv_chunk_dir,
                            'results_post_deconv_post_merge_'+str(i)),
                     templates=self.templates,
                     spike_train=self.spike_train,
                     templates_upsampled=self.sparse_upsampled_templates,
                     spike_train_upsampled=self.spike_train_upsampled,
                     connected_components=self.connected_components)
                 
            print ("  deconv templates (post-merge): ", self.templates.shape)
    
            # Note: new self.templates and self.spike_train is computed above
            # no need to return them to deconv
            # also, this will be overwritten for multiple iterations of merge function
            np.save(os.path.join(self.deconv_chunk_dir,
                            'spike_train_post_deconv_post_merge.npy'),
                    self.spike_train)
            np.save(os.path.join(self.deconv_chunk_dir,
                            'templates_post_deconv_post_merge.npy'),
                    self.templates)


    def resize_templates(self, template_len=50):

        n_channels = self.CONFIG.recordings.n_channels
        idx_list_local = idx_list[:initial_chunk]
        idx = [idx_list_local[0][0], idx_list_local[-1][1], 
               idx_list_local[0][2], idx_list_local[0][3]]

        spike_index_filename = os.path.join(CONFIG.data.root_folder,
                                             output_directory,
                                             'spike_train_cluster.npy')

        print ("  original templates: ", templates.shape)
        templates = recompute_templates(idx,
                                     templates,
                                     spike_index_filename,
                                     buffer_size, 
                                     standardized_filename, 
                                     n_channels,
                                     CONFIG,
                                     self.template_len)
                                     
        templates = np.float32(templates).swapaxes(0,1).swapaxes(1,2)
        print ("  resized templates: ", templates.shape)


    def match_pursuit_function(self):
                            
        print ("")
        print ("Initializing Match Pursuit, # segments: ", 
                self.idx_list_local.shape[0], 
                " start: ", self.idx_list_local[0][0], " end: ", 
                self.idx_list_local[-1][1], " start(sec): ", 
                round(self.idx_list_local[0][0]/float(self.CONFIG.recordings.sampling_rate),1),
                " end(sec): ", 
                round(self.idx_list_local[-1][1]/float(self.CONFIG.recordings.sampling_rate),1))

        # initialize match pursuit
        mp_object = MatchPursuit_objectiveUpsample(
                                  temps=self.templates,
                                  deconv_chunk_dir=self.deconv_chunk_dir,
                                  standardized_filename=self.standardized_filename,
                                  max_iter=self.max_iter,
                                  upsample=self.upsample_max_val,
                                  threshold=self.threshold,
                                  conv_approx_rank=self.conv_approx_rank,
                                  n_processors=self.CONFIG.resources.n_processors,
                                  multi_processing=self.CONFIG.resources.multi_processing)
        
        print ("  running Match Pursuit...")

        if not os.path.isdir(self.deconv_chunk_dir+'/segs'):
            os.makedirs(self.deconv_chunk_dir+'/segs')

        # collect segments not yet completed
        args_in = []
        for k in range(len(self.idx_list_local)):
            fname_out = (self.deconv_chunk_dir+'/segs'+
                         "/seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
            if os.path.exists(fname_out)==False:
                args_in.append([[self.idx_list_local[k], k],
                                self.buffer_size])

        if len(args_in)>0:
            if self.CONFIG.resources.multi_processing:
                parmap.map(mp_object.run,
                            args_in,
                            processes=self.CONFIG.resources.n_processors,
                            pm_pbar=True)

                #p = mp.Pool(processes = self.CONFIG.resources.n_processors)
                #p.map_async(mp_object.run, args_in).get(988895)
                #p.close()
            else:
                for k in range(len(args_in)):
                    mp_object.run(args_in[k])
        
        # collect spikes
        res = []
        for k in range(len(self.idx_list_local)):
            fname_out = (self.deconv_chunk_dir+
                         "/segs//seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
                         
            data = np.load(fname_out)
            res.append(data['spike_train'])

        print ("  gathering spike trains")
        self.dec_spike_train = np.vstack(res)
        
        # corrected spike trains using shift due to deconv tricks
        self.dec_spike_train = mp_object.correct_shift_deconv_spike_train(
                                                    self.dec_spike_train)
        print ("  initial deconv spike train: ", 
                            self.dec_spike_train.shape)

        '''
        # ********************************************
        # * LOAD CORRECT TEMPLATES FOR RESIDUAL COMP *
        # ********************************************
        '''
        # get upsampled templates and mapping for computing residual
        self.sparse_upsampled_templates, self.deconv_id_sparse_temp_map = (
                                mp_object.get_sparse_upsampled_templates())


        # save original spike ids (before upsampling
        # Cat: TODO: get this value from global/CONFIG
        self.spike_train = self.dec_spike_train.copy()
        self.spike_train[:, 1] = np.int32(self.spike_train[:, 1]/
                                                self.upsample_max_val)
        self.spike_train_upsampled = self.dec_spike_train.copy()
        self.spike_train_upsampled[:, 1] = self.deconv_id_sparse_temp_map[
            self.spike_train_upsampled[:, 1]]
        np.savez(self.deconv_chunk_dir + "/deconv_results.npz",
                 spike_train=self.spike_train,
                 templates=self.templates,
                 spike_train_upsampled=self.spike_train_upsampled,
                 templates_upsampled=self.sparse_upsampled_templates)

        np.save(os.path.join(self.deconv_chunk_dir,
                            'templates_post_deconv_pre_merge'),
                            self.templates)
                            
        np.save(os.path.join(self.deconv_chunk_dir,
                            'spike_train_post_deconv_pre_merge'),
                            self.spike_train)

    def compute_residual_function(self):
                                  
        # Note: this uses spike times occuring at beginning of spike
        fname = os.path.join(self.deconv_chunk_dir,
                                         'residual.bin')
                                         
        if os.path.exists(fname)==True:
            return

        # re-read entire block to get waveforms 
        # get indexes for entire chunk from local chunk list
        idx_chunk = [self.idx_list_local[0][0], self.idx_list_local[-1][1], 
                     self.idx_list_local[0][2], self.idx_list_local[0][3]]

        # read data block using buffer
        n_channels = self.CONFIG.recordings.n_channels
        
        #print (standardized_filename)
        # recording_chunk = binary_reader(idx_chunk, 
                                        # self.buffer_size, 
                                        # self.standardized_filename, 
                                        # self.n_channels)

        # compute residual for data chunk and save to disk
        # Cat TODO: parallelize this and also figure out a faster way to 
        #           process this data
        # Note: offset spike train to account for recording_chunk buffer size
        # this also enables working with spikes that are near the edges
        dec_spike_train_offset = self.dec_spike_train
        dec_spike_train_offset[:,0] += self.buffer_size

        #np.save(deconv_chunk_dir+'/dec_spike_train_offset_upsampled.npy',
        #        dec_spike_train_offset)

        print ("  init residual object")
        wf_object = Residual(self.sparse_upsampled_templates,
                                          dec_spike_train_offset,
                                          self.buffer_size,
                                          self.CONFIG.resources.n_processors,
                                          self.deconv_chunk_dir,
                                          self.n_sec_chunk,
                                          self.idx_list_local,
                                          self.standardized_filename)


        # compute residual using initial templates obtained above
        min_ptp = 0.0
        print ("  residual computation excludes units < ", min_ptp, "SU")
        wf_object.compute_residual_new(self.CONFIG, min_ptp)
    
        # new method for saving residuals
        wf_object.save_residual()
        #wf_object.data = wf_object.data[self.buffer_size:-self.buffer_size].reshape(-1)
        #wf_object.data.tofile(fname)



    def reclustering_function(self):
                         
        print ("\nPost-deconv reclustering...")
        # flag to indicate whether clustering original data or post-deconv
        deconv_flag = True
       
        self.recluster_dir = os.path.join(self.deconv_chunk_dir,
                                    "recluster")
             
        units = np.arange(self.templates.shape[2])
        args_in = []
        for unit in units:
            fname_out = (self.recluster_dir+
                         "/unit_{}.npz".format(
                         str(unit).zfill(6)))

            if os.path.exists(fname_out)==False:
                args_in.append([
                    deconv_flag,
                    unit, 
                    self.CONFIG,
                    self.spike_train,
                    self.recluster_dir, 
                    #self.spike_train_cluster,  #MIGHT WISH TO EXCLUDE THIS FOR NOW
                    self.templates]
                    )
                
        # run residual-reclustering function
        if len(args_in)>0:
            if self.CONFIG.resources.multi_processing:
                p = mp.Pool(processes = self.CONFIG.resources.n_processors)
                p.map_async(Cluster, args_in).get(988895)
                p.close()
            else:
                for unit in range(len(args_in)):
                    Cluster(args_in[unit])
                         
        print ("  reclustering complete")

def merge_pairs(templates, spike_train, templates_upsampled, spike_train_upsampled, merge_list, CONFIG2):
    
    merge_matrix = np.zeros((templates.shape[2], templates.shape[2]),'int32')
    for merge_pair in merge_list:
        if merge_pair != None:
            merge_matrix[merge_pair[0],merge_pair[1]]=1
            merge_matrix[merge_pair[1],merge_pair[0]]=1

    # compute graph based on pairwise connectivity
    G = nx.from_numpy_matrix(merge_matrix)
    
    merge_array=[]
    for cc in nx.connected_components(G):
        merge_array.append(list(cc))

    weights = np.zeros(templates.shape[2])
    unique_ids, n_spikes = np.unique(spike_train[:,1], return_counts=True)
    weights[unique_ids] = n_spikes
    
    spike_train_new = np.copy(spike_train)
    templates_new = np.zeros((templates.shape[0], templates.shape[1], len(merge_array)))
    
    for new_id, units in enumerate(merge_array):
        
        # update templates
        if len(units) > 1:
            # align first
            aligned_temps, shifts = align_templates(templates[:,:, units])
            temp_ = np.average(aligned_temps, weights=weights[units], axis=2)
            templates_new[:,:,new_id] = temp_
            
        elif len(units) == 1:
            templates_new[:,:,new_id] = templates[:,:,units[0]]

        # update spike train id
        # also update time and upsampled templates based on the shift
        for ii, unit in enumerate(units):
            
            idx = spike_train[:,1] == unit

            # updated spike train id
            spike_train_new[idx,1] = new_id

            if len(units) > 1:
                
                # update spike train time
                spt_old = spike_train[idx,0]
                shift_int = int(np.round(shifts[ii]))
                spike_train_new[idx,0] = spt_old - shift_int
                spike_train_upsampled[idx,0] = spt_old - shift_int
                
                # update upsampled templates
                upsampled_ids = np.unique(spike_train_upsampled[idx,1])
                aligned_upsampled_temps = shift_chans(
                    templates_upsampled[:, :, upsampled_ids].transpose(2,0,1), 
                    np.ones(len(upsampled_ids))*shift_int).transpose(1,2,0)
                templates_upsampled[:, :, upsampled_ids] = aligned_upsampled_temps
                
                

    return (spike_train_new, templates_new, 
            spike_train_upsampled, templates_upsampled,
            merge_array)

def align_templates(templates):
    
    templates = templates.transpose(2,0,1)
    max_idx = templates.ptp(1).max(1).argmax(0)
    ref_template = templates[max_idx]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]

    temps = templates[:, :, max_chan]

    best_shifts = align_get_shifts_with_ref(
                    temps, ref_template)

    aligned_templates = shift_chans(templates, best_shifts)
    
    return aligned_templates.transpose(1,2,0), best_shifts
    
def delete_spikes(templates, spike_train):

    # need to transpose axes for analysis below
    templates = templates.swapaxes(0,1)

    # remove templates < 3SU
    # Cat: TODO: read this threshold and flag from CONFIG
    template_threshold = 3
    
    ptps = templates.ptp(0).max(0)
    idx_remove = np.where(ptps<=template_threshold)[0]
    print ("  deleted spikes from # clusters < 3SU: ", idx_remove.shape[0])

    # Cat: TODO: speed this up!
    for idx_ in idx_remove:
        temp_idx = np.where(spike_train[:,1]==idx_)[0]
        spike_train = np.delete(spike_train, temp_idx, axis=0)        

    return spike_train


    

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
    wf_start = int(.15 * (wlen-1))
    wf_end = -int(.20 * (wlen-1))
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]

    ref_upsampled = wf_up[-1]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])

    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2)+1)):
        ref_shifted[:,i] = ref_upsampled[s+wf_start:s+wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[1],wlen_trunc])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end]

    return np.float32(wf_final[:,::upsample_factor]), best_shifts
    


def recompute_templates_from_raw(templates,
                                 deconv_chunk_dir, 
                                 idx, 
                                 buffer_size, 
                                 standardized_filename, 
                                 n_channels,
                                 CONFIG):
    '''
        Function reloads spike trains from deconv+residual reclustering
        and recomputes templates using raw record;
        - it then overwrites the original sorted files
    '''

    # Cat: TODO: data_start is not 0, should be
    data_start = 0
    offset = 200
    spike_size = int(self.CONFIG.recordings.spike_size_ms * 2
                          * self.CONFIG.recordings.sampling_rate / 1000) + 1
    global recording_chunk
    recording_chunk = binary_reader(idx, 
                                    buffer_size, 
                                    standardized_filename, 
                                    n_channels)

    # make deconv directory
    raw_dir = deconv_chunk_dir.replace('recluster','raw_templates')
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)
    
    # compute raw tempaltes in parallel
    units = []
    for k in range(templates.shape[2]):
        fout = (raw_dir +
                "/unit_{}.npz".format(
                str(k).zfill(6)))
        if os.path.exists(fout):
            continue
        units.append(k)

    if CONFIG.resources.multi_processing:
        parmap.map(raw_templates_parallel,
               units,
               deconv_chunk_dir,
               raw_dir,
               data_start,
               offset,
               spike_size,
               CONFIG,
               processes=CONFIG.resources.n_processors,
               pm_pbar=True)
    else:
        for unit in units:
            raw_templates_parallel(
            unit,
            deconv_chunk_dir,
            raw_dir,
            data_start,
            offset,
            spike_size,
            CONFIG)


def recompute_templates(idx,
                         templates,
                         spike_index_filename,
                         buffer_size, 
                         standardized_filename, 
                         n_channels,
                         CONFIG,
                         template_len):
                                     
    '''
        Function that reloads templates 
    '''

    # Cat: TODO: data_start is not 0, should be
    data_start = 0
    offset = 200
    
    spike_size = template_len*2+1
    
    global recording_chunk
    recording_chunk = binary_reader(idx, 
                                    buffer_size, 
                                    standardized_filename, 
                                    n_channels)
    print (templates.shape)
    units = np.arange(templates.shape[2])

    #if False:
    if CONFIG.resources.multi_processing:
        new_templates = parmap.map(resize_templates_parallel,
               units,
               spike_index_filename,
               data_start,
               offset,
               spike_size,
               processes=CONFIG.resources.n_processors,
               pm_pbar=True)
    else:
        for unit in units:
            resize_templates_parallel(
            unit,
            spike_index_filename,
            data_start,
            offset,
            spike_size)                              

    return new_templates

def resize_templates_parallel(unit, 
                              spike_index_filename,
                              data_start,
                              offset,
                              spike_size):

    # load spikes and templates from cluster step
    spike_index_cluster = np.load(spike_index_filename)

    idx = np.where(spike_index_cluster[:,1]==unit)[0]
    spike_train = spike_index_cluster[idx]

    # recompute templates based on raw data
    templates = []
    wf = load_waveforms_from_memory(recording_chunk,
                        data_start,
                        offset,
                        spike_train,
                        spike_size)

    template = np.median(wf, axis=0)

    return template
    
def raw_templates_parallel(unit, 
                           deconv_chunk_dir,
                           raw_dir,
                           data_start,
                           offset,
                           spike_size,
                           CONFIG):

    # load spikes and templates from deconv step
    fname = (deconv_chunk_dir+"/unit_{}.npz".format(str(unit).zfill(6)))
    data = np.load(fname)

    templates_postrecluster = data['templates_postrecluster']
    templates_cluster= data['templates_cluster']
    spike_index_postrecluster = data['spike_index_postrecluster']
    spike_index_cluster = data['spike_index_cluster']

    # recompute templates based on raw data
    templates = []
    for s in range(len(spike_index_postrecluster)):
        # shift spike times to accmodate offset buffer
        spike_train = spike_index_postrecluster[s]-200

        wf = load_waveforms_from_memory(recording_chunk,
                            data_start,
                            offset,
                            spike_train,
                            spike_size)

        #
        template = np.median(wf, axis=0)
        #template = scipy.stats.trim_mean(wf, 0.10, axis=0)
        templates.append(template)


    fout = (raw_dir +
            "/unit_{}.npz".format(
                str(unit).zfill(6)))

    np.savez(fout,
             spike_index=spike_index_postrecluster,
             templates=templates)

    if len(spike_index_postrecluster)==0:
        templates = templates_postrecluster = [templates_cluster.copy() * 0.001]
        spike_index_postrecluster = np.zeros((1,0),'int32')

    # pick channel of first template if multiple
    channel_raw = templates[0].ptp(0).argmax(0)
    channel_original = templates_cluster.ptp(0).argmax(0)

    #templates = np.array(templates)
    # plot templates
    plotting=True
    if plotting:

        fig = plt.figure(figsize =(40,20))
        grid = plt.GridSpec(40,20,wspace = 0.0,hspace = 0.2)
        ax = fig.add_subplot(grid[:, :])

        yscale = 4.
        xscale = 5.
        labels = []
        ctr=0
        # plot original cluster unit template
        ax.plot(CONFIG.geom[:,0]-30+
                np.arange(-templates_cluster.shape[0]//2.,templates_cluster.shape[0]//2,1)[:,np.newaxis]/xscale,
                CONFIG.geom[:,1] + templates_cluster*yscale,
                c=sorted_colors[ctr%100])

        patch_j = mpatches.Patch(color = sorted_colors[ctr%100],
                        label = "cluster :"+ str(spike_index_cluster.shape[0])+
                        ", ch: "+str(channel_original))
        labels.append(patch_j)
        ctr+=1

        # plot templates - using residual data
        for k in range(spike_index_postrecluster.shape[0]):
            ax.plot(CONFIG.geom[:,0]-15+
                    np.arange(-templates_postrecluster[k].shape[0]//2,
                             templates_postrecluster[k].shape[0]//2,1)[:,np.newaxis]/xscale,
                             CONFIG.geom[:,1] + templates_postrecluster[k]*yscale,
                             c=sorted_colors[ctr%100])

            patch_j = mpatches.Patch(color = sorted_colors[ctr%100],
                            label = "recluster+res: "+ str(spike_index_postrecluster[k].shape[0])+
                            ", ch: "+str(channel_raw))
            labels.append(patch_j)
            ctr+=1

        # plot templates - using raw data
        for k in range(spike_index_postrecluster.shape[0]):
            ax.plot(CONFIG.geom[:,0]+
                  np.arange(-templates[k].shape[0]//2,
                             templates[k].shape[0]//2,1)[:,np.newaxis]/xscale,
                             CONFIG.geom[:,1] + templates[k]*yscale,
                             c=sorted_colors[ctr%100])

            patch_j = mpatches.Patch(color = sorted_colors[ctr%100],
                            label = "recluster+raw: "+ str(spike_index_postrecluster[k].shape[0])+
                            ", ch: "+str(channel_original))
            labels.append(patch_j)
            ctr+=1

        # plot chan nubmers
        for i in range(CONFIG.recordings.n_channels):
            ax.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), alpha=0.4,
                                                            fontsize=30)
            # fill bewteen 2SUs on each channel
            ax.fill_between(CONFIG.geom[i,0] + np.arange(-61,0,1)/3.,
                -yscale + CONFIG.geom[i,1], yscale + CONFIG.geom[i,1],
                color='black', alpha=0.05)

        # plot max chan with big red dot
        ax.scatter(CONFIG.geom[channel_raw,0]+5, CONFIG.geom[channel_raw,1], s = 1000,
                                                color = 'red',alpha=0.3)
        # finish plotting
        ax.legend(handles = labels, fontsize=15)
        fig.suptitle("Unit: "+str(unit)+" chan: "+str(channel_raw), fontsize=25)
        fig.savefig(raw_dir+"/unit_"+str(unit)+".png")
        plt.close(fig)




def offset_spike_train(CONFIG, dec_spike_train_offset):
    # Cat: need to offset spike train as deconv outputs spike times at
    #      beginning of waveform; do this after the residual computation step
    #      which is based on beginning waveform residual computation
    #      So converting back to nn_spike time alignment for clustering steps
    # Cat: TODO read from CONFIG file; make sure this is corrected
    deconv_offset = int(CONFIG.recordings.spike_size_ms*
                                CONFIG.recordings.sampling_rate/1000.)
    print ("  offseting deconv spike train (timesteps): ",deconv_offset)
    dec_spike_train_offset[:,0]+= deconv_offset

    # - colapse unit ids using the expanded templates above
    #   as some templates were duplicated 30 times with shifts
    # - need to reset unit ids back to original templates and collapse
    #   over the spike trains
    # Note: deconv_spike_train does not have data buffer offset in it
    
    # Cat: TODO: read this value from CONFIG or another place; important!!
    upsample_max_val = 32.
    dec_spike_train_offset[:,1] = np.int32(dec_spike_train_offset[:,1]/upsample_max_val)

    return dec_spike_train_offset

    

def deconv_residual_recluster(data_in): 
    
    unit = data_in[0]
    dec_spike_train_offset = data_in[1]
    spike_train_cluster_new = data_in[2]
    idx_chunk = data_in[3]
    template_original = data_in[4]
    CONFIG = data_in[5]
    deconv_chunk_dir = data_in[6]
    data_start = data_in[7]
    offset = data_in[8]
    residuaL_clustering_flag = data_in[9]

    # Cat: TODO: read this from CONFIG
    n_dim_pca_compression = 5
    
    deconv_filename = (deconv_chunk_dir+"/unit_"+str(unit).zfill(6)+'.npz')
    if os.path.exists(deconv_filename)==False:
        
        # select deconv spikes and read waveforms
        unit_sp = dec_spike_train_offset[dec_spike_train_offset[:, 1] == unit, :]

        #print (unit, unit_sp)
        # save all clustered data
        if unit_sp.shape[0]==0: 
            print ("  unit: ", str(unit), " has no spikes...")
            np.savez(deconv_filename, spike_index=[], 
                        templates=[],
                        templates_std=[],
                        weights=[])
            return
               
        if unit_sp.shape[0]!= np.unique(unit_sp[:,0]).shape[0]:
            print ("  unit: ", unit, " non unique spikes found...")
            idx_unique = np.unique(unit_sp[:,0], return_index = True)[1]
            unit_sp = unit_sp[idx_unique]

        # Cat: TODO: load wider waveforms just as in clustering
        # Cat TODO: Need to load from CONFIG; careful as the templates are
        #           now being extended during cluster preamble using flexible val
        spike_size = 111
        #template = template[25:-25,:]
        
        
        # Cat: TODO read this from disk
        deconv_max_spikes = 10000
        if unit_sp.shape[0]>deconv_max_spikes:
            idx_deconv = np.random.choice(np.arange(unit_sp.shape[0]),
                                          size=deconv_max_spikes,
                                          replace=False)
            unit_sp = unit_sp[idx_deconv]         

        # Cat: TODO: here we add addtiional offset for buffer inside residual matrix
        # read waveforms by adding templates to residual
        residuaL_clustering_flag=True
        if residuaL_clustering_flag:
            spike_size = int(self.CONFIG.recordings.spike_size_ms * 2
                              * self.CONFIG.recordings.sampling_rate / 1000) + 1
            wf = get_wfs_from_residual(unit_sp,
                                       template_original, 
                                       deconv_chunk_dir,
                                       spike_size)
        else:    
            # read waveforms from recording chunk in memory
            # load waveforms with some padding then clip them below
            # Cat: TODO: spike_padding to be read/fixed in CONFIG
            #unit_sp[:,0]+=50
            #spike_size = template.shape[0]//2
            offset = 0
            wf = load_waveforms_from_memory(recording_chunk, 
                                            data_start, 
                                            offset, 
                                            unit_sp, 
                                            spike_size)
       
        # if wf.shape[1]==111:
        #     spike_start = 25
        #     spike_end = -25
        # elif wf.shape[1]==XX:
        #     spike_start =0
        #     spike_end = wf.shape[1]
        # else:
        #     print ("  spike width irregular fix this...")
        #     quit()


        #np.save(deconv_chunk_dir+'/wfs_'+str(unit).zfill(6)+'.npy', wf)
        # Cat: TODO: during deconv reclustering may not wish to exclude off-max
        #               channel templates
        channel = wf.mean(0).ptp(0).argmax(0)

        # Cat: TODO: set this from CONFIG;
        min_spikes_local = CONFIG.cluster.min_spikes

        triageflag = False
        alignflag = True
        plotting = False
        if unit%10==0:
            plotting = False
            
        scale = 10 
        n_feat_chans = 5
        n_dim_pca = 3
        wf_start = 0
        wf_end = 40
        mfm_threshold = 0.90
        knn_triage_threshold = 0.90
        upsample_factor = 5
        nshifts = 15
        
                
        chans = [] 
        gen = 0     #Set default generation for starting clustering stpe
        assignment_global = []
        spike_index = []
        templates = []
        feat_chans_cumulative = []
        
        # plotting parameters
        if plotting:
            #x = np.zeros(100, dtype = int)          
            #fig = plt.figure(figsize =(50,25))
            #grid = plt.GridSpec(10,5,wspace = 0.0,hspace = 0.2)
            #ax_t = fig.add_subplot(grid[13:, 6:])
            
            x = np.zeros(100, dtype = int)
            fig = plt.figure(figsize =(100,100))
            grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
            ax_t = fig.add_subplot(grid[13:, 6:])
            
        else:
            fig = []
            grid = []
            ax_t = []
            x = []
            
        deconv_flag = True
        RRR3_noregress_recovery_dynamic_features(unit, wf[:, spike_start:spike_end], 
            unit_sp, gen, fig, grid, x,
            ax_t, triageflag, alignflag, plotting, n_feat_chans, 
            n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
            upsample_factor, nshifts, assignment_global, spike_index, scale,
            knn_triage_threshold, deconv_flag, templates, min_spikes_local)


        # finish plotting 
        if plotting: 
            #ax_t = fig.add_subplot(grid[13:, 6:])
            for i in range(CONFIG.recordings.n_channels):
                ax_t.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), alpha=0.4, 
                                                                fontsize=30)
                # fill bewteen 2SUs on each channel
                ax_t.fill_between(CONFIG.geom[i,0] + np.arange(-spike_size//2,spike_size//2,1)/3.,
                    -scale + CONFIG.geom[i,1], scale + CONFIG.geom[i,1], 
                    color='black', alpha=0.05)
                    
                # plot original templates
                ax_t.plot(CONFIG.geom[:,0]+
                    np.arange(-template_original.shape[0]//2,
                               template_original.shape[0]//2,1)[:,np.newaxis]/3., 
                               CONFIG.geom[:,1] + template_original*scale, 
                               'r--', c='red')
                        
            # plot max chan with big red dot                
            ax_t.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], s = 2000, 
                                                    color = 'red')

            labels=[]
            if len(spike_index)>0: 
                sic_temp = np.concatenate(spike_index, axis = 0)
                assignment_temp = np.concatenate(assignment_global, axis = 0)
                idx = sic_temp[:,1] == unit
                clusters, sizes = np.unique(assignment_temp[idx], return_counts= True)
                clusters = clusters.astype(int)
                chans.extend(channel*np.ones(clusters.size))
                
                for i, clust in enumerate(clusters):
                    patch_j = mpatches.Patch(color = sorted_colors[clust%100], 
                                             label = "deconv = {}".format(sizes[i]))
                    labels.append(patch_j)

            patch_original = mpatches.Patch(color = 'red', label = 
                             "cluster in chunk/total: "+ 
                             str(spikes_in_chunk.shape[0])+"/"+
                             str(idx3.shape[0]))
            labels.append(patch_original)
            
            ax_t.legend(handles = labels, fontsize=30)

            # plto title
            fig.suptitle("Unit: "+str(unit), fontsize=25)
            fig.savefig(deconv_chunk_dir+"/unit{}.png".format(unit))
            plt.close(fig)


        # Cat: TODO: note clustering is done on PCA denoised waveforms but
        #            templates are computed on original raw signal

        # find original spikes in clustered chunk
        idx3 = np.where(spike_train_cluster_new[:,1]==unit)[0]
        spikes_in_chunk = np.where(np.logical_and(spike_train_cluster_new[idx3][:,0]>idx_chunk[0], 
                                                  spike_train_cluster_new[idx3][:,0]<=idx_chunk[1]))[0]
        # save original templates/spikes and reclustered
        np.savez(deconv_filename, 
                        spike_index_postrecluster=spike_index, 
                        templates_postrecluster=templates,
                        spike_index_cluster= spikes_in_chunk, 
                        templates_cluster=template_original)
                        
        print ("**** Unit ", str(unit), ", found # clusters: ", len(spike_index))
    
    # overwrite this variable just in case multiprocessing doesn't destroy it
    wf = None
    
    return channel


def get_wfs_from_residual(unit_sp, template, deconv_chunk_dir, 
                          n_times):
                                  
    """Gets clean spikes for a given unit."""
    
    # Note: residual contains buffers
    fname = deconv_chunk_dir+'/residual.npy'
    data = np.load(deconv_chunk_dir+'/residual.npy')

    # Add the spikes of the current unit back to the residual
    x = np.arange(-n_times//2,n_times//2,1)
    temp = data[x + unit_sp[:, :1], :] + template
    
    data = None
    return temp


def visible_chans(temps):
    a = temps.ptp(0) #np.max(temps, axis=0) - np.min(temps, 0)
    vis_chan = a > 1

    return vis_chan

        
def pairwise_filter_conv_local(deconv_chunk_dir, n_time, n_unit, temporal, 
                         singular, spatial, approx_rank, vis_chan, temps):
    
    #print (deconv_chunk_dir+"/parwise_conv.npy")
    if os.path.exists(deconv_chunk_dir+"/pairwise_conv.npy")==False:
        print ("IN LOOP")
        conv_res_len = n_time * 2 - 1
        pairwise_conv = np.zeros([n_unit, n_unit, conv_res_len])
        for unit1 in range(n_unit):
            u, s, vh = temporal[unit1], singular[unit1], spatial[unit1]
            vis_chan_idx = vis_chan[:, unit1]
            for unit2 in range(n_unit):
                for i in range(approx_rank):
                    pairwise_conv[unit2, unit1, :] += np.convolve(
                        np.matmul(temps[:, vis_chan_idx, unit2], vh[i, vis_chan_idx].T),
                        s[i] * u[:, i].flatten(), 'full')

        np.save(deconv_chunk_dir+"/pairwise_conv.npy", pairwise_conv)
    else:
        pairwise_conv = np.load(deconv_chunk_dir+"/pairwise_conv.npy")
        
    return pairwise_conv
    

def fix_spiketrains(up_up_map, spike_train):
    
    # assign unique spike train ids    
    spike_train_fixed = spike_train.copy()
    ctr=0
    #for k in np.unique(spike_train[:,1])[1:]:
    for k in np.arange(1,up_up_map.shape[0],1):

        idx = np.where(spike_train[:,1]==k)[0]
        
        # assign unique template id
        if up_up_map[k]==up_up_map[k-1]:
            spike_train_fixed[idx,1] = ctr
        else:
            ctr+=1
            spike_train_fixed[idx,1] = ctr
        
        #if k%1000==0: 
        print ("  ", k, up_up_map[k], idx.shape, ctr)

    return spike_train_fixed

    
