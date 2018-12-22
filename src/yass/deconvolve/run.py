import os
import logging
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm
import parmap
import scipy
import networkx as nx

from statsmodels import robust
from scipy.signal import argrelmin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from yass.deconvolve.match_pursuit import (MatchPursuit_objectiveUpsample, 
                                            Residual)
                                            
from yass.cluster.util import (binary_reader, RRR3_noregress_recovery_dynamic_features,
                               global_merge_max_dist, PCA, 
                               load_waveforms_from_memory,
                               make_CONFIG2, upsample_parallel, 
                               clean_templates, find_clean_templates)
from yass import read_config
from yass.cluster.cluster import Cluster

from diptest import diptest as dp

import multiprocessing as mp

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                 for name, color in colors.items())
sorted_colors = [name for hsv, name in by_hsv]


def run(spike_train_cluster,
        templates,
        output_directory='tmp/',
        recordings_filename='standardized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standardized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index_cluster.shape: {}'.format(templates.shape,
                                                 spike_train_cluster.shape))



    Deconv(spike_train_cluster,
            templates,
            output_directory,
            recordings_filename,
            CONFIG)

    # Note: new self.templates and self.spike_train is computed above
    # no need to return them to deconv
    spike_train = np.load(os.path.join(CONFIG.path_to_output_directory,
                    'spike_train_post_deconv_post_merge.npy'))

    templates = np.load(os.path.join(CONFIG.path_to_output_directory,
                    'templates_post_deconv_post_merge.npy'))

    return spike_train, templates

class Deconv(object):

    def __init__(self, spike_train, templates, output_directory, 
                 recordings_filename, CONFIG):

        self.spike_train = spike_train
        self.templates = templates.swapaxes(1,2).swapaxes(0,1)
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
        
        # compute segment list for parallel processing
        self.compute_idx_list()

        # Cat: TODO: read from CONFIG
        self.threshold = 10.
        self.conv_approx_rank = 10
        self.default_upsample_value=0
        self.upsample_max_val = 32.

        # Cat: TODO: max iteration not currently used
        self.max_iter = 5000
        
        # set the number of cluster iterations to be done
        # Cat: TODO: read from CONFIG
        self.n_iterations_merge = 1

        # iterative deconv parameters
        self.n_iterations_deconv = 0
        self.n_seconds_initial = 1200
        self.initial_chunk = int(self.n_seconds_initial//self.CONFIG.resources.n_sec_chunk) 

        # make deconv directory
        self.deconv_dir = os.path.join(self.CONFIG.path_to_output_directory,
                                  'deconv')
        if not os.path.isdir(self.deconv_dir):
            os.makedirs(self.deconv_dir)
            
        if self.n_iterations_deconv>0: 
            if not os.path.isdir(self.deconv_dir+'/initial/'):
                os.makedirs(self.deconv_dir+'/initial/')
       
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


    def deconv_iterative():
        print (" ITERATIVE DECONV NOT YET IMPLEMENTED...")
        idx_list_local = self.idx_list[:chunk_size]
        deconv_chunk_dir = os.path.join(self.CONFIG.path_to_output_directory,
                                        'deconv',
                                        'initial/')
        if not os.path.isdir(deconv_chunk_dir):
            os.makedirs(deconv_chunk_dir)
        
        # loop over iterations
        for k in range(self.n_iterations_deconv): 

            # run match pursuit and return templates and spike_trains
            (sparse_upsampled_templates, 
             dec_spike_train, 
             deconv_id_sparse_temp_map, 
             spike_train_cluster_prev_iteration) = self.match_pursuit_function()

            # run post-deconv merge; recomputes self.templates and self.spike_train
            self.merge()
            
            # OPTIONAL residual + reclustering steps
            # compute residual
            #self.compute_residual_function()

            # recluster 
            #self.templates, spike_train_cluster = reclustering_function()
                                                  # CONFIG,
                                                  # templates,
                                                  # deconv_chunk_dir,
                                                  # spike_train_cluster_prev_iteration,
                                                  # idx_list_local,
                                                  # initial_chunk,
                                                  # output_directory, 
                                                  # recordings_filename)


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
        
        # run post-deconv merge
        self.merge() 


    def merge(self): 
                            
        print ("Post-deconv merge...")
        print ("  computing residual to generate clean spikes")
        self.compute_residual_function()

        jitter=5
        upsample = 5
        
        for i in range(self.n_iterations_merge):
            print ("  running merge iteration: ", i+1 , "/", self.n_iterations_merge)
            print (self.spike_train.shape)
            tm = TemplateMerger(self.templates, self.spike_train, 
                                jitter, upsample,
                                self.CONFIG, iteration=i, recompute=False)
           
            merge_list = tm.get_merge_pairs()

            (self.spike_train, 
             self.templates,
             self.connected_components) = merge_pairs(self.templates, 
                                                 self.spike_train, 
                                                 merge_list, 
                                                 self.CONFIG)

            # save spike train
            np.savez(os.path.join(self.CONFIG.path_to_output_directory,
                            'deconv',
                            'results_post_deconv_post_merge_'+str(i)),
                     templates=self.templates,
                     spike_train=self.spike_train,
                     connected_components=self.connected_components)
                 
            print ("  deconv templates (post-merge): ", self.templates.shape)
    
            # Note: new self.templates and self.spike_train is computed above
            # no need to return them to deconv
            np.save(os.path.join(self.CONFIG.path_to_output_directory,
                            'spike_train_post_deconv_post_merge.npy'),
                    self.spike_train)
            np.save(os.path.join(self.CONFIG.path_to_output_directory,
                            'templates_post_deconv_post_merge.npy'),
                    self.templates)
                    
            
        #return spike_train, self.templates


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

        # collect segments not yet completed
        args_in = []
        for k in range(len(self.idx_list_local)):
            fname_out = (self.deconv_chunk_dir+
                         "/seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
            if os.path.exists(fname_out)==False:
                args_in.append([[self.idx_list_local[k], k],
                                self.buffer_size])

        if len(args_in)>0:
            if self.CONFIG.resources.multi_processing:
                p = mp.Pool(processes = self.CONFIG.resources.n_processors)
                p.map_async(mp_object.run, args_in).get(988895)
                p.close()
            else:
                for k in range(len(args_in)):
                    mp_object.run(args_in[k])
        
        # collect spikes
        res = []
        for k in range(len(self.idx_list_local)):
            fname_out = (self.deconv_chunk_dir+
                         "/seg_{}_deconv.npz".format(
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
        np.savez(self.deconv_chunk_dir + "/deconv_results.npz",
                 spike_train=self.spike_train,
                 templates=self.templates,
                 spike_train_upsampled=self.dec_spike_train,
                 templates_upsampled=self.sparse_upsampled_templates)

        np.save(os.path.join(self.CONFIG.path_to_output_directory,
                            'templates_post_deconv_pre_merge'),
                            self.templates)
                            
        np.save(os.path.join(self.CONFIG.path_to_output_directory,
                            'spike_train_post_deconv_pre_merge'), 
                            self.spike_train)
            
        #return (self.sparse_upsampled_templates, self.dec_spike_train,
        #        deconv_id_sparse_temp_map, spike_train_cluster_prev_iteration)


    def compute_residual_function(self):
                                  
        # Note: this uses spike times occuring at beginning of spike
        fname = os.path.join(self.CONFIG.path_to_output_directory, 
                                         'deconv',
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
        recording_chunk = binary_reader(idx_chunk, 
                                        self.buffer_size, 
                                        self.standardized_filename, 
                                        self.n_channels)

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
        wf_object.data = wf_object.data[self.buffer_size:-self.buffer_size].reshape(-1)
        wf_object.data.tofile(fname[:-4]+'.bin')


def merge_pairs(templates, spike_train, merge_list, CONFIG2):
    
    merge_matrix = np.zeros((templates.shape[2], templates.shape[2]),'int32')
    for merge_pair in merge_list:
        if merge_pair != None:
            merge_matrix[merge_pair[0],merge_pair[1]]=1

    # compute graph based on pairwise connectivity
    G = nx.from_numpy_matrix(merge_matrix)
    
    # merge spikes and templates
    print ("  merging units")
    final_spike_indexes = []
    templates_final = []
    ctr=0
    merge_array=[]
    for cc in nx.connected_components(G):
        # gather spikes 
        sic = np.zeros(0, dtype = int)
        weights=[]
        merge_array.append(list(cc))
        for j in cc:
            idx = np.where(spike_train[:,1]==j)[0]
            sic = np.concatenate([sic, spike_train[:,0][idx]])
            weights.append(idx.shape[0])
        temp = np.concatenate([sic[:,np.newaxis], ctr*np.ones([sic.size,1],dtype = 'int32')],axis = 1)
        final_spike_indexes.append(temp)
        
        # gather templates
        #temp_ = templates[:,:,list(cc)].mean(2)
        temp_ = np.average(templates[:,:,list(cc)], weights=weights,axis=2)
        templates_final.append(temp_)
        
        ctr+=1
    
    final_spike_indexes = np.vstack(final_spike_indexes)
    
    templates_final = np.array(templates_final).transpose(1,2,0)

    return (final_spike_indexes, templates_final, 
            merge_array)
    
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
    


def reclustering_function(CONFIG,
                          templates,
                          deconv_chunk_dir,
                          spike_train_cluster_new,
                          idx_list_local,
                          initial_chunk,
                          output_directory, 
                          recordings_filename):

    idx_chunk = [idx_list_local[0][0], idx_list_local[-1][1], 
                 idx_list_local[0][2], idx_list_local[0][3]]
    
    n_sec_chunk = CONFIG.resources.n_sec_chunk

    # make lists of arguments to be passed to 
    print ("  reclustering initial deconv chunk output...")
    CONFIG2 = make_CONFIG2(CONFIG)
    
    # load spike train and set train to beginning of chunk for indexing
    dec_spike_train_offset = np.load(deconv_chunk_dir+
                                        '/dec_spike_train_offset.npy')
    dec_spike_train_offset[:,0]-=idx_chunk[0]
    print ("  NO. UNIQUE SPIKE IDS: ", 
                        np.unique(dec_spike_train_offset[:,1]).shape)
   
    ''' ************************************************************
        ******************* READ RAW DATA CHUNK ********************
        ************************************************************
    '''
    # read recording chunk and share as global variable
    # Cat: TODO: recording_chunk should be a shared variable in 
    #            multiprocessing module;
    idx = idx_chunk
    data_start = idx[0]
    offset = idx[2]
    n_channels = CONFIG.recordings.n_channels
    buffer_size = 200
    standardized_filename = os.path.join(CONFIG.data.root_folder,
                                         output_directory,
                                         'preprocess', 
                                         recordings_filename)

    residual_clustering_flag = True
    if residual_clustering_flag:
        print ("  reclustering using residuals ")
    else:
        print ("  reclustering using raw data ")
        global recording_chunk
        recording_chunk = binary_reader(idx, 
                                        buffer_size, 
                                        standardized_filename, 
                                        n_channels)

    ''' ************************************************************
        ************** SETUP & RUN RECLUSTERING ********************
        ************************************************************
    '''

    # make argument list
    args_in = []

    # flag to indicate whether clustering original data or post-deconv
    deconv_flag = 'deconv'

    # index to be used for directory naming
    chunk_index = 0

    units = np.arange(templates.shape[2])
    for unit in units:
        fname_out = (deconv_chunk_dir+
                     "/recluster/unit_{}.npz".format(
                     str(unit).zfill(6)))
        if os.path.exists(fname_out)==False:
            args_in.append([deconv_flag,
                            unit,
                            idx_chunk,
                            chunk_index,
                            CONFIG2,
                            dec_spike_train_offset,
                            deconv_chunk_dir,
                            spike_train_cluster_new,
                            templates[:,:,unit]])

    # run residual-reclustering function
    if len(args_in)>0:
        if CONFIG.resources.multi_processing:
            p = mp.Pool(processes = CONFIG.resources.n_processors)
            p.map_async(Cluster, args_in).get(988895)
            p.close()
        else:
            for unit in range(len(args_in)):
                Cluster(args_in[unit])


    ''' ************************************************************
        ******* RECOMPUTE TEMPLATES USING NEW SPIKETRAINS  *********
        ************************************************************
    '''
    print ("  recomputing templates from raw data")
    recompute_templates_from_raw(templates,
                                 deconv_chunk_dir+"/recluster/",
                                 idx, 
                                 buffer_size, 
                                 standardized_filename, 
                                 n_channels,
                                 CONFIG2)


    ''' ************************************************************
        ******************* RUN TEMPLATE MERGE  ********************
        ************************************************************
    '''
    # run template merge
    print ("  merging templates")
    out_dir = 'deconv'
    raw_dir = deconv_chunk_dir+'/raw_templates/'
    spike_train, templates = global_merge_max_dist(raw_dir,
                                                   CONFIG2,
                                                   out_dir, 
                                                   units)

    templates = templates.swapaxes(0,2).swapaxes(0,1)
    np.savez(deconv_chunk_dir+"/deconv_results_post_recluster.npz", 
            spike_train=spike_train, 
            templates=templates)
    
    return templates, spike_train

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

    
def template_spike_dist(templates, spikes, jitter=0, upsample=1, vis_ptp=2., **kwargs):
    """compares the templates and spikes.

    parameters:
    -----------
    templates: numpy.array shape (K, C, T)
    spikes: numpy.array shape (M, C, T)
    jitter: int
        Align jitter amount between the templates and the spikes.
    upsample int
        Upsample rate of the templates and spikes.
    """
    # Only use the channels that the template has visibility.
    vis_chan = templates.ptp(-1).max(0) >= vis_ptp
    templates = templates[:, vis_chan, :]
    spikes = spikes[:, vis_chan, :]
    spikes_old = spikes.copy()

    # Upsample the templates
    if upsample > 1:
        n_t = templates.shape[-1]
        templates = scipy.signal.resample(templates, n_t * upsample, axis=-1)
        spikes = scipy.signal.resample(spikes, n_t * upsample, axis=-1)
        
    n_times = templates.shape[-1]
    n_chan = templates.shape[1]
    n_unit = templates.shape[0]
    n_spikes = spikes.shape[0]

    # Get a smaller window around the templates.
    if jitter > 0:
        jitter = jitter * upsample
        templates = templates[:, :, jitter // 2:-jitter // 2]
        idx = np.arange(n_times - jitter) + np.arange(jitter)[:, None]
        # indices: [spike number, channel, jitter, time]
        spikes = spikes[:, :, idx].transpose([0, 2, 1, 3]).reshape(
            [-1, n_chan, n_times - jitter])

    # Pairwise distance of templates and spikes
    dist = scipy.spatial.distance.cdist(
            templates.reshape([n_unit, -1]),
            spikes.reshape([n_spikes * max(jitter, 1), -1]),
            **kwargs)
            
    # Get best jitter distance
    if jitter > 0:
        return dist.reshape([n_unit, n_spikes, jitter]).min(axis=-1)
    return dist


def template_spike_dist_align(templates, spikes, jitter=0, upsample=1, vis_ptp=2., **kwargs):
    """compares the templates and spikes.

    parameters:
    -----------
    templates: numpy.array shape (K, C, T)
    spikes: numpy.array shape (M, C, T)
    jitter: int
        Align jitter amount between the templates and the spikes.
    upsample int
        Upsample rate of the templates and spikes.
    """

    from yass.cluster.cluster import align_get_shifts_with_ref, upsample_resample, shift_chans

    # Only use the channels that the template has visibility.
    vis_chan = templates.ptp(-1).max(0) >= vis_ptp
    templates = templates[:, vis_chan, :]
    spikes = spikes[:, vis_chan, :]

    print (templates.shape)
    max_chans = templates.ptp(2).argmax(1)
    print ("max chans: ", max_chans)

    waveforms = np.vstack((spikes[:500,max_chans[0]], 
                           spikes[500:,max_chans[1]]))
                           
    print (waveforms.shape)
    waveforms = np.vstack(templates[:,max_chans], waveforms)
    print (waveforms.shape)
    
    shifts = align_get_shifts_with_ref(waveforms, templates[0,max_chans[0]])
    print ("sfhits: ", shifts)
    #best_shifts = align_get_shifts_with_ref()
    #self.spt_global -= best_shifts
    waveforms = shift_chans(waveforms, shifts)

    templates = waveforms[:2]
    spikes = waveforms[2:]
    
    # # Upsample the templates
    # if upsample > 1:
        # n_t = templates.shape[-1]
        # templates = scipy.signal.resample(templates, n_t * upsample, axis=-1)
        # spikes = scipy.signal.resample(spikes, n_t * upsample, axis=-1)

    # n_times = templates.shape[-1]
    # n_chan = templates.shape[1]
    # n_unit = templates.shape[0]
    # n_spikes = spikes.shape[0]

    # # Get a smaller window around the templates.
    # if jitter > 0:
        # jitter = jitter * upsample
        # templates = templates[:, :, jitter // 2:-jitter // 2]
        # idx = np.arange(n_times - jitter) + np.arange(jitter)[:, None]
        # # indices: [spike number, channel, jitter, time]
        # spikes = spikes[:, :, idx].transpose([0, 2, 1, 3]).reshape(
            # [-1, n_chan, n_times - jitter])

    # templates = [2 , 49,  61+/- clipping]
    # spikes = [n_spikes, 49, 61 +/- clipping]
    # then flatten over 2nd and 3rd dimensions

    # Pairwise distance of templates and spikes
    dist = scipy.spatial.distance.cdist(
            templates.reshape([n_unit, -1]),
            spikes.reshape([n_spikes * max(jitter, 1), -1]),
            **kwargs)

    # Get best jitter distance
    if jitter > 0:
        return dist.reshape([n_unit, n_spikes, jitter]).min(axis=-1)
    return dist

class TemplateMerger(object):

    def __init__(self, templates, spike_train, jitter, upsample, CONFIG,
                 iteration=0, recompute=False):
        """
        parameters:
        -----------
        templates: numpy.ndarray shape (K, C, T)
            templates
        spike_train: numpy.ndarray shape (N, 2)
            First column is times and second is cluster id.
        align_jitter: int
            Number of jitter
        upsample: int
            upsample integer
        """
                
        # set filename to residual file
        self.filename = os.path.join(CONFIG.path_to_output_directory, 
                                     'deconv',
                                     'residual.bin')
        self.CONFIG = CONFIG
        self.iteration = iteration
        self.jitter, self.upsample = jitter, upsample

        print (" templates in: ", templates.shape)
        self.templates = templates.transpose(2,1,0)
        self.recompute=recompute
        self.spike_train = spike_train
        self.n_unit = self.templates.shape[0]
        
        # TODO find pairs that for proposing merges
        # get temp vs. temp affinity, use for merge proposals
        fname = os.path.join(CONFIG.path_to_output_directory, 
                             'deconv', 
                             'affinity_'+str(self.iteration)+'.npy')
                             
        if os.path.exists(fname)==False or self.recompute:
            print ("  computing affinity matrix") 
            self.affinity_matrix = template_spike_dist(
                self.templates, self.templates,
                jitter=self.jitter, upsample=self.upsample)
            np.save(fname, self.affinity_matrix)
        else:
            self.affinity_matrix = np.load(fname)
            
        
        # Template norms
        self.norms = template_spike_dist(
            self.templates, self.templates[:1, :, :] * 0,
            jitter=self.jitter, upsample=self.upsample, vis_ptp=0.).flatten()
        
        # Distance metric with diagonal set to large numbers
        dist_mat = np.zeros_like(self.affinity_matrix)
        for i in range(self.n_unit):
            dist_mat[i, i] = 1e4
        dist_mat += self.affinity_matrix
        
        # proposed merge pairs
        self.merge_candidate = scipy.optimize.linear_sum_assignment(dist_mat)[1]
        
        fname = os.path.join(CONFIG.path_to_output_directory, 
                             'deconv', 
                             'merge_candidate_'+str(self.iteration)+'.npy')
        np.save(fname, self.merge_candidate)
        
        # Check the ratio of distance between proposed pairs compared to
        # their norms, if the value is greater that .3 then test the unimodality
        # test on the distribution of distinces of residuals.
        self.dist_norm_ratio = dist_mat[
            range(self.n_unit),
            self.merge_candidate] / (self.norms[self.merge_candidate] + self.norms)

        fname = os.path.join(CONFIG.path_to_output_directory, 
                             'deconv', 
                             'dist_norm_ratio_'+str(self.iteration)+'.npy')
        np.save(fname, self.dist_norm_ratio)
        
    def get_merge_pairs(self):
        
        units = np.where(self.dist_norm_ratio < 0.5)[0]
        
        fname = os.path.join(self.CONFIG.path_to_output_directory, 
                             'deconv',
                             'merge_list_'+str(self.iteration)+'.npy')
                             
        if os.path.exists(fname)==False or self.recompute:
            if self.CONFIG.resources.multi_processing:
                merge_list = parmap.map(self.merge_templates_parallel, 
                             list(zip(units, self.merge_candidate[units])),
                             processes=self.CONFIG.resources.n_processors,
                             pm_pbar=True)
            # single core version
            else:
                merge_list = []
                for unit in units:
                    temp = self.merge_templates_parallel(
                                        [unit, self.merge_candidate[unit]])
                    merge_list.append(temp)
            np.save(fname, merge_list)
        else:
            merge_list = np.load(fname)
            
        return merge_list

    def merge_templates_parallel(self, data_in, threshold=0.9, n_samples=1000):
        """Whether to merge two templates or not.

        parameters:
        -----------
        unit1: int
            Index of the first template
        unit2: int
            Index of the second template
        threshold: float
            The threshold for the unimodality test.
        n_samples: int
            Maximum number of cleaned spikes from each unit.

        returns:
        --------
        Bool. If True, the two templates should be merged. False, otherwise.
        """
        unit1 = data_in[0]
        unit2 = data_in[1]
        
        fname = os.path.join(self.CONFIG.path_to_output_directory, 
                    'deconv', 
                    'l2features_'+str(unit1)+'_'+str(unit2)+'_'+str(self.iteration))
                    
        if os.path.exists(fname)==False or self.recompute:
        
            # get n_sample of cleaned spikes per template.
            spt1_idx = np.where(self.spike_train[:, 1] == unit1)[0][:n_samples]
            spt2_idx = np.where(self.spike_train[:, 1] == unit2)[0][:n_samples]
            spt1 = self.spike_train[spt1_idx, :]
            spt2 = self.spike_train[spt2_idx, :]

            # TODO(Cat): this filename and Config somehow
            spikes_1 = read_spikes(
                self.filename, unit1, self.templates.transpose([2, 1, 0]), spt1,
                self.CONFIG, residual_flag=True).transpose([0, 2, 1])
            spikes_2 = read_spikes(
                self.filename, unit2, self.templates.transpose([2, 1, 0]), spt2,
                self.CONFIG, residual_flag=True).transpose([0, 2, 1])
            spike_ids = np.append(
                np.ones(len(spikes_1)), np.zeros(len(spikes_2)), axis=0)
            l2_features = template_spike_dist(
                self.templates[[unit1, unit2], :, :],
                np.append(spikes_1, spikes_2, axis=0),
                jitter=self.jitter, upsample=self.upsample)
                        
            dp_val = test_unimodality(np.log(l2_features).T, spike_ids)
            #print (" units: ", unit1, unit2, " dp_val: ", dp_val)
            # save data
            np.savez(fname,
                     #spt1_idx=spt1_idx, spt2_idx=spt2_idx,
                     #spikes_1=spikes_1, spikes_2=spikes_2,
                     spike_ids=spike_ids,
                     l2_features=l2_features,
                     dp_val=dp_val)

        else:
            data = np.load(fname)
            dp_val = data['dp_val']
        
        if dp_val > threshold:
            return (unit1, unit2)

        return None

        
def binary_reader_waveforms(filename, n_channels, n_times, spikes, channels=None, data_type='float32'):
    ''' Reader for loading raw binaries
    
        standardized_filename:  name of file contianing the raw binary
        n_channels:  number of channels in the raw binary recording 
        n_times:  length of waveform 
        spikes: 1D array containing spike times in sample rate of raw data
        channels: load specific channels only
        data_type: float32 for standardized data
    
    '''

    # ***** LOAD RAW RECORDING *****
    if channels is None:
        wfs = np.zeros((spikes.shape[0], n_times, n_channels), data_type)
    else:
        wfs = np.zeros((spikes.shape[0], n_times, channels.shape[0]), data_type)

    with open(filename, "rb") as fin:
        for ctr,s in enumerate(spikes):
            # index into binary file: time steps * 4  4byte floats * n_channels
            fin.seek(s * 4 * n_channels, os.SEEK_SET)
            wfs[ctr] = np.fromfile(
                fin,
                dtype='float32',
                count=(n_times * n_channels)).reshape(n_times, n_channels)[:,channels]

    fin.close()
    return wfs

def read_spikes(filename, unit, templates, spike_train, CONFIG, residual_flag=False):
    ''' Function to read spikes from raw binaries
        
        filename: name of raw binary to be loaded
        unit: template # to be loaded
        templates:  [n_times, n_chans, n_templates] array holding all templates
        spike_train:  [times, ids] array holding all spike times
    '''

    # load spikes for particular unit
    spikes = spike_train[spike_train[:,1]==unit,0]

    # set load parameters
    n_channels = CONFIG.recordings.n_channels
    spike_size = int(CONFIG.recordings.spike_size_ms*CONFIG.recordings.sampling_rate//1000*2+1)
    channels = np.arange(CONFIG.recordings.n_channels)

    spike_waveforms = binary_reader_waveforms(filename,
                                         n_channels,
                                         spike_size,
                                         spikes, #- spike_size//2,  # can use this for centering
                                         channels)

    # if loading residual need to add template back into 
    if residual_flag:
        spike_waveforms+=templates[:,:,unit]

    return spike_waveforms

def test_unimodality(pca_wf, assignment, max_spikes = 10000):

    '''
    Parameters
    ----------
    pca_wf:  pca projected data
    assignment:  spike assignments
    max_spikes: optional
    '''

    n_samples = np.max(np.unique(assignment, return_counts=True)[1])

    # compute diptest metric on current assignment+LDA

    
    ## find indexes of data
    idx1 = np.where(assignment==0)[0]
    idx2 = np.where(assignment==1)[0]
    min_spikes = min(idx1.shape, idx2.shape)[0]

    # limit size difference between clusters to maximum of 5 times
    ratio = 1
    idx1=idx1[:min_spikes*ratio][:max_spikes]
    idx2=idx2[:min_spikes*ratio][:max_spikes]

    idx_total = np.concatenate((idx1,idx2))

    ## run LDA on remaining data
    lda = LDA(n_components = 1)
    #print (pca_wf[idx_total].shape, assignment[idx_total].shape) 
    trans = lda.fit_transform(pca_wf[idx_total], assignment[idx_total])
    diptest = dp(trans.ravel())

    ## also compute gaussanity of distributions
    ## first pick the number of bins; this metric is somewhat sensitive to this
    # Cat: TODO number of bins is dynamically set; need to work on this
    #n_bins = int(np.log(n_samples)*3)
    #y1 = np.histogram(trans, bins = n_bins)
    #normtest = stats.normaltest(y1[0])

    return diptest[1] #, normtest[1]
