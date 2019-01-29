import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm
import parmap
import scipy
import networkx as nx
import logging
import multiprocessing as mp

from diptest import diptest as dp

from yass import read_config

from statsmodels import robust
from scipy.signal import argrelmin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#from yass.cluster.util import read_spikes

class TemplateMerge(object):

    def __init__(self, templates, spike_train, jitter, upsample, CONFIG,
                 deconv_chunk_dir,
                 iteration=0, recompute=False, affinity_only=False):
                     
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
        
        # 
        self.deconv_chunk_dir = deconv_chunk_dir
        
        # set filename to residual file
        self.filename_residual = os.path.join(deconv_chunk_dir.replace('merge',''), 
                                     'residual.bin')
        self.CONFIG = CONFIG
        self.iteration = iteration
        self.jitter, self.upsample = jitter, upsample

        self.templates = templates.transpose(2,0,1)
        print ("  templates in: ", templates.shape)
        self.recompute=recompute
        self.spike_train = spike_train
        self.n_unit = self.templates.shape[0]
        
        # TODO find pairs that for proposing merges
        # get temp vs. temp affinity, use for merge proposals
        fname = os.path.join(deconv_chunk_dir, 
                             'affinity_'+str(self.iteration)+'.npy')
                             
        if os.path.exists(fname)==False or self.recompute:
            print ("  computing affinity matrix...") 
            self.affinity_matrix = template_spike_dist_linear_align(
                self.templates, self.templates,
                jitter=self.jitter, upsample=self.upsample)
            print ("  done computing affinity matrix")
            np.save(fname, self.affinity_matrix)
        else:
            print ("  affinity matrix prev computed...loading")
            self.affinity_matrix = np.load(fname)
        
        if affinity_only:
            return 
                
        # Template norms
        self.norms = template_spike_dist_linear_align(
            self.templates, self.templates[:1, :, :] * 0,
            jitter=self.jitter, upsample=self.upsample, vis_ptp=0.).flatten()

        # Distance metric with diagonal set to large numbers
        dist_mat = np.zeros_like(self.affinity_matrix)
        for i in range(self.n_unit):
            dist_mat[i, i] = 1e4
        dist_mat += self.affinity_matrix
        
        # proposed merge pairs
        #print ("  computing hungarian alg assignment...")
        fname = os.path.join(deconv_chunk_dir,
                             'merge_candidate_'+str(self.iteration)+'.npy')

        if os.path.exists(fname)==False:
            print ( "  TODO: write faster merge algorithm...")
            
            #self.merge_candidate = scipy.optimize.linear_sum_assignment(dist_mat)[1]
            #self.merge_candidate = self.find_merge_candidates()
            self.merge_candidate = np.argsort(dist_mat, axis=1)[:,0]

            np.save(fname, self.merge_candidate)
        else:
            self.merge_candidate = np.load(fname)
       
        
        # Check the ratio of distance between proposed pairs compared to
        # their norms, if the value is greater that .3 then test the unimodality
        # test on the distribution of distinces of residuals.
        self.dist_norm_ratio = dist_mat[
            range(self.n_unit),
            self.merge_candidate] / (self.norms[self.merge_candidate] + self.norms)

        fname = os.path.join(deconv_chunk_dir,
                             'dist_norm_ratio_'+str(self.iteration)+'.npy')
        np.save(fname, self.dist_norm_ratio)

    def find_merge_candidates(self):
        print ( "  TODO: write serial merging algorithm...")
        
        pass


    def get_merge_pairs(self):
        ''' Run all pairs of merge candidates through the l2 feature computation        
        '''
        
        units = np.where(self.dist_norm_ratio < 0.5)[0]
        
        fname = os.path.join(self.deconv_chunk_dir,
                             'merge_list_'+str(self.iteration)+'.npy')
                             
        print ("  computing l2features and distances in parallel")
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


    def get_merge_pairs_units(self):
        ''' Check all units against nearest units in increasing distance.
            This algorithm does not use hugnarian algorithm to assign 
            candidates; rather, check until diptests start to become too low
        '''
        
        units = np.where(self.dist_norm_ratio < 0.5)[0]
        
        fname = os.path.join(self.deconv_chunk_dir,
                             'merge_list_'+str(self.iteration)+'.npy')
                             
        print ("  computing l2features and distances in parallel")
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




    def merge_templates_parallel(self, data_in, threshold=0.99, n_samples=1000):
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
        
        fname = os.path.join(self.deconv_chunk_dir, 
                    'l2features_'+str(unit1)+'_'+str(unit2)+'_'+str(self.iteration)+'.npz')
                    
        if os.path.exists(fname)==False or self.recompute:
        
            # get n_sample of cleaned spikes per template.
            spt1_idx = np.where(self.spike_train[:, 1] == unit1)[0][:n_samples]
            spt2_idx = np.where(self.spike_train[:, 1] == unit2)[0][:n_samples]
            spt1 = self.spike_train[spt1_idx, :]
            spt2 = self.spike_train[spt2_idx, :]

            # TODO(Cat): this filename and Config somehow
            spikes_1 = read_spikes(
                self.filename_residual, unit1, self.templates, spt1,
                self.CONFIG, residual_flag=True)
            spikes_2 = read_spikes(
                self.filename_residual, unit2, self.templates, spt2,
                self.CONFIG, residual_flag=True)
            spike_ids = np.append(
                np.ones(len(spikes_1)), np.zeros(len(spikes_2)), axis=0)
            l2_features = template_spike_dist_linear_align(
                self.templates[[unit1, unit2], :, :],
                np.append(spikes_1, spikes_2, axis=0),
                jitter=self.jitter, upsample=self.upsample)
                        
            dp_val, _, _ = test_unimodality(np.log(l2_features).T, spike_ids)
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

        
    def merge_templates_parallel_units(self):
        
        pass


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

    return diptest[1], trans.ravel(), assignment[idx_total]#, normtest[1]


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

    print ("  upsampling")
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

    print ("  cdist computation")
    # Pairwise distance of templates and spikes
    dist = scipy.spatial.distance.cdist(
            templates.reshape([n_unit, -1]),
            spikes.reshape([n_spikes * max(jitter, 1), -1]),
            **kwargs)
    print ("  done cdist computation")

    # Get best jitter distance
    if jitter > 0:
        return dist.reshape([n_unit, n_spikes, jitter]).min(axis=-1)
    return dist


def template_spike_dist_linear_align(templates, spikes, jitter=0, upsample=1, vis_ptp=2., **kwargs):
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

    from yass.cluster.cluster import align_get_shifts_with_ref, upsample_resample, shift_chans

    #print ("templates: ", templates.shape)
    #print ("spikes: ", spikes.shape)
    # new way using alignment only on max channel
    # maek reference template based on templates
    max_idx = templates.ptp(1).max(1).argmax(0)
    ref_template = templates[max_idx]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]

    # stack template max chan waveforms only
    #max_chans = templates.ptp(2).argmax(1)
    #temps = []
    #for k in range(max_chans.shape[0]):
    #    temps.append(templates[k,max_chans[k]])
    #temps = np.vstack(temps)
    temps = templates[:, :, max_chan]
    #print ("tempsl stacked: ", temps.shape)
    
    #upsample_factor=5
    best_shifts = align_get_shifts_with_ref(
                    temps, ref_template, upsample)
    #print (" best shifts: ", best_shifts.shape)
    templates_aligned = shift_chans(templates, best_shifts)
    #print ("  new aligned templates: ", templates_aligned.shape)

    # find spike shifts
    #max_chans = spikes.ptp(2).argmax(1)
    #print ("max chans: ", max_chans.shape)
    #spikes_aligned = []
    #for k in range(max_chans.shape[0]):
    #    spikes_aligned.append(spikes[k,max_chans[k]])
    #spikes_aligned = np.vstack(spikes_aligned)
    #print ("spikes aligned max chan: ", spikes_aligned.shape)
    spikes_aligned = spikes[:,:,max_chan]
    best_shifts = align_get_shifts_with_ref(
                            spikes_aligned, ref_template, upsample)
    
    spikes_aligned = shift_chans(spikes, best_shifts)
    #print ("  new aligned spikes: ", spikes_aligned.shape)
    
    n_unit = templates_aligned.shape[0]
    n_spikes = spikes_aligned.shape[0]

    vis_chan = templates.ptp(1).max(0) >= vis_ptp
    templates = templates[:, :, vis_chan]
    spikes = spikes[:, :, vis_chan]

    #print ("  start cdist computation")
    # Pairwise distance of templates and spikes
    dist = scipy.spatial.distance.cdist(
           templates_aligned.reshape([n_unit, -1]),
           spikes_aligned.reshape([n_spikes, -1]))

    return dist
    


def binary_reader_waveforms(standardized_filename, n_channels, n_times, spikes, channels=None):

    # ***** LOAD RAW RECORDING *****
    if channels is None:
        wfs = np.zeros((spikes.shape[0], n_times, n_channels), 'float32')
    else:
        wfs = np.zeros((spikes.shape[0], n_times, channels.shape[0]), 'float32')

    skipped_idx = []
    with open(standardized_filename, "rb") as fin:
        ctr_wfs=0
        ctr_skipped=0
        for spike in spikes:
            # index into binary file: time steps * 4  4byte floats * n_channels
            fin.seek(spike * 4 * n_channels, os.SEEK_SET)
            try:
                wfs[ctr_wfs] = np.fromfile(
                    fin,
                    dtype='float32',
                    count=(n_times * n_channels)).reshape(
                                            n_times, n_channels)[:,channels]
                ctr_wfs+=1
            except:
                # skip loading of spike and decrease wfs array size by 1
                #print ("  spike to close to end, skipping and deleting array")
                wfs=np.delete(wfs, wfs.shape[0]-1,axis=0)
                skipped_idx.append(ctr_skipped)

            ctr_skipped+=1
    fin.close()
    
    return wfs, skipped_idx

    
def read_spikes(filename, unit, templates, spike_train, CONFIG, 
                channels=None, residual_flag=False, spike_size=None):
    ''' Function to read spikes from raw binaries
        
        filename: name of raw binary to be loaded
        unit: template # to be loaded
        templates:  [n_times, n_chans, n_templates] array holding all templates
        spike_train:  [times, ids] array holding all spike times
    '''

    # load spikes for particular unit
    if len(spike_train.shape)>1:
        spikes = spike_train[spike_train[:,1]==unit,0]
    else:
        spikes = spike_train
        
    # always load all channels and then index into subset otherwise
    # order won't be correct
    n_channels = CONFIG.recordings.n_channels
    
    # load default spike_size unless otherwise inidcated
    if spike_size==None:
        spike_size = int(CONFIG.recordings.spike_size_ms*CONFIG.recordings.sampling_rate//1000*2+1)

    if channels == None:
        channels = np.arange(n_channels)

    spike_waveforms, skipped_idx = binary_reader_waveforms(filename,
                                             n_channels,
                                             spike_size,
                                             spikes, #- spike_size//2,  # can use this for centering
                                             channels)

    # if loading residual need to add template back into 
    # Cat: TODO: this is bit messy; loading extrawide noise, but only adding
    #           narrower templates
    if residual_flag:
        if spike_size is None:
            spike_waveforms+=templates[unit, :, channels]
        # need to add templates in middle of noise wfs which are wider
        else:
            spike_size_default = int(CONFIG.recordings.spike_size_ms*
                                      CONFIG.recordings.sampling_rate//1000*2+1)
            offset = spike_size - spike_size_default
            spike_waveforms[:,offset//2:offset//2+spike_size_default]+=templates[unit][:, channels]
        
    return spike_waveforms #, skipped_idx
