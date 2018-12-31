# Class to do parallelized clustering

import os
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy import signal
from scipy import stats
from scipy.signal import argrelmax
from scipy.spatial import cKDTree
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from diptest import diptest as dp
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

from yass.explore.explorers import RecordingExplorer
from yass.geometry import n_steps_neigh_channels
from yass import mfm
from yass.util import absolute_path_to_asset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

colors = np.array([
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
'darkmagenta','yellow','hotpink'])

sorted_colors=colors


class Cluster(object):
    """Class for doing clustering."""

    def __init__(self, data_in):
            
        """Sets up the cluster class for each core
        Parameters: ...
              
        """
        
        # load data and check if prev completed
        if self.load_data(data_in):  return

        # local channel clustering
        if self.deconv_flag: 
            print("\nchan "+str(self.channel)+", unit: "+str(self.unit) +", START LOCAL CLUSTERING")
        else:
            print("\nchan "+str(self.channel)+", START LOCAL CLUSTERING")

        self.initialize(initial_spt=self.spike_indexes_chunk[:, 0], local=True)
        self.cluster(current_indices=self.starting_indices, gen=0, local=True)
        self.finish(fname='channel_{}'.format(self.channel))

        # distant channel clustering
        spike_train_local = np.copy(self.spike_train)
        spike_train_final = []
        templates_final = []
        for ii, spike_train_k in enumerate(spike_train_local):
            print("\nchan {}, START CLUSTERING UNIT {}/{}".format(self.channel, ii, len(spike_train_local)))

            self.initialize(initial_spt=spike_train_k, local=False)
            self.cluster(current_indices=self.starting_indices, gen=0, local=False)
            self.finish(fname='channel_{}_local_unit_{}'.format(self.channel, ii))
            
            spike_train_final += self.spike_train
            templates_final += self.templates

        # save clusters
        self.save_result(spike_train_final, templates_final)


    def cluster(self, current_indices, gen, local):

        ''' Recursive clustering function
            channel: current channel being clusterd
            wf = wf_PCA: denoised waveforms (# spikes, # time points, # chans)
            sic = spike_indices of spikes on current channel
            gen = generation of cluster; increases with each clustering step        
        '''

        # Exit if cluster too small
        if current_indices.shape[0] <= self.CONFIG.cluster.min_spikes: return
        
        if self.verbose:
            print("chan "+str(self.channel)+', gen '+str(gen)+', # spikes: '+ str(current_indices.shape[0]))

        # generation 0 steps
        if gen==0:
            # load waveforms for channel based clustering only
            #if self.deconv_flag==False:
            self.load_waveforms(local)
            # align waveforms
            self.align_step(local)
            # denoise waveforms on active channels
            self.denoise_step(local)

        #np.save('/media/cat/500GB/liam/49channels/data1_set1_5mins/tmp/deconv/initial/0/recluster/wf_global_'+str(local)+
        #        '_'+str(self.unit)+'.npy', self.wf_global)

        # featurize it
        save_pca=True
        if len(self.skipped_idx)>0:
            current_indices = np.delete(current_indices, self.skipped_idx, axis=0)
            
        pca_wf = self.featurize_step(gen, current_indices, local, save_pca)
        
        # knn triage
        idx_keep = self.knn_triage_step(gen, pca_wf)
        if idx_keep.shape[0] <= self.CONFIG.cluster.min_spikes: return

        # if anything is triaged, re-featurize and re-cluster
        if idx_keep.shape[0] < pca_wf.shape[0]:
            current_indices = current_indices[idx_keep]
            save_pca=False
            pca_wf = self.featurize_step(gen, current_indices, local, save_pca)

        # run initial cluster step
        vbParam = self.run_mfm(gen, self.subsample_step(gen, pca_wf))

        ##### TRIAGE 1 #####
        # adaptive knn triage
        #idx_keep = self.knn_triage_dynamic(gen, vbParam, pca_wf)
        #if idx_keep.shape[0] <= self.CONFIG.cluster.min_spikes: return

        # if anything is triaged, re-featurize and re-cluster
        #if idx_keep.shape[0] < pca_wf.shape[0]:
        #    current_indices = current_indices[idx_keep]
        #    pca_wf = self.featurize_step(gen, current_indices)
        #    vbParam = self.run_mfm(gen, self.subsample_step(gen, pca_wf))

        ##### TRIAGE 2 #####
        # if we subsampled then recover soft-assignments using above:
        idx_recovered, vbParam = self.recover_step(gen, vbParam, pca_wf)
        if idx_recovered.shape[0] <= self.CONFIG.cluster.min_spikes: return
        
        # if anything is triaged further, update the info
        if idx_recovered.shape[0] < pca_wf.shape[0]:
            current_indices = current_indices[idx_recovered]
            pca_wf = pca_wf[idx_recovered]
            
        ##### TRIAGE 3 #####
        # kill any units with less than min_spikes
        idx_survived, vbParam = self.kill_small_units(gen, vbParam)
        if idx_survived.shape[0] <= self.CONFIG.cluster.min_spikes: return
        
        # if anything is triaged further, update the info
        if idx_survived.shape[0] < pca_wf.shape[0]:
            current_indices = current_indices[idx_survived]
            pca_wf = pca_wf[idx_survived]
            
        '''*************************************************        
           *********** REVIEW AND SAVE RESULTS *************
           *************************************************        
        '''
        # Case #1: single mfm cluster found
        if vbParam.rhat.shape[1] == 1:
            self.single_cluster_step(gen, current_indices, pca_wf)

        # Case #2: multiple clusters
        else:
            # this is outside of multi_cluster_step to make 
            # pca_wf = None before going to the next generation
            cc_assignment, stability = self.cluster_annealing(vbParam)
            if self.plotting and gen<20:
                self.plot_clustering_scatter(gen, pca_wf, cc_assignment,
                    stability, 'mfm multi split')
            pca_wf = None

            self.multi_cluster_step(gen, current_indices, cc_assignment, local)
                                                    
    def load_data(self, data_in):
        ''''''

        ''' *******************************************
            ************ LOADED PARAMETERS ************
            *******************************************
        '''

        # this indicates channel-wise clustering - NOT postdeconv recluster
        self.deconv_flag = data_in[0]
        self.channel = data_in[1]
        self.CONFIG = data_in[2]

        # spikes in the current chunk
        self.spike_indexes_chunk = data_in[3]
        self.chunk_dir = data_in[4]

        # Check if channel alreedy clustered
        self.filename_postclustering = (self.chunk_dir + "/channel_"+
                                            str(self.channel).zfill(6)+".npz")

        # additional parameters if doing deconv:
        if self.deconv_flag:
            self.unit = self.channel.copy()
            print ("  reclustering unit: ", self.unit)

            #self.spike_train_cluster_original = data_in[5]
            self.templates_deconv = data_in[5]
            self.template_original = self.templates_deconv[:,:,self.unit]

            # keep track of this for possible debugging later
            self.spike_train_cluster_original = self.spike_indexes_chunk
            
            # reset channel for unit to it's ptp channel
            self.max_chan = self.template_original.ptp(0).argmax(0)
            self.channel = self.max_chan

            self.deconv_max_spikes = 3000
            self.filename_postclustering = (self.chunk_dir + "/unit_"+
                                                        str(self.unit).zfill(6)+".npz")

            self.filename_residual = os.path.join(self.chunk_dir.replace(
                                                    'recluster',''),
                                                  "residual.bin")

        if os.path.exists(self.filename_postclustering):
            return True

        # check to see if 'result/' folder exists otherwise make it
        self.figures_dir = self.chunk_dir+'/figures/'
        if not os.path.isdir(self.figures_dir):
            os.makedirs(self.figures_dir)
                
        ''' ********************************************
            *********** DEFAULT PARAMETERS *************
            ******************************************** 
        '''
        # default parameters
        self.n_channels = self.CONFIG.recordings.n_channels
        self.min_spikes_local = self.CONFIG.cluster.min_spikes
        self.standardized_filename = os.path.join(self.CONFIG.path_to_output_directory, 'preprocess', 'standardized.bin')
        
        self.geometry_file = os.path.join(self.CONFIG.data.root_folder,
                                          self.CONFIG.data.geometry)

        # CAT: todo read params below from file:
        self.plotting = False
        self.verbose = False
        self.starting_gen = 0
        self.knn_triage_threshold = 0.95 * 100
        self.knn_triage_flag = True
        self.selected_PCA_rank = 5
        self.yscale = 10.
        self.xscale = 2.
        self.triageflag = True
        self.n_feat_chans = 5
        self.mfm_threshold = 0.90
        self.upsample_factor = 5
        self.nshifts = 15
        self.n_dim_pca = 3
        self.n_dim_pca_compression = 5
        self.shift_allowance = 10
        self.spike_size_padded = 81
        self.max_cluster_spikes = 50000
        
        # limit on featurization window;
        # Cat: TODO this needs to be further set using window based on spike_size and smapling rate
        self.spike_size = int(self.CONFIG.recordings.spike_size_ms*2
                              *self.CONFIG.recordings.sampling_rate/1000
                              + self.shift_allowance*2)+1

        # array to hold shifts; need to initialize 
        self.global_shifts=None
        self.pca_wf_gen0=None

        # load subsampled spikes 
        self.load_spikes()

        # return flag that clustering not yet complete
        return False


    def load_spikes(self):

        # Cat: TODO: Is this index search expensive for hundreds of chans and many
        #       millions of spikes?  Might want to do once rather than repeat
        indexes = np.where(self.spike_indexes_chunk[:,1]==self.channel)[0]

        # limit clustering to at most 50,000 spikes
        # Cat: TODO: both flag and value should be read from CONFIG
        if self.deconv_flag==False:
            if indexes.shape[0]>self.max_cluster_spikes:
                idx = np.random.choice(np.arange(indexes.shape[0]),
                                                  size=self.max_cluster_spikes,
                                                  replace=False)
                indexes = indexes[idx]
        
        # reclustering only done on 5k spieks max
        else:
            if indexes.shape[0]>self.max_cluster_spikes//10:
                idx = np.random.choice(np.arange(indexes.shape[0]),
                                                  size=self.max_cluster_spikes//10,
                                                  replace=False)
                indexes = indexes[idx]


        # check that spkes times not too lcose to edges:
        # first determine length of processing chunk based on lenght of rec
        #  i.e. divide by 4 byte float and # of chans
        # Cat: TODO FIX/CHECK THIS;
        fp_len = int(os.path.getsize(self.standardized_filename)/
                     4/self.CONFIG.recordings.n_channels)

        # limit indexes away from edge of recording
        idx_inbounds = np.where(np.logical_and(
                        self.spike_indexes_chunk[indexes,0]>=self.spike_size//2,
                        self.spike_indexes_chunk[indexes,0]<(fp_len-self.spike_size)))[0]
        indexes = indexes[idx_inbounds]

        # check to see if any duplicate spike times occur
        if np.unique(indexes).shape[0] != indexes.shape[0]:
            print ("   >>>>>>>>>>> DUPLICATE SPIKE TIMES <<<<<<<<<<<<<<<<<<<<")
        indexes = np.unique(indexes)

        # limit spike indexes to only subsampled and inbound spikse
        self.spike_indexes_chunk = self.spike_indexes_chunk[indexes]


    def initialize(self, initial_spt, local):

        self.spike_train = []
        self.templates = []

        # save detected spike times for channel (redundant step)
        if local:
            self.spiketime_detect = initial_spt.copy()
            
        self.spt_global = initial_spt.astype('float64')

        self.starting_indices = np.arange(len(self.spt_global))

        self.initialize_template_space()

        if self.plotting:
            self.x = np.zeros(100, dtype = int)
            self.fig1 = plt.figure(figsize =(60,60))
            self.grid1 = plt.GridSpec(20,20,wspace = 1,hspace = 2)
            

            # setup template plot; scale based on electrode array layout
            xlim = self.CONFIG.geom[:,0].ptp(0)
            ylim = self.CONFIG.geom[:,1].ptp(0)#/float(xlim)
            self.fig2 = plt.figure(figsize =(100,max(ylim/float(xlim)*100,10)))
            self.ax2 = self.fig2.add_subplot(111)


    def initialize_template_space(self):

        # load template space related files
        self.pca_main = pickle.load(open(absolute_path_to_asset(
            os.path.join('template_space', 'pca_main.pkl')), 'rb'))
        self.pca_sec = pickle.load(open(absolute_path_to_asset(
            os.path.join('template_space', 'pca_sec.pkl')), 'rb'))
        self.pca_main_noise_std = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'pca_main_noise_std.npy')))
        self.pca_sec_noise_std = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'pca_sec_noise_std.npy')))

        # ref template
        self.ref_template = np.load(absolute_path_to_asset(
            os.path.join('template_space', 'ref_template.npy')))

        # turn off edges for less collision
        window = [15, 40]
        self.pca_main.components_[:, :window[0]] = 0
        self.pca_main.components_[:, window[1]:] = 0
        self.pca_sec.components_[:, :window[0]] = 0
        self.pca_sec.components_[:, window[1]:] = 0


    def load_waveforms(self, local):

        if self.verbose:
            print ("chan "+str(self.channel)+", gen 0, loading waveforms")
        
        neighbors = n_steps_neigh_channels(self.CONFIG.neigh_channels, 1)
        
        #if self.deconv_flag==False:
        self.neighbor_chans = np.where(neighbors[self.channel])[0]
        #else:
        #    self.neighbor_chans = np.where(neighbors[self.max_chan])[0]
            
        if local:
            self.loaded_channels = self.neighbor_chans
        else:
            self.loaded_channels = np.arange(self.CONFIG.recordings.n_channels)

        # load waveforms from raw data 
        if self.deconv_flag==False:
            self.wf_global = binary_reader_waveforms(self.standardized_filename,
                            self.CONFIG.recordings.n_channels,
                            self.spike_size,
                            self.spt_global.astype('int32')-(self.spike_size//2),
                            self.loaded_channels)
                            
        # post-deconv recluster loads wavefroms as residuals + templates
        else:
            self.wf_global, self.skipped_idx = read_spikes(self.filename_residual, 
                                         self.unit, 
                                         self.templates_deconv, 
                                         #self.spike_train_deconv,
                                         self.spt_global.astype('int32'),
                                         self.CONFIG, 
                                         self.loaded_channels,
                                         self.deconv_flag,
                                         spike_size=self.spike_size_padded)
        
        # clip waveforms; seems necessary for neuropixel probe due to artifacts
        self.wf_global = self.wf_global.clip(min=-1000, max=1000)


    def align_step(self, local):
        if self.verbose:
            print ("chan "+str(self.channel)+", gen 0, aligning")

        # delete any spikes that could not be loaded in previous step
        if len(self.skipped_idx)>0:
            self.spt_global = np.delete(self.spt_global, 
                                        self.skipped_idx,axis=0)
        
        # align waveforms by finding best shfits
        if local:
            mc = np.where(self.loaded_channels==self.channel)[0][0]
            best_shifts = align_get_shifts_with_ref(
                self.wf_global[:, self.shift_allowance:-self.shift_allowance, mc],
                self.ref_template)
            self.spt_global -= best_shifts
            self.global_shifts = best_shifts.copy()
        else:
            best_shifts = self.spt_global.astype('int32') - self.spt_global
        
        self.wf_global = shift_chans(self.wf_global, best_shifts)


    def denoise_step(self, local):

        if local:
            self.denoise_step_local()
        else:
            self.denoise_step_distant()

        if self.verbose:
            print ("chan "+str(self.channel)+", gen 0, waveorms denoised to {} dimensions".format(self.denoised_wf.shape[1]))


    def denoise_step_local(self):
        # align, note: aligning all channels to max chan which is appended to the end
        # note: max chan is first from feat_chans above, ensure order is preserved
        # note: don't want for wf array to be used beyond this function
        # Alignment: upsample max chan only; linear shift other chans

        n_data, _, n_chans = self.wf_global.shape
        self.denoised_wf = np.zeros((n_data, self.pca_main.components_.shape[0], n_chans),
                                    dtype='float32')

        for ii in range(n_chans):
            if self.loaded_channels[ii] == self.channel:
                self.denoised_wf[:, :, ii] = np.matmul(
                    self.wf_global[:, self.shift_allowance:-self.shift_allowance, ii],
                    self.pca_main.components_.T)/self.pca_main_noise_std[np.newaxis]
            else:
                self.denoised_wf[:, :, ii] = np.matmul(
                    self.wf_global[:, self.shift_allowance:-self.shift_allowance, ii],
                    self.pca_sec.components_.T)/self.pca_sec_noise_std[np.newaxis]

        self.denoised_wf = np.reshape(self.denoised_wf, [n_data, -1])

        good_features = np.median(np.square(self.denoised_wf), axis=0) > 0.5
        self.denoised_wf = self.denoised_wf[:, good_features]


    def denoise_step_distant(self):

        energy = np.median(self.wf_global, axis=0)
        max_energy = np.min(energy, axis=0)

        th = np.max((-0.5, max_energy[self.channel]))
        max_energy_loc_c = np.where(max_energy <= th)[0]
        max_energy_loc_t = energy.argmin(axis=0)
        max_energy_loc = np.hstack((max_energy_loc_t[max_energy_loc_c][:, np.newaxis],
                                    max_energy_loc_c[:, np.newaxis]))

        neighbors = n_steps_neigh_channels(self.CONFIG.neigh_channels, 1)
        t_diff = 3
        main_channel_loc = np.where(self.loaded_channels == self.channel)[0][0]
        index = np.where(max_energy_loc[:,1]== main_channel_loc)[0][0]
        keep = connecting_points(max_energy_loc, index, neighbors, t_diff)

        max_energy_loc = max_energy_loc[keep]

        # exclude main and secondary channels
        if np.sum(~np.in1d(max_energy_loc[:,1], self.neighbor_chans)) > 0:
            max_energy_loc = max_energy_loc[~np.in1d(max_energy_loc[:,1], self.neighbor_chans)]
        else:
            max_energy_loc = max_energy_loc[max_energy_loc[:,1]==main_channel_loc]

        self.denoised_wf = np.zeros((self.wf_global.shape[0], len(max_energy_loc)), dtype='float32')
        for ii in range(len(max_energy_loc)):
            self.denoised_wf[:, ii] = self.wf_global[:, max_energy_loc[ii,0], max_energy_loc[ii,1]]


    def active_chans_step(self, local):
            
        if self.verbose:
                print ("chan "+str(self.channel)+", gen 0, getting active channels")

        energy = np.max(np.median(np.square(self.wf_global), axis=0), axis=0)
        active_chans = np.where(energy > 0.5)[0]
        
        if not local:
            active_chans = active_chans[~np.in1d(active_chans, self.neighbor_chans)]

        if len(active_chans) == 0:
            active_chans = np.where(self.loaded_channels==self.channel)[0]
        
        self.active_chans = active_chans
           
        if local:
            self.denoised_wf
    

    def featurize_step(self, gen, indices, local, save_pca):

        if self.verbose:
            print("chan "+str(self.channel)+', gen '+str(gen)+', featurizing')

        if self.denoised_wf.shape[1] > self.selected_PCA_rank:
            stds = np.std(self.denoised_wf[indices], axis=0)
            good_d = np.where(stds > 1.05)[0]
            if len(good_d) < self.selected_PCA_rank:
                good_d = np.argsort(stds)[::-1][:self.selected_PCA_rank]

            data_to_fit = self.denoised_wf[indices][:, good_d]

            n_samples, n_features = data_to_fit.shape
            pca = PCA(n_components=min(self.selected_PCA_rank, n_features))

            pca_wf = pca.fit_transform(data_to_fit)
           
        else:
            pca_wf = self.denoised_wf[indices].copy()

        if gen==0 and local and save_pca:
            self.pca_wf_gen0= pca_wf.copy()
            
        return pca_wf.astype('float32')
     

    def subsample_step(self, gen, pca_wf):
 
        if self.verbose:
            print("chan "+str(self.channel)+', gen '+str(gen)+', random subsample')
       
        if not self.deconv_flag and (pca_wf.shape[0]> self.CONFIG.cluster.max_n_spikes):
            idx_subsampled = np.random.choice(np.arange(pca_wf.shape[0]),
                             size=self.CONFIG.cluster.max_n_spikes,
                             replace=False)
        
            pca_wf = pca_wf[idx_subsampled]

        return pca_wf
    
    
    def run_mfm(self, gen, pca_wf):
        
        mask = np.ones((pca_wf.shape[0], 1))
        group = np.arange(pca_wf.shape[0])
        vbParam = mfm.spikesort(pca_wf[:,:,np.newaxis],
                                 mask,
                                 group,
                                self.CONFIG)
        if self.verbose:
            print("chan "+ str(self.channel)+', gen '\
                +str(gen)+", "+str(vbParam.rhat.shape[1])+" clusters from ",pca_wf.shape)

        return vbParam


    def knn_triage_dynamic(self, gen, vbParam, pca_wf):

        muhat = vbParam.muhat[:,:,0].T
        cov = vbParam.invVhat[:,:,:,0].T / vbParam.nuhat[:,np.newaxis, np.newaxis]

        if cov.shape[0] == 1:
            self.triage_value = 0
            idx_keep = np.arange(pca_wf.shape[0])

        else:
            min_spikes = 1200

            pca_wf_temp = np.zeros([min_spikes*cov.shape[0], cov.shape[1]])
            assignment_temp = np.zeros(min_spikes*cov.shape[0], dtype = int)
            for i in range(cov.shape[0]):
                pca_wf_temp[i*min_spikes:(i+1)*min_spikes]= np.random.multivariate_normal(muhat[i], cov[i], min_spikes)
                assignment_temp[i*min_spikes:(i+1)*min_spikes] = i

            kdist_temp = knn_dist(pca_wf_temp)
            kdist_temp = kdist_temp[:,1:]

            median_distances = np.zeros([cov.shape[0]])
            for i in range(median_distances.shape[0]):
                #median_distances[i] = np.median(np.median(kdist_temp[i*min_spikes:(i+1)*min_spikes], axis = 0), axis = 0)
                median_distances[i] = np.percentile(np.median(kdist_temp[i*min_spikes:(i+1)*min_spikes], axis = 1), 90)

            kdist = knn_dist(pca_wf)
            idx_keep = np.median(kdist[:,1:], axis = 1) < 1 * np.median(median_distances)
            self.triage_value = 1.0 - idx_keep.sum()/idx_keep.size

        if self.verbose:
            print("chan "+str(self.channel)+', gen '+str(gen)+', '+str(np.round(self.triage_value*100))+'% triaged from adaptive knn triage')

        return np.where(idx_keep)[0]


    def knn_triage_step(self, gen, pca_wf):
        
        idx_keep = self.knn_triage(self.knn_triage_threshold, pca_wf)
        idx_keep = np.where(idx_keep==1)[0]
        self.triage_value = self.knn_triage_threshold/100.

        if self.verbose:
            print("chan "+str(self.channel)+', gen '+str(gen)+', knn triage removed {} from {} spikes'.format(pca_wf.shape[0]-len(idx_keep), pca_wf.shape[0]))

        return idx_keep


    def knn_triage(self, th, pca_wf):

        tree = cKDTree(pca_wf)
        dist, ind = tree.query(pca_wf, k=11)
        dist = np.sum(dist, 1)
        idx_keep1 = dist <= np.percentile(dist, th)
        return idx_keep1


    def recover_step(self, gen, vbParam, pca_wf_all):
 
        # for post-deconv reclustering, we can safely cluster only 10k spikes or less
        if not self.deconv_flag:
            idx_recovered, vbParam = self.recover_spikes(vbParam, pca_wf_all)

        else:
            idx_recovered = np.arange(pca_wf_all.shape[0])

        if self.verbose:
            print ("chan "+ str(self.channel)+', gen '+str(gen)+", recovered ",
                                                str(idx_recovered.shape[0])+ " spikes")

        return idx_recovered, vbParam
    
    def recover_spikes(self, vbParam, pca, maha_dist = 1):
    
        N, D = pca.shape
        threshold = D*maha_dist
        # update rhat on full data
        maskedData = mfm.maskData(pca[:,:,np.newaxis], np.ones([N, 1]), np.arange(N))
        vbParam.update_local(maskedData)

        # calculate mahalanobis distance
        maha = mfm.calc_mahalonobis(vbParam, pca[:,:,np.newaxis])
        idx_recovered = np.where(~np.all(maha >= threshold, axis=1))[0]
        vbParam.rhat = vbParam.rhat[idx_recovered]

        # zero out low assignment vals
        # Cat: TODO move this parameter to init function
        self.recover_threshold = 0.001
        if True:
            vbParam.rhat[vbParam.rhat < self.recover_threshold] = 0
            vbParam.rhat = vbParam.rhat/np.sum(vbParam.rhat,
                                             1, keepdims=True)

        return idx_recovered, vbParam
     
    def kill_small_units(self, gen, vbParam):
 
        # for post-deconv reclustering, we can safely cluster only 10k spikes or less
        assignment = vbParam.rhat.argmax(1)
        unique_units, n_data = np.unique(assignment, return_counts=True)

        big_units = unique_units[n_data > self.CONFIG.cluster.min_spikes]
        n_unit_killed = vbParam.rhat.shape[1] - len(big_units)
        if len(big_units) > 0:
            idx_survived = np.where(np.in1d(assignment, big_units))[0]

            vbParam.rhat = vbParam.rhat[idx_survived][:, big_units]
            vbParam.rhat = vbParam.rhat/vbParam.rhat.sum(axis=1, keepdims=True)
            vbParam.ahat = vbParam.ahat[big_units]
            vbParam.lambdahat = vbParam.lambdahat[big_units]
            vbParam.nuhat = vbParam.nuhat[big_units]
            vbParam.muhat = vbParam.muhat[:,big_units]
            vbParam.Vhat = vbParam.Vhat[:,:,big_units]
            vbParam.invVhat = vbParam.invVhat[:,:,big_units]

        else:
            idx_survived = np.zeros(0)
            vbParam.rhat = np.zeros((0,0))

        if self.verbose:
            print ("chan "+ str(self.channel)+', gen '+str(gen)+", killed ",
                                                str(n_unit_killed)+' small units')

        if vbParam.rhat.shape[1] != len(big_units):
            raise ValueError('number of units in rhat is wrong!')
        return idx_survived, vbParam

    def calculate_stability(self, rhat):
        K = rhat.shape[1]
        mask = rhat > 0.0
        stability = np.zeros(K)
        for clust in range(stability.size):
            if mask[:,clust].sum() == 0.0:
                continue
            stability[clust] = np.average(mask[:,clust] * rhat[:,clust], axis = 0, weights = mask[:,clust])

        return stability

    def get_k_cc(self, maha, maha_thresh_min, k_target):

        # it assumes that maha_thresh_min gives 
        # at least k+1 number of connected components
        k_now = k_target + 1
        if len(self.get_cc(maha, maha_thresh_min)) != k_now:
            raise ValueError("something is not right")

        maha_thresh = maha_thresh_min
        while k_now > k_target:
            maha_thresh += 1
            cc = self.get_cc(maha, maha_thresh)
            k_now = len(cc)

        if k_now == k_target:
            return cc, maha_thresh

        else:
            maha_thresh_max = maha_thresh
            maha_thresh_min = maha_thresh - 1
            if len(self.get_cc(maha, maha_thresh_min)) <= k_target:
                raise ValueError("something is not right")

            ctr = 0
            maha_thresh_max_init = maha_thresh_max
            while True:
                ctr += 1
                maha_thresh = (maha_thresh_max + maha_thresh_min)/2.0
                cc = self.get_cc(maha, maha_thresh)
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
                    print(len(self.get_cc(maha, maha_thresh+0.001)))
                    print(len(self.get_cc(maha, maha_thresh-0.001)))
                    raise ValueError("something is not right")


    def get_cc(self, maha, maha_thresh):
        row, column = np.where(maha<maha_thresh)
        G = nx.DiGraph()
        for i in range(maha.shape[0]):
            G.add_node(i)
        for i, j in zip(row,column):
            G.add_edge(i, j)
        cc = [list(units) for units in nx.strongly_connected_components(G)]
        return cc


    def cluster_annealing(self, vbParam):

        N, K = vbParam.rhat.shape

        stability = self.calculate_stability(vbParam.rhat)
        if (K <= 2) or np.all(stability > 0.9):
            return vbParam.rhat.argmax(1), stability

        maha = mfm.calc_mahalonobis(vbParam, vbParam.muhat.transpose((1,0,2)))
        maha = np.maximum(maha, maha.T)
        #N, K = vbParam.rhat.shape
        #mu = np.copy(vbParam.muhat[:,:,0].T)
        #mudiff = mu[:,np.newaxis] - mu
        #prec = vbParam.Vhat[:,:,:,0].T * vbParam.nuhat[:,np.newaxis, np.newaxis]
        #maha = np.matmul(np.matmul(mudiff[:, :, np.newaxis], prec[:, np.newaxis]), mudiff[:, :, :, np.newaxis])[:, :, 0, 0]

        # decrease number of connected components one at a time.
        # in any step if all components are stables, stop and return
        # otherwise, go until there are only two connected components and return it
        maha_thresh_min = 0
        for k_target in range(K-1, 1, -1):
            # get connected components with k_target number of them
            cc, maha_thresh_min = self.get_k_cc(maha, maha_thresh_min, k_target)
            
            # calculate soft assignment for each cc
            rhat_cc = np.zeros([N,len(cc)])
            for i, units in enumerate(cc):
                rhat_cc[:, i] = np.sum(vbParam.rhat[:, units], axis=1)
            rhat_cc[rhat_cc<0.001] = 0.0
            rhat_cc = rhat_cc/np.sum(rhat_cc,axis =1 ,keepdims = True)

            # calculate stability for each component
            # and make decision            
            stability = self.calculate_stability(rhat_cc)
            if np.all(stability>0.90) or k_target == 2:
                return rhat_cc.argmax(1), stability

        
    def single_cluster_step(self, gen, current_indices, pca_wf):

        # exclude units whose maximum channel is not on the current 
        # clustered channel; but only during clustering, not during deconv
        #template = np.median(self.wf_global[current_indices], axis=0)
        template = np.mean(self.wf_global[current_indices], axis=0)
        assignment = np.zeros(len(current_indices))
        mc = self.loaded_channels[np.argmax(template.ptp(0))]
        if mc != self.channel and (self.deconv_flag==False): 
            if self.verbose:
                print ("  chan "+str(self.channel)+", template has maxchan "+str(mc), 
                        " skipping ...")
            
            # always plot scatter distributions
            if self.plotting and gen<20:
                split_type = 'mfm non_max-chan'
                end_flag = 'cyan'
                self.plot_clustering_scatter(gen, 
                            pca_wf, assignment, [1], split_type, end_flag)             
        else:         
            N = len(self.spike_train)
            if self.verbose:
                print("chan "+str(self.channel)+', gen '+str(gen)+", >>> cluster "+
                    str(N)+" saved, size: "+str(len(assignment))+"<<<")
                print ("")
            
            self.spike_train.append(self.spt_global[current_indices])
            self.templates.append(template)
            
            # plot template if done
            if self.plotting:
                self.plot_clustering_template(gen, template, 
                                              len(current_indices), N)

                # always plot scatter distributions
                if gen<20:
                    split_type = 'mfm single unit'
                    end_flag = 'red'
                    self.plot_clustering_scatter(gen,  
                            pca_wf, assignment, [1], split_type, end_flag)
             
    def multi_cluster_step(self, gen, current_indices, cc_assignment, local):
        
        for clust in np.unique(cc_assignment):
            idx = np.where(cc_assignment==clust)[0]

            if self.verbose:
                print("chan "+str(self.channel)+', gen '+str(gen)+
                    ", reclustering cluster with "+ str(idx.shape[0]) +' spikes')

            self.cluster(current_indices[idx], gen+1, local)


    def finish(self, fname=None):

        if self.plotting:
            if self.deconv_flag:
                spikes_original = np.where(self.spike_train_cluster_original == self.unit)[0]

            ####### finish cluster plots #######
            if self.deconv_flag:
                max_chan = self.template_original.ptp(0).argmax(0)
            else:
                max_chan = self.channel

            self.fig1.suptitle(fname, fontsize=100)
            if self.deconv_flag:
                self.fig1.savefig(self.chunk_dir + "/recluster/unit_{}_scatter.png".format(self.unit))
            else:
                #self.fig1.savefig(self.chunk_dir + "/channel_{}_scatter.png".format(self.channel))
                self.fig1.savefig(os.path.join(self.figures_dir,fname+'_scatter.png'))
            #plt.close(self.fig1)

            ####### finish template plots #######
            # plot channel numbers and shading
            for i in self.loaded_channels:
                self.ax2.text(self.CONFIG.geom[i,0], self.CONFIG.geom[i,1],
                              str(i), alpha=0.4, fontsize=10)
                              
                # fill bewteen 2SUs on each channel
                self.ax2.fill_between(self.CONFIG.geom[i,0] +
                     np.arange(-self.spike_size,0,1)/self.xscale, -self.yscale +
                     self.CONFIG.geom[i,1], self.yscale + self.CONFIG.geom[i,1],
                     color='black', alpha=0.05)
                
            # plot max chan with big red dot
            self.ax2.scatter(self.CONFIG.geom[max_chan,0],
                              self.CONFIG.geom[max_chan,1], s = 2000,
                              color = 'red')

            # plot original templates for post-deconv reclustering
            if self.deconv_flag:
                self.ax2.plot(self.CONFIG.geom[:, 0] +
                          np.arange(-self.template_original.shape[0] // 2,
                          self.template_original.shape[0] // 2, 1)[:, np.newaxis] / self.xscale,
                          self.CONFIG.geom[:, 1] + self.template_original * self.yscale,
                          'r--', c='red')

            labels = []
            if self.deconv_flag:
                patch_j = mpatches.Patch(color='red', label="size = {}".format(spikes_original.shape[0]))
                labels.append(patch_j)

            # if at least 1 cluster is found, plot the template
            if len(self.spike_train)>0:
                for clust in range(len(self.spike_train)):
                    patch_j = mpatches.Patch(color = sorted_colors[clust%100],
                                             label = "size = {}".format(len(self.spike_train[clust])))
                    labels.append(patch_j)
            self.ax2.legend(handles=labels, fontsize=100)

            # plot title
            self.fig2.suptitle(fname, fontsize=100)
            if self.deconv_flag:
                self.fig2.savefig(self.chunk_dir + "/recluster/unit_{}_template.png".format(self.unit))
            else:
                #self.fig2.savefig(self.chunk_dir + "/channel_{}_template.png".format(self.channel))
                self.fig2.savefig(os.path.join(self.figures_dir,fname+'_template.png'))
            #plt.close(self.fig2)
            plt.close('all')


    def save_result(self, spike_train=None, templates=None):

        if self.deconv_flag:
            spikes_original = np.where(self.spike_train_cluster_original == self.unit)[0]

        # Cat: TODO: note clustering is done on PCA denoised waveforms but
        #            templates are computed on original raw signal
        # recompute templates to contain full width information... 

        if self.deconv_flag:
            np.savez(self.filename_postclustering,
                        spike_index=spike_train,
                        templates=templates,
                        spike_index_prerecluster= spikes_original,
                        templates_prerecluster=self.template_original)
        else:
            np.savez(self.filename_postclustering,
                     spiketime=spike_train,
                     templates=templates,
                     global_shifts=self.global_shifts,
                     pca_wf_gen0=self.pca_wf_gen0,
                     spiketime_detect=self.spiketime_detect)

        if self.deconv_flag==False:
            print ("**** Channel: ", str(self.channel), " starting spikes: ",
                len(self.spike_indexes_chunk), ", found # clusters: ",
                len(spike_train))
        else:
            print ("**** Unit: ", str(self.unit), " starting spikes: ",
                len(self.spike_indexes_chunk), ", found # clusters: ",
                len(spike_train))

        self.wf_global = None
        self.denoised_wf = None
        self.spike_train = None
        self.templates = None
  
    # def diptest_step(self, EM_split, assignment2, idx_recovered, vbParam2, pca_wf_all):
        
        # if EM_split: 
            # gmm = GaussianMixture(n_components=2)
        
        # ctr=0 
        # dp_val = 1.0
        # idx_temp_keep = np.arange(idx_recovered.shape[0])
        # cluster_idx_keep = np.arange(vbParam2.muhat.shape[1])
        # # loop over cluster until at least 3 loops and take lowest dp value
        # while True:    
            # # use EM algorithm to get binary split
            # if EM_split: 
                # gmm.fit(pca_wf_all[idx_recovered])
                # labels = gmm.predict_proba(pca_wf_all[idx_recovered])

                # temp_rhat = labels
                # temp_assignment = np.zeros(labels.shape[0], 'int32')
                # idx = np.where(labels[:,1]>0.5)[0]
                # temp_assignment[idx]=1
            
            # # use mfm algorithm to find temp-assignment
            # else:
                # temp_assignment = self.mfm_binary_split2(
                    # vbParam2.muhat[:, cluster_idx_keep],
                    # assignment2[idx_recovered],
                    # cluster_idx_keep)


            # # check if any clusters smaller than min spikes
            # counts = np.unique(temp_assignment, return_counts=True)[1]

            # # update indexes if some clusters too small
            # if min(counts)<self.CONFIG.cluster.min_spikes:
                # print ("  REMOVING SMALL CLUSTER DURING diptest")
                # bigger_cluster_id = np.argmax(counts)
                # idx_temp_keep = np.where(temp_assignment==bigger_cluster_id)[0]
                # idx_recovered = idx_recovered[idx_temp_keep]

                # # This decreases the clusters kept in muhat
                # cluster_idx_keep = np.unique(assignment2[idx_recovered])
                
                # # exit if cluster gets decimated below threshld
                # if idx_recovered.shape[0]<self.CONFIG.cluster.min_spikes:
                    # return dp_val, assignment2[idx_recovered], idx_recovered

                # # if removed down to a single cluster, recluster it (i.e. send dpval=0.0)
                # if cluster_idx_keep.shape[0] < 2:
                    # return 0.0, assignment2[idx_recovered], idx_recovered

            # # else run the unimodality test
            # # Cat: todo: this is not perfect, still misses some bimodal distributions
            # else:
                # # test EM for unimodality
                # dp_new = self.test_unimodality(pca_wf_all[idx_recovered], temp_assignment)
                
                # # set initial values
                # if ctr==0:
                    # assignment3 = temp_assignment
                
                # # search for lowest split score (not highest)
                # # goal is to find most multimodal split, not unimodal
                # if dp_new <dp_val:
                    # dp_val= dp_new
                    # assignment3 = temp_assignment

                # # if at least 3 loops using EM-split, or any loop iteration for mfm
                # if ctr>2 or not EM_split:
                    # # need to also ensure that we've not deleted any spikes after we
                    # #  saved the last lowest-dp avlue assignment
                    # if assignment3.shape[0] != temp_assignment.shape[0]:
                        # assignment3 = temp_assignment
                    # break
                
                # ctr+=1
            
        # return dp_val, assignment3, idx_recovered


    def robust_stds(self, wf_align):
        
        stds = np.median(np.abs(wf_align - np.median(wf_align, axis=0, keepdims=True)), axis=0)*1.4826
        return stds


    def mfm_binary_split2(self, muhat, assignment_orig, cluster_index=None):

        centers = muhat[:, :, 0].T
        K, D = centers.shape
        if cluster_index is None:
            cluster_index = np.arange(K)

        label = AgglomerativeClustering(n_clusters=2).fit(centers).labels_
        assignment = np.zeros(len(assignment_orig), 'int16')
        for j in range(2):
            print (j)
            print (np.where(label == j)[0])
            #clusters = cluster_index[np.where(label == j)[0]]
            clusters = cluster_index[np.where(label == j)[0]]
            for k in clusters:
                assignment[assignment_orig == k] = j

        return assignment

    def save_step(self, dp_val, mc, gen, idx_recovered,
                          pca_wf_all, vbParam2, assignment2, assignment3,
                          sic_current, template_current, feat_chans):
                              
        # make sure cluster is on max chan, otherwise omit it
        if mc != self.channel and (self.deconv_flag==False): 
            print ("  channel: ", self.channel, " template has maxchan: ", mc, 
                    " skipping ...")
            
            # always plot scatter distributions
            if gen<20:
                split_type = 'mfm-binary - non max chan'
                end_flag = 'cyan'                       
                self.plot_clustering_scatter(gen,  
                    assignment3,
                    assignment2[idx_recovered],
                    pca_wf_all[idx_recovered],
                    vbParam2.rhat[idx_recovered],
                    split_type,
                    end_flag)
                        
            return 
        
        N = len(self.assignment_global)
        if self.verbose:
            print("chan "+str(self.channel)+' gen: '+str(gen)+" >>> cluster "+
                  str(N)+" saved, size: "+str(idx_recovered.shape)+"<<<")
        
        self.assignment_global.append(N * np.ones(assignment3.shape[0]))
        self.spike_index.append(sic_current[idx_recovered])
        #template = np.median(template_current[idx_recovered],0)
        template = np.mean(template_current[idx_recovered],0)
        self.templates.append(template)

        # plot template if done
        if self.plotting:
            self.plot_clustering_template(gen, template, idx_recovered, 
                                         feat_chans, N)

            # always plot scatter distributions
            if gen<20:
                # hack to expand the assignments back out to size of original
                # data stream
                assignment3 = np.zeros(pca_wf_all[idx_recovered].shape[0],'int32')
                split_type = 'mfm-binary, dp: '+ str(round(dp_val,5))
                end_flag = 'green'
                self.plot_clustering_scatter(gen,  
                    assignment3,
                    assignment2[idx_recovered],
                    pca_wf_all[idx_recovered],
                    vbParam2.rhat[idx_recovered],
                    split_type,
                    end_flag)     



    def connected_channels(self, channel_list, ref_channel, neighbors, keep=None):
        if keep is None:
            keep = np.zeros(len(neighbors), 'bool')
        if keep[ref_channel] == 1:
            return keep
        else:
            keep[ref_channel] = 1
            chans = channel_list[neighbors[ref_channel][channel_list]]
            for c in chans:
                keep = self.connected_channels(channel_list, c, neighbors, keep=keep)
            return keep


    def get_feat_channels_mad(self, wf_align):
        '''  Function that uses MAD statistic like robust variance estimator
             to select channels
        '''
        # compute robust stds over units
        #stds = np.median(np.abs(wf_align - np.median(wf_align, axis=0, keepdims=True)), axis=0)*1.4826
        # trim vesrion of stds
        #stds = np.std(stats.trimboth(wf_align, 0.025), 0)
        stds = self.robust_stds(wf_align)

        # max per channel
        std_max = stds.max(0)
        
        # order channels by largest diptest value
        feat_chans = np.argsort(std_max)[::-1]
        #feat_chans = feat_chans[std_max[feat_chans] > 1.2]

        max_chan = wf_align.mean(0).ptp(0).argmax(0)

        return feat_chans, max_chan, stds


    def featurize(self, wf, robust_stds, feat_chans, max_chan):
        
        # select argrelmax of mad metric greater than trehsold
        #n_feat_chans = 5

        n_features_per_channel = 2
        wf_final = np.zeros((0,wf.shape[0]), 'float32')
        # select up to 2 features from max amplitude chan;
        trace = robust_stds[:,max_chan]
        idx = argrelmax(trace, axis=0, mode='clip')[0]

        if idx.shape[0]>0:
            idx_sorted = np.argsort(trace[idx])[::-1]
            idx_thresh = idx[idx_sorted[:n_features_per_channel]]
            temp = wf[:,idx_thresh,max_chan]
            wf_final = np.vstack((wf_final, temp.T))
            #wf_final.append(wf[:,idx_thresh,max_chan])
            
        ## loop over all feat chans and select max 2 argrelmax time points as features
        n_feat_chans = np.min((self.n_feat_chans, wf.shape[2]))
        for k in range(n_feat_chans):

            # don't pick max channel again, already picked above
            if feat_chans[k]==max_chan: continue
            
            trace = robust_stds[:,feat_chans[k]]
            idx = argrelmax(trace, axis=0, mode='clip')[0]
            if idx.shape[0]>0:
                idx_sorted = np.argsort(trace[idx])[::-1]
                idx_thresh = idx[idx_sorted[:n_features_per_channel]]
                temp = wf[:,idx_thresh,feat_chans[k]]
                wf_final = np.vstack((wf_final, temp.T))

        # Cat: TODO: this may crash if weird data goes in
        #print (" len wf arra: ", len(wf_final))
        #wf_final = np.array(wf_final)
        #wf_final = wf_final.swapaxes(0,1).reshape(wf.shape[0],-1)
        wf_final = wf_final.T

        # run PCA on argrelmax points;
        # Cat: TODO: read this from config
        pca = PCA(n_components=min(self.selected_PCA_rank, wf_final.shape[1]))
        pca.fit(wf_final)
        pca_wf = pca.transform(wf_final)

        # convert boolean to integer indexes
        idx_keep_feature = np.arange(wf_final.shape[0])

        return idx_keep_feature, pca_wf, wf_final


    def test_unimodality(self, pca_wf, assignment, max_spikes = 10000):

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

                                
    def plot_clustering_scatter(self, gen, pca_wf, assignment, stability,
                                split_type, end_point='false'):

        if (self.x[gen]<20) and (gen <20):

            # add generation index
            ax = self.fig1.add_subplot(self.grid1[gen, self.x[gen]])
            self.x[gen] += 1

            clusters, sizes = np.unique(assignment, return_counts=True)
            if len(clusters) != len(stability):
               raise ValueError('there is an empty cluster!')

            # make legend
            labels = []
            for clust in range(len(clusters)):
                patch_j = mpatches.Patch(color = sorted_colors[clust], 
                    label = "size = {}, stability = {}".format(sizes[clust], np.round(stability[clust],2)))
                labels.append(patch_j)
            
            # make scater plots
            if pca_wf.shape[1]>1:
                ax.scatter(pca_wf[:,0], pca_wf[:,1], 
                    c = sorted_colors[assignment.astype(int)] ,alpha=0.05)

                # add red dot for converged clusters; cyan to off-channel
                if end_point!='false':
                    ax.scatter(pca_wf[:,0].mean(), pca_wf[:,1].mean(), c= end_point, s = 2000, alpha=.5)
            else:
                for clust in clusters:
                    ax.hist(pca_wf[np.where(assignment==clust)[0]], 100)

            # finish plotting
            ax.legend(handles = labels, fontsize=10, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
            ax.set_title(split_type+': '+str(sizes.sum())+' spikes, triage %: '+ str(np.round(self.triage_value*100,2)), fontsize = 10) 

          
    def plot_clustering_template(self, gen, template, n_data, unit_id):
        
        # plot template
        #local_scale = min
        geom_loaded = self.CONFIG.geom[self.loaded_channels]
        R, C = template.shape

        self.ax2.plot(geom_loaded[:, 0]+
                  np.arange(-R//2, R//2,1)[:, np.newaxis]/self.xscale,
                  geom_loaded[:, 1] + template*self.yscale, c=colors[unit_id%100],
                  linewidth = 10,
                  alpha=min(max(0.4, n_data/1000.), 1))



def connecting_points(points, index, neighbors, t_diff, keep=None):
    if keep is None:
        keep = np.zeros(len(points), 'bool')

    if keep[index] == 1:
        return keep
    else:
        keep[index] = 1
        spatially_close = np.where(neighbors[points[index, 1]][points[:, 1]])[0]
        close_index = spatially_close[np.abs(points[spatially_close, 0] - points[index, 0]) <= t_diff]

        for j in close_index:
            keep = connecting_points(points, j, neighbors, t_diff, keep)

        return keep


def align_get_shifts_with_ref(wf, ref, upsample_factor = 5, nshifts = 7):

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
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)

    

def upsample_resample(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces


def knn_dist(pca_wf):
    tree = cKDTree(pca_wf)
    dist, ind = tree.query(pca_wf, k=30)
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
                print ("  spike to close to end, skipping and deleting array")
                wfs=np.delete(wfs, wfs.shape[0]-1,axis=0)
                skipped_idx.append(ctr_skipped)

            ctr_skipped+=1
    fin.close()

    return wfs, skipped_idx
           

def read_spikes(filename, unit, templates, spike_train, CONFIG, 
                channels, residual_flag=False, spike_size=None):
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
            spike_waveforms+=templates[:,channels,unit]
        # need to add templates in middle of noise wfs which are wider
        else:
            offset = spike_size - int(CONFIG.recordings.spike_size_ms*
                                      CONFIG.recordings.sampling_rate//1000*2+1)
            spike_waveforms[:,offset//2:-offset//2]+=templates[:,channels,unit]
        
    return spike_waveforms, skipped_idx
    

def shift_chans(wf, best_shifts):
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    # Cat: TODO read this from CNOFIG
    wf_shifted = []
    wfs_final=[]
    for k, shift_ in enumerate(best_shifts):
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(wf[k],ceil,axis=0)
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            temp = np.roll(wf[k],ceil,axis=0)*(shift_-floor)+np.roll(wf[k],floor, axis=0)*(ceil-shift_)
        wfs_final.append(temp)
    wf_shifted = np.array(wfs_final)
    
    return wf_shifted
