import numpy as np
import scipy
import time, os
import parmap
import copy
from tqdm import tqdm
import time
from yass.cluster.util import (binary_reader, load_waveforms_from_memory)

# ********************************************************
# ********************************************************
# ********************************************************

def parallel_conv_filter(data_in, 
                        n_time,
                        up_up_map,
                        unit_overlap,
                        up_factor,
                        vis_chan,
                        approx_rank,
                        deconv_dir):

    proc_index = data_in[0]
    unit_array = data_in[1]

    # Cat: must load these structures from disk for multiprocessing step; 
    #       where there are many templates; due to multiproc 4gb limit 
    fname = deconv_dir+"/svd.npz"
    data = np.load(fname)
    temporal_up = data['temporal_up']
    temporal = data['temporal']
    singular = data['singular']
    spatial = data['spatial']

    pairwise_conv_array = []
    for unit2 in unit_array:
        conv_res_len = n_time * 2 - 1
        n_overlap = np.sum(unit_overlap[unit2, :])
        pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)
        orig_unit = unit2 // up_factor
        masked_temp = np.flipud(np.matmul(
                temporal_up[unit2] * singular[orig_unit][None, :],
                spatial[orig_unit, :, :]))

        for j, unit1 in enumerate(np.where(unit_overlap[unit2, :])[0]):
            u, s, vh = temporal[unit1], singular[unit1], spatial[unit1] 
            vis_chan_idx = vis_chan[:, unit1]
            mat_mul_res = np.matmul(
                    masked_temp[:, vis_chan_idx], vh[:approx_rank, vis_chan_idx].T)

            for i in range(approx_rank):
                pairwise_conv[j, :] += np.convolve(
                        mat_mul_res[:, i],
                        s[i] * u[:, i].flatten(), 'full')
    
        pairwise_conv_array.append(pairwise_conv)
        
    np.save(deconv_dir+'/temp_temp_chunk_'+str(proc_index), 
                                                pairwise_conv_array)

    #return pairwise_conv



# ********************************************************
# ********************************************************
# ********************************************************

class MatchPursuit_objectiveUpsample(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, temps, deconv_chunk_dir, standardized_filename, 
                 max_iter, refrac_period=60, upsample=1, threshold=10., 
                 conv_approx_rank=10, n_processors=1,multi_processing=False,
                 vis_su=2., 
                 keep_iterations=False):
        """Sets up the deconvolution object.

        Parameters:
        -----------
        data: numpy array of shape (T, C)
            Where T is number of time samples and C number of channels.
        temps: numpy array of shape (t, C, K)
            Where t is number of time samples and C is the number of
            channels and K is total number of units.
        conv_approx_rank: int
            Rank of SVD decomposition for approximating convolution
            operations for templates.
        threshold: float
            amount of energy differential that is admissible by each
            spike. The lower this threshold, more spikes are recovered.
        vis_su: float
            threshold for visibility of template channel in terms
            of peak to peak standard unit.
        keep_iterations: boolean
            Keeps the spike train per iteration if True. Otherwise,
            does not keep the history.
        """
        
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.temps = temps.astype(np.float32)
        self.orig_temps = temps.astype(np.float32)
        print ("  in match pursuit templates (n_time, n_chan, n_unit): ", self.temps.shape)
                
        self.deconv_dir = deconv_chunk_dir
        self.standardized_filename = standardized_filename
        self.max_iter = max_iter
        self.n_processors = n_processors
        self.multi_processing = multi_processing

        
        # Upsample and downsample time shifted versions
        # Dynamic Upsampling Setup; function for upsampling based on PTP
        # Cat: TODO find better ptp-> upsample function
        self.upsample_templates_mp(int(upsample))
            
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.vis_su_threshold = vis_su
        self.vis_chan = None
        self.visible_chans()
        self.template_overlaps()
        self.spatially_mask_templates()
        # Upsample the templates
        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor
        self.orig_template_idx = np.arange(0, self.n_unit, self.up_factor)
        print ("  # upsample templates: ", self.n_unit)
        
        # Computing SVD for each template.
        print ("  computing SVD on templates ")
        self.compress_templates()
        
        # Compute pairwise convolution of filters
        print ("  computing temp_temp")
        self.pairwise_filter_conv()
        
        # compute norm of templates
        self.norm = np.zeros([self.orig_n_unit, 1], dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(
                    np.square(self.temps[:, self.vis_chan[:, i], i]))
        
        #fname_out = (self.deconv_dir+"/seg_{}_deconv.npz".format(
        #                                    str(self.seg_ctr).zfill(6)))
        
        # np.save(self.deconv_dir+'/vis_chans.npy', self.vis_chan)
        # np.save(self.deconv_dir+'/norms.npy', self.norm)
        # quit()
        
        # Setting up data properties
        self.keep_iterations = keep_iterations
        #self.update_data(data)
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        
        # Energey reduction for assigned spikes.
        self.dist_metric = np.array([])
        
        # Single time preperation for high resolution matches
        # matching indeces of peaks to indices of upsampled templates
        factor = self.up_factor
        radius = factor // 2 + factor % 2
        self.up_window = np.arange(-radius - 1, radius + 1)[:, None]
        self.up_window_len = len(self.up_window)
        off = (factor + 1) % 2
        
        # Indices of single time window the window around peak after upsampling
        self.zoom_index = (radius + 1) * factor + np.arange(-factor // 2, radius)
        peak_to_template_idx = np.append(
                np.arange(radius + off, factor),
                np.arange(radius + off))
        self.peak_to_template_idx = np.pad(
                peak_to_template_idx, (1, 0), 'edge')
        if off:
            self.peak_to_template_idx[0] -= 1
        peak_time_jitter = np.array([1, 0]).repeat(radius)
        peak_time_jitter[radius - 1] = 0
        self.peak_time_jitter = np.pad(peak_time_jitter, (1, 0), 'edge')
        
        # Refractory Perios Setup.
        self.refrac_radius = refrac_period
        
        # Account for upsampling window so that np.inf does not fall into the
        # window around peak for valid spikes.
        #self.adjusted_refrac_radius = max(
        #        1, self.refrac_radius - self.up_factor // 2)
        
        self.adjusted_refrac_radius = 10
        
        # Stack for turning on invalud units for next iteration
        self.turn_off_stack = []

    def upsample_templates_mp(self, upsample):
        if upsample != 1:
            
            if True:
                max_upsample = 32
                # original function
                self.unit_up_factor = np.power(
                        4, np.floor(np.log2(np.max(self.temps.ptp(axis=0), axis=0))))
                self.up_factor = min(max_upsample, int(np.max(self.unit_up_factor)))
                self.unit_up_factor[self.unit_up_factor > max_upsample] = max_upsample
                self.up_up_map = np.zeros(
                        self.n_unit * self.up_factor, dtype=np.int32)
                for i in range(self.n_unit):
                    u_idx = i * self.up_factor
                    u_factor = self.unit_up_factor[i]
                    skip = self.up_factor // u_factor
                    self.up_up_map[u_idx:u_idx + self.up_factor] = u_idx  + np.arange(
                            #0, self.up_factor, u_factor).repeat(u_factor)
                            0, self.up_factor, skip).repeat(skip)
            else:
                    #unit_up_factor2 = np.power(
                    #        2, np.floor(np.log2(ptps)))*2
                    ptps = np.max(self.temps.ptp(axis=0), axis=0)
                    self.unit_up_factor = np.ones(ptps.shape)
                    self.up_factor = upsample
                    self.unit_up_factor[ptps > 4] = 2
                    self.unit_up_factor[ptps > 8] = 8
                    self.unit_up_factor[ptps > 16] = 32
                    self.unit_up_factor[ptps > 32] = 64
                    self.unit_up_factor[ptps > 64] = 128
                    self.up_up_map = np.zeros(
                            self.n_unit * self.up_factor, dtype=np.int32)
                    for i in range(self.n_unit):
                    #for i in range(100):
                        u_idx = i * self.up_factor
                        u_factor = self.unit_up_factor[i]
                        skip = self.up_factor // u_factor
                        self.up_up_map[u_idx:u_idx + self.up_factor] = u_idx  + np.arange(
                                0, self.up_factor, skip).repeat(skip)

        else:
                # Upsample and downsample time shifted versions
                self.up_factor = upsample
                self.unit_up_factor = upsample
                self.up_up_map = range(self.n_unit * self.up_factor)

        fname = self.deconv_dir + "/up_up_maps.npz"
        np.savez(fname,
                 up_up_map = self.up_up_map,
                 unit_up_factor = self.unit_up_factor)


    def update_data(self):
        """Updates the data for the deconv to be run on with same templates."""
        self.data = self.data.astype(np.float32)
        self.data_len = self.data.shape[0]
        
        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1
        self.dot = np.zeros(
                [self.orig_n_unit, self.obj_len],
                dtype=np.float32)
                
        # Indicator for computation of the objective.
        self.obj_computed = False
        
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])
        self.iter_spike_train = []

        
    def visible_chans(self):
        if self.vis_chan is None:
            a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
            self.vis_chan = a > self.vis_su_threshold
        return self.vis_chan


    def template_overlaps(self):
        """Find pairwise units that have overlap between."""
        vis = self.vis_chan.T
        self.unit_overlap = np.sum(
            np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
        self.unit_overlap = self.unit_overlap > 0
        self.unit_overlap = np.repeat(self.unit_overlap, self.up_factor, axis=0)
    

    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        idx = np.logical_xor(
                np.ones(self.temps.shape, dtype=bool), self.vis_chan)
        self.temps[idx] = 0.
        

    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""
        
        fname = self.deconv_dir+"/svd.npz"
        if os.path.exists(fname)==False:
            self.temporal, self.singular, self.spatial = np.linalg.svd(
                np.transpose(np.flipud(self.temps), (2, 0, 1)))
            # Keep only the strongest components
            self.temporal = self.temporal[:, :, :self.approx_rank]
            self.singular = self.singular[:, :self.approx_rank]
            self.spatial = self.spatial[:, :self.approx_rank, :]
            # Upsample the temporal components of the SVD
            # in effect, upsampling the reconstruction of the
            # templates.
            if self.up_factor == 1:
                # No upsampling is needed.
                self.temporal_up = self.temporal
                return
            self.temporal_up = scipy.signal.resample(
                    self.temporal, self.n_time * self.up_factor, axis=1)
            idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(self.up_factor)[:, None]
            self.temporal_up = np.reshape(
                    self.temporal_up[:, idx, :], [-1, self.n_time, self.approx_rank]).astype(np.float32)

            np.savez(fname,
                     temporal_up = self.temporal_up,
                     temporal = self.temporal, 
                     singular = self.singular,
                     spatial = self.spatial)
        else:
            data = np.load(fname)
            self.temporal_up = data['temporal_up']
            self.temporal = data['temporal']
            self.singular = data['singular']
            self.spatial = data['spatial']


    # Cat: TODO: Parallelize this function
    def pairwise_filter_conv_parallel(self):
    
        # Cat: TODO: this may still crash memory in some cases; can split into additional bits
        units = np.array_split(np.unique(self.up_up_map), self.n_processors)
        if self.multi_processing:
            parmap.map(parallel_conv_filter, 
                            list(zip(np.arange(len(units)),units)), 
                            self.n_time,
                            self.up_up_map,
                            self.unit_overlap,
                            self.up_factor,
                            self.vis_chan,
                            self.approx_rank,
                            self.deconv_dir,
                            processes=self.n_processors,
                            pm_pbar=True)
        else:
            units = np.unique(self.up_up_map)

            for k in range(len(units)):
                print ("unit : ", k)
                parallel_conv_filter( 
                            [k,[units[k]]],
                            self.n_time,
                            self.up_up_map,
                            self.unit_overlap,
                            self.up_factor,
                            self.vis_chan,
                            self.approx_rank,
                            self.deconv_dir)
        
        # load temp_temp saved files from disk due to memory overload otherwise
        temp_array = []
        for i in range(len(units)):
            fname = self.deconv_dir+'/temp_temp_chunk_'+str(i)+'.npy'
            temp_pairwise_conv = np.load(fname)
            temp_array.extend(temp_pairwise_conv)
            os.remove(fname)

        # initialize empty list and fill only correct locations
        print ("  gathering temp_temp results...")
        pairwise_conv=[]
        for i in range(self.n_unit):
            pairwise_conv.append(None)

        ctr=0
        for unit2 in np.unique(self.up_up_map):
            pairwise_conv[unit2] = temp_array[ctr]
            ctr+=1
        
        pairwise_conv = np.array(pairwise_conv)
        print (pairwise_conv.shape)
        
        # save to disk, don't keep in memory
        np.save(self.deconv_dir+"/pairwise_conv.npy", pairwise_conv)



    # Cat: TODO: Parallelize this function
    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""

        if os.path.exists(self.deconv_dir+"/pairwise_conv.npy") == False:

            # Cat: TODO: original temp_temp computation, do not erase it, keep
            #           it for debugging and testing pursposes
            #if not self.multi_processing:
            ##if True:
                #print (" turned multi-processing off here")
                ##print ("  (todo parallelize pairse filter conv)...")
                #conv_res_len = self.n_time * 2 - 1
                #self.pairwise_conv = []
                #for i in range(self.n_unit):
                    #self.pairwise_conv.append(None)
                #available_upsampled_units = np.unique(self.up_up_map)
                #for unit2 in tqdm(available_upsampled_units, '  computing temptemp'):
                    ## Set up the unit2 conv all overlaping original units.
                    #n_overlap = np.sum(self.unit_overlap[unit2, :])
                    #self.pairwise_conv[unit2] = np.zeros([n_overlap, conv_res_len], dtype=np.float32)
                    #orig_unit = unit2 // self.up_factor
                    #masked_temp = np.flipud(np.matmul(
                            #self.temporal_up[unit2] * self.singular[orig_unit][None, :],
                            #self.spatial[orig_unit, :, :]))
                    #for j, unit1 in enumerate(np.where(self.unit_overlap[unit2, :])[0]):
                        #u, s, vh = self.temporal[unit1], self.singular[unit1], self.spatial[unit1] 
                        #vis_chan_idx = self.vis_chan[:, unit1]

                        #mat_mul_res = np.matmul(
                                #masked_temp[:, vis_chan_idx], vh[:self.approx_rank, vis_chan_idx].T)
                        #for i in range(self.approx_rank):
                            #self.pairwise_conv[unit2][j, :] += np.convolve(
                                    #mat_mul_res[:, i],
                                    #s[i] * u[:, i].flatten(), 'full')
                
                #self.pairwise_conv = np.array(self.pairwise_conv)
            #else:        
        
            self.pairwise_filter_conv_parallel()
                

        #else:
        #    self.pairwise_conv = np.load(self.deconv_dir+'/pairwise_conv.npy')
    
    
    def get_reconstructed_upsampled_templates(self):
        """Get the reconstructed upsampled versions of the original templates.
        If no upsampling was requested, returns the SVD reconstructed version
        of the original templates.
        """
        rec = np.matmul(
                self.temporal_up * np.repeat(self.singular, self.up_factor, axis=0)[:, None, :],
                np.repeat(self.spatial, self.up_factor, axis=0))
        return np.fliplr(rec).transpose([1, 2, 0])

    def get_sparse_upsampled_templates(self):
        """Returns the fully upsampled sparse version of the original templates.
        returns:
        --------
        Tuple of numpy.ndarray. First element is of shape (t, C, M) is the set
        upsampled shifted templates that have been used in the dynamic
        upsampling approach. Second is an array of lenght K (number of original
        units) * maximum upsample factor. Which maps cluster ids that are result
        of deconvolution to 0,...,M-1 that corresponds to the sparse upsampled
        templates.
        """
        
        fname = self.deconv_dir+"/sparse_templates.npy"
        if os.path.exists(fname)==False:
        
            print ("  getting sparse upsampled templates (TODO: Parallelize)...")
            down_sample_idx = np.arange(
                    0, self.n_time * self.up_factor, self.up_factor)
            down_sample_idx = down_sample_idx + np.arange(
                    0, self.up_factor)[:, None]
            result = []
            
            # Reordering the upsampling. This is done because we upsampled the time
            # reversed temporal components of the SVD reconstruction of the
            # templates. This means That the time-reveresed 10x upsampled indices
            # respectively correspond to [0, 9, 8, ..., 1] of the 10x upsampled of
            # the original templates.
            all_temps = []
            reorder_idx = np.append(
                    np.arange(0, 1),
                    np.arange(self.up_factor - 1, 0, -1))

            # Sequentialize the number of up_up_map. For instance,
            # [0, 0, 0, 0, 4, 4, 4, 4, ...] turns to [0, 0, 0, 0, 1, 1, 1, 1, ...].
            deconv_id_sparse_temp_map = []
            tot_temps_so_far = 0
            
                            
            for i in range(self.orig_n_unit):
                up_temps = scipy.signal.resample(
                        self.orig_temps[:, :, i],
                        self.n_time * self.up_factor)[down_sample_idx, :]
                up_temps = up_temps.transpose([1, 2, 0])
                up_temps = up_temps[:, :, reorder_idx]
                skip = self.up_factor // self.unit_up_factor[i]
                keep_upsample_idx = np.arange(0, self.up_factor, skip).astype(np.int32)
                deconv_id_sparse_temp_map.append(np.arange(
                        self.unit_up_factor[i]).repeat(skip) + tot_temps_so_far)
                tot_temps_so_far += self.unit_up_factor[i]
                all_temps.append(up_temps[:, :, keep_upsample_idx])


            deconv_id_sparse_temp_map = np.concatenate(
                                    deconv_id_sparse_temp_map, axis=0)
                
            all_temps = np.concatenate(all_temps, axis=2)
            
            np.save(fname, all_temps)
            np.save(os.path.split(fname)[0]+'/deconv_id_sparse_temp_map.npy',
                                                    deconv_id_sparse_temp_map)
        
        else:
                        
            all_temps = np.load(fname)
            deconv_id_sparse_temp_map = np.load(os.path.split(fname)[0]+
                                            '/deconv_id_sparse_temp_map.npy')
        
        return all_temps, deconv_id_sparse_temp_map
        
    def get_sparse_upsampled_templates_parallel(self, unit):
        
        i = unit

        up_temps = scipy.signal.resample(
                self.orig_temps[:, :, i],
                self.n_time * self.up_factor)[down_sample_idx, :]
        up_temps = up_temps.transpose([1, 2, 0])
        up_temps = up_temps[:, :, reorder_idx]
        skip = self.up_factor // self.unit_up_factor[i]
        keep_upsample_idx = np.arange(0, self.up_factor, skip).astype(np.int32)

        deconv_id_sparse_temp_map.append(np.arange(
                self.unit_up_factor[i]).repeat(skip) + tot_temps_so_far)

        tot_temps_so_far += self.unit_up_factor[i]

        all_temps.append(up_temps[:, :, keep_upsample_idx])
        
        
        
        
    def get_upsampled_templates(self):
        """Returns the fully upsampled version of the original templates."""
        down_sample_idx = np.arange(0, self.n_time * self.up_factor, self.up_factor)
        down_sample_idx = down_sample_idx + np.arange(0, self.up_factor)[:, None]
        
        # original data stream
        #up_temps = scipy.signal.resample(
        #        self.orig_temps, self.n_time * self.up_factor)[down_sample_idx, :, :]
        
        if self.multi_processing:
            res = parmap.map(self.upsample_templates_parallel, 
                            self.orig_temps.T,
                            self.n_time, 
                            self.up_factor,
                            down_sample_idx,
                            processes=self.n_processors,
                            pm_pbar=True)
        else:
            res = []
            for k in range(self.orig_temps.T.shape[0]):
                print ("unit : ", k)
                res.append(self.upsample_templates_parallel(
                            self.orig_temps.T[k],
                            self.n_time, 
                            self.up_factor,
                            down_sample_idx))                
                        
        up_temps = np.array(res)
        print ("  upsampled templates shape: ", up_temps.shape)
        up_temps = up_temps.transpose(
            [2, 3, 0, 1]).reshape([self.n_chan, -1, self.n_time]).transpose([2, 0, 1])
        self.n_unit = self.n_unit * self.up_factor
        # Reordering the upsampling. This is done because we upsampled the time
        # reversed temporal components of the SVD reconstruction of the
        # templates. This means That the time-reveresed 10x upsampled indices
        # respectively correspond to [0, 9, 8, ..., 1] of the 10x upsampled of
        # the original templates.
        reorder_idx = np.tile(
                np.append(
                    np.arange(0, 1),
                    np.arange(self.up_factor - 1, 0, -1)),
                self.orig_n_unit)
        reorder_idx += np.arange(
                0, self.up_factor * self.orig_n_unit,
                self.up_factor).repeat(self.up_factor)
        return up_temps[:, :, reorder_idx]
        
    def upsample_templates_parallel(template, n_time, up_factor,
                                    down_sample_idx):
        return scipy.signal.resample(
            template.T, n_time * up_factor)[down_sample_idx, :]
        

    def correct_shift_deconv_spike_train(self, dec_spike_train):
        """Get time shift corrected version of the deconvovled spike train.
        This corrected version only applies if you consider getting upsampled
        templates with get_upsampled_templates() method.
        """
        correct_spt = copy.copy(dec_spike_train)
        correct_spt[correct_spt[:, 1] % self.up_factor > 0, 0] += 1
        return correct_spt
                     

    def compute_objective(self):
        """Computes the objective given current state of recording."""
        if self.obj_computed:
            return self.obj
        n_rows = self.orig_n_unit * self.approx_rank
        conv_result = np.zeros(
                [n_rows, self.data_len + self.n_time - 1], dtype=np.float32)
        
        filters = self.temporal.transpose([0, 2, 1]).reshape([n_rows, -1])

        if True: 
            matmul_result = np.matmul(
                    self.spatial.reshape([n_rows, -1]) * self.singular.reshape([-1, 1]),
                    self.data.T)

            for i in range(n_rows):
                conv_result[i, :] = np.convolve(
                        matmul_result[i, :], filters[i, :], mode='full')
        
        # this is the lower memory version that runs 3 x slower
        else:
            temp_spatial = self.spatial.reshape([n_rows, -1])
            temp_singular = self.singular.reshape([-1, 1])
            for i in range(n_rows):
                temp_matmul = np.matmul(
                    temp_spatial[i] * temp_singular[i],
                    self.data.T)
                conv_result[i, :] = np.convolve(
                        temp_matmul, filters[i, :], mode='full')
        
        for i in range(1, self.approx_rank):
            conv_result[np.arange(0, n_rows, self.approx_rank), :] +=\
                    conv_result[np.arange(i, n_rows, self.approx_rank), :]
        self.obj = 2 * conv_result[np.arange(0, n_rows, self.approx_rank), :] - self.norm

        # Set indicator to true so that it no longer is run
        # for future iterations in case subtractions are done
        # implicitly.
        self.obj_computed = True
        return self.obj
        
        
    def high_res_peak(self, times, unit_ids):
        """Finds best matching high resolution template.
        Given an original unit id and the infered spike times
        finds out which of the shifted upsampled templates of
        the unit best matches at that time to the residual.
        Parameters:
        -----------
        times: numpy.array of numpy.int
            spike times for the unit.
        unit_ids: numpy.array of numpy.int
            Respective to times, id of each spike corresponding
            to the original units.
        Returns:
        --------
            tuple in the form of (numpy.array, numpy.array, numpy.array)
            respectively the offset of shifted templates and a necessary time
            shift to correct the spike time, and the index of spike times that
            do not violate refractory period.
        """
        if self.up_factor == 1 or len(times) < 1:
            return 0, 0, range(len(times))
        idx = times + self.up_window
        peak_window = self.obj[unit_ids, idx]
        # Find times that the window around them do not inlucde np.inf.
        # In other words do not violate refractory period.
        invalid_idx = np.logical_or(
            np.isinf(peak_window[0, :]), np.isinf(peak_window[-1, :]))
        # Turn off the invlaid units for next iterations.
        turn_off_idx = times[invalid_idx] + np.arange(
                - self.refrac_radius, 1)[:, None]
        self.obj[unit_ids[invalid_idx], turn_off_idx] = - np.inf
        valid_idx = np.logical_not(invalid_idx)
        peak_window = peak_window[:, valid_idx]
        if peak_window.shape[1]  == 0:
            return np.array([]), np.array([]), valid_idx 
        high_resolution_peaks = scipy.signal.resample(
                peak_window, self.up_window_len * self.up_factor, axis=0)
        shift_idx = np.argmax(
                high_resolution_peaks[self.zoom_index, :], axis=0)
        return self.peak_to_template_idx[shift_idx], self.peak_time_jitter[shift_idx], valid_idx
        
        
    # def find_peaks_old(self):
        # """Finds peaks in subtraction differentials of spikes."""
        # max_across_temp = np.max(self.obj, axis=0)
        # spike_times = scipy.signal.argrelmax(
                # max_across_temp, order=self.refrac_radius)[0]
        # spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        # dist_metric = max_across_temp[spike_times]
        # # TODO(hooshmand): this requires a check of the last element(s)
        # # of spike_times only not of all of them since spike_times
        # # is sorted already.
        # valid_idx = spike_times < self.data_len - self.n_time
        # dist_metric = dist_metric[valid_idx]
        # spike_times = spike_times[valid_idx]
        # # Upsample the objective and find the best shift (upsampled)
        # # template.
        # spike_ids = np.argmax(self.obj[:, spike_times], axis=0)
        # upsampled_template_idx, time_shift, valid_idx = self.high_res_peak(
                # spike_times, spike_ids)
        # spike_ids = spike_ids[valid_idx] * self.up_factor + upsampled_template_idx
        # spike_times = spike_times[valid_idx] - time_shift
        # # Note that we shift the discovered spike times from convolution
        # # Space to actual raw voltate space by subtracting self.n_time
        # result = np.append(
            # spike_times[:, None] - self.n_time + 1,
            # spike_ids[:, None], axis=1)
        
        # return result, dist_metric[valid_idx]

    def find_peaks(self):
        """Finds peaks in subtraction differentials of spikes."""
        max_across_temp = np.max(self.obj, axis=0)
        spike_times = scipy.signal.argrelmax(
                max_across_temp, order=self.refrac_radius)[0]
        spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        dist_metric = max_across_temp[spike_times]
        # TODO(hooshmand): this requires a check of the last element(s)
        # of spike_times only not of all of them since spike_times
        # is sorted already.
        valid_idx = spike_times < self.data_len - self.n_time
        dist_metric = dist_metric[valid_idx]
        spike_times = spike_times[valid_idx]
        # Upsample the objective and find the best shift (upsampled)
        # template.
        spike_ids = np.argmax(self.obj[:, spike_times], axis=0)
        upsampled_template_idx, time_shift, valid_idx = self.high_res_peak(
                spike_times, spike_ids)
        # The spikes that had NAN in the window and could not be updampled
        # should fall-back on default value.
        spike_ids *= self.up_factor
        if np.sum(valid_idx) > 0:
            spike_ids[valid_idx] += upsampled_template_idx
            spike_times[valid_idx] -= time_shift
        # Note that we shift the discovered spike times from convolution
        # Space to actual raw voltate space by subtracting self.n_time
        result = np.append(
            spike_times[:, None] - self.n_time + 1,
            spike_ids[:, None], axis=1)

        return result, dist_metric[valid_idx]

        
    def enforce_refractory(self, spike_train):
        """Enforces refractory period for units."""
        window = np.arange(- self.adjusted_refrac_radius, self.adjusted_refrac_radius)
        n_spikes = spike_train.shape[0]
        win_len = len(window)
        # The offset self.n_time - 1 is necessary to revert the spike times
        # back to objective function indices which is the result of convoultion
        # operation.
        time_idx = (spike_train[:, 0:1] + self.n_time - 1) + window
        # Re-adjust cluster id's so that they match
        # with the original templates
        unit_idx = spike_train[:, 1:2] // self.up_factor
        self.obj[unit_idx, time_idx[:, 1:-1]] = - np.inf
        
        
    def subtract_spike_train(self, spt):
        """Substracts a spike train from the original spike_train."""
        present_units = np.unique(spt[:, 1])
        for i in present_units:
            conv_res_len = self.n_time * 2 - 1
            unit_sp = spt[spt[:, 1] == i, :]
            spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1] 
            # Grid idx of subset of channels and times
            unit_idx = self.unit_overlap[i]
            idx = np.ix_(unit_idx, spt_idx.ravel())
            self.obj[idx] -= np.tile(
                    2 * self.pairwise_conv[self.up_up_map[i]], len(unit_sp))

        self.enforce_refractory(spt)
        
        
    def get_iteration_spike_train(self):
        return self.iter_spike_train

        
    def run(self, data_in):

        start_time = time.time()
        verbose = False
        self.idx_list = data_in[0][0]
        self.seg_ctr = data_in[0][1]
        self.buffer_size = data_in[1]
        
        # ********* run deconv ************
        fname_out = (self.deconv_dir+"/seg_{}_deconv.npz".format(
                                            str(self.seg_ctr).zfill(6)))

        # read raw data for segment using idx_list vals
        #self.load_data_from_memory()
        self.data = binary_reader(self.idx_list, self.buffer_size, 
                             self.standardized_filename,
                             self.n_chan).astype(np.float32)

        #self.data = data.astype(np.float32)
        self.data_len = self.data.shape[0]
        
        # load pairwise conv filter OR TRY TO USE GLOBAL SHARED VARIABLE
        self.pairwise_conv = np.load(self.deconv_dir+"/pairwise_conv.npy")
        #self.pairwise_conv = pairwise_conv

        # update data
        self.update_data()
        
        # compute objective function
        start_time = time.time()
        self.compute_objective()
        print ('  deconv seg {0}, objective matrix took: {1:.2f}'.
                format(self.seg_ctr, time.time()-start_time))
                
        ctr = 0
        tot_max = np.inf
        while tot_max > self.threshold and ctr < self.max_iter:
            spt, dist_met = self.find_peaks()
            
            if len(spt) == 0:
                break
            
            self.dec_spike_train = np.append(self.dec_spike_train, spt, axis=0)
            
            self.subtract_spike_train(spt)
            
            if self.keep_iterations:
                self.iter_spike_train.append(spt)
            self.dist_metric = np.append(self.dist_metric, dist_met)
                        
            if verbose: 
                print ("Iteration {0} Found {1} spikes with {2:.2f} energy reduction.".format(
                ctr, spt.shape[0], np.sum(dist_met)))

            ctr += 1

        print ('  deconv seg {0}, # iter: {1}, tot_spikes: {2}, tot_time: {3:.2f}'.
                format(
                self.seg_ctr, ctr, self.dec_spike_train.shape[0],
                time.time()-start_time))

        # ******** ADJUST SPIKE TIMES TO REMOVE BUFFER AND OFSETS *******
        # order spike times
        idx = np.argsort(self.dec_spike_train[:,0])
        self.dec_spike_train = self.dec_spike_train[idx]

        # find spikes inside data block, i.e. outside buffers
        idx = np.where(np.logical_and(self.dec_spike_train[:,0]>=self.idx_list[2],
                                      self.dec_spike_train[:,0]<self.idx_list[3]))[0]
        self.dec_spike_train = self.dec_spike_train[idx]

        # offset spikes to start of index
        self.dec_spike_train[:,0]+= self.idx_list[0] - self.idx_list[2]
        
        np.savez(fname_out, spike_train = self.dec_spike_train, 
                            dist_metric = self.dist_metric)

        self.data = None
        #return self.dec_spike_train
        
    
# ********************************************************************
# *************** RESIDUAL COMPUTATION FUNCTION **********************
# ********************************************************************

class Residual(object):
    
    def __init__(self, 
                 temps,
                 dec_spike_train, 
                 buffer_size, 
                 n_processors, 
                 deconv_chunk_dir, 
                 n_sec_chunk,
                 idx_list_local,
                 standardized_filename):
        
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.buffer_size = buffer_size
        self.standardized_filename = standardized_filename

        self.dec_spike_train = dec_spike_train
        self.n_processors = n_processors
        self.deconv_chunk_dir = deconv_chunk_dir
        self.n_sec_chunk = n_sec_chunk
        self.idx_list_local = idx_list_local

    # Legacy code, do not remove yet
    # def get_unit_spikes(self, unit, unit_sp):
    #     """Gets clean spikes for a given unit."""
    #     #unit_sp = dec_spike_train[dec_spike_train[:, 1] == unit, :]
    #
    #     # Add the spikes of the current unit back to the residual
    #     temp = self.data[np.arange(0, self.n_time) + unit_sp[:, :1], :] + self.temps[:, :, unit]
    #     return temp
        
    def compute_residual_new(self, CONFIG, min_ptp):
        ''' Function to subtract residuals using parallel CPU
            How to: grab chunks of data with a buffer on each side (e.g. 200)
                    and delete all spike tempaltes at location of spike times                   
            
            The problem: when subtracting waveforms from buffer areas
                         it becomes tricky to add the residuals together
                         from 2 neighbouring chunks because they will 
                         add the residual from the other chunk's data back in
                         essentially invaldiating the subtraction in the buffer
            
            The solution: keep track of energy subtracted in the buffer zone
            
            The pythonic solution: 
                - make an np.zero() data_blank array and also
                derasterize it at the same time as data array; 
                - subtract the buffer bits of the data_blank array from residual array
                (this essentially subtracts the data 2 x so that it can be added back
                in by other chunk)
                - concatenate the parallel chunks together by adding the residuals in the
                buffer zones. 
        '''
        
        print ("  Computing residual in parallel ")

        # compute residual only with tempaltes > min_ptp
        self.min_ptp = min_ptp

        # take 10sec chunks with 200 buffer on each side
        len_chunks = CONFIG.recordings.sampling_rate *self.n_sec_chunk

        # split the data into lists of indexes and spike_times for parallelization
        # get indexes for entire chunk from local chunk list
        end_index = self.idx_list_local[-1][1]
        spike_times=[]
        indexes=[]
        for ctr,k in enumerate(range(0, end_index, len_chunks)):
            start = k
            end = k+len_chunks

            # split data for each chunk
            indexes.append([start,end, ctr])

            # assign spikes in each chunk; offset 
            idx_in_chunk = np.where(np.logical_and(
                                    self.dec_spike_train[:,0]>=start-self.buffer_size,
                                    self.dec_spike_train[:,0]<end+self.buffer_size))[0]

            spikes_in_chunk = self.dec_spike_train[idx_in_chunk]
            # reset spike times to zero for each chunk but add in bufer_size
            #  that will be read in with data 
            spikes_in_chunk[:,0] = spikes_in_chunk[:,0] - start #+ self.buffer_size
            spike_times.append(spikes_in_chunk)

        spike_times = np.array(spike_times)

        # Cat: TODO: read multiprocessing flag from CONFIG
        if CONFIG.resources.multi_processing:
            parmap.map(self.subtract_parallel, 
                         list(zip(indexes,spike_times)), 
                         self.n_unit, self.n_time, 
                         self.deconv_chunk_dir,
                         processes=CONFIG.resources.n_processors,
                         pm_pbar=True)
        else:
            for k in range(len(indexes)):
                self.subtract_parallel(
                           [indexes[k],spike_times[k]],
                           self.n_unit, self.n_time,
                           self.deconv_chunk_dir)
        
        # initilize residual array with buffers; 
        self.data = np.zeros((end_index+self.buffer_size*2,self.n_chan),dtype=np.float32)
        for k in range(len(indexes)):
            fname = self.deconv_chunk_dir+ '/residual_seg_'+str(indexes[k][2])+'.npy'
            res = np.load(fname)
            os.remove(fname)
            self.data[indexes[k][0]+self.buffer_size:indexes[k][1]+self.buffer_size]+= res[self.buffer_size:-self.buffer_size]


    def subtract_parallel(self, data_in, n_unit, n_time, 
                          deconv_chunk_dir):
        '''
        '''
        
        indexes = data_in[0]
        local_spike_train=data_in[1]
        proc_index = indexes[2]

        fname = deconv_chunk_dir+ '/residual_seg_'+str(proc_index)+'.npy'
        if os.path.exists(fname)==True:
            return

        # note only derasterize up to last bit, don't remove spikes from 
        # buffer_size.. end because those will be looked at by next chunk
        # load indexes and then index into original data

        idx_chunk = [indexes[0], indexes[1], self.buffer_size]
        data = binary_reader(idx_chunk, 
                             self.buffer_size, 
                             self.standardized_filename, 
                             self.n_chan)
        data_blank = np.zeros(data.shape)

        deconv_id_sparse_temp_map = np.load(deconv_chunk_dir+'/deconv_id_sparse_temp_map.npy')
        sparse_templates = np.load(deconv_chunk_dir+'/sparse_templates.npy')

        ptps = sparse_templates.ptp(0).max(0)
        exclude_idx = np.where(ptps<self.min_ptp)[0]

        # loop over units and subtract energy
        unique_units = np.unique(local_spike_train[:,1])
        for i in unique_units:
            unit_sp = local_spike_train[local_spike_train[:, 1] == i, :]

            # subtract data from both the datachunk and blank version
            template_id = int(deconv_id_sparse_temp_map[i])

            # exclude some templates from residual computation
            if template_id in exclude_idx:
                continue
            
            data[np.arange(0, n_time) + unit_sp[:, :1], :] -= sparse_templates[:, :, template_id]

            #data_blank[np.arange(0, n_time) + unit_sp[:, :1], :] -= sparse_templates[:, :, template_id]

        # Cat: TODO: This still leaves edge artifacts in; should fix at some point
        #       So the problem is that if a large spike is on the buffer border, only
        #       the onside part will be deleted, the part in the buffer will be left alone
        #       the correct way to fix this is to add overlapping buffers back to raw...

        # remove the buffer contributions x 2 so they can be properly added in after
        # note: this will make a small error in the first buffer and last buffer for
        # entire dataset; can fix this with some conditional
        #data[:self.buffer_size]-=data_blank[:self.buffer_size]
        #data[-self.buffer_size:]-=data_blank[-self.buffer_size:]
        
        # hard save or parmap can't hold entire data set
        np.save(fname, data)
        
