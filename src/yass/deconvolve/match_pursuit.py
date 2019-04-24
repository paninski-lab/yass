import numpy as np
import scipy
import time, os
import parmap
import copy
from tqdm import tqdm
import time
import logging

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
    fname = os.path.join(deconv_dir,"svd.npz")
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


class MatchPursuit_objectiveUpsample(object):
    """Class for doing greedy matching pursuit deconvolution."""

    def __init__(self, fname_templates, save_dir, reader, 
                 max_iter=1000, upsample=1, threshold=10., 
                 conv_approx_rank=10, n_processors=1,
                 multi_processing=False, vis_su=1.,
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
        
        logger = logging.getLogger(__name__)

        self.verbose = False

        temps = np.load(fname_templates).transpose(1, 2, 0)
        self.temps = temps.astype(np.float32)

        self.n_time, self.n_chan, self.n_unit = temps.shape
        self.deconv_dir = save_dir
        self.reader = reader
        self.max_iter = max_iter
        self.n_processors = n_processors
        self.multi_processing = multi_processing
        
        # Upsample and downsample time shifted versions
        # Dynamic Upsampling Setup; function for upsampling based on PTP
        # Cat: TODO find better ptp-> upsample function
        self.upsample_max_val = upsample
        self.upsample_templates_mp(int(upsample))
            
        self.threshold = threshold
        self.approx_rank = conv_approx_rank
        self.vis_su_threshold = vis_su
        self.visible_chans()
        self.template_overlaps()
        self.spatially_mask_templates()
        # Upsample the templates
        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor
        
        logger.info("computing SVD on templates")
        # Computing SVD for each template.
        self.compress_templates()
        
        # Compute pairwise convolution of filters
        logger.info("computing temp_temp")
        self.pairwise_filter_conv()

        # compute norm of templates
        self.norm = np.zeros([self.orig_n_unit, 1], dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(
                    np.square(self.temps[:, self.vis_chan[:, i], i]))

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
        self.up_window = np.arange(-radius, radius + 1)[:, None]
        self.up_window_len = len(self.up_window)
        off = (factor + 1) % 2

        # Indices of single time window the window around peak after upsampling
        self.zoom_index = radius * factor + np.arange(-radius, radius+1)
        self.peak_to_template_idx = np.append(np.arange(radius, -1, -1),
                                              (factor-1)-np.arange(radius))
        self.peak_time_jitter = np.append([0],np.array([0, 1]).repeat(radius))

        # Refractory Perios Setup.
        # DO NOT MAKE IT SMALLER THAN self.n_time - 1 !!!
        self.refrac_radius = self.n_time - 1
        
        # Account for upsampling window so that np.inf does not fall into the
        # window around peak for valid spikes.
        self.adjusted_refrac_radius = 10

    def upsample_templates_mp(self, upsample):
        if upsample != 1:
            
            if True:
                max_upsample = upsample
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
                self.unit_up_factor = np.ones(self.n_unit)
                self.up_up_map = range(self.n_unit * self.up_factor)

        fname = os.path.join(self.deconv_dir, "up_up_maps.npz")
        np.savez(fname,
                 up_up_map = self.up_up_map,
                 unit_up_factor = self.unit_up_factor)


    def update_data(self):
        """Updates the data for the deconv to be run on with same templates."""
        self.data = self.data.astype(np.float32)
        self.data_len = self.data.shape[0]
        
        # Computing SVD for each template.
        self.obj_len = self.data_len + self.n_time - 1
                
        # Indicator for computation of the objective.
        self.obj_computed = False
        
        # Resulting recovered spike train.
        self.dec_spike_train = np.zeros([0, 2], dtype=np.int32)
        self.dist_metric = np.array([])
        self.iter_spike_train = []

        
    def visible_chans(self):
        a = np.max(self.temps, axis=0) - np.min(self.temps, 0)
        self.vis_chan = a > self.vis_su_threshold

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
        
        fname = os.path.join(self.deconv_dir, "svd.npz")
        if os.path.exists(fname)==False:
            self.temporal, self.singular, self.spatial = np.linalg.svd(
                np.transpose(self.temps, (2, 0, 1)))
            # Keep only the strongest components
            self.temporal = self.temporal[:, :, :self.approx_rank]
            self.singular = self.singular[:, :self.approx_rank]
            self.spatial = self.spatial[:, :self.approx_rank, :]
            # Upsample the temporal components of the SVD
            # in effect, upsampling the reconstruction of the
            # templates.
            if self.up_factor == 1:
                # No upsampling is needed.
                temporal_up = self.temporal
            else:
                temporal_up = scipy.signal.resample(
                        self.temporal, self.n_time * self.up_factor, axis=1)
                idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(self.up_factor)[:, None]
                temporal_up = np.reshape(
                        temporal_up[:, idx, :], [-1, self.n_time, self.approx_rank]).astype(np.float32)

            self.temporal = np.flip(self.temporal, axis=1)
            temporal_up = np.flip(temporal_up, axis=1)

            np.savez(fname,
                     temporal_up = temporal_up,
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
            fname = os.path.join(self.deconv_dir, 'temp_temp_chunk_'+str(i)+'.npy')
            temp_pairwise_conv = np.load(fname)
            temp_array.extend(temp_pairwise_conv)
            os.remove(fname)

        # initialize empty list and fill only correct locations
        pairwise_conv=[]
        for i in range(self.n_unit):
            pairwise_conv.append(None)

        ctr=0
        for unit2 in np.unique(self.up_up_map):
            pairwise_conv[unit2] = temp_array[ctr]
            ctr+=1
        
        pairwise_conv = np.array(pairwise_conv)
        
        # save to disk, don't keep in memory
        np.save(os.path.join(self.deconv_dir, "pairwise_conv.npy"), pairwise_conv)



    # Cat: TODO: Parallelize this function
    def pairwise_filter_conv(self):
        """Computes pairwise convolution of templates using SVD approximation."""

        if os.path.exists(
            os.path.join(self.deconv_dir, "pairwise_conv.npy")) == False:

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
        
        fname = os.path.join(self.deconv_dir, "sparse_templates.npy")
        if os.path.exists(fname)==False:
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
                        self.temps[:, :, i],
                        self.n_time * self.up_factor)[down_sample_idx, :]
                up_temps = up_temps.transpose([1, 2, 0])
                #up_temps = up_temps[:, :, reorder_idx]
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
                self.temps[:, :, i],
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
                
        if self.multi_processing:
            res = parmap.map(self.upsample_templates_parallel, 
                            self.temps.T,
                            self.n_time, 
                            self.up_factor,
                            down_sample_idx,
                            processes=self.n_processors,
                            pm_pbar=True)
        else:
            res = []
            for k in range(self.temps.T.shape[0]):
                res.append(self.upsample_templates_parallel(
                            self.temps.T[k],
                            self.n_time, 
                            self.up_factor,
                            down_sample_idx))                
                        
        up_temps = np.array(res)
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

        conv_result = np.zeros(
                [self.orig_n_unit, self.data_len + self.n_time - 1], dtype=np.float32)
        for rank in range(self.approx_rank):
            matmul_result = np.matmul(
                    self.spatial[:, rank] * self.singular[:, [rank]],
                    self.data.T)

            filters = self.temporal[:, :, rank]
            for unit in range(self.orig_n_unit):
                conv_result[unit, :] += np.convolve(
                        matmul_result[unit, :], filters[unit], mode='full')
        self.obj = 2 * conv_result - self.norm
        # Set indicator to true so that it no longer is run
        # for future iterations in case subtractions are done
        # implicitly.
        self.obj_computed = True

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
            max_across_temp[self.n_time-1:self.obj.shape[1] - self.n_time],
            order=self.refrac_radius)[0] + self.n_time - 1
        spike_times = spike_times[max_across_temp[spike_times] > self.threshold]
        dist_metric = max_across_temp[spike_times]

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
            spike_times[valid_idx] += time_shift
        # Note that we shift the discovered spike times from convolution
        # Space to actual raw voltate space by subtracting self.n_time
        result = np.append(
            spike_times[:, None] - self.n_time + 1,
            spike_ids[:, None], axis=1)

        return result, dist_metric[valid_idx]

        
    def enforce_refractory(self, spike_train):
        """Enforces refractory period for units."""
        window = np.arange(-self.adjusted_refrac_radius, self.adjusted_refrac_radius+1)
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
            conv_res_len = self.n_time*2 - 1
            unit_sp = spt[spt[:, 1] == i, :]
            spt_idx = np.arange(0, conv_res_len) + unit_sp[:, :1] 
            # Grid idx of subset of channels and times
            unit_idx = self.unit_overlap[i]
            idx = np.ix_(unit_idx, spt_idx[::2].ravel())
            self.obj[idx] -= np.tile(
                    2 * self.pairwise_conv[self.up_up_map[i]], len(unit_sp[::2]))

            idx = np.ix_(unit_idx, spt_idx[1::2].ravel())
            self.obj[idx] -= np.tile(
                    2 * self.pairwise_conv[self.up_up_map[i]], len(unit_sp[1::2]))

        self.enforce_refractory(spt)
        
        
    def get_iteration_spike_train(self):
        return self.iter_spike_train


    def run(self, batch_ids, fnames_out):

        # set default pairwise-conv flag
        self.pairwise_conv = None
        
        # loop over each assigned segment
        for batch_id, fname_out in zip(batch_ids, fnames_out):
            
            # Cat: TODO is this redundant? i.e. is this check done outside already?
            if os.path.exists(fname_out):
                continue
            
            # load pairwise conv filter only once per core:
            if self.pairwise_conv is None:
                self.pairwise_conv = np.load(os.path.join(
                    self.deconv_dir, "pairwise_conv.npy"))
                
            start_time = time.time()
            
            # ********* run deconv ************

            # read raw data for segment using idx_list vals
            #self.load_data_from_memory()
            self.data = self.reader.read_data_batch(batch_id, add_buffer=True)

            #self.data = data.astype(np.float32)
            self.data_len = self.data.shape[0]

            # update data
            self.update_data()
            
            # compute objective function
            start_time = time.time()
            self.compute_objective()
            if self.verbose:
                logger.info('deconv seg {0}, objective matrix took: {1:.2f}'.
                        format(batch_id, time.time()-start_time))
                    
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
                            
                if self.verbose: 
                    logger.info("Iteration {0} Found {1} spikes with {2:.2f} energy reduction.".format(
                    ctr, spt.shape[0], np.sum(dist_met)))

                ctr += 1

            if self.verbose:
                logger.info('deconv seg {0}, # iter: {1}, tot_spikes: {2}, tot_time: {3:.2f}'.
                    format(
                    batch_id, ctr, self.dec_spike_train.shape[0],
                    time.time()-start_time))

            # ******** ADJUST SPIKE TIMES TO REMOVE BUFFER AND OFSETS *******
            # order spike times
            idx = np.argsort(self.dec_spike_train[:,0])
            self.dec_spike_train = self.dec_spike_train[idx]

            # find spikes inside data block, i.e. outside buffers
            idx = np.where(np.logical_and(
                self.dec_spike_train[:,0] >= self.reader.buffer,
                self.dec_spike_train[:,0]< self.data.shape[0] - self.reader.buffer))[0]
            self.dec_spike_train = self.dec_spike_train[idx]

            # offset spikes to start of index
            batch_offset = self.reader.idx_list[batch_id, 0] - self.reader.buffer
            self.dec_spike_train[:,0] += batch_offset
            
            np.savez(fname_out,
                     spike_train = self.dec_spike_train,
                     dist_metric = self.dist_metric)
                     
