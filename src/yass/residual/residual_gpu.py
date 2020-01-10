import os
import logging
import numpy as np
import time
from tqdm import tqdm
import torch
import sys
import cudaSpline as deconv
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, splder, sproot
import parmap


# these functions are same asin match_pursuit; copied just in case    
def transform_template_parallel(template, knots=None, prepad=7, postpad=3, order=3):

    if knots is None:
        #knots = np.arange(len(template.data[0]) + prepad + postpad)
        knots = np.arange(template.shape[1] + prepad + postpad)
        #print ("template.shape[0]: ", template.shape[1])
    # loop over every channel?
    splines = [
        fit_spline_cpu(curve, knots=knots, prepad=prepad, postpad=postpad, order=order) 
        for curve in template
    ]
    coefficients = np.array([spline[1][prepad-1:-1*(postpad+1)] for spline in splines], dtype='float32')
    
    return coefficients
        
        
def fit_spline_cpu(curve, knots=None, prepad=0, postpad=0, order=3):
    if knots is None:
        knots = np.arange(len(curve) + prepad + postpad)
    return splrep(knots, np.pad(curve, (prepad, postpad), mode='symmetric'), k=order)
    
    
class RESIDUAL_GPU2(object):
    
    def __init__(self,
                reader,
                recordings_filename,
                recording_dtype,
                CONFIG,
                fname_shifts,
                fname_scales,
                fname_templates,
                output_directory,
                dtype_out,
                fname_out,
                fname_spike_train,
                update_templates):
                  
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        
        self.logger = logging.getLogger(__name__)

        self.reader = reader
        self.recordings_filename = recordings_filename
        self.recording_dtype = recording_dtype
        self.CONFIG = CONFIG
        #self.output_directory = output_directory
        self.data_dir = output_directory
        self.dtype_out = dtype_out
        self.fname_out = fname_out
        self.fname_templates = fname_templates
        self.fname_shifts = fname_shifts
        self.fname_scales = fname_scales
        self.fname_residual = os.path.join(self.data_dir,'residual.bin')
        self.fname_spike_train = fname_spike_train
        
        
        # updated templates options
        # Cat: TODO read from CONFIG
        #self.update_templates = update_templates
        self.update_templates = update_templates
        
        # grab the starting templates
        if self.update_templates:
            #print ("loading
            self.fname_templates = os.path.join(os.path.split(self.data_dir)[0],
                                    'deconv','template_updates',
                                    'templates_'+
                                    str(CONFIG.deconvolution.template_update_time)+
                                    'sec.npy')

        # Cat: TODO read from CONFIG File
        self.template_update_time = CONFIG.deconvolution.template_update_time

        # fixed value for CUDA code; do not change
        self.tempScaling = 1.0

        # load parameters and data
        self.load_data()

        # load templates
        self.load_templates()
        
        # make bsplines
        self.make_bsplines_parallel()
        
        # run residual computation
        self.subtract_step()

    def load_data(self):
       
        # 
        self.n_chan = self.CONFIG.recordings.n_channels
                                
        # Cat: TODO: read buffer from CONFIG
        self.reader.buffer = 200
            
        # load spike train
        self.spike_train = np.load(self.fname_spike_train)

        # time offset loading
        self.time_offsets = np.load(self.fname_shifts)
        
        # scale fit
        self.scales = np.load(self.fname_scales)
                
        # Cat: TODO is this being used anymore?
        self.waveform_len = (self.CONFIG.recordings.sampling_rate/1000*
                             self.CONFIG.recordings.spike_size_ms)

        # compute chunks of data to be processed
        # self.n_sec = self.CONFIG.resources.n_sec_chunk_gpu
        # self.chunk_len = self.n_sec*self.CONFIG.recordings.sampling_rate
        # self.rec_len = self.CONFIG.rec_len

        print ("# of chunks: ", self.reader.n_batches)


    def load_templates(self):
        #print ("  loading templates...", self.fname_templates)
        # load templates
        self.temps = np.load(self.fname_templates).transpose(2,1,0).astype('float32')
        #print ("loaded temps:", self.temps.shape)
        self.temps_gpu = torch.from_numpy(self.temps.transpose(2,0,1)).float().cuda()
        #self.temps_gpu = torch.from_numpy(self.temps).long().cuda()


        # note shrink tempaltes by factor of 2 as there is a 2x hardcoded in CPP function
        self.template_vals=[]
        for k in range(self.temps.shape[2]):
            self.template_vals.append(torch.from_numpy(0.5*self.temps[:,:,k]).cuda())

        #print (" example filter (n_chans, n_times): ", self.template_vals[0].shape)

        # compute vis units 
        self.template_inds = []
        self.vis_units = np.arange(self.n_chan)
        for k in range(len(self.template_vals)):
            self.template_inds.append(torch.from_numpy(self.vis_units).cuda())

        
        #self.vis_units_parallel = []
        #for k in range(self.temps.shape[2]):
        #    self.vis_units_parallel
    
    
    def make_bsplines(self):
        #print ("  making bsplines... (TODO Parallelize)")
        # make template objects
        self.templates = deconv.BatchedTemplates(
                        [deconv.Template(vals, inds) for vals, inds in 
                            zip(self.template_vals, self.template_inds)])

        # make bspline objects
        def fit_spline(curve, knots=None,  prepad=0, postpad=0, order=3):
            if knots is None:
                knots = np.arange(len(curve) + prepad + postpad)
            return splrep(knots, np.pad(curve, (prepad, postpad), mode='symmetric'), k=order)

        def transform_template(template, knots=None, prepad=7, postpad=3, order=3):
            if knots is None:
                knots = np.arange(len(template.data[0]) + prepad + postpad)
            splines = [
                fit_spline(curve, knots=knots, prepad=prepad, postpad=postpad, order=order) 
                for curve in template.data.cpu().numpy()
            ]
            coefficients = np.array([spline[1][prepad-1:-1*(postpad+1)] for spline in splines], dtype='float32')
            return deconv.Template(torch.from_numpy(coefficients).cuda(), template.indices)

        # make bspline coefficients
        self.coefficients = deconv.BatchedTemplates(
                    [transform_template(template) for template in self.templates])


    # TODO IMPLEMENT THIS!
    def make_bsplines_parallel(self):
        
        #print (self.temps_gpu.shape, len(self.template_inds), self.template_inds[0].shape)
        self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temps_gpu, self.template_inds)])

        #print ("  making template bsplines")
        #fname = os.path.join(data_dir,'voltage_bsplines_'+
        #          str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu) + '.npy')
        
        #if os.path.exists(fname)==False:
        if True:
            
            # Cat; TODO: don't need to pass tensor/cuda templates to parallel function
            #            - can just pass the raw cpu templates
            # multi-core bsplines
            if self.CONFIG.resources.multi_processing:
                templates_cpu = []
                for template in self.temp_temp_cpp:
                    templates_cpu.append(template.data.cpu().numpy())

                coefficients = parmap.map(transform_template_parallel, templates_cpu, 
                                            processes=self.CONFIG.resources.n_processors,
                                            pm_pbar=False)
            # single core
            else:
                coefficients = []
                for template in self.temp_temp_cpp:
                    template_cpu = template.data.cpu().numpy()
                    coefficients.append(transform_template_parallel(template_cpu))
            
            #np.save(fname, coefficients)
       # else:
         #   print ("  ... loading coefficients from disk")
        #    coefficients = np.load(fname)
        
        #print ("  ... moving coefficients to cuda objects")
        coefficients_cuda = []
        for p in range(len(coefficients)):
            coefficients_cuda.append(deconv.Template(torch.from_numpy(coefficients[p]).cuda(), self.temp_temp_cpp[p].indices))
            # print ('self.temp_temp_cpp[p].indices: ', self.temp_temp_cpp[p].indices)
            # print ("self.vis_units: ", self.vis_units[p])
            # coefficients_cuda.append(deconv.Template(torch.from_numpy(coefficients[p]).cuda(), self.vis_units[p]))
        
        self.coefficients = deconv.BatchedTemplates(coefficients_cuda)

        del self.temp_temp_cpp
        del coefficients_cuda
        del coefficients
        torch.cuda.empty_cache()


    def subtract_step(self):
        
        # loop over chunks and do work
        t0 = time.time()
        verbose = False
        debug = False
        
        residual_array = []
        self.reader.buffer = 1000

        # open residual file for appending on the fly
        f = open(self.fname_residual,'wb')

        batch_id=0
        #for chunk in tqdm(self.reader.idx_list):
        for chunk in tqdm(self.reader.idx_list):
            
            time_sec = (batch_id*self.CONFIG.resources.n_sec_chunk_gpu_deconv)
                          
           # print ((self.update_templates), 
           #     (((time_sec)%self.template_update_time)==0),
           #     (batch_id!=0))
            # updated templates options
            if ((self.update_templates) and 
                (((time_sec)%self.template_update_time)==0) and
                (batch_id!=0)):
                
                # Cat: TODO: note this function reads the average templates forward to
                #       correctly match what was computed during current window
                #       May wish to try other options
                
                self.fname_templates = os.path.join(os.path.split(self.data_dir)[0],
                                    'deconv','template_updates',
                                    'templates_'+
                                     str(self.CONFIG.deconvolution.template_update_time+
                                            batch_id*self.CONFIG.resources.n_sec_chunk_gpu_deconv)+
                                    'sec.npy')
                                    
                #print ("NEW TEMPLATES...")
                #print ("   reloading: ", self.fname_templates)
                # reload templates
                self.load_templates()
                self.make_bsplines_parallel()
           
            # load chunk starts and ends for indexing below
            chunk_start = chunk[0]
            chunk_end = chunk[1]
        
            # read and pad data with buffer
            # self.data_temp = self.reader.read_data_batch(batch_id, add_buffer=True)
            data_chunk = self.reader.read_data_batch(batch_id, add_buffer=True).T

            # transfer raw data to cuda
            objective = torch.from_numpy(data_chunk).cuda()
            if verbose: 
                print ("Input size: ",objective.shape, int(sys.getsizeof(objective)), "MB")

            # Cat: TODO: may wish to pre-compute spike indiexes in chunks using cpu-multiprocessing
            #            because this constant search is expensive;
            # index into spike train at chunks:
            # Cat: TODO: this may miss spikes that land exactly at time 61.
            idx = np.where(np.logical_and(self.spike_train[:,0]>=(chunk_start-self.waveform_len), 
                            self.spike_train[:,0]<=(chunk_end+self.waveform_len)))[0]
            if verbose: 
                print (" # idx of spikes in chunk ", idx.shape, idx)
            
            # offset time indices by added buffer above
            times_local = (self.spike_train[idx,0]+self.reader.buffer-chunk_start
                                                  -self.waveform_len//2)
            time_indices = torch.from_numpy(times_local).long().cuda()
            # spike_list.append(times_local+chunk_start)
            if verbose: 
                print ("spike times: ", time_indices.shape, time_indices)

            # select template ids
            templates_local = self.spike_train[idx,1]
            template_ids = torch.from_numpy(templates_local).long().cuda()
            # id_list.append(templates_local)
            if verbose: 
                print (" template ids: ", template_ids.shape, template_ids)

            # select superres alignment shifts
            time_offsets_local = self.time_offsets[idx]
            time_offsets_local = torch.from_numpy(time_offsets_local).float().cuda()
            
            # select superres alignment shifts
            scales_local = self.scales[idx]
            scales_local = torch.from_numpy(scales_local).float().cuda()



            # dummy values
            self.tempScaling_array = time_offsets_local*0.0+1.0

            if verbose: 
                print ("time offsets: ", time_offsets_local.shape, time_offsets_local)
            
            if verbose:
                t5 = time.time()
                
            if False:
                np.save('/home/cat/times.npy', time_indices.cpu().data.numpy())
                np.save('/home/cat/objective.npy', objective.cpu().data.numpy())
                np.save('/home/cat/template_ids.npy', template_ids.cpu().data.numpy())
                np.save('/home/cat/time_offsets_local.npy', time_offsets_local.cpu().data.numpy())
                
            # of of spikes to be subtracted per iteration
            # Cat: TODO: read this from CONFIG;
            # Cat: TODO this may crash if a single spike is left; 
            #       needs to be wrapped in a list
            chunk_size = 10000
            for chunk in range(0, time_indices.shape[0], chunk_size):
                print (time_indices[chunk:chunk+chunk_size])

                torch.cuda.synchronize()
                if time_indices[chunk:chunk+chunk_size].shape[0]==0:
                    # Add spikes back in;
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size][None],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size][None],
                                        self.coefficients,
                                        self.tempScalingscales_local[chunk:chunk+chunk_size][None])
                                        
                            
                else:
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size],
                                        self.coefficients,
                                        self.tempScaling*scales_local[chunk:chunk+chunk_size])
                                        
            torch.cuda.synchronize()

            if verbose:
                print ("subtraction time: ", time.time()-t5)

            temp_out = objective[:,self.reader.buffer:-self.reader.buffer].cpu().data.numpy().copy(order='F')
            f.write(temp_out.T)
            
            batch_id+=1
            #if batch_id > 3:
            #    break
        f.close()

        print ("Total residual time: ", time.time()-t0)
            

#*****************************************************************************
#*****************************************************************************
#*****************************************************************************

class RESIDUAL_DRIFT(object):
    
    def __init__(self,
                d_gpu
                ):
                  
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        
        self.logger = logging.getLogger(__name__)

        # fixed value for CUDA code; do not change
        self.tempScaling = 1.0

        # load data from d_gpu object
        self.CONFIG = d_gpu.CONFIG

        # 
        self.n_chan = self.CONFIG.recordings.n_channels
                
        # Cat: TODO is this being used anymore?
        self.waveform_len = (self.CONFIG.recordings.sampling_rate/1000*
                             self.CONFIG.recordings.spike_size_ms)

        # load deconvolution shifts, ids, spike times
        if len(d_gpu.neuron_array)>0: 
            ids = torch.cat(d_gpu.neuron_array)
            spike_array = torch.cat(d_gpu.spike_array)
            
            self.spike_train = torch.cat((ids[None], spike_array[None]),0).transpose(1,0)
            print ("self.spike_train: ", self.spike_train.shape)
            
            
            self.time_offsets = torch.cat(d_gpu.shift_list)
        else:
            print (" NO SPIKES TO DECONVOLVE ")
            return

        # data
        self.data = d_gpu.data

        # templates
        self.temps_gpu = d_gpu.temps

        # compute vis units/template inds
        self.template_inds = []
        self.vis_units = np.arange(self.n_chan)
        print ("d_gpu.temps: ", d_gpu.temps.shape)
        for k in range(d_gpu.temps.shape[2]):
            self.template_inds.append(torch.from_numpy(self.vis_units).cuda())

        # make bsplines
        self.coefficients = d_gpu.coefficients
        
        # run residual computation
        self.subtract_step_single_chunk()


    def subtract_step_single_chunk(self):
        
        # loop over chunks and do work
        t0 = time.time()
        verbose = False
        debug = False
        
        # read and pad data with buffer
        # transfer raw data to cuda
        objective = self.data #.transpose(1,0)

        # offset time indices by added buffer above
        time_indices = self.spike_train[:,0]+1000

        # select template ids
        template_ids = self.spike_train[:,1] 

        # select superres alignment shifts
        time_offsets_local = self.time_offsets

        # dummy values
        tempScaling_array = time_offsets_local*0.0+1.0

        print ("objective: ", objective.shape)
        print ("time_indices: ", time_indices.shape)
        print ("template_ids: ", template_ids.shape)
        print ("time_offsets_local: ", time_offsets_local.shape)
        print ("tempScaling_array: ", tempScaling_array.shape)

        # of of spikes to be subtracted per iteration
        # Cat: TODO: read this from CONFIG;
        # Cat: TODO this may crash if a single spike is left; 
        #       needs to be wrapped in a list
        chunk_size = 10000
        for chunk in range(0, time_indices.shape[0], chunk_size):
            torch.cuda.synchronize()
            if time_indices[chunk:chunk+chunk_size].shape[0]==0:
                                                                                           
                # Add spikes back in;
                deconv.subtract_splines(
                                    objective,
                                    time_indices[chunk:chunk+chunk_size][None],
                                    time_offsets_local[chunk:chunk+chunk_size],
                                    template_ids[chunk:chunk+chunk_size][None],
                                    self.coefficients,
                                    #self.tempScaling
                                    tempScaling_array
                                    )
                                                                
            else:      
                deconv.subtract_splines(
                                    objective,
                                    time_indices[chunk:chunk+chunk_size],
                                    time_offsets_local[chunk:chunk+chunk_size],
                                    template_ids[chunk:chunk+chunk_size],
                                    self.coefficients,
                                    #self.tempScaling
                                    tempScaling_array
                                    )                                        
                                        
            torch.cuda.synchronize()

        return objective
            
            

# ***************************************************************************************************
# ***************************************************************************************************
# ***************************************************************************************************
# ***************************************************************************************************
# Cat: TODO: this function is incorrect for computing residual in voltage space; 
#       - it's for residuals in objective function space
class RESIDUAL_GPU3(object):
    
    def __init__(self,
                reader,
                recordings_filename,
                recording_dtype,
                CONFIG,
                fname_shifts,
                fname_templates,
                output_directory,
                dtype_out,
                fname_out,
                fname_spike_train,
                update_templates=False):
        
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        
        self.logger = logging.getLogger(__name__)

        self.reader = reader
        self.recordings_filename = recordings_filename
        self.recording_dtype = recording_dtype
        self.CONFIG = CONFIG
        self.output_directory = output_directory
        self.dtype_out = dtype_out
        self.fname_out = fname_out
        self.fname_templates = fname_templates
        self.fname_shifts = fname_shifts
        self.fname_residual = os.path.join(self.output_directory,'residual.bin')
        self.fname_spike_train = fname_spike_train
        self.dir_deconv = os.path.join(os.path.split(self.output_directory)[0],'deconv')
        self.dir_bsplines = os.path.join(self.dir_deconv,'svd')
        
        # updated templates options
        self.update_templates = update_templates

        if self.update_templates:
            self.template_update_time = CONFIG.deconvolution.template_update_time
            self.fname_templates = os.path.join(os.path.join(self.dir_deconv,
                                'template_updates'),
                            'templates_'+str(self.template_update_time)+'sec.npy')
            #print ("starting templates :  ", self.fname_templates)

            print ("TODO: load precomputed drift template bsplines...")

        # fixed value for CUDA code; do not change
        self.tempScaling = 2.0
        
        # Cat: TODO: read from CONFIG
        self.vis_chan_thresh= 1.0
        
        # initialize chunk 
        self.chunk_id=0

        # templates being updated
        # Cat: TODO: this is hacky; to be read from CONFIG file
        self.dir_template_updates = os.path.join(self.dir_deconv,'template_updates')
        files = os.listdir(self.dir_template_updates)
        if len(files)>0:
            self.update_templates = True
        else:
            self.update_templates = False
        
        # load parameters and data
        self.load_data()

        # run residual computation
        self.subtract_step()
        

    def load_data(self):

        t0 = time.time()
        
        # 
        #self.data_dir = self.output_directory

        # 
        self.n_chan = self.CONFIG.recordings.n_channels
                                
        # Cat: TODO: read buffer from CONFIG
        self.reader.buffer = 200
            
        # load spike train
        self.spike_train = np.load(self.fname_spike_train)
        
        #
        self.waveform_len = (self.CONFIG.recordings.sampling_rate/1000*
                             self.CONFIG.recordings.spike_size_ms)

        # time offset loading
        self.time_offsets = np.load(self.fname_shifts)

        # set default to first chunk
        if True:
            self.load_temps()
            self.load_vis_units()
            self.load_temp_temp()
            self.initialize_cpp()
            self.templates_to_bsplines()


    def load_vis_units(self):
        # save vis_units for residual recomputation and other steps

        # if False:
            # fname = os.path.join(self.dir_bsplines,'vis_units_'+
                          # str((self.chunk_id+1)*self.CONFIG.deconvolution.template_update_time) + '_1.npy')
            
            # print ("  loading vis_units name: ", fname)
            # self.vis_units = np.load(fname, allow_pickle=True)
            # print ("  self.vis_units: ", self.vis_units.shape, self.vis_units[0])
        
        # else:
            # #self.vis_units = np.arange(self.n_chan)
            # self.all_chans = np.arange(45)
            # self.vis_units = []
            # for k in range(self.temps.shape[0]):
                # self.vis_units.append(torch.from_numpy(self.all_chans).cuda())
            
        self.vis_units=[]
        #self.N_CHAN, self.STIME, self.K = self.temps.shape
        for k in range(self.K):
            #self.vis_units_gpu.append(torch.FloatTensor(np.where(self.vis_units[k])[0]).long().cuda())
            self.vis_units.append(torch.FloatTensor(np.arange(self.N_CHAN)).long().cuda())
        #self.vis_units = self.vis_units_gpu
        
    
    def load_temp_temp(self):
        print ("  loading temp_temp")
        fname = os.path.join(self.dir_bsplines,'temp_temp_sparse_svd_'+
              str((self.chunk_id+1)*self.CONFIG.deconvolution.template_update_time) + '_1.npy')
        
        self.temp_temp = np.load(fname, allow_pickle=True)
        print (" temp_temp svd: fname ",  fname)
        print (" temp_temp: ",  self.temp_temp.shape)
            

    def initialize_cpp(self):

        # make a list of pairwise batched temp_temp and their vis_units
        # Cat: TODO: this isn't really required any longer;
        #            - the only thing required from this in parallel bsplines function is
        #              self.temp_temp_cpp.indices - is self.vis_units
        #              
        print (self.temp_temp.shape, self.temp_temp[0].shape)
        print (len(self.vis_units), self.vis_units[0].shape[0])
        print (self.fname_templates)
        self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])

        print ("self.temp_temp_cpp[0]: ", self.temp_temp_cpp[0])
        print ("self.temp_temp[0]: ", self.temp_temp[0])
        print ("self.vis_units[0]: ", self.vis_units[0])


    def load_temps(self):
        ''' Load templates and set parameters
        '''
        
        # load templates
        #print ("Loading template: ", self.fname_templates)
        self.temps = np.load(self.fname_templates, allow_pickle=True).transpose(2,1,0)
        self.N_CHAN, self.STIME, self.K = self.temps.shape
        print ("self.temps fname; ", self.fname_templates)

    
    def visible_chans(self):
        #if self.vis_chan is None:
        a = np.max(self.temps, axis=1) - np.min(self.temps, 1)
        
        # Cat: TODO: must read visible channel/unit threshold from file;
        self.vis_chan = a > self.vis_chan_thresh

        a_self = self.temps.ptp(1).argmax(0)
        for k in range(a_self.shape[0]):
            self.vis_chan[a_self[k],k]=True

        # fname = os.path.join(self.svd_dir,'vis_chans.npy')
        # np.save(fname, self.vis_chan)


    def template_overlaps(self):
        """Find pairwise units that have overlap between."""
        vis = self.vis_chan.T
        self.unit_overlap = np.sum(
            np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
        self.unit_overlap = self.unit_overlap > 0
        self.vis_units = self.unit_overlap

                    
    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        for k in range(self.temps.shape[2]):
            zero_chans = np.where(self.vis_chan[:,k]==0)[0]
            self.temps[zero_chans,:,k]=0.
          
    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""

        print ("   making SVD data... (todo: move to GPU)")
        ## compute everythign using SVD
        # Cat: TODO: is this necessary?  
        #      can just overwrite all the svd stuff every template update
        if self.update_templates_backwards:
            fname = os.path.join(self.svd_dir,'templates_svd_'+
                      str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npz')
        else:
            fname = os.path.join(self.svd_dir,'templates_svd_'+
                      str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npz')
            
        if os.path.exists(fname)==False:
            #print ("self.temps: ", self.temps.shape)
            #np.save("/home/cat/temps.npy", self.temps)
            
            self.temporal, self.singular, self.spatial = np.linalg.svd(
                np.transpose(np.flipud(np.transpose(self.temps,(1,0,2))),(2, 0, 1)))
            
            # Keep only the strongest components
            self.temporal = self.temporal[:, :, :self.RANK]
            self.singular = self.singular[:, :self.RANK]
            self.spatial = self.spatial[:, :self.RANK, :]

            # Upsample the temporal components of the SVD
            # in effect, upsampling the reconstruction of the
            # templates.
            
            # Cat: TODO: No upsampling is needed; to remove temporal_up from code
            self.temporal_up = self.temporal
           
            np.savez(fname, temporal=self.temporal, singular=self.singular, 
                     spatial=self.spatial, temporal_up=self.temporal_up)
            
        else:
            # load data for for temp_temp computation
            print ("Loading SVD data from disk...")
            data = np.load(fname, allow_pickle=True)
            self.temporal_up = data['temporal_up']
            self.temporal = data['temporal']
            self.singular = data['singular']
            self.spatial = data['spatial']     
            
    def compute_temp_temp_svd(self):

        print ("  making temp_temp filters (todo: move to GPU)")
        
        # Cat: TODO: this might break if no template update sparse svd is saved
        fname = os.path.join(self.dir_bsplines,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.deconvolution.template_update_time) + '_1.npy')
        
        print ("sparse svd: ", fname)
        print (".... loading temp-temp from disk")
        
        if False:
            self.temp_temp = np.load(fname, allow_pickle=True)
        
        else:
            
            self.load_temps()
            self.visible_chans()
            self.template_overlaps()
            self.spatially_mask_templates()
            self.compress_templates()
            
            self.up_up_map = None
            deconv_dir = ''

            # 
            units = np.arange(self.temps.shape[2])
            
            self.n_time = self.CONFIG.spike_size
            self.update_templates_backwards = None
            self.svd_dir = None
            self.chunk_id = None


            # Cat: TODO: work on multi CPU and GPU versions
            if self.CONFIG.resources.multi_processing:
                units_split = np.array_split(units, self.CONFIG.resources.n_processors)
                self.temp_temp = parmap.map(parallel_conv_filter2, 
                                              units_split, 
                                              self.n_time,
                                              self.up_up_map,
                                              deconv_dir,
                                              self.update_templates_backwards,
                                              self.svd_dir,
                                              self.chunk_id,
                                              self.CONFIG.resources.n_sec_chunk_gpu_deconv,
                                              self.vis_chan,
                                              self.unit_overlap,
                                              self.RANK,
                                              self.temporal,
                                              self.singular,
                                              self.spatial,
                                              self.temporal_up,
                                              processes=self.CONFIG.resources.n_processors,
                                              pm_pbar=False)

    
    def templates_to_bsplines(self):

        print ("  making template bsplines")
        fname = os.path.join(self.dir_bsplines,'bsplines_'+
                  str((self.chunk_id+1)*self.CONFIG.deconvolution.template_update_time) + '.npy')
        print ("Bspline file: ", fname)
        
        if os.path.exists(fname)==False:
        #if True:
            
            # Cat; TODO: don't need to pass tensor/cuda templates to parallel function
            #            - can just pass the raw cpu templates
            # multi-core bsplines
            if self.CONFIG.resources.multi_processing:
                templates_cpu = []
                for template in self.temp_temp_cpp:
                    templates_cpu.append(template.data.cpu().numpy())

                import parmap
                coefficients = parmap.map(transform_template_parallel, templates_cpu, 
                                            processes=self.CONFIG.resources.n_processors,
                                            pm_pbar=False)
            # single core
            else:
                coefficients = []
                for template in self.temp_temp_cpp:
                    template_cpu = template.data.cpu().numpy()
                    coefficients.append(transform_template_parallel(template_cpu))
            
            #np.save(fname, coefficients)
        else:
            print ("  ... loading coefficients from disk")
            coefficients = np.load(fname, allow_pickle=True)

        # # 
        print (" recomputed coefficients: ", coefficients[0].shape)
        print (" recomputed coefficients: ", coefficients[0])

        coefficients = np.load(fname)
        print (" loaded coefficients: ", coefficients[0].shape)
        print (" loaded coefficients: ", coefficients[0])
        

        print ("  ... moving coefficients to cuda objects")
        coefficients_cuda = []
        for p in range(len(coefficients)):
            coefficients_cuda.append(deconv.Template(torch.from_numpy(coefficients[p]).cuda(), self.temp_temp_cpp[p].indices))

        self.coefficients = deconv.BatchedTemplates(coefficients_cuda)

        del self.temp_temp
        del self.temp_temp_cpp
        del coefficients_cuda
        del coefficients
        torch.cuda.empty_cache()
    
           
    def subtract_step(self):
        
        # loop over chunks and do work
        t0 = time.time()
        verbose = False
        debug = False
        
        residual_array = []
        self.reader.buffer = 200

        # open residual file for appending on the fly
        f = open(self.fname_residual,'wb')

        #self.chunk_id =0
        batch_ctr = 0
        batch_id = 0
        print (" STARTING RESIDUAL COMPUTATION...")
        for chunk in tqdm(self.reader.idx_list):
            
            time_sec = (batch_id*self.CONFIG.resources.n_sec_chunk_gpu_deconv)
                            
            #print ("time_sec: ", time_sec)
            
            # updated templates options
            if ((self.update_templates) and 
                (((time_sec)%self.template_update_time)==0) and
                (batch_id!=0)):
                
                #print ("UPDATING TEMPLATES, time_sec: ", time_sec)
                
                self.chunk_id +=1

                # Cat: TODO: note this function reads the average templates +60sec to
                #       correctly match what was computed during current window
                #       May wish to try other options
                self.fname_templates = os.path.join(os.path.split(self.data_dir)[0],
                        'deconv','template_updates',
                        'templates_'+str(time_sec+self.template_update_time)+'sec.npy')
                print ("updating templates from:  ", self.fname_templates)
                # 
               #print (" updating bsplines...")
                if False:
                    self.load_templates()
                    self.make_bsplines()
                else:
                    self.load_vis_units()
                    self.load_temp_temp()
                    self.initialize_cpp()
                    self.templates_to_bsplines()                    
           
            # load chunk starts and ends for indexing below
            chunk_start = chunk[0]
            chunk_end = chunk[1]
        
            # read and pad data with buffer
            # self.data_temp = self.reader.read_data_batch(batch_id, add_buffer=True)
            data_chunk = self.reader.read_data_batch(batch_id, add_buffer=True).T

            # transfer raw data to cuda
            objective = torch.from_numpy(data_chunk).cuda()
            if verbose: 
                print ("Input size: ",objective.shape, int(sys.getsizeof(objective)), "MB")

            # Cat: TODO: may wish to pre-compute spike indiexes in chunks using cpu-multiprocessing
            #            because this constant search is expensive;
            # index into spike train at chunks:
            # Cat: TODO: this may miss spikes that land exactly at time 61.
            idx = np.where(np.logical_and(self.spike_train[:,0]>=(chunk_start-self.waveform_len), 
                            self.spike_train[:,0]<=(chunk_end+self.waveform_len)))[0]
            if verbose: 
                print (" # idx of spikes in chunk ", idx.shape, idx)
            
            # offset time indices by added buffer above
            times_local = (self.spike_train[idx,0]+self.reader.buffer-chunk_start
                                                  -self.waveform_len//2)
            time_indices = torch.from_numpy(times_local).long().cuda()
            # spike_list.append(times_local+chunk_start)
            if verbose: 
                print ("spike times/time_indices: ", time_indices.shape, time_indices)

            # select template ids
            templates_local = self.spike_train[idx,1]
            template_ids = torch.from_numpy(templates_local).long().cuda()
            if verbose: 
                print (" template ids: ", template_ids.shape, template_ids)

            # select superres alignment shifts
            time_offsets_local = self.time_offsets[idx]
            time_offsets_local = torch.from_numpy(time_offsets_local).float().cuda()

            if verbose: 
                print ("time offsets: ", time_offsets_local.shape, time_offsets_local)
            
            if verbose:
                t5 = time.time()
                
            # of of spikes to be subtracted per iteration
            # Cat: TODO: read this from CONFIG;
            # Cat: TODO this may crash if a single spike is left; 
            #       needs to be wrapped in a list
            if True:
                chunk_size = 10000
                for chunk in range(0, time_indices.shape[0], chunk_size):
                    #print ("Chunk: ", chunk)
                    torch.cuda.synchronize()
                    if time_indices[chunk:chunk+chunk_size].shape[0]==0:
                        deconv.subtract_splines(
                                            objective,
                                            time_indices[chunk:chunk+chunk_size][None],
                                            time_offsets_local[chunk:chunk+chunk_size],
                                            template_ids[chunk:chunk+chunk_size][None],
                                            self.coefficients,
                                            self.tempScaling)
                    else:      
                        #print (" multi-spike subtraction: ", chunk)

                        deconv.subtract_splines(
                                            objective,
                                            time_indices[chunk:chunk+chunk_size],
                                            time_offsets_local[chunk:chunk+chunk_size],
                                            template_ids[chunk:chunk+chunk_size],
                                            self.coefficients,
                                            self.tempScaling)
                                            

                                            
            # do unit-wise subtraction; thread safe
            else:
                for unit in np.unique(template_ids.cpu().data.numpy()):
                    #print ('unit: ', unit)
                    torch.cuda.synchronize()
                    
                    idx_unit = np.where(template_ids.cpu().data.numpy()==unit)[0]
                    if idx_unit.shape[0]==1:
                        deconv.subtract_splines(
                                            objective,
                                            time_indices[idx_unit][None],
                                            time_offsets_local[idx_unit],
                                            template_ids[idx_unit][None],
                                            self.coefficients,
                                            self.tempScaling)
                    elif idx_unit.shape[0]>1:
                        deconv.subtract_splines(
                                            objective,
                                            time_indices[idx_unit],
                                            time_offsets_local[idx_unit],
                                            template_ids[idx_unit],
                                            self.coefficients,
                                            self.tempScaling)
                                            
            torch.cuda.synchronize()

            if verbose:
                print ("subtraction time: ", time.time()-t5)

            temp_out = objective[:,self.reader.buffer:-self.reader.buffer].cpu().data.numpy().copy(order='F')
            f.write(temp_out.T)
            
            batch_id+=1
            #if batch_id > 3:
            #    break
        f.close()

        print ("Total residual time: ", time.time()-t0)
            
