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
        self.data_dir = self.output_directory

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
            self.load_templates()
            self.make_bsplines()
        else:
            self.templates_to_bsplines()


    def load_templates(self):
        #print ("  loading templates...")
        # load templates
        self.temps = np.load(self.fname_templates).transpose(2,1,0).astype('float32')
        #print ("loaded temps:", self.temps.shape)

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

        #print(" example vis unit ", self.template_inds[0][:10])    
        
        # time offset loading
        self.time_offsets = np.load(self.fname_shifts)
        #print ("time offsets: ", self.time_offsets)

    
    def make_bsplines(self):
        #print ("  making bsplines...")
        # make template objects
        self.templates = deconv.BatchedTemplates(
                        [deconv.Template(vals, inds) for vals, inds in 
                            zip(self.template_vals, self.template_inds)])

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
                    
        #print ("self.coefficients: ", self.coefficients)
        #print ("self.coefficients[0].data: ", self.coefficients[0].data)
        
    def templates_to_bsplines(self):

        # load templates (as cuda/tensor objects)
        if self.update_templates:
            fname = os.path.join(self.dir_bsplines,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
        else:
            fname = os.path.join(self.dir_bsplines,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
        self.temp_temp = np.load(fname, allow_pickle=True)

        # load visible units for bspline/cuda objects 
        if self.update_templates:
            fname = os.path.join(self.dir_bsplines,'vis_units_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npz.npy')
        else:
            fname = os.path.join(self.dir_bsplines,'vis_units_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
        self.vis_units = np.load(fname)
        
        self.vis_units_gpu=[]
        for k in range(self.vis_units.shape[1]):
            self.vis_units_gpu.append(torch.FloatTensor(np.where(self.vis_units[k])[0]).long().cuda())
        self.vis_units = self.vis_units_gpu        
        
        # initialize template objects on cuda
        self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])
        
        # loading saved coefficients
        fname = os.path.join(self.dir_bsplines,'bsplines_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
        coefficients = np.load(fname)

        print ("  ... moving coefficients to cuda objects")
        coefficients_cuda = []
        for p in range(len(coefficients)):
            coefficients_cuda.append(deconv.Template(torch.from_numpy(coefficients[p]).cuda(), self.temp_temp_cpp[p].indices))
        
        self.coefficients = deconv.BatchedTemplates(coefficients_cuda)
        #print ("self.coefficients[0]: ", self.coefficients[0].shape)
            
        # print ("self.coefficients: ", self.coefficients)
        # print ("self.coefficients[0].data: ", self.coefficients[0].data)
                
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
                if True:
                    self.load_templates()
                    self.make_bsplines()
                else:
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
                    print ('unit: ', unit)
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
            
   
class RESIDUAL_GPU2(object):
    
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
                fname_spike_train):
        
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
        
        
        # updated templates options
        # Cat: TODO read from CONFIG
        self.update_templates = False
        
        # Cat: TODO read from CONFIG File
        self.template_update_time = 60

        # fixed value for CUDA code; do not change
        self.tempScaling = 2.0

        # load parameters and data
        self.load_data()

        # run residual computation
        self.subtract_step()

    def load_data(self):

        t0 = time.time()
        
        # 
        self.data_dir = self.output_directory

        # 
        self.n_chan = self.CONFIG.recordings.n_channels
                                
        # Cat: TODO: read buffer from CONFIG
        self.reader.buffer = 200
            
        # load spike train
        self.spike_train = np.load(self.fname_spike_train)
        
        # Cat: TODO is this being used anymore?
        self.waveform_len = (self.CONFIG.recordings.sampling_rate/1000*
                             self.CONFIG.recordings.spike_size_ms)

        # compute chunks of data to be processed
        # self.n_sec = self.CONFIG.resources.n_sec_chunk_gpu
        # self.chunk_len = self.n_sec*self.CONFIG.recordings.sampling_rate
        # self.rec_len = self.CONFIG.rec_len

        print ("# of chunks: ", self.reader.n_batches)

        # load templates
        self.load_templates()
        
        # make bsplines
        self.make_bsplines()
        

    def load_templates(self):
        print ("  loading templates...")
        # load templates
        self.temps = np.load(self.fname_templates).transpose(2,1,0).astype('float32')
        #print ("loaded temps:", self.temps.shape)

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

        #print(" example vis unit ", self.template_inds[0][:10])    
        
        # time offset loading
        self.time_offsets = np.load(self.fname_shifts)
        #print ("time offsets: ", self.time_offsets)

    
    def make_bsplines(self):
        print ("  making bsplines...")
        # make template objects
        self.templates = deconv.BatchedTemplates(
                        [deconv.Template(vals, inds) for vals, inds in 
                            zip(self.template_vals, self.template_inds)])

        # make bspline objects
        #print (" # templates: ", len(self.templates))
        #print (" example template size: ", self.templates[0].data.shape)

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


    def subtract_step(self):
        
        # loop over chunks and do work
        t0 = time.time()
        verbose = False
        debug = False
        
        residual_array = []
        self.reader.buffer = 200

        # open residual file for appending on the fly
        f = open(self.fname_residual,'wb')

        batch_id=0
        #for chunk in tqdm(self.reader.idx_list):
        for chunk in tqdm(self.reader.idx_list):
            
            time_sec = (batch_id*self.CONFIG.resources.n_sec_chunk_gpu_deconv)
                            
            # updated templates options
            if ((self.update_templates) and 
                ((time_sec)%self.template_update_time)==0):
                
                # Cat: TODO: note this function reads the average templates +60sec to
                #       correctly match what was computed during current window
                #       May wish to try other options
                self.fname_templates = os.path.join(os.path.split(self.data_dir)[0],
                        'deconv','template_updates',
                        'templates_'+str(time_sec+self.template_update_time)+'sec.npy')
                #print ("reloading: ", self.fname_templates)
                # reload templates
                self.load_templates()
                
                self.make_bsplines()
           
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

            if verbose: 
                print ("time offsets: ", time_offsets_local.shape, time_offsets_local)
            
            if verbose:
                t5 = time.time()
                
            # of of spikes to be subtracted per iteration
            # Cat: TODO: read this from CONFIG;
            # Cat: TODO this may crash if a single spike is left; 
            #       needs to be wrapped in a list
            chunk_size = 10000
            for chunk in range(0, time_indices.shape[0], chunk_size):
                torch.cuda.synchronize()
                if time_indices[chunk:chunk+chunk_size].shape[0]==0:
                    # deconv.subtract_splines(
                                        # objective,
                                        # time_indices[chunk:chunk+chunk_size][None],
                                        # time_offsets_local[chunk:chunk+chunk_size],
                                        # template_ids[chunk:chunk+chunk_size][None],
                                        # self.coefficients)
                                                                                               
                    # Add spikes back in;
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size][None],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size][None],
                                        self.coefficients,
                                        self.tempScaling)
                                        
                            
                else:      
                    # deconv.subtract_splines(
                                        # objective,
                                        # time_indices[chunk:chunk+chunk_size],
                                        # time_offsets_local[chunk:chunk+chunk_size],
                                        # template_ids[chunk:chunk+chunk_size],
                                        # self.coefficients)
                                        
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size],
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
            
