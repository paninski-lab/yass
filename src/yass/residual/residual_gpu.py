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


class RESIDUAL_GPU(object):
    
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
        
        # load parameters and data
        self.load_data()

        # run residual computation
        self.subtract_step()

    def load_data(self):

        t0 = time.time()
        
        # 
        data_dir = self.output_directory

        #
        rec_len = self.CONFIG.resources.n_sec_chunk_gpu

        # Cat: TODO: read buffer from disk
        self.buffer = 200

        # 
        n_chan = self.CONFIG.recordings.n_channels

        # # read raw voltage data
        # print ("  reading binary data into memory (n_chan, n_times)")
        # self.data_temp = np.fromfile(os.path.join(self.CONFIG.data.root_folder,'tmp',
                                    # 'preprocess','standardized.bin'),
                                    # 'float32').reshape(-1,n_chan).T
        # print ("  data shape: ", self.data_temp.shape)

        # batch_ids = []
        # fnames_seg = []
        # for batch_id in range(self.reader.n_batches):
            # batch_ids.append(batch_id)
                             
        self.reader.buffer = 200
            
        # load spike train
        self.spike_train = np.load(self.fname_spike_train)
        #print ("  spike train loaded: ", self.spike_train)
        
        
        # subtract hafl spike width again as gpu_deconv expects beginning not middle of waveform
        #self.spike_train[:,0] = self.spike_train[:,0]-(self.CONFIG.recordings.sampling_rate/1000*
        #                self.CONFIG.recordings.spike_size_ms)//2
                        
        #self.spike_train[:,0] = self.spike_train[:,0]-(self.CONFIG.recordings.sampling_rate/1000*
        #                self.CONFIG.recordings.spike_size_ms)//2

        self.waveform_len = self.CONFIG.spike_size

        # compute chunks of data to be processed
        n_sec = self.CONFIG.resources.n_sec_chunk_gpu
        chunk_len = n_sec*self.CONFIG.recordings.sampling_rate
        rec_len = self.CONFIG.rec_len

        # make list of chunks to loop over
        # self.chunks = []
        # ctr = 0 
        # while True:
            # if ctr*chunk_len >= rec_len:
                # break
            # self.chunks.append([ctr*chunk_len, ctr*chunk_len+chunk_len])
            # ctr+=1
        print ("# of chunks: ", self.reader.n_batches)
            
        # load templates
        print ("TEMPLATE NAMES: ", self.fname_templates)
        temps = np.load(self.fname_templates).transpose(2,1,0).astype('float32')
        print ("loaded temps:", temps.shape)

        # note shrink tempaltes by factor of 2 as there is a 2x multiplication inside cuda bspline function
        template_vals=[]
        for k in range(temps.shape[2]):
            template_vals.append(torch.from_numpy(0.5*temps[:,:,k]).cuda())
            #template_vals.append(torch.from_numpy(0.25*temps[:,:,k]).cuda())

        print (" example filter (n_chans, n_times): ", template_vals[0].shape)
        print (template_vals[0].shape)
        print ("len temp_vals: ", len(template_vals))

        # compute vis units 
        template_inds = []
        #vis_units = np.arange(len(template_vals))
        vis_units = np.arange(n_chan)
        for k in range(len(template_vals)):
            template_inds.append(torch.from_numpy(vis_units).cuda())

        print(" example vis unit ", template_inds[0][:10])    
        print (" # template_inds: ", len(template_inds))
        print (type(template_inds[0]))
        # example vis unit  tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='cuda:0')

        # time offset loading
        self.time_offsets = np.load(self.fname_shifts)
        print ("time offsets: ", self.time_offsets)

        # make template objects
        templates = deconv.BatchedTemplates(
            [deconv.Template(vals, inds) for vals, inds in zip(template_vals, template_inds)]
        )

        # make bspline objects
        print (" # templates: ", len(templates))
        print (" example template size: ", templates[0].data.shape)

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
        print ("  making bspline coefficients")
        self.coefficients = deconv.BatchedTemplates([transform_template(template) for template in templates])
        print (" # of coefficients: ", len(self.coefficients))
        #print (" example coefficient shape: ", self.coefficients[0].data.shape)

                             
        print ("  PRELOADING COMPLETED: ", np.round(time.time()-t0,2),"sec")


    def subtract_step(self):
        
        # loop over chunks and do work
        t0 = time.time()
        verbose = False
        debug = False
        
        residual_array = []
        self.reader.buffer=200
        
        f = open(self.fname_residual,'wb')
        
        pad_chunk = np.zeros((self.CONFIG.recordings.n_channels,self.buffer),'float32')

        ctr=0
        spike_list = []
        id_list = []
        for chunk in tqdm(self.reader.idx_list):
            tlocal = time.time()
            chunk_start = chunk[0]
            chunk_end = chunk[1]
                
            # pad data with buffer
            batch_id = ctr
            #self.data_temp = self.reader.read_data_batch(batch_id, add_buffer=True)
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
            times_local = (self.spike_train[idx,0]+self.buffer-chunk_start
                                                  -self.waveform_len//2)
            time_indices = torch.from_numpy(times_local).long().cuda()
            spike_list.append(times_local+chunk_start)
            if verbose: 
                print ("spike times: ", time_indices.shape, time_indices)

            # select template ids
            templates_local = self.spike_train[idx,1]
            template_ids = torch.from_numpy(templates_local).long().cuda()
            id_list.append(templates_local)
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
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size][None],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size][None],
                                        self.coefficients)
                else:      
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size],
                                        self.coefficients)
            torch.cuda.synchronize()

            if verbose:
                print ("subtraction time: ", time.time()-t5)
            
            #tfin = time.time()
            #if ctr%10==0:
            #    print (" chunk:", ctr+1, "/", self.reader.n_batches, 
            #            chunk, time.time() - tlocal)
            temp_out = objective[:,self.buffer:-self.buffer].cpu().data.numpy().copy(order='F')
            f.write(temp_out.T)
            
            ctr+=1
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
            
            time_sec = (batch_id*self.CONFIG.resources.n_sec_chunk_gpu)
                            
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
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size][None],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size][None],
                                        self.coefficients)
                else:      
                    deconv.subtract_splines(
                                        objective,
                                        time_indices[chunk:chunk+chunk_size],
                                        time_offsets_local[chunk:chunk+chunk_size],
                                        template_ids[chunk:chunk+chunk_size],
                                        self.coefficients)
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
            
   
