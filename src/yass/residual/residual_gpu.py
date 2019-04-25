import os
import logging
import numpy as np
import time
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

        print ("  current deconv requires loading all data in memory (TODO switch to chunk reading)")
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

        batch_ids = []
        fnames_seg = []
        for batch_id in range(self.reader.n_batches):
            batch_ids.append(batch_id)
                             
        self.reader.buffer = 0
            
        # load spike train
        self.spike_train = np.load(self.fname_spike_train)
        print ("  spike train loaded: ", self.spike_train)
        
        idx = self.spike_train[:,0].argsort()
        print ("  spike train loaded sorted: ", self.spike_train[idx][:10],
                '...', self.spike_train[idx][-10:])
        
        # subtract hafl spike width again as gpu_deconv expects beginning not middle of waveform
        # self.spike_train[:,0] = self.spike_train[:,0]-((
                        # self.CONFIG.recordings.sampling_rate/1000*
                        # self.CONFIG.recordings.spike_size_ms)*3)//2
        self.spike_train[:,0] = self.spike_train[:,0]-(self.CONFIG.recordings.sampling_rate/1000*
                        self.CONFIG.recordings.spike_size_ms)//2

        # compute chunks of data to be processed
        n_sec = self.CONFIG.resources.n_sec_chunk_gpu
        chunk_len = n_sec*self.CONFIG.recordings.sampling_rate
        #rec_len = self.data_temp.shape[1]
        rec_len = self.CONFIG.rec_len

        # make list of chunks to loop over
        self.chunks = []
        ctr = 0 
        while True:
            if ctr*chunk_len >= rec_len:
                break
            self.chunks.append([ctr*chunk_len, ctr*chunk_len+chunk_len])
            ctr+=1
        print ("# of chunks: ", len(self.chunks))
            
        # load templates
        temps = np.load(self.fname_templates).transpose(2,1,0).astype('float32')
        print ("loaded temps:", temps.shape)

        # note shrink tempaltes by factor of 2 as there is a 2x multiplication inside cuda bspline function
        template_vals=[]
        for k in range(temps.shape[2]):
            template_vals.append(torch.from_numpy(0.5*temps[:,:,k]).cuda())

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
        verbose = True
        debug = False
        
        residual_array = []
        
        f = open(self.fname_residual,'wb')
        
        pad_chunk = np.zeros((self.CONFIG.recordings.n_channels,self.buffer),'float32')
        #for ctr, chunk in enumerate(chunks):
        for ctr, chunk in enumerate(self.chunks):
            tlocal = time.time()
            chunk_start = chunk[0]
            chunk_end = chunk[1]
        
            # get relevantspike times
            batch_id = ctr
            self.data_temp = self.reader.read_data_batch(batch_id, add_buffer=True)
            
            # clip data
            #data_chunk = self.data_temp[:, chunk_start:chunk_end]
            data_chunk = self.data_temp.T
            
           
            # pad data with buffer
            data_chunk = np.hstack((pad_chunk, np.hstack((data_chunk, pad_chunk))))
            if verbose:
                np.save('/home/cat/data_chunk.npy', data_chunk)
            
            objective = torch.from_numpy(data_chunk).cuda()
            print (" Ojbective: ", objective.shape)
            if verbose: 
                print ("Input size: ",objective.shape, int(sys.getsizeof(objective)), "MB")

            # Cat: TODO: may wish to pre-compute spike indiexes in chunks using cpu-multiprocessing
            # index into spike train at chunks:
            idx = np.where(np.logical_and(self.spike_train[:,0]>=chunk_start, 
                            self.spike_train[:,0]<chunk_end))[0]
            if verbose: 
                print (" # idx of spikes in chunk ", idx.shape, idx)
            
            # offset time indices by added buffer above
            times_local = self.spike_train[idx,0]+self.buffer-chunk_start
            idx_keep = np.where(times_local!=200)[0]
            times_local = times_local[idx_keep]

            # add more spike times
            if debug:
                times_local = np.concatenate((times_local, [5000,5500,5000,5500,5000,5500,5000,5500]))

            time_indices = torch.from_numpy(times_local).long().cuda()

            if verbose: 
                print ("spike times: ", time_indices.shape, time_indices)
                np.save('/home/cat/time_indices.npy', time_indices.cpu().data.numpy())

            # select template ids
            templates_local = self.spike_train[idx,1]
            if debug:
                templates_local = np.concatenate((templates_local, [3,483, 483,513,613,716,716,717]))
            
            template_ids = templates_local[idx_keep]

            template_ids = torch.from_numpy(template_ids).long().cuda()
            if verbose: 
                print (" template ids: ", template_ids.shape, template_ids)
                np.save('/home/cat/template_ids.npy', template_ids.cpu().data.numpy())
                

            # select superres alignment shifts
            time_offsets_local = self.time_offsets[idx]
            time_offsets_local = time_offsets_local[idx_keep]

            if debug:
                time_offsets_local = np.concatenate((time_offsets_local, [0.,0.,0.,0.,0.,0.,0.,0.]))
            
            #time_offsets_local = torch.from_numpy(time_offsets_local).cuda()
            time_offsets_local = torch.from_numpy(time_offsets_local).float().cuda()
            if verbose: 
                np.save('/home/cat/time_offsets.npy', time_offsets_local.cpu().data.numpy())
                print ("time offsets: ", time_offsets_local.shape, time_offsets_local)
            
            #objective_copy = objective.clone()
            # Cat; TODO deleting time indices offset; unclear why required
            # Cat: TODO: offset should be loaded from CONFIG data.
            if verbose:
                t5 = time.time()
            torch.cuda.synchronize()
            deconv.subtract_splines(objective,
                                    time_indices,
                                    time_offsets_local,
                                    template_ids,
                                    self.coefficients)
            torch.cuda.synchronize()
            
            if verbose:
                print ("subtraction time: ", time.time()-t5)
            
            tfin = time.time()
            if ctr%10==0:
                print (" chunk:", ctr+1, "/", len(self.chunks), 
                        chunk, time.time() - tlocal)
            temp_out = objective[:,self.buffer:-self.buffer].cpu().data.numpy().copy(order='F')
            f.write(temp_out.T)
            
            quit()
        f.close()

        
        print ("Total residual time: ", time.time()-t0)
            
        

    # def save_residual(self):
    
        # f = open(self.fname_out,'wb')
        # for fname in self.fnames_seg:

            # res = np.load(fname).astype(self.dtype_out)
            # f.write(res)
        # f.close()
        
        # # delete residual chunks after successful merging/save
        # for fname in self.fnames_seg:
            # os.remove(fname)
