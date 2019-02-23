import numpy as np
import sys, os, math
import matplotlib.pyplot as plt
import datetime as dt
import scipy, scipy.signal


#import torch
#from torch import nn
#from torch.autograd import Variable
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ian's module to be imported later
#import deconv

class deconvGPU(object):

    def __init__(self, n_sec, root_dir, upsample_flag=False):
        
        # initalize parameters for 
        self.set_params(n_sec, root_dir, upsample_flag)

        
    def initialize(self):
        
        # load templates and svd componenets
        self.load_temps()
        
        # find vis-chans
        self.visible_chans()
        
        # find vis-units
        self.template_overlaps()
        
        # set all nonvisible channels to 0. to help with SVD
        self.spatially_mask_templates()
           
        # compute template convolutions
        if self.svd_flag:
            self.compute_temp_temp_svd()
        else:
            self.compute_temp_temp()
   
        # move data to gpu
        self.data_to_gpu()
                
        # upsample all self convolution traces (i.e. on main template's row)
        #   this will be used to match deconv traces
        #self.shift_templates_all()

        # initialize Ian's objects
        if self.cpp_subtract:
            self.initialize_cpp()
        
        
    def run(self, chunk):
        
        # load raw data and templates
        self.load_data(chunk)
        
        # make objective function
        self.make_objective()
        
        # set inifinities where obj < 0 or some other value
        # Cat:TODO not currently used; not sure this helps.
        #self.set_objective_infinities()
        
        # run 
        self.subtraction_step()
                
        # empty cache
        torch.cuda.empty_cache()

    
    def initialize_cpp(self):
        
        # make a list of pairwise batched temp_temp and their vis_units
        self.templates_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])   

        
    def set_params(self, n_sec, root_dir, upsample_flag=False):

        # set root directory for loading:
        self.root_dir = root_dir
        
        # number of seconds to load from recording
        self.n_sec = n_sec
        
        # Cat: TODO: Load sample rate from disk
        self.sample_rate = 20000
        
        # length of conv filter
        self.n_times = torch.arange(-60,61,1).long().to(device)#,dtype=torch.int32, device='cuda')

        # set length of lockout window
        self.lockout_window = 60
        
        # same but as a constant
        self.n_time = 61
                
        # buffer
        self.buffer =200
        
        # set max deconv thersho
        self.deconv_thresh = 10

        # svd compression flag
        self.svd_flag = False
        
        # rank for SVD
        self.RANK = 5

        # upsampling
        self.upsample_flag = upsample_flag
        
        # save spike train array
        self.spike_train=np.zeros((0,2),'int64')

        # list to track spikes found in each iteration for debugging
        self.spike_list = []

              
    def compute_temp_temp_svd(self):

        print ("  computing temp_temp svd on cpu (TODO: move this to GPU)")
        # 
        self.compress_templates()

        fname = os.path.join(self.root_dir,'tmp/deconv/temp_temp_sparse_svd.npy')
        if os.path.exists(fname)==False:

            # recompute vis chans and vis units 
            # Cat: TODO Fix this so dont' have to repeat it here
            self.up_up_map = None
            up_factor = 1
            deconv_dir = ''

            
            #
            self.visible_chans()
            
            #
            self.template_overlaps()
            
            # 
            units = np.arange(self.temps.shape[2])
            self.temp_temp= self.parallel_conv_filter(units, 
                                                    self.n_time,
                                                    self.up_up_map,
                                                    deconv_dir)
                
            temp_gpu = []
            for k in range(len(self.temp_temp)):
                temp_gpu.append(torch.from_numpy(self.temp_temp[k]).float().to(device))

            self.temp_temp = temp_gpu
            np.save(fname, self.temp_temp)
                                 
        else:
            self.temp_temp = np.load(fname)


        
        # load also the diagonal of the temp temp function:
        fname = os.path.join(self.root_dir,"tmp/deconv/temp_temp_diagonal_svd.npy")
        if os.path.exists(fname)==False:
            self.temp_temp_diagonal = []
            for k in range(len(self.temp_temp)):
                self.temp_temp_diagonal.append(self.temp_temp[k][self.orig_index[k]].cpu().data.numpy())

            self.temp_temp_diagonal = np.vstack(self.temp_temp_diagonal)
            np.save(fname, self.temp_temp_diagonal)
        else:
            self.temp_temp_diagonal = np.load(fname)
    
    
    def parallel_conv_filter(self,units, 
                            n_time,
                            up_up_map,
                            deconv_dir):

       
        # Cat: must load these structures from disk for multiprocessing step; 
        #       where there are many templates; due to multiproc 4gb limit 
        temporal = self.temporal 
        singular = self.singular 
        spatial = self.spatial 
        temporal_up = self.temporal_up 

        vis_chan = self.vis_chan
        up_factor = 1
        unit_overlap = self.unit_overlap
        approx_rank = self.RANK
        
        #for unit2 in units:
        conv_res_len = n_time * 2 - 1
        pairwise_conv_array = []
        for unit2 in units:
            if unit2%100==0:
                print (" temp_temp: ", unit2)
            n_overlap = np.sum(unit_overlap[unit2, :])
            pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)
            orig_unit = unit2 // up_factor
            masked_temp = np.flipud(np.matmul(
                    temporal_up[unit2] * singular[orig_unit][None, :],
                    spatial[orig_unit, :, :]))

            #pairwise_conv*=0.
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

        return pairwise_conv_array

        #np.save(deconv_dir+'/temp_temp_chunk_'+str(proc_index), 
        #                                            pairwise_conv_array)
                 
            
    def compute_temp_temp(self):
                
        fname = os.path.join(self.root_dir,"tmp/deconv/temp_temp_sparse.npy")

        if os.path.exists(fname)==False: 
            temps = np.load(os.path.join(self.root_dir,"tmp/templates_cluster.npy")).transpose(1,0,2)
            print ("Computing temp_temp")
            print ("  clustered templates (n_templates, n_chans, n_times): ", temps.shape)

            # Cat: TODO: can make this less GPU memory intesnive and just load each indivdiaully
            temps_reversed = np.flip(temps,axis=2).copy()
            temps_reversed = torch.from_numpy(temps_reversed).float().to(device)

            # input data stack: templates, channels, times
            print ("  input templates: ", temps_reversed.shape)

            # same as input_var except with extra dimension
            filter_var = temps_reversed[:,None]
            print ("  filter shapes: ", filter_var.shape)

            units = np.arange(temps.shape[0])
            temp_temp = []
            start =dt.datetime.now().timestamp()
            for unit in units:

                # select vis_chans_local
                vis_chan_local = np.where(self.vis_chan[:,unit])[0]
                #print ("self.vis_chan :", self.vis_chan.shape)
                #print (" vis_chan_local: ", vis_chan_local)

                # select vis_units_local
                vis_unit = np.where(self.vis_units[unit])[0]
    
                i_var = temps_reversed[vis_unit][:,vis_chan_local,:]
                f_var = filter_var[unit,:,vis_chan_local]
        
                # 
                output_var = nn.functional.conv1d(i_var, f_var, padding=60)

                temp_temp.append(output_var[:,0])
                if unit%100==0:
                    print ("  unit: ", unit, "input: ", i_var.shape, 
                           ", filter: ", f_var.shape,
                           ", output: ", output_var.shape,'\n')


            print ("  total time : ", dt.datetime.now().timestamp()-start)
            
            np.save(fname, temp_temp)
            self.temp_temp = temp_temp
        else:

            self.temp_temp = np.load(fname)
 
        # load also the diagonal of the temp temp function:
        fname = os.path.join(self.root_dir,"tmp/deconv/temp_temp_diagonal.npy")
        if os.path.exists(fname)==False:
            self.temp_temp_diagonal = []
            for k in range(len(self.temp_temp)):
                self.temp_temp_diagonal.append(self.temp_temp[k][self.orig_index[k]].cpu().data.numpy())

            self.temp_temp_diagonal = np.vstack(self.temp_temp_diagonal)
            np.save(fname, self.temp_temp_diagonal)
        else:
            self.temp_temp_diagonal = np.load(fname)
 


    
    def visible_chans(self):
        #if self.vis_chan is None:
        a = np.max(self.temps, axis=1) - np.min(self.temps, 1)
        self.vis_chan = a > 2.0
        #else:
        #    self.vis_chan = np.ones((self.n_chan, self.n_unit),'int32')

        #return self.vis_chan


    def template_overlaps(self):
        """Find pairwise units that have overlap between."""
        vis = self.vis_chan.T
        self.unit_overlap = np.sum(
            np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
        self.unit_overlap = self.unit_overlap > 0
        #self.unit_overlap = np.repeat(self.unit_overlap, 1, axis=0)
        
                
    def spatially_mask_templates(self):
        
        """Spatially mask templates so that non visible channels are zero."""
        for k in range(self.temps.shape[2]):
            zero_chans = np.where(self.vis_chan[:,k]==0)[0]
            self.temps[zero_chans,:,k]=0.


        
    def load_temps(self):
        ''' Load templates and set parameters
        
        '''
        #start = dt.datetime.now().timestamp()

        # load templates; keep only first 2048
        self.temps = np.load(self.root_dir + "/tmp/templates_cluster.npy").transpose(2,1,0)#[:, :, :2048]

        
        #make a copy for later use
        # Cat: TODO: this is temporary, can delete eventually
        self.N_CHAN, self.STIME, self.K = self.temps.shape
        print (" Loaded templates shape (n_chan, n_time, n_temps): ", self.temps.shape)
        
        
        # vis chans: indexes of channels on which a template is visible, i.e. > 2SU 
        fname = os.path.join(self.root_dir,"tmp/deconv/vis_chan.npy")
        print (fname)
        if os.path.exists(fname)==False:
            ptps = self.temps.ptp(1) #np.max(temps, axis=1) - np.min(temps, 1)
            vis_chan = ptps > 2.0
            vis_chan = vis_chan.T
            np.save(fname, vis_chan)
        else:
            vis_chan=np.load(fname)

        print ("vis chan: ", vis_chan.shape)
            
        # visible units: indexes of the units that overlap with a particular template
        fname = os.path.join(self.root_dir, 'tmp/deconv/vis_units.npy')
        fname_orig_index = os.path.join(self.root_dir, 'tmp/deconv/temp_temp_orig_index.npy')
        if os.path.exists(fname)==False:
            unit_overlap = np.sum(np.logical_and(vis_chan[:, None, :], vis_chan[None, :, :]), axis=2)
            self.vis_units = unit_overlap > 0
            np.save(fname, self.vis_units)

            # also track where each template's index is in the sparse vis_units array
            orig_index = []
            for k in range(self.vis_units.shape[0]):
                orig_index.append(np.sum(self.vis_units[k,:k]))
            
            orig_index = np.array(orig_index)
            np.save(fname_orig_index, orig_index)
        else:
            self.vis_units=np.load(fname)
            orig_index = np.load(fname_orig_index)

        print ("vis units: ", self.vis_units.shape)

        self.orig_index = torch.from_numpy(orig_index).long().to(device)

        
    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""
        
        ## compute everythign using SVD
        fname = os.path.join(self.root_dir,'tmp/deconv/templates_svd.npz')
        if os.path.exists(fname)==False:
        #if True:
            print ("  computing SVD on templates (TODO: implement torch.svd()")
            print (" templates: ", self.temps.shape)
    
            #self.approx_rank = self.RANK
            self.temporal, self.singular, self.spatial = np.linalg.svd(
                np.transpose(np.flipud(np.transpose(self.temps,(1,0,2))),(2, 0, 1)))
            
            # Keep only the strongest components
            self.temporal = self.temporal[:, :, :self.RANK]
            self.singular = self.singular[:, :self.RANK]
            self.spatial = self.spatial[:, :self.RANK, :]
            
            print ("self.temps: ", self.temps.shape)
            #print ("self.temps.transpose: ", self.temps_cpu.transpose(1,0,2)[0][0])
            
            # Upsample the temporal components of the SVD
            # in effect, upsampling the reconstruction of the
            # templates.
            if self.upsample_flag==False:
                # No upsampling is needed.
                self.temporal_up = self.temporal
            else:
                self.temporal_up = scipy.signal.resample(
                        self.temporal, self.n_time * self.up_factor, axis=1)
                idx = np.arange(0, self.n_time * self.up_factor, self.up_factor) + np.arange(self.up_factor)[:, None]
                self.temporal_up = np.reshape(
                        self.temporal_up[:, idx, :], [-1, self.n_time, self.RANK]).astype(np.float32)                
           
            np.savez(fname, temporal=self.temporal, singular=self.singular, 
                     spatial=self.spatial, temporal_up=self.temporal_up)
            
        else:
            print ("  loading SVD (from disk)")
                
            # load data for for temp_temp computation
            data = np.load(fname)
            self.temporal_up = data['temporal_up']
            self.temporal = data['temporal']
            self.singular = data['singular']
            self.spatial = data['spatial']                
            
      
    def data_to_gpu(self):
    
        # compute norm
        norm = np.zeros((self.vis_chan.shape[1],1),'float32')
        print (self.temps.transpose(1,0,2).shape, self.vis_chan.shape)
        for i in range(self.vis_chan.shape[1]):
            #print (self.vis_chan[:,i].shape, self.vis_chan[:,i])
            #print (i, self.temps.transpose(1,0,2)[:, self.vis_chan[:,i], i].shape)
            norm[i] = np.sum(np.square(self.temps.transpose(1,0,2)[:, self.vis_chan[:,i], i]))
        
        #move data to gpu
        #self.temps = torch.from_numpy(self.temps).float().to(device)
        self.norm = torch.from_numpy(norm).float().to(device)
        
        # load vis chans on gpu
        self.vis_chan_gpu=[]
        for k in range(self.vis_chan.shape[1]):
            self.vis_chan_gpu.append(torch.from_numpy(np.where(self.vis_chan[:,k])[0]).long().to(device))
        self.vis_chan = self.vis_chan_gpu
            
        # load vis_units onto gpu
        self.vis_units_gpu=[]
        for k in range(self.vis_units.shape[1]):
            self.vis_units_gpu.append(torch.FloatTensor(np.where(self.vis_units[k])[0]).long().to(device))
        self.vis_units = self.vis_units_gpu
        
        # move svd items to gpu
        if self.svd_flag:
            self.n_rows = self.temps.shape[2] * self.RANK
            self.spatial_gpu = torch.from_numpy(self.spatial.reshape([self.n_rows, -1])).float().to(device)
            self.singular_gpu = torch.from_numpy(self.singular.reshape([-1, 1])).float().to(device)
            self.temporal_gpu = np.flip(self.temporal,1)
            self.filters_gpu = torch.from_numpy(self.temporal_gpu.transpose([0, 2, 1]).reshape([self.n_rows, -1])).float().to(device)[None,None]
            
  
    def load_data(self, chunk):
        '''  Function to load raw data 
        '''

        start = dt.datetime.now().timestamp()

        # set indexes
        start_time = chunk[0]*self.sample_rate
        end_time = chunk[1]*self.sample_rate
        self.chunk_len = end_time-start_time
        
        # load raw data
        fname = os.path.join(self.root_dir, 'tmp/preprocess/standardized.bin')
        self.data = np.zeros((self.N_CHAN, self.buffer*2+self.chunk_len))
        with open(fname, 'rb') as fin:
            fin.seek(start_time * 4 * self.N_CHAN, os.SEEK_SET)
            self.data[:,self.buffer:-self.buffer] = np.fromfile(
                fin,
                dtype='float32',
                count=(self.chunk_len * self.N_CHAN)).reshape(self.chunk_len,self.N_CHAN).T#.astype(np.int32)
            
        print ("Input size: ",self.data.shape, int(sys.getsizeof(self.data.astype(np.float32))/1E6), "MB")
        print ("Load raw data (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
        print ("---------------------------------------")
        print ('')
                        

    def make_objective(self):
        start = dt.datetime.now().timestamp()
        print ("Computing objective (todo: remove for loop)")

        # transfer data to GPU
        self.obj_gpu = torch.zeros((self.temps.shape[2], self.data.shape[1]+self.STIME-1),
                                       dtype=torch.float, device='cuda')
            
        # make objective using full rank (i.e. no SVD)
        if self.svd_flag==False:
            
            traces = torch.from_numpy(self.data).to(device) 
            temps = torch.from_numpy(self.temps).to(device)
            for i in range(self.K):
                self.obj_gpu[i,:] = nn.functional.conv1d(traces[None, self.vis_chan[i],:], 
                                                         temps[None,self.vis_chan[i],:,i],
                                                         padding=self.STIME-1)[0][0]
                
        # use SVD to make objective:
        else:
         
            print ("  computing objective function (TODO: speed up loops)")
            # ****************************************

            # move data to gpu
            data_gpu = torch.from_numpy(self.data).float().to(device)
            torch.cuda.synchronize()

            # U_ x H_ (spatial * vals)
            # Faster way, but need to store mm in memory
            if False:
#                 print ("  fast computation, more memory (TODO: chunk this step)")
                mm = torch.mm(self.spatial_gpu*self.singular_gpu, data_gpu)[None,None]
                torch.cuda.synchronize()

                print ("mm: ", mm.shape)
                print ("filters: ", self.filters_gpu[:,:,0, :].shape)

                for i in range(self.n_rows):
                    self.obj_gpu[i//self.RANK, :] += nn.functional.conv1d(mm[:,:,i, :], 
                                                               self.filters_gpu[:,:,i, :], 
                                                               padding=self.STIME-1)[0][0]
            # slower but compute mm on the fly
            else:
#                 print ("  slower option, less memory (NOt fully implemented)")
                # looping over ranks
                for r in range(self.RANK):
                    mm = torch.mm(self.spatial_gpu[r::self.RANK]*self.singular_gpu[r::self.RANK], 
                                  data_gpu)[None,None]

                    for i in range(self.temps.shape[2]):
                        self.obj_gpu[i,:] += nn.functional.conv1d(mm[:,:,i,:], 
                                                             self.filters_gpu[:,:,i*self.RANK+r, :], 
                                                             padding=self.STIME-1)[0][0]
        
                    #del mm
        torch.cuda.synchronize()
        
        # compute objective function = 2 x Convolution term - norms
        self.obj_gpu*= 2. 
        self.obj_gpu-= self.norm
        #self.obj_gpu*=-1.
        
#         print ("  conv matrix: ", self.obj_gpu.shape, " time; ", dt.datetime.now().timestamp()-time2)

        print ("Total time obj func (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
        print ("---------------------------------------")
        print ('')
             
    
    def set_objective_infinities(self):
        
        self.inf_thresh = -1E10
        zero_trace = torch.zeros(self.obj_gpu.shape[1],dtype=torch.float, device='cuda')
        
        # loop over all templates and set to -Inf
        for k in range(self.obj_gpu.shape[0]):
            idx = torch.where(self.obj_gpu[k]<self.inf_thresh,
                          zero_trace+1,
                          zero_trace)
            idx = torch.nonzero(idx)[:,0]
            self.obj_gpu[k][idx]=-float("Inf")
    
        
            
    def save_spikes(self):
        
        # Fourth step: save spikes
        # Cat: TODO: don't run this every single iteration; run only once per deconv chunk
        # first, append spike times to existing spike_train
        # make a 2D array to hold spikes: shape: (spike_time, neuron_id)
        #temp = np.zeros((self.relmax_peaks_idx.shape[0],2),'int64')
        #temp[:,0] = self.relmax_peaks_idx[:,0].cpu().data.numpy()
        #temp[:,1] = self.gpu_argmax[self.relmax_peaks_idx][:,0].cpu().data.numpy()
        #self.spike_train = np.vstack((self.spike_train, temp))

        self.spike_list.append([self.offset, self.relmax_peaks_idx[:,0], 
                                self.gpu_argmax[self.relmax_peaks_idx][:,0]])
        #
        #self.spike_shifts = np.hstack((self.spike_shifts, self.temp_temp_shifts))
        
        
    def subtraction_step(self):
        
        start = dt.datetime.now().timestamp()

        # initialize arrays
        self.obj_array =[]
        self.n_iter=0
        for k in range(self.max_iter):
            if self.save_objective:
                self.obj_array.append(self.obj_gpu.cpu().data.numpy().copy())
            
            # find peaks
            search_time = self.find_peaks()
            
            ## test if trehsold reached; 
            ## Cat: TODO: this info might already be availble
            t_max,_ = torch.max(self.gpu_max[self.lockout_window:-self.lockout_window],0)
            if t_max<self.deconv_thresh:
                print ("... threshold reached, exiting...")
                break

            # upsample and find shifts
            #self.find_shifts()
            
            # subtract spikes
            if self.cpp_subtract==False:
                times_, total_time, continue_flag = self.subtract()
            else:
                total_time, continue_flag = self.subtract_cpp()
            
            if self.print_iterations:
                if self.n_iter%50==0:
                    print ("  iter: ", self.n_iter, "  obj_funct peak: ", t_max.item(), 
                       " # of spikes ", self.relmax_peaks_idx.shape[0], 
                       " subract time: ", np.round(total_time,3), 
                       " search time: ", np.round(search_time,3), " sec")#,
                       #" total time: ", np.round(total_time,3), " sec")

            # save spiketimes
            self.save_spikes()
            
            # increase index
            self.n_iter+=1

            if continue_flag==False:
                break
            
        print ("Total subtraction step: ", np.round(dt.datetime.now().timestamp()-start,3))
        
        
    def find_peaks(self):
        ''' Function to use torch.max and an algorithm to find peaks
            Note:  torch code returns all rel max peaks currently;
                   then calling scipy.signal.argrelmax to find rel peaks 
                   within a set window (e.g. +/- 30 (i.e. 60 total))
                   
            Todo #1: convert argrelmax code to pytorch/C 
                     Note; partial solution found using max_pool1d;
            
        '''
        
        # Cat: TODO: make sure you can also deconvolve ends of data;
        #      currently using padding here...

        start = dt.datetime.now().timestamp()
        # First step: find peaks across entire energy function across dimension 0
        #       input: (n_neurons, n_times)
        #       output:  n_times (i.e. the max energy function value at each point in time)
        # Search excluding ends using lockout_window
        self.gpu_max, self.gpu_argmax = torch.max(
                self.obj_gpu, 0)
                #self.obj_gpu[:,self.lockout_window:-self.lockout_window], 0)
                #self.obj_gpu[:,:-2*self.lockout_window], 0)
        torch.cuda.synchronize()

        # Second step: find relative peaks across max function above for some lockout window
        #       input: n_times (i.e. values of energy at each point in time)
        #       output:  1D array = relative peaks across time for given lockout_window
        window_maxima = torch.nn.functional.max_pool1d_with_indices(self.gpu_max.view(1,1,-1), 
                                                                    self.lockout_window*2, 1, 
                                                                    padding=self.lockout_window)[1].squeeze()
        candidates = window_maxima.unique()
        self.relmax_peaks_idx = candidates[(window_maxima[candidates]==candidates).nonzero()]
        
        # Third step: only deconvolve spikes where obj_function max > threshold
        idx = torch.where(self.gpu_max[self.relmax_peaks_idx]>self.deconv_thresh, 
                          self.gpu_max[self.relmax_peaks_idx]*0+1, 
                          self.gpu_max[self.relmax_peaks_idx]*0)
        idx = torch.nonzero(idx)[:,0]
        
        self.relmax_peaks_idx = self.relmax_peaks_idx[idx]
        #self.relmax_peaks_idx += self.lockout_window
#         print (self.relmax_peaks_idx.shape)
#         print ("self.relmax_peaks_idx : ", self.relmax_peaks_idx )

        #print ("self.relmax_peaks_idx: ", self.relmax_peaks_idx)
        return (dt.datetime.now().timestamp()-start)         


    def shift_templates_all(self):
        ''' Computes a linearly interpolated, shifted version of the self-convolution 
            trace (i.e. temp_temp[i,i]) for each template.  This is then used to align
            to the peaks in the objective function.  
            
            Cat: TODO: this should be just saved to disk if the window and increments are fixed
            
            Cat: TODO: this should probably also be tested using sinc/spline interpolation
                       
            input: self.temp_temp_diagonal (n_templates, 121)
                   this is the convolution of each template with itself (i.e. temp_temp[i,i])
        '''
        
        self.l_shift = -0.8
        self.r_shift = 0.8
        self.increment = 0.01
        
        # load shifted versions
        # Cat: TODO: only shift detected units; not a priority
        self.temps_all_shifted = np.zeros((self.temp_temp_diagonal.shape[0], 
                                           np.arange(self.l_shift, self.r_shift, self.increment).shape[0], 
                                           self.temp_temp_diagonal.shape[1]), 'float32')

        print (" self.temps_all_shifted: ", self.temps_all_shifted.shape)
        
        # Cat: TODO: check if offline interpolated upsample is better (and not too slow)
        #       size of data only N-templates X 121 
        shifts = np.arange(self.l_shift, self.r_shift, self.increment)
        for k, shift_ in enumerate(shifts):
            if int(shift_)==shift_:
                ceil = int(shift_)
                temp = np.roll(self.temp_temp_diagonal,ceil,axis=1)
            else:
                ceil = int(math.ceil(shift_))
                floor = int(math.floor(shift_))
                temp = (np.roll(self.temp_temp_diagonal,ceil, axis=1)*(shift_-floor)+
                        np.roll(self.temp_temp_diagonal,floor, axis=1)*(ceil-shift_))

            self.temps_all_shifted[:,k] = temp

        # transfer to gpu tensor; 
        self.temps_all_shifted = torch.from_numpy(self.temps_all_shifted).float().to(device)
        print ("temps_all_shifted: ",self.temps_all_shifted.shape)

        
    def find_shifts(self):
        '''  Function called every deconv iteration;
             Loads obj_function around detected peak +/- 60 timesteps and 
             finds best shift of upsampled temp_temp[i,i] (i.e. diagonal 
             self-convolution traces).
        
        '''

        # neuron ids are argmax of peaks in the peak function
        neuron_ids = self.gpu_argmax[self.relmax_peaks_idx] # neuron_ids[rel_max]
        
        # loop over neurons and match trace snipit with 
        # Cat: TODO: don't fix 60..61 time values... read from CONFIG
        # Cat: TODO: if this is slow, there are repeated computations here...
        self.temp_temp_shifts = []
        for ctr, neuron_id in enumerate(neuron_ids):

            # select obj_function snipit using neuron id and peak time           
            # Cat: TODO: must offset index by self.lockout_window;
            #     so formulat should be 
            #     self.obj_gpu[..., ... -self.lockout_window + 60:... +self.lockout_window + 61]
            #obj_trace = self.obj_gpu[neuron_id, self.relmax_peaks_idx[ctr]-60:self.relmax_peaks_idx[ctr]+61]
            obj_trace = self.obj_gpu[neuron_id, self.relmax_peaks_idx[ctr]:self.relmax_peaks_idx[ctr]+121][0]

            # find best shift of template self convolution (previously computed) 
            #   with obj_trace snipit
            best_id = self.find_best_shift(obj_trace, self.temps_all_shifted[neuron_id][0])

            # save shift for shifting during subtract step
            self.temp_temp_shifts.append(best_id)
            
            
    def find_best_shift(self, obj_trace, trace_shifted):
        '''  Finds the best shift among a range of traces
        
        '''
        # match only energey in the centre parts of the traces
        # Cat: TODO: these values must be parameters read from CONFIG
        diffs = obj_trace[40:80]-trace_shifted[:,40:80]
        
        # find maximum which is peak during subtraction
        max_vals, _ = diffs.max(1)
        
        # find the smallest of the maxima
        best_shift = max_vals.argmin(0)

        return best_shift
    
    
    def shift_temp_temp_gpu(self, ctr, neuron_id):
        
        # select from list of shifts
        #ctr = torch.tensor(ctr).long().to(device)[0]
        #idx = idx.clone().detach()
        #print (idx)
        
        # load integer shift from previously loaded
        shift_ = self.temp_temp_shifts[ctr]

        # recompute the floating point shift
        shifts = np.arange(self.l_shift, self.r_shift, self.increment)
        shift_ = shifts[shift_]
        
        # Cat: TODO; simplify this conditional (not priority)
        if int(shift_)==shift_:
            ceil = torch.tensor(int(shift_)).long().to(device)
            temp_shifted = self.roll_gpu(self.temp_temp[neuron_id], ceil, 1)  
        else:
            ceil = torch.tensor(int(math.ceil(shift_))).long().to(device)
            floor = torch.tensor(int(math.floor(shift_))).long().to(device)

            temp_shifted = (self.roll_gpu(self.temp_temp[neuron_id],ceil,1)*(shift_-floor)+
                            self.roll_gpu(self.temp_temp[neuron_id],floor,1)*(ceil-shift_))
        
        return temp_shifted
        
    def roll_gpu(self, x, shift, dim):

        if shift==0:
            return x

        return torch.cat((x[:, -shift:], x[:, :-shift]), dim=dim)
        
    
    def subtract(self):
        ''' Function to subtract temp_temp from computed peak locations
        '''

        # keep track of processing time for subtraction step
        start_all = dt.datetime.now().timestamp()

        times_=[]
        # loop over groups of identical spikes; 
        if False: 
            for ctr, neuron_id in enumerate(torch.unique(self.gpu_argmax[self.relmax_peaks_idx])):

                # find location of times for neuron_id
                # Cat: TODO: can we do this search 1 time instead of so many times
                idx = torch.where(self.gpu_argmax[self.relmax_peaks_idx]==neuron_id, 
                                  self.gpu_argmax[self.relmax_peaks_idx]*0+1, 
                                  self.gpu_argmax[self.relmax_peaks_idx]*0)
                idx = torch.nonzero(idx)[:,0]
                times = self.relmax_peaks_idx[idx]
                #print (times)
                # add the offset from the gpu_max search above
                times = times + self.lockout_window

                start = dt.datetime.now().timestamp()
                
                # shift the templates every single time
                # Cat: TODO: better way to do this is to just store in memory or on disk
                
                if self.upsample_flag:
                    temp_temp_shifted = self.shift_temp_temp_gpu(ctr, neuron_id)
                    self.obj_gpu[self.vis_units[neuron_id][:,np.newaxis],self.n_times + times[:,np.newaxis]]-= 2*temp_temp_shifted
                else:
                    self.obj_gpu[self.vis_units[neuron_id][:,np.newaxis],self.n_times + times[:,np.newaxis]]-= 2*self.temp_temp[neuron_id]
                
                # mark refractory periods with zeros
                # Cat: TODO: may need to step back 1/2 width of spike length
                #self.obj_gpu[neuron_id,self.n_times + times[:,None]]= -float("Inf")
                self.obj_gpu[neuron_id,self.n_times[30:90] + times[:,None]]= -float("Inf")
                times_.append(dt.datetime.now().timestamp()-start)
        else:
            # or loop over every single spike; no search
            #for ctr, neuron_id in enumerate(torch.unique(self.gpu_argmax[self.relmax_peaks_idx])):
            for ctr, neuron_id in enumerate(self.gpu_argmax[self.relmax_peaks_idx]):

                #print (" neuronId: ", neuron_id)

                time = self.relmax_peaks_idx[ctr]
                #print (times)
                # add the offset from the gpu_max search above
                time = time #+ self.buffer

                #times = self.relmax_peaks_idx[ctr]
                #print (" neuron_id: ", neuron_id,  ", times: ", time)
                
                start = dt.datetime.now().timestamp()
                if self.upsample_flag:
                    temp_temp_shifted = self.shift_temp_temp_gpu(ctr, neuron_id)
                    self.obj_gpu[self.vis_units[neuron_id][:,np.newaxis],self.n_times + time]-= 2*temp_temp_shifted
                else:
                    self.obj_gpu[self.vis_units[neuron_id][:,np.newaxis],self.n_times + time]-= 2*self.temp_temp[neuron_id]
                    #self.obj_gpu[self.vis_units[neuron_id][:,np.newaxis],self.n_times + times[:,np.newaxis]]-= 2*self.temp_temp[neuron_id]

                # mark refractory periods with zeros
                # Cat: TODO: may need to step back 1/2 width of spike length
                #self.obj_gpu[neuron_id,self.n_times + times[:,None]]= -1E10
                #times_.append(dt.datetime.now().timestamp()-start)
                #self.obj_gpu[neuron_id,self.n_times[30:90] + time]= -float("Inf")
                self.obj_gpu[neuron_id,self.n_times + time]= -float("Inf")
                times_.append(dt.datetime.now().timestamp()-start)
            
        return times_, dt.datetime.now().timestamp()-start_all, True

        
    def subtract_cpp(self):
        
        start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
        
        spike_times = self.relmax_peaks_idx.squeeze()-self.lockout_window
        spike_temps = self.gpu_argmax[self.relmax_peaks_idx].squeeze()
        
        if self.relmax_peaks_idx.size()[0]==1:
            return dt.datetime.now().timestamp()-start, False
        
        #t0 = time.time()
        deconv.subtract_spikes(data=self.obj_gpu,
                               spike_times=spike_times,
                               spike_temps=spike_temps,
                               templates=self.templates_cpp,
                               do_refrac_fill=True,
                               refrac_fill_val=-1e10)
        torch.cuda.synchronize()
        
        return dt.datetime.now().timestamp()-start, True
        
        
