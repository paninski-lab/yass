import numpy as np
import sys, os, math
import datetime as dt
import scipy, scipy.signal
from scipy.interpolate import splrep, splev, make_interp_spline, splder, sproot

# doing imports inside module until travis is fixed
# Cat: TODO: move these to the top once Peter's workstation works
import torch
from torch import nn
#from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# cuda package to do GPU based spline interpolation and subtraction
import cudaSpline as deconv

class deconvGPU(object):

    def __init__(self, CONFIG, fname_templates, out_dir):

        # initalize parameters for 
        self.set_params(CONFIG, fname_templates, out_dir)

    def set_params(self, CONFIG, fname_templates, out_dir):

        # 
        self.CONFIG = CONFIG

        # set root directory for loading:
        self.root_dir = self.CONFIG.data.root_folder
        
        #
        self.out_dir = out_dir

        #
        self.fname_templates = fname_templates
        
        # number of seconds to load from recording
        self.n_sec = self.CONFIG.resources.n_sec_chunk_gpu
        
        # Cat: TODO: Load sample rate from disk
        self.sample_rate = self.CONFIG.recordings.sampling_rate
        
        # Cat: TODO: unclear if this is always the case
        self.n_time = self.CONFIG.recordings.spike_size_ms*self.sample_rate//1000+1
        
        # set length of lockout window
        # Cat: TODO: unclear if this is always correct
        self.lockout_window = self.n_time -1

        # length of conv filter
        self.n_times = torch.arange(-self.lockout_window,self.n_time,1).long().to(device)#,dtype=torch.int32, device='cuda')
        
        # buffer
        # Cat: TODO: Read from CONFIG
        self.buffer = 200
        
        # set max deconv thersho
        self.deconv_thresh = self.CONFIG.deconvolution.threshold

        # svd compression flag
        self.svd_flag = True
        
        # rank for SVD
        # Cat: TODO: read from CONFIG
        self.RANK = 5
        
        # make a 3 point array to be used in quadratic fit below
        self.peak_pts = torch.arange(-1,+2).to(device)
        
        # list to track spikes found in each iteration for debugging
        self.spike_list = []
        self.shift_list = []
               
        
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
            
        # Cat: TODO we should dissable all non-SVD options
        else:
            self.compute_temp_temp()
   
        # move data to gpu
        self.data_to_gpu()
                
        # initialize Ian's objects
        self.initialize_cpp()

        # conver templates to bpslines
        self.templates_to_bsplines()
           
            
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
        self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])   
        
        
    def templates_to_bsplines(self):

        print ("  making template bsplines (TODO: make template bsplines in parallel on CPU")
        print ("                           (TODO: save template bsplines to disk to not recompute)")
        
        print ("     # of templates converting to bsplines: ", len(self.temp_temp_cpp))
        self.coefficients = deconv.BatchedTemplates([self.transform_template(template) for template in self.temp_temp_cpp])
        

    def fit_spline(self, curve, knots=None,  prepad=0, postpad=0, order=3):
        if knots is None:
            knots = np.arange(len(curve) + prepad + postpad)
        return splrep(knots, np.pad(curve, (prepad, postpad), mode='symmetric'), k=order)


    def transform_template(self, template, knots=None, prepad=7, postpad=3, order=3):

        if knots is None:
            knots = np.arange(len(template.data[0]) + prepad + postpad)
        splines = [
            self.fit_spline(curve, knots=knots, prepad=prepad, postpad=postpad, order=order) 
            for curve in template.data.cpu().numpy()
        ]
        coefficients = np.array([spline[1][prepad-1:-1*(postpad+1)] for spline in splines], dtype='float32')
        return deconv.Template(torch.from_numpy(coefficients).cuda(), template.indices)

     
    def compute_temp_temp_svd(self):

        print ("  computing temp_temp svd on cpu (TODO: move this to GPU)")
        # 
        self.compress_templates()

        fname = os.path.join(self.out_dir,'temp_temp_sparse_svd.npy')
        if os.path.exists(fname)==False:

            # recompute vis chans and vis units 
            # Cat: TODO Fix this so dont' have to repeat it here
            self.up_up_map = None
            deconv_dir = ''
            
            #
            # self.visible_chans()
            
            # #
            # self.template_overlaps()
            
            # 
            units = np.arange(self.temps.shape[2])
            
            # Cat: TODO: work on multi CPU and GPU versions
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
        
        # # load also the diagonal of the temp temp function:
        # # Cat: TODO: is this required any longer? 
        # fname = os.path.join(self.out_dir,"temp_temp_diagonal_svd.npy")
        # if os.path.exists(fname)==False:
            # self.temp_temp_diagonal = []
            # for k in range(len(self.temp_temp)):
                # self.temp_temp_diagonal.append(self.temp_temp[k][self.orig_index[k]].cpu().data.numpy())

            # self.temp_temp_diagonal = np.vstack(self.temp_temp_diagonal)
            # np.save(fname, self.temp_temp_diagonal)
        # else:
            # self.temp_temp_diagonal = np.load(fname)
    
    
    def parallel_conv_filter(self,units, 
                            n_time,
                            up_up_map,
                            deconv_dir):

        # Cat: must load these structures from disk for multiprocessing step; 
        #       where there are many templates; due to multiproc 4gb limit 
        temporal = self.temporal #data['temporal']
        singular = self.singular #data['singular']
        spatial = self.spatial #data['spatial']
        temporal_up = self.temporal_up #data['temporal_up']

        vis_chan = self.vis_chan
        unit_overlap = self.unit_overlap
        approx_rank = self.RANK
        
        #for unit2 in units:
        conv_res_len = n_time * 2 - 1
        pairwise_conv_array = []
        print ("  TODO: parallelize temp_temp computation")
        for unit2 in units:
            if unit2%100==0:
                print (" temp_temp: ", unit2)
            n_overlap = np.sum(unit_overlap[unit2, :])
            pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)
            orig_unit = unit2 
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

        return pairwise_conv_array

            
    def compute_temp_temp(self):
                
        fname = os.path.join(self.out_dir,"temp_temp_sparse.npy")

        if os.path.exists(fname)==False: 
            temps = np.load(os.path.join(self.fname_templates)).transpose(1,0,2)
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

                # select vis_units_local
                vis_unit = np.where(self.vis_units[unit])[0]
    
                i_var = temps_reversed[vis_unit][:,vis_chan_local,:]
                f_var = filter_var[unit,:,vis_chan_local]
        
                # convolve 
                output_var = nn.functional.conv1d(i_var, f_var, padding=60)

                # Cat: TODO: squeeze these arrays (and others)
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
 
        # # load also the diagonal of the temp temp function:
        # fname = os.path.join(self.out_dir,"temp_temp_diagonal.npy")
        # if os.path.exists(fname)==False:
            # self.temp_temp_diagonal = []
            # for k in range(len(self.temp_temp)):
                # self.temp_temp_diagonal.append(self.temp_temp[k][self.orig_index[k]].cpu().data.numpy())

            # self.temp_temp_diagonal = np.vstack(self.temp_temp_diagonal)
            # np.save(fname, self.temp_temp_diagonal)
        # else:
            # self.temp_temp_diagonal = np.load(fname)
 
    
    def visible_chans(self):
        #if self.vis_chan is None:
        a = np.max(self.temps, axis=1) - np.min(self.temps, 1)
        
        # Cat: TODO: must read visible channel/unit threshold from file;
        self.vis_chan = a > 2.0
        a_self = self.temps.ptp(1).argmax(0)
        for k in range(a_self.shape[0]):
            self.vis_chan[a_self[k],k]=True


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

        
    def load_temps(self):
        ''' Load templates and set parameters
        '''
        
        # load templates
        self.temps = np.load(self.fname_templates).transpose(2,1,0)
        self.N_CHAN, self.STIME, self.K = self.temps.shape
        print (" Loaded templates shape (n_chan, n_time, n_temps): ", self.temps.shape)
        self.temps_gpu = torch.from_numpy(self.temps).float().to(device)
        
        # # vis chans: indexes of channels on which a template is visible, i.e. > 2SU 
        # fname = os.path.join(self.out_dir,"vis_chan.npy")
        # if os.path.exists(fname)==False:
            # ptps = self.temps.ptp(1) #np.max(temps, axis=1) - np.min(temps, 1)
            # vis_chan = ptps > 2.0
            # vis_chan = vis_chan.T
            # np.save(fname, vis_chan)
        # else:
            # vis_chan=np.load(fname)

        # print ("vis chan: ", vis_chan.shape)
            
        # # visible units: indexes of the units that overlap with a particular template
        # fname = os.path.join(self.out_dir, 'vis_units.npy')
        # fname_orig_index = os.path.join(self.out_dir, 'temp_temp_orig_index.npy')
        # if os.path.exists(fname)==False:
            # unit_overlap = np.sum(np.logical_and(vis_chan[:, None, :], vis_chan[None, :, :]), axis=2)
            # self.vis_units = unit_overlap > 0
            # np.save(fname, self.vis_units)

            # # also track where each template's index is in the sparse vis_units array
            # orig_index = []
            # for k in range(self.vis_units.shape[0]):
                # orig_index.append(np.sum(self.vis_units[k,:k]))
            
            # orig_index = np.array(orig_index)
            # np.save(fname_orig_index, orig_index)
        # else:
            # self.vis_units=np.load(fname)
            # orig_index = np.load(fname_orig_index)

        #print ("vis units: ", self.vis_units.shape)

        #self.orig_index = torch.from_numpy(orig_index).long().to(device)

        
    def compress_templates(self):
        """Compresses the templates using SVD and upsample temporal compoents."""
        
        ## compute everythign using SVD
        fname = os.path.join(self.out_dir,'templates_svd.npz')
        if os.path.exists(fname)==False:
        #if True:
            print ("  computing SVD on templates (TODO: implement torch.svd()")
    
            #self.approx_rank = self.RANK
            self.temporal, self.singular, self.spatial = np.linalg.svd(
                np.transpose(np.flipud(np.transpose(self.temps,(1,0,2))),(2, 0, 1)))
            
            # Keep only the strongest components
            self.temporal = self.temporal[:, :, :self.RANK]
            self.singular = self.singular[:, :self.RANK]
            self.spatial = self.spatial[:, :self.RANK, :]
            
            print ("self.temps: ", self.temps.shape)
            #print ("self.temps.transpose: ", self.temps_cpu.transpose(1,0,2)[0][0])
            
            print ("temporal: ", self.temporal.shape)
            print ("self.temporal: ", self.temporal[0][0])
            print (" singular: ", self.singular.shape)
            print (" spatial: ", self.spatial.shape)            
            
            # Upsample the temporal components of the SVD
            # in effect, upsampling the reconstruction of the
            # templates.
            # Cat: TODO: No upsampling is needed; to remove temporal_up from code
            self.temporal_up = self.temporal
           
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
        for i in range(self.vis_chan.shape[1]):
            norm[i] = np.sum(np.square(self.temps.transpose(1,0,2)[:, self.vis_chan[:,i], i]))
        
        #move data to gpu
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
            
        #print ("Temps load (run once only): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
        #print ("---------------------------------------")
        #print ('')

  
    def load_data(self, chunk):
        '''  Function to load raw data 
        '''

        start = dt.datetime.now().timestamp()

        # set indexes
        start_time = chunk[0]*self.sample_rate
        end_time = chunk[1]*self.sample_rate
        self.chunk_len = end_time-start_time
        
        # load raw data
        fname = os.path.join(self.root_dir, 'tmp','preprocess','standardized.bin')
        self.data = np.zeros((self.N_CHAN, self.buffer*2+self.chunk_len))
        # Cat: TODO: datatype len hardcoded to 4 below
        with open(fname, 'rb') as fin:
            fin.seek(start_time * 4 * self.N_CHAN, os.SEEK_SET)
            self.data[:,self.buffer:-self.buffer] = np.fromfile(
                fin,
                dtype='float32',
                count=(self.chunk_len * self.N_CHAN)).reshape(self.chunk_len,self.N_CHAN).T#.astype(np.int32)
        
        self.data = torch.from_numpy(self.data).float().to(device)
        torch.cuda.synchronize()
        #print ("Input size: ",self.data.shape, int(sys.getsizeof(self.data.astype(np.float32))/1E6), "MB")
        if self.verbose:

            print ("Input size: ",self.data.shape, int(sys.getsizeof(self.data)), "MB")
            print ("Load raw data (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
            print ("---------------------------------------")
            print ('')
                       

    def make_objective(self):
        start = dt.datetime.now().timestamp()
        if self.verbose:
            print ("Computing objective ")

        # transfer data to GPU
        self.obj_gpu = torch.zeros((self.temps.shape[2], self.data.shape[1]+self.STIME-1),
                                       dtype=torch.float, device='cuda')
            
        # make objective using full rank (i.e. no SVD)
        if self.svd_flag==False:
            
            #traces = torch.from_numpy(self.data).to(device) 
            traces = self.data
            temps = torch.from_numpy(self.temps).to(device)
            for i in range(self.K):
                self.obj_gpu[i,:] = nn.functional.conv1d(traces[None, self.vis_chan[i],:], 
                                                         temps[None,self.vis_chan[i],:,i],
                                                         padding=self.STIME-1)[0][0]
                
        # use SVD to make objective:
        else:
         
            # move data to gpu
            data_gpu = self.data

            # U_ x H_ (spatial * vals)
            # looping over ranks
            for r in range(self.RANK):
                mm = torch.mm(self.spatial_gpu[r::self.RANK]*self.singular_gpu[r::self.RANK], 
                              data_gpu)[None,None]

                for i in range(self.temps.shape[2]):
                    self.obj_gpu[i,:] += nn.functional.conv1d(mm[:,:,i,:], 
                                                         self.filters_gpu[:,:,i*self.RANK+r, :], 
                                                         padding=self.STIME-1)[0][0]

        torch.cuda.synchronize()
        
        # compute objective function = 2 x Convolution term - norms
        self.obj_gpu*= 2. 
        self.obj_gpu-= self.norm
        
        if self.verbose:
            print ("Total time obj func (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
            print ("---------------------------------------")
            print ('')
             
    
    # Cat: TODO Is this function used any longer?
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
        
        # save offset of chunk time; spiketimes and neuron ids
        self.spike_list.append([self.offset, self.spike_times[:,0], 
                                self.neuron_ids[:,0]])
        self.shift_list.append(self.xshifts)
        
        
    def subtraction_step(self):
        
        start = dt.datetime.now().timestamp()

        # initialize arrays
        self.obj_array=[]
        self.obj_array_residual=[]
        self.n_iter=0
        for k in range(self.max_iter):
            
            # find peaks
            search_time = self.find_peaks()
            
            ## test if trehsold reached; 
            ## Cat: TODO: this info might already be availble
            t_max,_ = torch.max(self.gpu_max[self.lockout_window:-self.lockout_window],0)
            if t_max<self.deconv_thresh:
                if self.verbose:
                    print ("... threshold reached, exiting...")
                break

            elif self.spike_times.shape[0]==0:
                if self.verbose:
                    print ("... no detected spikes, exiting...")
                break                
                
            # save 3 point arrays for quad fit
            start1 = dt.datetime.now().timestamp()
            self.find_shifts()
            shift_time = dt.datetime.now().timestamp()- start1
            
            # debug flag; DO NOT use otherwise
            if self.save_objective:
                self.obj_array.append(self.obj_gpu.cpu().data.numpy().copy())
                
            # subtract spikes
            total_time = self.subtract_cpp()
            
            # debug flag; DO NOT use otherwise
            if self.save_objective:
                self.obj_array_residual.append(self.obj_gpu.cpu().data.numpy().copy())
                
            if self.verbose:
                if self.n_iter%self.print_iteration_counter==0:
                    print ("  iter: ", self.n_iter, "  obj_funct peak: ", np.round(t_max.item(),1), 
                       " # of spikes ", self.spike_times.shape[0], 
                       ", search time: ", np.round(search_time,6), 
                       ", quad fit time: ", np.round(shift_time,6),
                       ", subtract time: ", np.round(total_time,6), "sec")
            
            # save spiketimes
            self.save_spikes()
            
            # increase index
            self.n_iter+=1
            
        if self.verbose:
            print ("Total subtraction step: ", np.round(dt.datetime.now().timestamp()-start,3))
        
    
    def find_shifts(self):
        '''  Function that fits quadratic to 3 points centred on each peak of obj_func 
        '''
        
        #print (self.neuron_ids.shape, self.spike_times.shape)
        if self.neuron_ids.shape[0]>1:
            idx_tripler = (self.neuron_ids, self.spike_times.squeeze()[:,None]+self.peak_pts)
        else:
            idx_tripler = (self.neuron_ids, self.spike_times+self.peak_pts)
        
       # print ("idx tripler: ", idx_tripler)
        self.threePts = self.obj_gpu[idx_tripler]
        self.shift_from_quad_fit_3pts_flat_equidistant_constants(self.threePts.transpose(0,1))

        
    # compute shift for subtraction in objective function space
    def shift_from_quad_fit_3pts_flat_equidistant_constants(self, pts):
        ''' find x-shift after fitting quadratic to 3 points
            Input: [n_peaks, 3] which are values of three points centred on obj_func peak
            Assumes: equidistant spacing between sample times (i.e. the x-values are hardcoded below)
        '''

        self.xshifts = ((((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/2)/
                  (-2*((pts[0]-pts[1])-(((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/(2)))))-1        


        
    def find_peaks(self):
        ''' Function to use torch.max and an algorithm to find peaks
        '''
        
        # Cat: TODO: make sure you can also deconvolve ends of data;
        #      currently using padding here...

        # First step: find peaks across entire energy function across dimension 0
        #       input: (n_neurons, n_times)
        #       output:  n_times (i.e. the max energy function value at each point in time)
        #       note: windows are padded
        start = dt.datetime.now().timestamp()
        self.gpu_max, self.neuron_ids = torch.max(self.obj_gpu, 0)
        torch.cuda.synchronize()
        end_max = dt.datetime.now().timestamp()-start

        # Second step: find relative peaks across max function above for some lockout window
        #       input: n_times (i.e. values of energy at each point in time)
        #       output:  1D array = relative peaks across time for given lockout_window
        window_maxima = torch.nn.functional.max_pool1d_with_indices(self.gpu_max.view(1,1,-1), 
                                                                    self.lockout_window*2, 1, 
                                                                    padding=self.lockout_window)[1].squeeze()
        candidates = window_maxima.unique()
        self.spike_times = candidates[(window_maxima[candidates]==candidates).nonzero()]
        
        # Third step: only deconvolve spikes where obj_function max > threshold
        idx = torch.where(self.gpu_max[self.spike_times]>self.deconv_thresh, 
                          self.gpu_max[self.spike_times]*0+1, 
                          self.gpu_max[self.spike_times]*0)
        idx = torch.nonzero(idx)[:,0]
        
        self.spike_times = self.spike_times[idx]
        self.neuron_ids = self.neuron_ids[self.spike_times]
        #print ("  max_search: ", np.round(end_max,8), " everything else: ", np.round(dt.datetime.now().timestamp() - start1, 8))
        return (dt.datetime.now().timestamp()-start)         
    
        
    def subtract_cpp(self):
        
        start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
        
        spike_times = self.spike_times.squeeze()-self.lockout_window
        spike_temps = self.neuron_ids.squeeze()
        
        # if single spike, wrap it in list
        if self.spike_times.size()[0]==1:
            spike_times = spike_times[None]
            spike_temps = spike_temps[None]
            
        deconv.subtract_splines(
                    self.obj_gpu,
                    spike_times,
                    self.xshifts,
                    spike_temps,
                    self.coefficients)

        torch.cuda.synchronize()

        # also fill in self-convolution traces with low energy so the
        #   spikes cannot be detected again (i.e. enforcing refactoriness)
        deconv.refrac_fill(energy=self.obj_gpu,
                                  spike_times=spike_times,
                                  spike_ids=spike_temps,
                                  fill_length=self.n_time,  # variable fill length here
                                  fill_offset=self.n_time//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                                  fill_value=-1E10)

        torch.cuda.synchronize()
        
        return (dt.datetime.now().timestamp()-start)
                     
        
