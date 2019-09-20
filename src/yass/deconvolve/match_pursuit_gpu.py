import numpy as np
import sys, os, math
import datetime as dt
import scipy, scipy.signal
import parmap
from scipy.interpolate import splrep, splev, make_interp_spline, splder, sproot

# doing imports inside module until travis is fixed
# Cat: TODO: move these to the top once Peter's workstation works
import torch
from torch import nn
#from torch.autograd import Variable

# cuda package to do GPU based spline interpolation and subtraction
import cudaSpline as deconv


# # ****************************************************************************
# # ****************************************************************************
# # ****************************************************************************

def parallel_conv_filter2(units, 
                          n_time,
                          up_up_map,
                          deconv_dir,
                          update_templates_backwards,
                          svd_dir,
                          chunk_id,
                          n_sec_chunk_gpu,
                          vis_chan,
                          unit_overlap,
                          approx_rank,
                          temporal,
                          singular,
                          spatial,
                          temporal_up):

    # loop over asigned units:
    conv_res_len = n_time * 2 - 1
    pairwise_conv_array = []
    for unit2 in units:
        #if unit2%100==0:
        #    print (" temp_temp: ", unit2)
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


        
# # ****************************************************************************
# # ****************************************************************************
# # ****************************************************************************
                     
class deconvGPU(object):

    def __init__(self, CONFIG, fname_templates, out_dir):
        
        #
        self.out_dir = out_dir
        
        # initialize directory for saving
        self.seg_dir = os.path.join(self.out_dir,'segs')
        if not os.path.isdir(self.seg_dir):
            os.mkdir(self.seg_dir)
            
        self.svd_dir = os.path.join(self.out_dir,'svd')
        if not os.path.isdir(self.svd_dir):
            os.mkdir(self.svd_dir)

        self.temps_dir = os.path.join(self.out_dir,'template_updates')
        if not os.path.isdir(self.temps_dir):
            os.mkdir(self.temps_dir)

        # initalize parameters for 
        self.set_params(CONFIG, fname_templates, out_dir)

    def set_params(self, CONFIG, fname_templates, out_dir):

        # 
        self.CONFIG = CONFIG

        # set root directory for loading:
        #self.root_dir = self.CONFIG.data.root_folder
        
        #
        self.out_dir = out_dir

        #
        self.fname_templates = fname_templates
        
        # number of seconds to load from recording
        self.n_sec = self.CONFIG.resources.n_sec_chunk_gpu_deconv
        
        # Cat: TODO: Load sample rate from disk
        self.sample_rate = self.CONFIG.recordings.sampling_rate
        
        # Cat: TODO: unclear if this is always the case
        self.n_time = self.CONFIG.spike_size
        
        # set length of lockout window
        # Cat: TODO: unclear if this is always correct
        self.lockout_window = self.n_time-1

        # 
        self.fill_value = 1E4
        
        # objective function scaling for the template term;
        self.tempScaling = 2.0

        # refractory period
        # Cat: TODO: move to config
        refrac_ms = 1
        self.refractory = int(self.CONFIG.recordings.sampling_rate/1000*refrac_ms)

        # length of conv filter
        #self.n_times = torch.arange(-self.lockout_window,self.n_time,1).long().cuda()

        # set max deconv thersho
        self.deconv_thresh = self.CONFIG.deconvolution.threshold

        # svd compression flag
        #self.svd_flag = True
        
        # make a 3 point array to be used in quadratic fit below
        #self.peak_pts = torch.arange(-1,+2).cuda()
        
        
    def initialize(self):

        # length of conv filter
        self.n_times = torch.arange(-self.lockout_window,self.n_time,1).long().cuda()
        # make a 3 point array to be used in quadratic fit below
        self.peak_pts = torch.arange(-1,+2).cuda()
        
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
            self.compress_templates()
            self.compute_temp_temp_svd()

        # # Cat: TODO we should dissable all non-SVD options?!
        # else:
            # self.compute_temp_temp()
   
        # move data to gpu
        self.data_to_gpu()
                
        # initialize Ian's objects
        self.initialize_cpp()

        # conver templates to bpslines
        self.templates_to_bsplines()

        # if self.scd:
            # # also need to make inverted templates for addition step
            # # self.initialize_cpp_inverted()
            
            # # conver templates to bpslines
            # self.templates_to_bsplines_inverted()
        
            # # make dummy templates for recovery of refractory period
            # # self.initialize_cpp_refractory()
            
    def run(self, chunk_id):
        
        # rest lists for each segment of time
        self.offset_array = []
        self.spike_array = []
        self.neuron_array = []
        self.shift_list = []
        self.add_spike_temps = []
        self.add_spike_times = []
        
        # save iteration 
        self.chunk_id = chunk_id

        # load raw data and templates
        self.load_data(chunk_id)
        
        # make objective function
        self.make_objective()
               
        # run 
        self.subtraction_step()
                
        # empty cache
        torch.cuda.empty_cache()


    def initialize_cpp(self):

        # make a list of pairwise batched temp_temp and their vis_units
        # Cat: TODO: this isn't really required any longer;
        #            - the only thing required from this in parallel bsplines function is
        #              self.temp_temp_cpp.indices - is self.vis_units
        #              
        self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])

        
    def templates_to_bsplines(self):

        print ("  making template bsplines")
        fname = os.path.join(self.svd_dir,'bsplines_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
        
        if os.path.exists(fname)==False:
            
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
            
            np.save(fname, coefficients)
        else:
            print ("  ... loading coefficients from disk")
            coefficients = np.load(fname)

        print ("  ... moving coefficients to cuda objects")
        coefficients_cuda = []
        for p in range(len(coefficients)):
            coefficients_cuda.append(deconv.Template(torch.from_numpy(coefficients[p]).cuda(), self.temp_temp_cpp[p].indices))
            # print ('self.temp_temp_cpp[p].indices: ', self.temp_temp_cpp[p].indices)
            # print ("self.vis_units: ", self.vis_units[p])
            # coefficients_cuda.append(deconv.Template(torch.from_numpy(coefficients[p]).cuda(), self.vis_units[p]))
        
        self.coefficients = deconv.BatchedTemplates(coefficients_cuda)

        del self.temp_temp
        del self.temp_temp_cpp
        del coefficients_cuda
        del coefficients
        torch.cuda.empty_cache()
            
     
    def compute_temp_temp_svd(self):

        print ("  making temp_temp filters (todo: move to GPU)")
        if self.update_templates_backwards:
            fname = os.path.join(self.svd_dir,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
        else:
            fname = os.path.join(self.svd_dir,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
        
        if os.path.exists(fname)==False:

            # recompute vis chans and vis units 
            # Cat: TODO Fix this so dont' have to repeat it here
            self.up_up_map = None
            deconv_dir = ''

            # 
            units = np.arange(self.temps.shape[2])
            
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
            else:
                units_split = np.array_split(units, self.CONFIG.resources.n_processors)
                self.temp_temp = []
                for units_ in units_split:
                    self.temp_temp.append(parallel_conv_filter2(units_, 
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
                                                      self.temporal_up))
                                                      
            # gather results
            temp_temp_local = [None]*units.shape[0]
            for ctr1, u1 in enumerate(units_split):
                for ctr2, u2 in enumerate(u1):
                    temp_temp_local[units_split[ctr1][ctr2]] = self.temp_temp[ctr1][ctr2]

            # transfer list to GPU
            self.temp_temp = []
            for k in range(len(temp_temp_local)):
                self.temp_temp.append(torch.from_numpy(temp_temp_local[k]).float().cuda())

            # save GPU list as numpy object
            np.save(fname, self.temp_temp)
                                 
        else:
            print (".... loading temp-temp from disk")
            self.temp_temp = np.load(fname, allow_pickle=True)
               
    
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

        # save vis_units for residual recomputation and other steps
        fname = os.path.join(self.svd_dir,'vis_units_'+
                      str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
        np.save(fname, self.vis_units)
                        
                        
    def spatially_mask_templates(self):
        """Spatially mask templates so that non visible channels are zero."""
        for k in range(self.temps.shape[2]):
            zero_chans = np.where(self.vis_chan[:,k]==0)[0]
            self.temps[zero_chans,:,k]=0.


    def load_temps(self):
        ''' Load templates and set parameters
        '''
        
        # load templates
        #print ("Loading template: ", self.fname_templates)
        self.temps = np.load(self.fname_templates, allow_pickle=True).transpose(2,1,0)
        self.N_CHAN, self.STIME, self.K = self.temps.shape
        # this transfer to GPU is not required any longer
        # self.temps_gpu = torch.from_numpy(self.temps).float().cuda()
        
        # compute max chans for data
        #print ("Making max chans, ptps, etc. for iteration 0: ", self.temps.shape)
        self.max_chans = self.temps.ptp(1).argmax(0)

        # compute ptps for data
        self.ptps = self.temps.ptp(1).max(0)

        # Robust PTP location computation; find argmax and argmin of 
        self.ptp_locs = []
        for k in range(self.temps.shape[2]):
            max_temp = self.temps[self.max_chans[k],:,k].argmax(0)
            min_temp = self.temps[self.max_chans[k],:,k].argmin(0)
            self.ptp_locs.append([max_temp,min_temp])
        
        
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
            data = np.load(fname, allow_pickle=True)
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
        self.norm = torch.from_numpy(norm).float().cuda()
        
        # load vis chans on gpu
        self.vis_chan_gpu=[]
        for k in range(self.vis_chan.shape[1]):
            self.vis_chan_gpu.append(torch.from_numpy(np.where(self.vis_chan[:,k])[0]).long().cuda())
        self.vis_chan = self.vis_chan_gpu
            
        # load vis_units onto gpu
        self.vis_units_gpu=[]
        for k in range(self.vis_units.shape[1]):
            self.vis_units_gpu.append(torch.FloatTensor(np.where(self.vis_units[k])[0]).long().cuda())
        self.vis_units = self.vis_units_gpu
        
        # move svd items to gpu
        if self.svd_flag:
            self.n_rows = self.temps.shape[2] * self.RANK
            self.spatial_gpu = torch.from_numpy(self.spatial.reshape([self.n_rows, -1])).float().cuda()
            self.singular_gpu = torch.from_numpy(self.singular.reshape([-1, 1])).float().cuda()
            self.temporal_gpu = np.flip(self.temporal,1)
            self.filters_gpu = torch.from_numpy(self.temporal_gpu.transpose([0, 2, 1]).reshape([self.n_rows, -1])).float().cuda()[None,None]

  
    def load_data(self, chunk_id):
        '''  Function to load raw data 
        '''
        
        try:
            del self.data 
            torch.cuda.empty_cache()
        except:
            pass
            
        start = dt.datetime.now().timestamp()

        # read dat using reader class
        self.data_cpu = self.reader.read_data_batch(
            chunk_id, add_buffer=True).T
        
        self.offset = self.reader.idx_list[chunk_id, 0] - self.reader.buffer
        self.data = torch.from_numpy(self.data_cpu).float().cuda()

        #print (" self.data: ", self.data.shape, ", size: ", sys.getsizeof(self.data.storage()))

        if self.verbose:

            print ("Input size: ",self.data.shape, int(sys.getsizeof(self.data)), "MB")
            print ("Load raw data (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
            print ("---------------------------------------")
            print ('')
                       

    def make_objective(self):
        start = dt.datetime.now().timestamp()
        if self.verbose:
            print ("Computing objective ")
           
        try: 
            self.obj_gpu*=0.
        except:
            self.obj_gpu = torch.zeros((self.temps.shape[2], self.data.shape[1]+self.STIME-1),
                                   dtype=torch.float).cuda()
        
        #print (" self.obj_gpu: ", self.obj_gpu.shape, ", size: ", sys.getsizeof(self.obj_gpu.storage()))
        # make objective using full rank (i.e. no SVD)
        if self.svd_flag==False:
            
            #traces = torch.from_numpy(self.data).to(device) 
            traces = self.data
            temps = torch.from_numpy(self.temps).cuda()
            for i in range(self.K):
                self.obj_gpu[i,:] = nn.functional.conv1d(traces[None, self.vis_chan[i],:], 
                                                         temps[None,self.vis_chan[i],:,i],
                                                         padding=self.STIME-1)[0][0]
                
        # use SVD to make objective:
        else:         
            # U_ x H_ (spatial * vals)
            # looping over ranks
            for r in range(self.RANK):
                mm = torch.mm(self.spatial_gpu[r::self.RANK]*self.singular_gpu[r::self.RANK], 
                              self.data)[None,None]

                #print (" mm: ", mm.shape, ", size: ", sys.getsizeof(mm.storage()))

                for i in range(self.temps.shape[2]):
                    self.obj_gpu[i,:] += nn.functional.conv1d(mm[:,:,i,:], 
                                                         self.filters_gpu[:,:,i*self.RANK+r, :], 
                                                         padding=self.STIME-1)[0][0]
            
            del mm
            torch.cuda.empty_cache()
            
        torch.cuda.synchronize()
        
        # compute objective function = 2 x Convolution term - norms
        self.obj_gpu*= 2.
        
        # standard objective function  
        if True:
            self.obj_gpu-= self.norm
        # adaptive value 
        else:
            self.obj_gpu-= 1.25*self.norm

        if self.verbose:
            print ("Total time obj func (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
            print ("---------------------------------------")
            print ('')
             
    
    # Cat: TODO Is this function used any longer?
    def set_objective_infinities(self):
        
        self.inf_thresh = -1E10
        zero_trace = torch.zeros(self.obj_gpu.shape[1],dtype=torch.float).cuda()
        
        # loop over all templates and set to -Inf
        for k in range(self.obj_gpu.shape[0]):
            idx = torch.where(self.obj_gpu[k]<self.inf_thresh,
                          zero_trace+1,
                          zero_trace)
            idx = torch.nonzero(idx)[:,0]
            self.obj_gpu[k][idx]=-float("Inf")
        
            
    def save_spikes(self):
        # # save offset of chunk time; spiketimes and neuron ids
        self.offset_array.append(self.offset)
        self.spike_array.append(self.spike_times[:,0])
        self.neuron_array.append(self.neuron_ids[:,0])
        self.shift_list.append(self.xshifts)
        
                
    def subtraction_step(self):
        
        start = dt.datetime.now().timestamp()

        # initialize arrays
        self.n_iter=0
        
        # tracks the number of addition steps during SCD
        self.add_iteration_counter=0
        self.save_spike_flag=True
        
        for k in range(self.max_iter):
            if False:
                #if k < 30:
                np.save(self.out_dir+'/objectives/chunk'+
                        str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', self.obj_gpu.cpu().data.numpy())
 
            # **********************************************
            # *********** SCD ADDITION STEP ****************
            # **********************************************
            # Note; this step needs to be carried out before peak search + subtraction to make logic simpler
            if self.scd:
                # old scd method where every 10 iterations, there's a random addition step of spikes from up to 5 prev iterations
                if False:
                    if (k%2==10) and (k>0):
                        if False:
                            # add up to spikes from up to 5 previous iterations
                            idx_iter = np.random.choice(np.arange(min(len(self.spike_array),self.scd_max_iteration)),
                                                        size=min(self.n_iter,self.scd_n_additions),
                                                        replace=False)
                            for idx_ in idx_iter: 
                                self.add_cpp(idx_)
                    
                # newer scd method: inject spikes from top 10 iterations and redeconvolve 
                else:
                    # updated exhuastive SCD over top 10 deconv iterations
                    # This conditional checks that loop is in an iteration that should be an addition step
                    if ((k%(self.n_scd_iterations*2))>=self.n_scd_iterations and \
                        (k%(self.n_scd_iterations*2))<(self.n_scd_iterations*2)) and \
                        (k<self.n_scd_stages*self.n_scd_iterations*2):
                    # if ((k>=10) and (k<20)) or ((k>=30) and (k<40)) or ((k>=50) and (k<60)):
                        self.save_spike_flag=False
                        # add spikes back in; then find peaks again below
                        self.add_cpp_allspikes(self.add_iteration_counter)                

                  
            # **********************************************
            # **************** FIND PEAKS ******************
            # **********************************************
            search_time = self.find_peaks()

            if self.spike_times.shape[0]==0:
                if self.verbose:
                    print ("... no detected spikes, exiting...")
                break                
            
            # **********************************************
            # **************** FIND SHIFTS *****************
            # **********************************************
            shift_time = self.find_shifts()
            
                
            # **********************************************
            # **************** SUBTRACTION STEP ************
            # **********************************************
            total_time = self.subtract_cpp()           
           

            # **********************************************
            # ************** SCD FINISHING UP **************
            # **********************************************
            # Note; after adding spikes back in - and running peak discover+subtraction
            #       - need to reassign rediscovered spikes back to the original list where they came from
            if self.scd:
                #if ((k>=10) and (k<=19)) or ((k>=30) and (k<40)) or ((k>=50) and (k<60)):
                if ((k%(self.n_scd_iterations*2))>=self.n_scd_iterations and \
                    (k%(self.n_scd_iterations*2))<(self.n_scd_iterations*2)) and \
                    (k<self.n_scd_stages*self.n_scd_iterations*2):
                    # insert spikes back to original iteration. 
                    self.spike_array[self.add_iteration_counter] = self.spike_times[:,0]
                    self.neuron_array[self.add_iteration_counter] = self.neuron_ids[:,0]
                    self.shift_list[self.add_iteration_counter] = self.xshifts
                
                    self.add_iteration_counter+=1

            # **********************************************
            # ************** POST PROCESSING ***************
            # **********************************************
            # save spiketimes only when doing deconv outside SCD loop
            if self.save_spike_flag:
                self.save_spikes()

            # reset regular spike save after finishing SCD (note: this should be done after final addition/subtraction
            #       gets added to the list of spikes;
            #       otherwise the spieks are saved twice
            if (k%(self.n_scd_iterations*2)==0):
                self.save_spike_flag=True
                self.add_iteration_counter=0
                
            # increase index
            self.n_iter+=1
        
            # post-processing steps;
            if self.verbose:
                if self.n_iter%self.print_iteration_counter==0:
                    print ("  iter: ", self.n_iter, "  obj_funct peak: ", 
                            np.round(t_max.item(),1), 
                           " # of spikes ", self.spike_times.shape[0], 
                           ", search time: ", np.round(search_time,6), 
                           ", quad fit time: ", np.round(shift_time,6),
                           ", subtract time: ", np.round(total_time,6), "sec")
                                   
        if self.verbose:
            print ("Total subtraction step: ", np.round(dt.datetime.now().timestamp()-start,3))
        
    
        #np.save('/home/cat/saved_array.npy', self.saved_gpu_array)
        
    def find_shifts(self):
        '''  Function that fits quadratic to 3 points centred on each peak of obj_func 
        '''
        
        start1 = dt.datetime.now().timestamp()
        #print (self.neuron_ids.shape, self.spike_times.shape)
        if self.neuron_ids.shape[0]>1:
            idx_tripler = (self.neuron_ids, self.spike_times.squeeze()[:,None]+self.peak_pts)
        else:
            idx_tripler = (self.neuron_ids, self.spike_times+self.peak_pts)
        
       # print ("idx tripler: ", idx_tripler)
        self.threePts = self.obj_gpu[idx_tripler]
        #np.save('/home/cat/trips.npy', self.threePts.cpu().data.numpy())
        self.shift_from_quad_fit_3pts_flat_equidistant_constants(self.threePts.transpose(0,1))

        return (dt.datetime.now().timestamp()- start1)

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
        torch.cuda.synchronize()
        self.gpu_max, self.neuron_ids = torch.max(self.obj_gpu, 0)
        torch.cuda.synchronize()
        end_max = dt.datetime.now().timestamp()-start

        # Second step: find relative peaks across max function above for some lockout window
        #       input: n_times (i.e. values of energy at each point in time)
        #       output:  1D array = relative peaks across time for given lockout_window
        # Cat: TODO: this may atually crash if a spike is located in exactly the 1 time step bewteen buffer and 2 xlockout widnow
        window_maxima = torch.nn.functional.max_pool1d_with_indices(self.gpu_max.view(1,1,-1), 
                                                                    self.lockout_window, 1, 
                                                                    padding=self.lockout_window//2)[1].squeeze()
        candidates = window_maxima.unique()
        self.spike_times = candidates[(window_maxima[candidates]==candidates).nonzero()]
       
        # Third step: only deconvolve spikes where obj_function max > threshold
        # Cat: TODO: also, seems like threshold might get stuck on artifact peaks
        idx = torch.where(self.gpu_max[self.spike_times]>self.deconv_thresh, 
                          self.gpu_max[self.spike_times]*0+1, 
                          self.gpu_max[self.spike_times]*0)
        idx = torch.nonzero(idx)[:,0]
        self.spike_times = self.spike_times[idx]

        # Fourth step: exclude spikes that occur in lock_outwindow at start;
        # Cat: TODO: check that this is correct, 
        #      unclear whetther spikes on edge of window get correctly excluded
        #      Currently we lock out first ~ 60 timesteps (for 3ms wide waveforms)
        #       and last 120 timesteps
        #      obj function is usually rec_len + buffer*2 + lockout_window
        #                   e.g. 100000 + 200*2 + 60 = 100460
        
        # original window
        if False:
            idx1 = torch.where((self.spike_times>self.lockout_window) &
                           #(self.spike_times<(self.obj_gpu.shape[1]-self.lockout_window)),
                           (self.spike_times<(self.obj_gpu.shape[1]-self.lockout_window*2)),
                           self.spike_times*0+1, 
                           self.spike_times*0)
        else:
            idx1 = torch.where((self.spike_times>self.lockout_window) &
                                (self.spike_times<(self.obj_gpu.shape[1]-self.lockout_window)),
                                self.spike_times*0+1, 
                                self.spike_times*0)


        idx2 = torch.nonzero(idx1)[:,0]
        self.spike_times = self.spike_times[idx2]
        #print ("self.spke_times: ", self.spike_times[-10:], self.obj_gpu.shape)

        # save only neuron ids for spikes to be deconvolved
        self.neuron_ids = self.neuron_ids[self.spike_times]
        return (dt.datetime.now().timestamp()-start)         
    
        
    def subtract_cpp(self):
        
        start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
        
        spike_times = self.spike_times.squeeze()-self.lockout_window
        spike_temps = self.neuron_ids.squeeze()
        
        # zero out shifts if superres shift turned off
        # Cat: TODO: remove this computation altogether if not required;
        #           will save some time.
        if self.superres_shift==False:
            self.xshifts = self.xshifts*0
        
        # if single spike, wrap it in list
        # Cat: TODO make this faster/pythonic
        if self.spike_times.size()[0]==1:
            spike_times = spike_times[None]
            spike_temps = spike_temps[None]

        deconv.subtract_splines(
                    self.obj_gpu,
                    spike_times,
                    self.xshifts,
                    spike_temps,
                    self.coefficients,
                    self.tempScaling)

        torch.cuda.synchronize()
        
        # also fill in self-convolution traces with low energy so the
        #   spikes cannot be detected again (i.e. enforcing refractoriness)
        # Cat: TODO: read from CONFIG
        
        if self.refractoriness:
            #print ("filling in timesteps: ", self.n_time)
            deconv.refrac_fill(energy=self.obj_gpu,
                                  spike_times=spike_times,
                                  spike_ids=spike_temps,
                                  #fill_length=self.n_time,  # variable fill length here
                                  #fill_offset=self.n_time//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                                  fill_length=self.refractory*2+1,  # variable fill length here
                                  fill_offset=self.n_time//2+self.refractory//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                                  fill_value=-self.fill_value)

        torch.cuda.synchronize()
            
        return (dt.datetime.now().timestamp()-start)
                     
    def sample_spikes(self,idx_iter):
        """
            OPTION 1: pick 10% (or more) of spikes from a particular iteration and add back in;
                      - advantage: don't need to worry about spike overlap;
                      - disadvantage: not as diverse as other injection steps
        
            OPTION 2: Same as OPTION 1 but also loop over a few other iterations           
            
            ------------------------ SLOWER OPTIONS -------------------------------------------            
        
            OPTION 3: pick 10% of spikes from first 10 iterations and preserve lockout
                      - advantage, more diverse 
                      - disadvantage: have to find fast algorithm to remove spikes too close together

            OPTION 4: pick 10% of spikes from any of the previous iterations and preserve lockout
                      - disadvantage: have to find fast algorithm to remove spikes too close together
        
        """
        
        # OPTION 1: pick a single previous iteration index; for now only use the first 10 iterations
        #           - one issue is that later iterations have few spikes and //10 yeilds 0 for example
        #           - to dsicuss 

        # pick 10 % random spikes from the selected iteration
        # Cat: TODO: maybe more pythonic ways (i.e. faster); but not clear
        idx_inject = np.random.choice(np.arange(self.spike_array[idx_iter].shape[0]), 
                                size = self.spike_array[idx_iter].shape[0]//2, replace=False)
        
        # Cat: TODO: this is a bit hacky way to stop picking from some iteration:
        if idx_inject.shape[0]<10:
            return ([], [], [], False)
            
        idx_not = np.delete(np.arange(self.spike_array[idx_iter].shape[0]),idx_inject)

        # pick spikes from those lists
        spike_times_list = self.spike_array[idx_iter][idx_inject]-self.lockout_window
        spike_ids_list = self.neuron_array[idx_iter][idx_inject]
        spike_shifts_list= self.shift_list[idx_iter][idx_inject]

        # delete spikes, ids etc that were selected above; 
        self.spike_array[idx_iter] = self.spike_array[idx_iter][idx_not]
        self.neuron_array[idx_iter] = self.neuron_array[idx_iter][idx_not]
        self.shift_list[idx_iter] = self.shift_list[idx_iter][idx_not]
        
        # return lists for addition below
        return spike_times_list, spike_ids_list, spike_shifts_list, True


    def sample_spikes_allspikes(self,idx_iter):
        """
            Same as sample_spikes() but pick all spikes from a previous iteration,
        """

        spike_times_list = self.spike_array[idx_iter]-self.lockout_window
        spike_ids_list = self.neuron_array[idx_iter]
        spike_shifts_list= self.shift_list[idx_iter]

        return spike_times_list, spike_ids_list, spike_shifts_list, True
        
        
    def add_cpp(self, idx_iter):
        #start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
                        
        # select randomly 10% of spikes from previous deconv; 
        spike_times, spike_temps, spike_shifts, flag = self.sample_spikes(idx_iter)

        # Cat: TODO is this flag required still?
        if flag == False:
            return 
            
        # also fill in self-convolution traces with low energy so the
        #   spikes cannot be detected again (i.e. enforcing refractoriness)
        # Cat: TODO: investgiate whether putting the refractoriness back in is viable
        if self.refractoriness:
            deconv.refrac_fill(energy=self.obj_gpu,
                              spike_times=spike_times,
                              spike_ids=spike_temps,
                              #fill_length=self.n_time,  # variable fill length here
                              #fill_offset=self.n_time//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                              fill_length=self.refractory*2+1,  # variable fill length here
                              fill_offset=self.n_time//2+self.refractory//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                              fill_value=self.fill_value)
                              
            # deconv.subtract_spikes(data=self.obj_gpu,
                                   # spike_times=spike_times,
                                   # spike_temps=spike_temps,
                                   # templates=self.templates_cpp_refractory_add,
                                   # do_refrac_fill = False,
                                   # refrac_fill_val = -1e10)

        torch.cuda.synchronize()
        
        # Add spikes back in;
        deconv.subtract_splines(
                            self.obj_gpu,
                            spike_times,
                            spike_shifts,
                            spike_temps,
                            self.coefficients,
                            -self.tempScaling)

        torch.cuda.synchronize()
        
        return 
        
        
    def add_cpp_allspikes(self, idx_iter):
        #start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
                        
        # select randomly 10% of spikes from previous deconv; 
        #spike_times, spike_temps, spike_shifts, flag = self.sample_spikes(idx_iter)
        
        # select all spikes from a previous iteration
        spike_times, spike_temps, spike_shifts, flag = self.sample_spikes_allspikes(idx_iter)

        torch.cuda.synchronize()

        if flag == False:
            return 
            
        # also fill in self-convolution traces with low energy so the
        #   spikes cannot be detected again (i.e. enforcing refractoriness)
        # Cat: TODO: investgiate whether putting the refractoriness back in is viable
        if self.refractoriness:
            deconv.refrac_fill(energy=self.obj_gpu,
                              spike_times=spike_times,
                              spike_ids=spike_temps,
                              #fill_length=self.n_time,  # variable fill length here
                              #fill_offset=self.n_time//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                              fill_length=self.refractory*2+1,  # variable fill length here
                              fill_offset=self.n_time//2+self.refractory//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                              fill_value=self.fill_value)
                              
                              
            # deconv.subtract_spikes(data=self.obj_gpu,
                                   # spike_times=spike_times,
                                   # spike_temps=spike_temps,
                                   # templates=self.templates_cpp_refractory_add,
                                   # do_refrac_fill = False,
                                   # refrac_fill_val = -1e10)

        torch.cuda.synchronize()
        
        # Add spikes back in;
        deconv.subtract_splines(
                            self.obj_gpu,
                            spike_times,
                            spike_shifts,
                            spike_temps,
                            self.coefficients,
                            -self.tempScaling)

        torch.cuda.synchronize()
        
        return 
# # ****************************************************************************
# # ****************************************************************************
# # ****************************************************************************


class deconvGPU2(object):
                   
   #'''  Greedy + exhaustive deconv - TO BE IMPLEMNETED
   #'''   
   
    def __init__(self, CONFIG, fname_templates, out_dir):
        
        #
        self.out_dir = out_dir
        
        # initialize directory for saving
        self.seg_dir = os.path.join(self.out_dir,'segs')
        if not os.path.isdir(self.seg_dir):
            os.mkdir(self.seg_dir)

        # initalize parameters for 
        self.set_params(CONFIG, fname_templates, out_dir)
        
