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
        self.fill_value = 1E6

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

        if self.scd:
            # also need to make inverted templates for addition step
            self.initialize_cpp_inverted()
            
            # conver templates to bpslines
            self.templates_to_bsplines_inverted()
        
            # make dummy templates for recovery of refractory period
            # self.initialize_cpp_refractory()
            
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


    # # not currently used
    # def make_splines_parallel(self):
        
        # res = parmap.map(bsplines_parallel, 
                         # list(zip(self.temp_temp, self.vis_units)),
                         # processes=self.CONFIG.resources.n_processors,
                         # pm_pbar=True)                         
                    
        # self.coefficients = deconv.BatchedTemplates(res)


    # def initialize_cpp_refractory(self):
        
        # # make a list of pairwise batched temp_temp and their vis_units
        # self.templates_cpp_refractory = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp_refractory, self.vis_units)])   
 
        # self.templates_cpp_refractory_add = deconv.BatchedTemplates([deconv.Template(-nzData, nzInd) for nzData, nzInd in zip(self.temp_temp_refractory, self.vis_units)])   


    def initialize_cpp(self):

        # make a list of pairwise batched temp_temp and their vis_units
        self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])


    def initialize_cpp_inverted(self):
        
        # make a list of pairwise batched temp_temp and their vis_units
        self.temp_temp_cpp_inverted = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp_inverted, self.vis_units)])   
 
         
    def templates_to_bsplines(self):

        print ("  making template bsplines (todo: parallelize)")
        
        # Cat: TODO: implement proper parallelized bspline computation
        self.coefficients = deconv.BatchedTemplates([self.transform_template(template) for template in self.temp_temp_cpp])
        self.temp_temp_cpp = None

        #fname = os.path.join(self.svd_dir,'bsplines_'+
        #          str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
                  
        #if os.path.exists(fname)==False:
            # make initial lists of templates
            # Cat: TODO: is this initizliation still required?!
            
            # print ("Skiipping rest") 
            # if False:
                # # make bsplines 
                # coefficients_local = [self.transform_template(template) for template in self.temp_temp_cpp]
            
                # # save data from bsplines to disk
                
                # list_out = []
                # for k in range(len(coefficients_local)):
                    # list_out.append(coefficients_local[k].data)

                # np.save(fname, list_out, allow_pickle=True)
                # self.coefficients = deconv.BatchedTemplates(coefficients_local)
            
        # else:
            # # load saved coefficients before made into deconv.BatchedTemplates
            # coefficients_local = np.load(fname, allow_pickle=True)
                    
            # # load dummy cpp templates 
            # temp_temp_list = []
            # for k in range(len(self.temp_temp)):
                # temp_temp_empty = np.zeros((self.temp_temp[k].shape[0],self.temp_temp[k].shape[1]+4),'float32')
                # temp_temp_list.append(torch.from_numpy(temp_temp_empty).float().to(device))

            # self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(temp_temp_list, self.vis_units)])

            # self.coefficients = []
            # for k in range(len(coefficients_local)):
                # self.temp_temp_cpp[k].data.copy_(coefficients_local[k])
                # self.coefficients.append(self.temp_temp_cpp[k])

            # self.coefficients = deconv.BatchedTemplates(self.coefficients)

    def templates_to_bsplines_inverted(self):

        print ("  making template bsplines - inverted (todo: parallelize)")
        
        self.coefficients_inverted = deconv.BatchedTemplates([self.transform_template(template) for template in self.temp_temp_cpp_inverted])


        #fname = os.path.join(self.svd_dir,'bsplines_inverted_'+
        #          str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
        
        # if os.path.exists(fname)==False:
            # # make initial lists of templates
            # # Cat: TODO: is this required?
            # #self.coefficients_inverted = deconv.BatchedTemplates([self.transform_template(template) for template in self.temp_temp_cpp_inverted])
            
            # # make bsplines 
            # coefficients_local = [self.transform_template(template) for template in self.temp_temp_cpp_inverted]
        
            # # save data from bsplines to disk
            # list_out = []
            # for k in range(len(coefficients_local)):
                # list_out.append(coefficients_local[k].data)

            # np.save(fname, list_out, allow_pickle=True)
            # self.coefficients_inverted = deconv.BatchedTemplates(coefficients_local)
            
        # else:
            # # load saved coefficients before made into deconv.BatchedTemplates
            # coefficients_local = np.load(fname, allow_pickle=True)
                    
            # # make dummy cpp templates 
            # temp_temp_list = []
            # for k in range(len(self.temp_temp_inverted)):
                # temp_temp_empty = np.zeros((self.temp_temp_inverted[k].shape[0],self.temp_temp_inverted[k].shape[1]+4),'float32')
                # temp_temp_list.append(torch.from_numpy(temp_temp_empty).float().to(device))

            # self.temp_temp_cpp_inverted = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(temp_temp_list, self.vis_units)])

            # self.coefficients_inverted = []
            # for k in range(len(coefficients_local)):
                # self.temp_temp_cpp_inverted[k].data.copy_(coefficients_local[k])
                # self.coefficients_inverted.append(self.temp_temp_cpp_inverted[k])

            # self.coefficients_inverted = deconv.BatchedTemplates(self.coefficients_inverted)



    def fit_spline(self, curve, knots=None, prepad=0, postpad=0, order=3):
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

        print ("  making temp_temp filters (todo: move to GPU)")
        if self.update_templates_backwards:
            fname = os.path.join(self.svd_dir,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
            fname_inverted = os.path.join(self.svd_dir,'temp_temp_sparse_svd_inverted_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
            fname_refractory = os.path.join(self.svd_dir,'temp_temp_sparse_svd_refractory'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '_1.npy')
                                    
        else:
            fname = os.path.join(self.svd_dir,'temp_temp_sparse_svd_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
            fname_inverted = os.path.join(self.svd_dir,'temp_temp_sparse_svd_inverted_'+
                  str((self.chunk_id+1)*self.CONFIG.resources.n_sec_chunk_gpu_deconv) + '.npy')
            fname_refractory = os.path.join(self.svd_dir,'temp_temp_sparse_svd_refractory_'+
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
            temp_gpu = []
            temp_gpu_inverted = []
            for k in range(len(temp_temp_local)):
                temp_gpu.append(torch.from_numpy(temp_temp_local[k]).float().cuda())
                temp_gpu_inverted.append(torch.from_numpy(-temp_temp_local[k]).float().cuda())

            self.temp_temp = temp_gpu
            self.temp_temp_inverted = temp_gpu_inverted
            
            np.save(fname, self.temp_temp)
            np.save(fname_inverted, self.temp_temp_inverted)
            
            # also make dummy temp_temp values for refractory period subtraction

            # if self.scd:
                # self.temp_temp_refractory = []
                # for unit in range(len(self.temp_temp)):
                    # temp3 = self.temp_temp[unit].cpu().data.numpy().copy()

                    # # find the index of the template in its own visible unit stack
                    # idx = np.where(np.where(self.vis_units[unit])[0]==unit)[0]
                    # temp3 *= 0
                    # temp3[idx] = -1E6
                    # self.temp_temp_refractory.append(torch.from_numpy(temp3).float().to(device))

                # np.save(fname_refractory, self.temp_temp_refractory)
                                 
        else:
            self.temp_temp = np.load(fname, allow_pickle=True)
            self.temp_temp_inverted = np.load(fname_inverted, allow_pickle=True)

            # if self.scd:
                # self.temp_temp_refractory = np.load(fname_refractory, allow_pickle=True)
                
    
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

        # fname = os.path.join(self.svd_dir,'vis_units.npy')
        # np.save(fname, self.vis_units)
                        
                        
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
        #print (" Loaded templates shape (n_chan, n_time, n_temps): ", self.temps.shape)
        self.temps_gpu = torch.from_numpy(self.temps).float().cuda()
        
        
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

        start = dt.datetime.now().timestamp()

        # Old way of buffering using zeros
        # if False:
            # self.buffer=200
            # self.chunk_len = (self.CONFIG.resources.n_sec_chunk_gpu_deconv*self.CONFIG.recordings.sampling_rate)
            # self.data = np.zeros((self.N_CHAN, self.buffer*2+self.chunk_len))
        # else:
        self.data_cpu = self.reader.read_data_batch(
            chunk_id, add_buffer=True).T
        
        self.offset = self.reader.idx_list[chunk_id, 0] - self.reader.buffer

        self.data = torch.from_numpy(self.data_cpu).float().cuda()
        #torch.cuda.synchronize()
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
                                   dtype=torch.float).cuda()
            
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
        #self.add_spike_times.append(spike_times)
        #self.add_spike_temps.a
        
                
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
                    self.coefficients)

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
                            self.coefficients_inverted)

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
                            self.coefficients_inverted)

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
        

    # def set_params(self, CONFIG, fname_templates, out_dir):

        # # 
        # self.CONFIG = CONFIG

        # # set root directory for loading:
        # # self.root_dir = self.CONFIG.data.root_folder
        
        # #
        # self.out_dir = out_dir

        # #
        # self.fname_templates = fname_templates
        
        # # number of seconds to load from recording
        # self.n_sec = self.CONFIG.resources.n_sec_chunk_gpu_deconv
        
        # # Cat: TODO: Load sample rate from disk
        # self.sample_rate = self.CONFIG.recordings.sampling_rate
        
        # # Cat: TODO: unclear if this is always the case
        # self.n_time = self.CONFIG.recordings.spike_size_ms*self.sample_rate//1000+1
        
        # # set length of lockout window
        # # Cat: TODO: unclear if this is always correct
        # self.lockout_window = self.n_time-1

        # # length of conv filter
        # self.n_times = torch.arange(-self.lockout_window,self.n_time,1).long().to(device)

        # # set max deconv thersho
        # self.deconv_thresh = self.CONFIG.deconvolution.threshold

        # # svd compression flag
        # self.svd_flag = True
        
        # # rank for SVD
        # # Cat: TODO: read from CONFIG
        # # self.RANK = 5
        
        # # make a 3 point array to be used in quadratic fit below
        # self.peak_pts = torch.arange(-1,+2).to(device)
        
        
    # def initialize(self):
        
        # # load templates and svd componenets
        # self.load_temps()
        
        # # find vis-chans
        # self.visible_chans()
        
        # # find vis-units
        # self.template_overlaps()
        
        # # set all nonvisible channels to 0. to help with SVD
        # self.spatially_mask_templates()
           
        # # compute template convolutions
        # if self.svd_flag:
            # self.compute_temp_temp_svd()
        # # Cat: TODO we should dissable all non-SVD options?!
        # else:
            # self.compute_temp_temp()
   
        # # move data to gpu
        # self.data_to_gpu()
                
        # # initialize Ian's objects
        # self.initialize_cpp()

        # # conver templates to bpslines
        # self.templates_to_bsplines()
           
            
    # def run(self, chunk_id):
        
        # # save iteration 
        # self.chunk_id = chunk_id
        
        # # load raw data and templates
        # self.load_data(chunk_id)
        
        # # make objective function
        # self.make_objective()
        
        # # set inifinities where obj < 0 or some other value
        # # Cat:TODO not currently used; not sure this helps.
        # #self.set_objective_infinities()
                
        # # run 
        # self.subtraction_step()
                
        # # empty cache
        # torch.cuda.empty_cache()

    
    # def initialize_cpp(self):
        
        # # make a list of pairwise batched temp_temp and their vis_units
        # self.temp_temp_cpp = deconv.BatchedTemplates([deconv.Template(nzData, nzInd) for nzData, nzInd in zip(self.temp_temp, self.vis_units)])   
        
        
    # def templates_to_bsplines(self):

        # print ("  making template bsplines (TODO: make template bsplines in parallel on CPU")
        # print ("                           (TODO: save template bsplines to disk to not recompute)")
        
        # print ("     # of templates converting to bsplines: ", len(self.temp_temp_cpp))
        # self.coefficients = deconv.BatchedTemplates([self.transform_template(template) for template in self.temp_temp_cpp])


    # def fit_spline(self, curve, knots=None,  prepad=0, postpad=0, order=3):
        # if knots is None:
            # knots = np.arange(len(curve) + prepad + postpad)
        # return splrep(knots, np.pad(curve, (prepad, postpad), mode='symmetric'), k=order)


    # def transform_template(self, template, knots=None, prepad=7, postpad=3, order=3):

        # if knots is None:
            # knots = np.arange(len(template.data[0]) + prepad + postpad)
        # splines = [
            # self.fit_spline(curve, knots=knots, prepad=prepad, postpad=postpad, order=order) 
            # for curve in template.data.cpu().numpy()
        # ]
        # coefficients = np.array([spline[1][prepad-1:-1*(postpad+1)] for spline in splines], dtype='float32')
        # return deconv.Template(torch.from_numpy(coefficients).cuda(), template.indices)

     
    # def compute_temp_temp_svd(self):

        # print ("  computing temp_temp svd on cpu (TODO: move this to GPU)")
        # # 
        # self.compress_templates()

        # fname = os.path.join(self.out_dir,'temp_temp_sparse_svd.npy')
        # if os.path.exists(fname)==False:

            # # recompute vis chans and vis units 
            # # Cat: TODO Fix this so dont' have to repeat it here
            # self.up_up_map = None
            # deconv_dir = ''

            # # 
            # units = np.arange(self.temps.shape[2])
            
            # # Cat: TODO: work on multi CPU and GPU versions
            # self.temp_temp= self.parallel_conv_filter(units, 
                                                    # self.n_time,
                                                    # self.up_up_map,
                                                    # deconv_dir)
                
            # temp_gpu = []
            # for k in range(len(self.temp_temp)):
                # temp_gpu.append(torch.from_numpy(self.temp_temp[k]).float().to(device))

            # self.temp_temp = temp_gpu
            # np.save(fname, self.temp_temp)
                                 
        # else:
            # self.temp_temp = np.load(fname)
        
        # # # load also the diagonal of the temp temp function:
        # # # Cat: TODO: is this required any longer? 
        # # fname = os.path.join(self.out_dir,"temp_temp_diagonal_svd.npy")
        # # if os.path.exists(fname)==False:
            # # self.temp_temp_diagonal = []
            # # for k in range(len(self.temp_temp)):
                # # self.temp_temp_diagonal.append(self.temp_temp[k][self.orig_index[k]].cpu().data.numpy())

            # # self.temp_temp_diagonal = np.vstack(self.temp_temp_diagonal)
            # # np.save(fname, self.temp_temp_diagonal)
        # # else:
            # # self.temp_temp_diagonal = np.load(fname)
    
    
    # def parallel_conv_filter(self,units, 
                            # n_time,
                            # up_up_map,
                            # deconv_dir):

        # # Cat: must load these structures from disk for multiprocessing step; 
        # #       where there are many templates; due to multiproc 4gb limit 
        # temporal = self.temporal #data['temporal']
        # singular = self.singular #data['singular']
        # spatial = self.spatial #data['spatial']
        # temporal_up = self.temporal_up #data['temporal_up']

        # vis_chan = self.vis_chan
        # unit_overlap = self.unit_overlap
        # approx_rank = self.RANK
        
        # #for unit2 in units:
        # conv_res_len = n_time * 2 - 1
        # pairwise_conv_array = []
        # print ("  TODO: parallelize temp_temp computation")
        # for unit2 in units:
            # if unit2%100==0:
                # print (" temp_temp: ", unit2)
            # n_overlap = np.sum(unit_overlap[unit2, :])
            # pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)
            # orig_unit = unit2 
            # masked_temp = np.flipud(np.matmul(
                    # temporal_up[unit2] * singular[orig_unit][None, :],
                    # spatial[orig_unit, :, :]))

            # for j, unit1 in enumerate(np.where(unit_overlap[unit2, :])[0]):
                # u, s, vh = temporal[unit1], singular[unit1], spatial[unit1] 

                # vis_chan_idx = vis_chan[:, unit1]
                # mat_mul_res = np.matmul(
                        # masked_temp[:, vis_chan_idx], vh[:approx_rank, vis_chan_idx].T)

                # for i in range(approx_rank):
                    # pairwise_conv[j, :] += np.convolve(
                            # mat_mul_res[:, i],
                            # s[i] * u[:, i].flatten(), 'full')

            # pairwise_conv_array.append(pairwise_conv)

        # return pairwise_conv_array

    # def compute_temp_temp(self):

        # fname = os.path.join(self.out_dir,"temp_temp_sparse.npy")

        # if os.path.exists(fname)==False: 
            # temps = np.load(os.path.join(self.fname_templates)).transpose(1,0,2)
            # print ("Computing temp_temp")
            # print ("  clustered templates (n_templates, n_chans, n_times): ", temps.shape)

            # # Cat: TODO: can make this less GPU memory intesnive and just load each indivdiaully
            # temps_reversed = np.flip(temps,axis=2).copy()
            # temps_reversed = torch.from_numpy(temps_reversed).float().to(device)

            # # input data stack: templates, channels, times
            # print ("  input templates: ", temps_reversed.shape)

            # # same as input_var except with extra dimension
            # filter_var = temps_reversed[:,None]
            # print ("  filter shapes: ", filter_var.shape)

            # units = np.arange(temps.shape[0])
            # temp_temp = []
            # start =dt.datetime.now().timestamp()
            # for unit in units:

                # # select vis_chans_local
                # vis_chan_local = np.where(self.vis_chan[:,unit])[0]

                # # select vis_units_local
                # vis_unit = np.where(self.vis_units[unit])[0]
    
                # i_var = temps_reversed[vis_unit][:,vis_chan_local,:]
                # f_var = filter_var[unit,:,vis_chan_local]
        
                # # convolve 
                # output_var = nn.functional.conv1d(i_var, f_var, padding=60)

                # # Cat: TODO: squeeze these arrays (and others)
                # temp_temp.append(output_var[:,0])
                # if unit%100==0:
                    # print ("  unit: ", unit, "input: ", i_var.shape, 
                           # ", filter: ", f_var.shape,
                           # ", output: ", output_var.shape,'\n')


            # print ("  total time : ", dt.datetime.now().timestamp()-start)
            
            # np.save(fname, temp_temp)
            # self.temp_temp = temp_temp
        # else:

            # self.temp_temp = np.load(fname)

        # # # load also the diagonal of the temp temp function:
        # # fname = os.path.join(self.out_dir,"temp_temp_diagonal.npy")
        # # if os.path.exists(fname)==False:
            # # self.temp_temp_diagonal = []
            # # for k in range(len(self.temp_temp)):
                # # self.temp_temp_diagonal.append(self.temp_temp[k][self.orig_index[k]].cpu().data.numpy())

            # # self.temp_temp_diagonal = np.vstack(self.temp_temp_diagonal)
            # # np.save(fname, self.temp_temp_diagonal)
        # # else:
            # # self.temp_temp_diagonal = np.load(fname)
 
    
    # def visible_chans(self):
        # #if self.vis_chan is None:
        # a = np.max(self.temps, axis=1) - np.min(self.temps, 1)
        
        # # Cat: TODO: must read visible channel/unit threshold from file;
        # self.vis_chan = a > self.vis_chan_thresh

        # a_self = self.temps.ptp(1).argmax(0)
        # for k in range(a_self.shape[0]):
            # self.vis_chan[a_self[k],k]=True


    # def template_overlaps(self):
        # """Find pairwise units that have overlap between."""
        # vis = self.vis_chan.T
        # self.unit_overlap = np.sum(
            # np.logical_and(vis[:, None, :], vis[None, :, :]), axis=2)
        # self.unit_overlap = self.unit_overlap > 0
        # self.vis_units = self.unit_overlap
                
    # def spatially_mask_templates(self):
        # """Spatially mask templates so that non visible channels are zero."""
        # for k in range(self.temps.shape[2]):
            # zero_chans = np.where(self.vis_chan[:,k]==0)[0]
            # self.temps[zero_chans,:,k]=0.

    # def load_temps(self):
        # ''' Load templates and set parameters
        # '''
        
        # # load templates
        # self.temps = np.load(self.fname_templates).transpose(2,1,0)
        # self.N_CHAN, self.STIME, self.K = self.temps.shape
        # print (" Loaded templates shape (n_chan, n_time, n_temps): ", self.temps.shape)
        # self.temps_gpu = torch.from_numpy(self.temps).float().to(device)
        
        
    # def compress_templates(self):
        # """Compresses the templates using SVD and upsample temporal compoents."""

        # ## compute everythign using SVD
        # fname = os.path.join(self.out_dir,'templates_svd.npz')
        # print ("self.out_dir ", self.out_dir)
        # if os.path.exists(fname)==False:
        # #if True:
            # print ("  computing SVD on templates (TODO: implement torch.svd()")
    
            # #self.approx_rank = self.RANK
            # self.temporal, self.singular, self.spatial = np.linalg.svd(
                # np.transpose(np.flipud(np.transpose(self.temps,(1,0,2))),(2, 0, 1)))
            
            # # Keep only the strongest components
            # self.temporal = self.temporal[:, :, :self.RANK]
            # self.singular = self.singular[:, :self.RANK]
            # self.spatial = self.spatial[:, :self.RANK, :]
            
            # print ("self.temps: ", self.temps.shape)
            # #print ("self.temps.transpose: ", self.temps_cpu.transpose(1,0,2)[0][0])
            
            # print ("temporal: ", self.temporal.shape)
            # print ("self.temporal: ", self.temporal[0][0])
            # print (" singular: ", self.singular.shape)
            # print (" spatial: ", self.spatial.shape)            
            
            # # Upsample the temporal components of the SVD
            # # in effect, upsampling the reconstruction of the
            # # templates.
            # # Cat: TODO: No upsampling is needed; to remove temporal_up from code
            # self.temporal_up = self.temporal
           
            # np.savez(fname, temporal=self.temporal, singular=self.singular, 
                     # spatial=self.spatial, temporal_up=self.temporal_up)
            
        # else:
            # print ("  loading SVD (from disk)")
                
            # # load data for for temp_temp computation
            # data = np.load(fname)
            # self.temporal_up = data['temporal_up']
            # self.temporal = data['temporal']
            # self.singular = data['singular']
            # self.spatial = data['spatial']                                     
            
      
    # def data_to_gpu(self):
    
        # # compute norm
        # norm = np.zeros((self.vis_chan.shape[1],1),'float32')
        # for i in range(self.vis_chan.shape[1]):
            # norm[i] = np.sum(np.square(self.temps.transpose(1,0,2)[:, self.vis_chan[:,i], i]))
        
        # #move data to gpu
        # self.norm = torch.from_numpy(norm).float().to(device)
        
        # # load vis chans on gpu
        # self.vis_chan_gpu=[]
        # for k in range(self.vis_chan.shape[1]):
            # self.vis_chan_gpu.append(torch.from_numpy(np.where(self.vis_chan[:,k])[0]).long().to(device))
        # self.vis_chan = self.vis_chan_gpu
            
        # # load vis_units onto gpu
        # self.vis_units_gpu=[]
        # for k in range(self.vis_units.shape[1]):
            # self.vis_units_gpu.append(torch.FloatTensor(np.where(self.vis_units[k])[0]).long().to(device))
        # self.vis_units = self.vis_units_gpu
        
        # # move svd items to gpu
        # if self.svd_flag:
            # self.n_rows = self.temps.shape[2] * self.RANK
            # self.spatial_gpu = torch.from_numpy(self.spatial.reshape([self.n_rows, -1])).float().to(device)
            # self.singular_gpu = torch.from_numpy(self.singular.reshape([-1, 1])).float().to(device)
            # self.temporal_gpu = np.flip(self.temporal,1)
            # self.filters_gpu = torch.from_numpy(self.temporal_gpu.transpose([0, 2, 1]).reshape([self.n_rows, -1])).float().to(device)[None,None]

  
    # def load_data(self, chunk_id):
        # '''  Function to load raw data 
        # '''

        # start = dt.datetime.now().timestamp()

        # # Old way of buffering using zeros
        # # if False:
            # # self.buffer=200
            # # self.chunk_len = (self.CONFIG.resources.n_sec_chunk_gpu_deconv*self.CONFIG.recordings.sampling_rate)
            # # self.data = np.zeros((self.N_CHAN, self.buffer*2+self.chunk_len))
        # # else:
        # self.data_cpu = self.reader.read_data_batch(
            # chunk_id, add_buffer=True).T

        
        # self.offset = self.reader.idx_list[chunk_id, 0] - self.reader.buffer

        # self.data = torch.from_numpy(self.data_cpu).float().to(device)
        # torch.cuda.synchronize()
        # #print ("Input size: ",self.data.shape, int(sys.getsizeof(self.data.astype(np.float32))/1E6), "MB")
        # if self.verbose:

            # print ("Input size: ",self.data.shape, int(sys.getsizeof(self.data)), "MB")
            # print ("Load raw data (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
            # print ("---------------------------------------")
            # print ('')
                       

    # def make_objective(self):
        # start = dt.datetime.now().timestamp()
        # if self.verbose:
            # print ("Computing objective ")

        # # transfer data to GPU
        # self.obj_gpu = torch.zeros((5, self.temps.shape[2], self.data.shape[1]+self.STIME-1),
                                       # dtype=torch.float, device='cuda')
            
        # # make objective using full rank (i.e. no SVD)
        # if self.svd_flag==False:
            
            # #traces = torch.from_numpy(self.data).to(device) 
            # traces = self.data
            # temps = torch.from_numpy(self.temps).to(device)
            # for i in range(self.K):
                # self.obj_gpu[i,:] = nn.functional.conv1d(traces[None, self.vis_chan[i],:], 
                                                         # temps[None,self.vis_chan[i],:,i],
                                                         # padding=self.STIME-1)[0][0]
                
        # # use SVD to make objective:
        # else:
            # # move data to gpu
            # data_gpu = self.data

            # # U_ x H_ (spatial * vals)
            # # looping over ranks
            # for r in range(self.RANK):
                # mm = torch.mm(self.spatial_gpu[r::self.RANK]*self.singular_gpu[r::self.RANK], 
                              # data_gpu)[None,None]

                # for i in range(self.temps.shape[2]):
                    # self.obj_gpu[0,i,:] += nn.functional.conv1d(mm[:,:,i,:], 
                                                         # self.filters_gpu[:,:,i*self.RANK+r, :], 
                                                         # padding=self.STIME-1)[0][0]
            # for k in range(1,5,1):
                # self.obj_gpu[k]= self.obj_gpu[0]

        # torch.cuda.synchronize()
        
        # # compute objective function = 2 x Convolution term - norms
        # self.obj_gpu*= 2.
        
        # # standard objective function  
        # #if True:
        # self.obj_gpu-= self.norm
        # # adaptive value 
        # #else:
        # #    self.obj_gpu-= 1.25*self.norm
            
        
        # if self.verbose:
            # print ("Total time obj func (run every chunk): ", np.round(dt.datetime.now().timestamp()-start,2),"sec")
            # print ("---------------------------------------")
            # print ('')
             
    
    # # # Cat: TODO Is this function used any longer?
    # # def set_objective_infinities(self):
        
        # # self.inf_thresh = -1E10
        # # zero_trace = torch.zeros(self.obj_gpu.shape[2],dtype=torch.float, device='cuda')
        
        # # # loop over all templates and set to -Inf
        # # for k in range(self.obj_gpu.shape[1]):
            # # idx = torch.where(self.obj_gpu[k]<self.inf_thresh,
                          # # zero_trace+1,
                          # # zero_trace)
            # # idx = torch.nonzero(idx)[:,0]
            # # self.obj_gpu[k][idx]=-float("Inf")
        
            
    # def save_spikes(self):
        
        # # # save offset of chunk time; spiketimes and neuron ids
        # self.offset_array.append(self.offset)
        # self.spike_array.append(self.spike_times[:,0])
        # self.neuron_array.append(self.neuron_ids[:,0])
        # self.shift_list.append(self.xshifts)
        
                
    # def subtraction_step(self):
        
        # start = dt.datetime.now().timestamp()

        # # initialize arrays
        # self.obj_array=[]
        # self.obj_array_residual=[]
        # self.n_iter=0
        # for k in range(self.max_iter):
            
            # # find peaks
            # search_time = self.find_peaks()
            
            # ## test if trehsold reached; 
            # ## Cat: TODO: this info might already be availble from above function
            # t_max,_ = torch.max(self.gpu_max[self.lockout_window:-self.lockout_window],0)
            # if t_max<self.deconv_thresh:
                # if self.verbose:
                    # print ("... threshold reached, exiting...")
                # break

            # elif self.spike_times.shape[0]==0:
                # if self.verbose:
                    # print ("... no detected spikes, exiting...")
                # break                
                
            # # save 3 point arrays for quad fit
            # start1 = dt.datetime.now().timestamp()
            # self.find_shifts()
            # shift_time = dt.datetime.now().timestamp()- start1
            
            # # debug flag; DO NOT use otherwise
            # if self.save_objective:
                # self.obj_array.append(self.obj_gpu.cpu().data.numpy().copy())
                
            # # subtract spikes
            # total_time = self.subtract_cpp()
            
            # # debug flag; DO NOT use otherwise
            # if self.save_objective:
                # self.obj_array_residual.append(self.obj_gpu.cpu().data.numpy().copy())
                
            # if self.verbose:
                # if self.n_iter%self.print_iteration_counter==0:
                    # print ("  iter: ", self.n_iter, "  obj_funct peak: ", np.round(t_max.item(),1), 
                       # " # of spikes ", self.spike_times.shape[0], 
                       # ", search time: ", np.round(search_time,6), 
                       # ", quad fit time: ", np.round(shift_time,6),
                       # ", subtract time: ", np.round(total_time,6), "sec")
            
            # # save spiketimes
            # self.save_spikes()
            
            # # increase index
            # self.n_iter+=1
            
        # if self.verbose:
            # print ("Total subtraction step: ", np.round(dt.datetime.now().timestamp()-start,3))
        
    
    # def find_shifts(self):
        # '''  Function that fits quadratic to 3 points centred on each peak of obj_func 
        # '''
        
        # #print (self.neuron_ids.shape, self.spike_times.shape)
        # if self.neuron_ids.shape[0]>1:
            # idx_tripler = (self.neuron_ids, self.spike_times.squeeze()[:,None]+self.peak_pts)
        # else:
            # idx_tripler = (self.neuron_ids, self.spike_times+self.peak_pts)
        
       # # print ("idx tripler: ", idx_tripler)
        
        # for k in range(5):
            # self.threePts = self.obj_gpu[k][idx_tripler]
            # #np.save('/home/cat/trips.npy', self.threePts.cpu().data.numpy())
            # xshifts = self.shift_from_quad_fit_3pts_flat_equidistant_constants(self.threePts.transpose(0,1))
        
            # self.xshifts.append(xshifts)
        

        
    # # compute shift for subtraction in objective function space
    # def shift_from_quad_fit_3pts_flat_equidistant_constants(self, pts):
        # ''' find x-shift after fitting quadratic to 3 points
            # Input: [n_peaks, 3] which are values of three points centred on obj_func peak
            # Assumes: equidistant spacing between sample times (i.e. the x-values are hardcoded below)
        # '''

        # xshifts = ((((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/2)/
                  # (-2*((pts[0]-pts[1])-(((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/(2)))))-1        
        
        # return xshifts
        
    # def find_peaks(self):
        # ''' Function to use torch.max and an algorithm to find peaks
        # '''
        
        # # Cat: TODO: make sure you can also deconvolve ends of data;
        # #      currently using padding here...

        # # First step: find peaks across entire energy function across dimension 0
        # #       input: (n_neurons, n_times)
        # #       output:  n_times (i.e. the max energy function value at each point in time)
        # #       note: windows are padded
        # start = dt.datetime.now().timestamp()
        # for k in range(5):
            # self.gpu_max, self.neuron_ids = torch.max(self.obj_gpu[k], 0)
            # torch.cuda.synchronize()
            # end_max = dt.datetime.now().timestamp()-start

            # # Second step: find relative peaks across max function above for some lockout window
            # #       input: n_times (i.e. values of energy at each point in time)
            # #       output:  1D array = relative peaks across time for given lockout_window
            # # Cat: TODO: this may atually crash if a spike is located in exactly the 1 time step bewteen buffer and 2 xlockout window
            # window_maxima = torch.nn.functional.max_pool1d_with_indices(self.gpu_max.view(1,1,-1), 
                                                                        # self.lockout_window*2, 1, 
                                                                        # padding=self.lockout_window)[1].squeeze()
            # candidates = window_maxima.unique()
            # self.spike_times = candidates[(window_maxima[candidates]==candidates).nonzero()]
           
            # # Third step: only deconvolve spikes where obj_function max > threshold
            # idx = torch.where(self.gpu_max[self.spike_times]>self.deconv_thresh, 
                              # self.gpu_max[self.spike_times]*0+1, 
                              # self.gpu_max[self.spike_times]*0)
            # idx = torch.nonzero(idx)[:,0]
            # self.spike_times = self.spike_times[idx]

            # # Fourth step: exclude spikes that occur in lock_outwindow at start;
            # idx1 = torch.where((self.spike_times>self.lockout_window) &
                               # #(self.spike_times<(self.obj_gpu.shape[1]-self.lockout_window)),
                               # (self.spike_times<self.obj_gpu[k].shape[1]),
            # #idx1 = torch.where(self.spike_times>self.lockout_window,
                               # self.spike_times*0+1, 
                               # self.spike_times*0)
            # idx2 = torch.nonzero(idx1)[:,0]
            # self.spike_times = self.spike_times[idx2]

            # # save only neuron ids for spikes to be deconvolved
            # self.neuron_ids = self.neuron_ids[self.spike_times]

        # return (dt.datetime.now().timestamp()-start)
        
        
    # def subtract_cpp(self):
        
        # start = dt.datetime.now().timestamp()
        
        # torch.cuda.synchronize()
        
        # spike_times = self.spike_times.squeeze()-self.lockout_window
        # spike_temps = self.neuron_ids.squeeze()
        
        # # if single spike, wrap it in list
        # if self.spike_times.size()[0]==1:
            # spike_times = spike_times[None]
            # spike_temps = spike_temps[None]
        
        # #shifts_coord_descent = torch.array
        # #for k in range(5):
        # deconv.subtract_splines(
                    # self.obj_gpu[2],
                    # spike_times,
                    # self.xshifts[2],
                    # spike_temps,
                    # self.coefficients)

        # torch.cuda.synchronize()

        # # also fill in self-convolution traces with low energy so the
        # #   spikes cannot be detected again (i.e. enforcing refactoriness)
        # deconv.refrac_fill(energy=self.obj_gpu[2],
                                  # spike_times=spike_times,
                                  # spike_ids=spike_temps,
                                  # fill_length=self.n_time,  # variable fill length here
                                  # fill_offset=self.n_time//2,       # again giving flexibility as to where you want the fill to start/end (when combined with preceeding arg
                                  # fill_value=-1E10)

        # torch.cuda.synchronize()
            
        # return (dt.datetime.now().timestamp()-start)
