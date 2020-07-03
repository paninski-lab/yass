import numpy as np
import sys, os, math
import datetime as dt
import scipy, scipy.signal
import parmap
from scipy.interpolate import splrep, splev, make_interp_spline, splder, sproot, interp1d
from tqdm import tqdm

# doing imports inside module until travis is fixed
# Cat: TODO: move these to the top once Peter's workstation works
import torch
from torch import nn
#from torch.autograd import Variable

# cuda package to do GPU based spline interpolation and subtraction
import cudaSpline as deconv
import rowshift as rowshift

from yass.postprocess.duplicate import abs_max_dist
from yass.deconvolve.util import WaveForms
from yass.deconvolve.utils import TempTempConv, reverse_shifts


# # ****************************************************************************
# # ****************************************************************************
# # ****************************************************************************


def parallel_conv_filter2(units, 
                          n_time,
                          up_up_map,
                          deconv_dir,
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
    
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
        #print("... deconv using GPU device: ", torch.cuda.current_device())
        
        #
        self.out_dir = out_dir
        
        # initialize directory for saving
        self.seg_dir = os.path.join(self.out_dir, 'segs')
        if not os.path.exists(self.seg_dir):
            os.mkdir(self.seg_dir)
            
        self.init_dir = os.path.join(self.out_dir, 'initialize')
        if not os.path.exists(self.init_dir):
            os.mkdir(self.init_dir)

        #self.temps_dir = os.path.join(self.out_dir, 'template_updates')
        #if not os.path.exists(self.temps_dir):
        #    os.mkdir(self.temps_dir)

        self.fname_templates = fname_templates

        # initalize parameters for 
        self.set_params(CONFIG, out_dir)

    def set_params(self, CONFIG, out_dir):

        # 
        self.CONFIG = CONFIG

        # 
        self.fill_value = 1E4

        # objective function scaling for the template term;
        self.tempScaling = 2.0

        # refractory period
        # Cat: TODO: move to config
        refrac_ms = 1
        self.refractory = int(self.CONFIG.recordings.sampling_rate/1000*refrac_ms)

        # set max deconv threshold
        self.deconv_thresh = self.CONFIG.deconvolution.threshold


    def initialize(self, move_data_to_gpu=True):
        
        #print("... deconv using GPU device: ", torch.cuda.current_device())

        # load templates and svd componenets
        self.load_temps()

        self.initialize_shift_svd()

        # convert templates to bpslines
        self.templates_to_bsplines()

        # large units for height fit
        if self.fit_height:
            self.ptps = self.temps.ptp(1).max(0)
            self.large_units = np.where(self.ptps > self.fit_height_ptp)[0]

        if move_data_to_gpu:
            # compute norms and move data to GPU
            self.data_to_gpu()


    def load_temps(self):
        ''' Load templates and set parameters
        '''
        
        # load templates
        #print ("Loading template: ", self.fname_templates)
        self.temps = np.load(self.fname_templates, allow_pickle=True).transpose(2,1,0)
        self.N_CHAN, self.STIME, self.K = self.temps.shape
        
        # set length of lockout window
        self.lockout_window = self.STIME - 1

    def initialize_shift_svd(self):
        
        fname_templates_denoised = os.path.join(
            self.init_dir, 'templates_denoised.npy')
        fname_spat_comp = os.path.join(
            self.init_dir, 'spat_comp.npy')
        fname_temp_comp = os.path.join(
            self.init_dir, 'temp_comp.npy')
        fname_align_shifts = os.path.join(
            self.init_dir, 'align_shifts.npy')
        fname_subtraction_offset = os.path.join(
            self.init_dir, 'subtraction_offset.npy')
        fname_peak_time_residual_offset = os.path.join(
            self.init_dir, 'peak_time_residual_offset.npy')
        fname_temp_temp = os.path.join(
            self.init_dir, 'temp_temp.npy')
        fname_vis_units = os.path.join(
            self.init_dir, 'vis_units.npy')
        
        # pad len is constant and is 1.5 ms on each side, i.e. a total of 3 ms
        self.pad_len = int(1.5 * self.CONFIG.recordings.sampling_rate / 1000.)
        # jitter_len is selected in a way that deconv works with 3 ms signals
        self.jitter_len = self.pad_len
        self.jitter_diff = 0
        #if self.CONFIG.recordings.spike_size_ms > 3:
        #    self.jitter_diff = (self.CONFIG.recordings.spike_size_ms - 3)
        #    self.jitter_diff = int(self.jitter_diff * self.CONFIG.recordings.sampling_rate / 1000. / 2.)
        self.jitter_len = self.pad_len + self.jitter_diff

        if (os.path.exists(fname_templates_denoised) and
            os.path.exists(fname_temp_temp) and
            os.path.exists(fname_vis_units) and
            os.path.exists(fname_spat_comp) and
            os.path.exists(fname_temp_comp) and
            os.path.exists(fname_align_shifts) and
            os.path.exists(fname_subtraction_offset) and
            os.path.exists(fname_peak_time_residual_offset)):

            self.temps = np.load(fname_templates_denoised, allow_pickle=True)
            self.spat_comp = np.load(fname_spat_comp, allow_pickle=True)
            self.temp_comp = np.load(fname_temp_comp, allow_pickle=True)
            self.align_shifts = np.load(fname_align_shifts, allow_pickle=True)
            self.subtraction_offset = int(np.load(fname_subtraction_offset, allow_pickle=True))
            self.peak_time_residual_offset = np.load(fname_peak_time_residual_offset)
            
        else:
            ttc = TempTempConv(
                self.CONFIG, 
                templates=self.temps.transpose(2,0,1),
                geom=self.CONFIG.geom, rank=self.RANK,
                pad_len=self.pad_len, jitter_len=self.jitter_len, sparse=True)

            # update self.temps to the denoised templates
            self.temps = ttc.residual_temps.transpose(1, 2, 0)
            self.spat_comp = ttc.spat_comp.transpose([0,2,1])
            self.temp_comp = ttc.temp_comp
            self.align_shifts = ttc.align_shifts
            self.subtraction_offset = int(ttc.peak_time_temp_temp_offset)
            self.peak_time_residual_offset = ttc.peak_time_residual_offset

            # save the results
            np.save(fname_templates_denoised, self.temps, allow_pickle=True)
            np.save(fname_spat_comp, self.spat_comp, allow_pickle=True)
            np.save(fname_temp_comp, self.temp_comp, allow_pickle=True)
            np.save(fname_align_shifts, self.align_shifts, allow_pickle=True)
            np.save(fname_subtraction_offset, self.subtraction_offset, allow_pickle=True)
            np.save(fname_peak_time_residual_offset, self.peak_time_residual_offset, allow_pickle=True)
            np.save(fname_temp_temp, ttc.temp_temp, allow_pickle=True)
            np.save(fname_vis_units, ttc.unit_overlap, allow_pickle=True)

        
    def templates_to_bsplines(self):

        fname = os.path.join(self.init_dir, 'bsplines.npy')
        if not os.path.exists(fname):
            print ("  making template bsplines")
            fname_temp_temp = os.path.join(self.init_dir, 'temp_temp.npy')
            temp_temp = np.load(fname_temp_temp, allow_pickle=True)

            # multi-core bsplines
            if self.CONFIG.resources.multi_processing:
                coefficients = parmap.map(transform_template_parallel, temp_temp, 
                                          processes=self.CONFIG.resources.n_processors//2,
                                          pm_pbar=False)
            # single core
            else:
                coefficients = []
                for template in temp_temp:
                    coefficients.append(transform_template_parallel(template))
            np.save(fname, coefficients, allow_pickle=True)

            self.coefficients = coefficients
        else:
            print ("  ... loading coefficients from disk")
            self.coefficients = np.load(fname, allow_pickle=True)

    def data_to_gpu(self):
        
        self.peak_pts = torch.arange(-1,+2).cuda()

        #norm
        norm = np.sum(np.square(self.temps), (0, 1))
        self.norms = torch.from_numpy(norm).float().cuda()
        
        # spatial and temporal component of svd
        self.spat_comp = torch.from_numpy(self.spat_comp).float().cuda()
        self.temp_comp = torch.from_numpy(self.temp_comp).float().cuda()        

        print ("  ... moving coefficients to cuda objects")

        # load vis units
        fname_vis_units = os.path.join(self.init_dir, 'vis_units.npy')
        vis_units = np.load(fname_vis_units, allow_pickle=True)

        #
        coefficients_cuda = []
        for p in range(len(self.coefficients)):
            coefficients_cuda.append(deconv.Template(
                torch.from_numpy(self.coefficients[p]).float().cuda(),
                torch.from_numpy(vis_units[p]).long().cuda()))
        self.coefficients = deconv.BatchedTemplates(coefficients_cuda)

        del coefficients_cuda
        torch.cuda.empty_cache()

        if self.fit_height:
            self.large_units = torch.from_numpy(self.large_units).cuda()

    def run(self, chunk_id):

        # rest lists for each segment of time
        self.spike_array = []
        self.neuron_array = []
        self.shift_list = []
        self.height_list = []
        self.add_spike_temps = []
        self.add_spike_times = []
        
        # save iteration 
        self.chunk_id = chunk_id

        # load raw data and templates
        self.load_data(chunk_id)
        
        # make objective function
        #self.make_objective()
        self.make_objective_shifted_svd()

        # run
        self.subtraction_step()

        # empty cache
        torch.cuda.empty_cache()

        # gather results (and move to cpu)
        self.gather_results()

    def gather_results(self):

        # make spike train
        # get all spike times and neuron ids
        if self.spike_array.shape[0]>0:
            spike_times = torch.cat(self.spike_array)
            neuron_ids = torch.cat(self.neuron_array)
            spike_train = torch.stack((spike_times, neuron_ids), dim=1).cpu().numpy()

            # fix spike times
            spike_train[:, 0] = spike_train[:,0] + self.STIME//2 - (2 * self.jitter_diff)
            for unit in range(self.K):
                spike_train[spike_train[:, 1] == unit, 0] += self.peak_time_residual_offset[unit]
            self.spike_train = spike_train

            # make shifts and heights
            self.shifts = torch.cat(self.shift_list).cpu().numpy()
            self.heights = torch.cat(self.height_list).cpu().numpy()
        
        # if no spikes are found return empty lists
        else:
            self.spike_train = np.zeros((0,2),'int32')

            # make shifts and heights
            self.shifts = np.zeros(0,'float32')
            self.heights = np.zeros(0,'float32')


        self.spike_array = None
        self.neuron_array = None
        self.shift_list = None
        self.height_list = None

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
                

    def make_objective_shifted_svd(self):
        start = dt.datetime.now().timestamp()
        if self.verbose:
            print ("Computing objective ")       
       
        #obj_function = np.zeros([NUNIT, data.shape[1] + 61 - 1])
        #self.obj_gpu = torch.zeros((self.K, self.data.shape[1]+self.STIME-1 + 2 * self.jitter_diff),
        #                            dtype=torch.float).cuda()
        self.obj_gpu = torch.cuda.FloatTensor(self.K, self.data.shape[1]+self.STIME-1 + 2 * self.jitter_diff).fill_(0)
        for unit in range(self.K):
            # Do the shifts that was required for aligning template
            shifts = reverse_shifts(self.align_shifts[unit])
            #print ("shifts: ", shifts.shape)
            
            # this needs to be taken out of this loop and done single time
            shifts_gpu = torch.from_numpy(shifts).long().cuda()
            
            # CUDA code
            rowshift.forward(self.data, shifts_gpu)

            # multiplication step
            mm = torch.mm(self.spat_comp[unit], self.data)
                    
            # Sum over Rank
            for i in range(self.RANK):
                self.obj_gpu[unit,:]+= nn.functional.conv1d(mm[i][None,None,:],
                                               self.temp_comp[unit,i][None,None,:], 
                                               padding = self.STIME-1)[0][0]
                #print ("Convolution result: ", temp_out.shape)
                  
            # Undo the shifts that we did earlier
            #in_place_roll_shift(data, -shifts)
            rowshift.backward(self.data, shifts_gpu)

        #obj_function = 2 * obj_function - temp_norms[:NUNIT][:, None]  #drop NUNIT;  # drop additional dimensions;
        #print ("obj_function: ", self.obj_gpu.shape)
        #print ("self.norms: ", self.norms.shape)
        self.obj_gpu = 2 * self.obj_gpu - self.norms[:,None]  #drop NUNIT;  # drop additional dimensions;

        del mm
        #del temp_out
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        

    def save_spikes(self):
        # # save offset of chunk time; spiketimes and neuron ids
        #self.offset_array.append(self.offset)
        self.spike_array.append(self.spike_times[:,0])
        self.neuron_array.append(self.neuron_ids[:,0])
        self.shift_list.append(self.xshifts)
        self.height_list.append(self.heights)
                
                
    def subtraction_step(self):
        
        start = dt.datetime.now().timestamp()

        # initialize arrays
        self.n_iter=0
        
        # tracks the number of addition steps during SCD
        self.add_iteration_counter=0
        self.save_spike_flag=True
        
        for k in range(self.max_iter):
 
            # **********************************************
            # *********** SCD ADDITION STEP ****************
            # **********************************************
            # Note; this step needs to be carried out before peak search + subtraction to make logic simpler
            if self.scd:                    
                # # newer scd method: inject spikes from top 10 iterations and redeconvolve 
                # updated exhuastive SCD over top 10 deconv iterations
                # This conditional checks that loop is in an iteration that should be an addition step
                if ((k%(self.n_scd_iterations*2))>=self.n_scd_iterations and \
                    (k%(self.n_scd_iterations*2))<(self.n_scd_iterations*2)) and \
                    (k<self.n_scd_stages*self.n_scd_iterations*2):

                    # turn off saving spikes flag so that new spikes aren't appended
                    #       - instead they are inserted back into the original lcoation see conditional below
                    self.save_spike_flag=False

                    self.tempScaling_array = self.shift_list[self.add_iteration_counter]*0.0 + 2.0

                    
                    # add spikes back in; then run forward deconv below
                    self.add_cpp_allspikes()                

                  
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
            # **************** FIT HEIGHT *****************
            # **********************************************
            fit_height_time = self.compute_height()

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
                        
                    # insert spikes back to original iteration - no need to add append them as a new list
                    self.spike_array[self.add_iteration_counter] = self.spike_times[:,0]
                    self.neuron_array[self.add_iteration_counter] = self.neuron_ids[:,0]
                    self.shift_list[self.add_iteration_counter] = self.xshifts
                    self.height_list[self.add_iteration_counter] = self.heights
                    self.add_iteration_counter+=1

            # reset regular spike save after finishing SCD (note: this should be done after final addition/subtraction
            #       gets added to the list of spikes;
            #       otherwise the spieks are saved twice
            if (k%(self.n_scd_iterations*2)==0):
                self.save_spike_flag=True
                self.add_iteration_counter=0

            # **********************************************
            # ************** POST PROCESSING ***************
            # **********************************************
            # save spiketimes only when doing deconv outside SCD loop
            if self.save_spike_flag:
                self.save_spikes()
                                
            # increase index
            self.n_iter+=1
        
            # post-processing steps;

            # np.savez('/media/cat/4TBSSD/liam/512channels/2005-04-26-0/data002/tmp/final_deconv/icd/'+
                    # str(k)+'.npz',
                    # k = k,
                    # save_spike_flag = self.save_spike_flag,
                    # spike_array = self.spike_array,
                    # neuron_array = self.neuron_array,
                    # shift_list = self.shift_list
                    # )
            
        
        #rint ("# of iterations; ", k)
        #quit()
        if self.verbose:
            print ("Total subtraction step: ", np.round(dt.datetime.now().timestamp()-start,3))

        
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
        
    def compute_height(self):
        '''  Function that fits quadratic to 3 points centred on each peak of obj_func 
        '''
        
        start1 = dt.datetime.now().timestamp()

        if self.fit_height:
            # get peak value
            peak_vals = self.quad_interp_3pt(self.threePts.transpose(1,0), self.xshifts)

            # height
            height = 0.5*(peak_vals/self.norms[self.neuron_ids[:,0]] + 1)
            height[height < 1 - self.max_height_diff] = 1
            height[height > 1 + self.max_height_diff] = 1
            
            idx_small_ = ~torch.any(self.neuron_ids == self.large_units[None],1)
            height[idx_small_] = 1
            
            self.heights = height
            
        else:
            self.heights = torch.ones(len(self.xshifts)).cuda()

        return (dt.datetime.now().timestamp()- start1)

    def quad_interp_peak(self, pts):
        ''' find x-shift after fitting quadratic to 3 points
            Input: [n_peaks, 3] which are values of three points centred on obj_func peak
            Assumes: equidistant spacing between sample times (i.e. the x-values are hardcoded below)
        '''

        num = ((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/2
        denom = -2*((pts[0]-pts[1])-(((pts[1]-pts[2])*(-1)-(pts[0]-pts[1])*(-3))/(2)))
        num[denom==0] = 1
        denom[denom==0] = 1
        return (num/denom)-1    

    def quad_interp_3pt(self, vals, shift):
        a = 0.5*vals[0] + 0.5*vals[2] - vals[1]
        b = -0.5*vals[0] + 0.5*vals[2]
        c = vals[1]

        return a*shift**2 + b*shift + c

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

        #np.save('/media/cat/2TB/liam/49channels/data1_allset_shifted_svd/tmp/block_2/deconv/neuron_ids_'+
        #         str(self.n_iter)+'.npy', 
        #         self.neuron_ids.cpu().data.numpy())

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
        idx1 = torch.where((self.spike_times>(self.subtraction_offset)) &
                            (self.spike_times<(self.obj_gpu.shape[1]-(self.subtraction_offset))),
                            self.spike_times*0+1, 
                            self.spike_times*0)
        idx2 = torch.nonzero(idx1)[:,0]
        #self.spike_times = self.spike_times[idx2]
        self.spike_times = self.spike_times[idx2]
        #print ("self.spke_times: ", self.spike_times[-10:], self.obj_gpu.shape)

        # save only neuron ids for spikes to be deconvolved
        self.neuron_ids = self.neuron_ids[self.spike_times]
        #np.save('/media/cat/2TB/liam/49channels/data1_allset_shifted_svd/tmp/block_2/deconv/neuron_ids_'+str(self.n_iter)+
        #         '_postpeak.npy', 
        #         self.neuron_ids.cpu().data.numpy())
        
        return (dt.datetime.now().timestamp()-start)         
    
        
    def subtract_cpp(self):
        
        start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
        
        if False:
            self.spike_times = self.spike_times[:1]
            self.neuron_ids = self.neuron_ids[:1]
            self.xshifts = self.xshifts[:1]
            self.heights = self.heights[:1]
            self.obj_gpu *=0.

        #spike_times = self.spike_times.squeeze()-self.lockout_window
        spike_times = self.spike_times.squeeze()-self.subtraction_offset
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

        #print ("spke_times: ", spike_times, spike_times)
        #print ("spke_times: ", spike_times[:20], spike_times[-20:])
        
        # save metadata
        if False:
            if self.n_iter<500:
                self.objectives_dir = os.path.join(self.out_dir,'objectives')
                if not os.path.isdir(self.objectives_dir):
                    os.mkdir(self.objectives_dir)
                    
                np.save(self.out_dir+'/objectives/spike_times_inside_'+ 
                                   str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                   spike_times.squeeze().cpu().data.numpy())
                np.save(self.out_dir+'/objectives/spike_ids_inside_'+
                                   str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                   spike_temps.squeeze().cpu().data.numpy())
                np.save(self.out_dir+'/objectives/obj_gpu_'+
                                   str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                   self.obj_gpu.cpu().data.numpy())
                np.save(self.out_dir+'/objectives/shifts_'+
                                   str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                   self.xshifts.cpu().data.numpy())
                np.save(self.out_dir+'/objectives/tempScaling_'+
                                   str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                   self.tempScaling)
                np.save(self.out_dir+'/objectives/heights_'+
                                   str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                   self.heights.cpu().data.numpy())
            
                if False:
                    for k in range(len(self.coefficients)):
                        np.save(self.out_dir+'/objectives/coefficients_'+str(k)+"_"+
                                       str(self.chunk_id)+"_iter_"+str(self.n_iter)+'.npy', 
                                       self.coefficients[k].data.cpu().numpy())
                    print ("spike_times: ", spike_times.shape)
                    print ("spike_times: ", type(spike_times.data[0].item()))
                    print ("spike_temps: ", spike_temps.shape)
                    print ("spike_temps: ", type(spike_temps.data[0].item()))
                    print ("self.obj_gpu: ", self.obj_gpu.shape)
                    print ("self.obj_gpu: ", type(self.obj_gpu.data[0][0].item()))
                    print ("self.xshifts: ", self.xshifts.shape)
                    print ("self.xshifts: ", type(self.xshifts.data[0].item()))
                    print ("self.tempScaling: ", self.tempScaling)
                    print ("self.heights: ", self.heights.shape)
                    print ("self.heights: ", type(self.heights.data[0].item()))
                    print ("self.coefficients[k]: ", self.coefficients[k].data.shape)
                    print ("self.coefficients[k]: ", type(self.coefficients[k].data[0][0].item()))
            else:
                quit()
        
        #self.obj_gpu = self.obj_gpu*0.
        #spike_times = spike_times -99
        deconv.subtract_splines(
                    self.obj_gpu,
                    spike_times,
                    self.xshifts,
                    spike_temps,
                    self.coefficients,
                    self.tempScaling*self.heights)
                               
        torch.cuda.synchronize()
        
        # also fill in self-convolution traces with low energy so the
        #   spikes cannot be detected again (i.e. enforcing refractoriness)
        # Cat: TODO: read from CONFIG

        if self.refractoriness:
            #print ("filling in timesteps: ", self.n_time)
            deconv.refrac_fill(energy=self.obj_gpu,
                                  spike_times=spike_times,
                                  spike_ids=spike_temps,
                                  fill_length=self.refractory*2+1,  # variable fill length here
                                  fill_offset=self.subtraction_offset-2-self.refractory,
                                  fill_value=-self.fill_value)

        torch.cuda.synchronize()
            
        return (dt.datetime.now().timestamp()-start)

    def sample_spikes_allspikes(self):
        """
            Same as sample_spikes() but picking all spikes from a previous iteration,
        """

        #spike_times_list = self.spike_array[self.add_iteration_counter]-self.lockout_window
        spike_times_list = self.spike_array[self.add_iteration_counter]-self.subtraction_offset
        spike_ids_list = self.neuron_array[self.add_iteration_counter]
        spike_shifts_list= self.shift_list[self.add_iteration_counter]
        spike_height_list = self.height_list[self.add_iteration_counter]

        return spike_times_list, spike_ids_list, spike_shifts_list, spike_height_list      
        
    def add_cpp_allspikes(self):
        #start = dt.datetime.now().timestamp()
        
        torch.cuda.synchronize()
                        
        # select all spikes from a previous iteration
        spike_times, spike_temps, spike_shifts, spike_heights = self.sample_spikes_allspikes()

        torch.cuda.synchronize()

        # also fill in self-convolution traces with low energy so the
        #   spikes cannot be detected again (i.e. enforcing refractoriness)
        # Cat: TODO: investgiate whether putting the refractoriness back in is viable
        if self.refractoriness:
            deconv.refrac_fill(energy=self.obj_gpu,
                              spike_times=spike_times,
                              spike_ids=spike_temps,
                              fill_length=self.refractory*2+1,  # variable fill length here
                              fill_offset=self.subtraction_offset-2-self.refractory,
                              fill_value=self.fill_value)

        torch.cuda.synchronize()

        # Add spikes back in;
        deconv.subtract_splines(
                            self.obj_gpu,
                            spike_times,
                            spike_shifts,
                            spike_temps,
                            self.coefficients,
                            -self.tempScaling*spike_heights)

        torch.cuda.synchronize()

        return 
