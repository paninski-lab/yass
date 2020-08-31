import scipy.spatial.distance
import numpy as np
from tqdm import tqdm_notebook
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from yass.reader import READER
from yass.config import Config
import torch
import cudaSpline as deconv
from scipy.interpolate import splrep
from numpy.linalg import inv as inv
import os 
from pathlib import Path
import pickle
import parmap
from tqdm import tqdm
import numpy as np
import torch
import torch.distributions as dist
#from IB2 import IB_Denoiser
from torch import nn
from math import sqrt
from yass.neuralnetwork import Detect, Denoise
from yass.cluster.util import make_CONFIG2
from scipy.stats import chi2

##### DENOISER SET UP
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["GIO_EXTRA_MODULES"] = "/usr/lib/x86_64-linux-gnu/gio/modules/"

def predict0(data, model, in_channels = None):
    #data.shape = n_spikes x n_times x n_channels
    N = data.shape[0]
    n_timesteps = data.shape[1]
    if in_channels is None:
        in_channels = range(data.shape[2])
    
    templates_output = np.zeros((n_timesteps, data.shape[2]))

    for chan in in_channels:
        data0 = data[:, :, chan]
        data1 = data0.reshape([1,N,1,n_timesteps])
        data_torch = torch.tensor(data1).to(model.params_denoiser['device']).type(torch.float32)
        predicted = predict1(data_torch, model)
        predicted = predicted.cpu().detach().numpy()
        del data_torch
        templates_output[:, chan] = predicted[0, 0, :]
    return templates_output

def predict1(data,model):
    
    assert(1== data.shape[0])
    
    N = data.shape[1]
    n_channels = 1
    n_timesteps = data.shape[3]
    data1= data[0,:,0:1,:]
 
    hs = model.encode(data1).view([1, N, model.h_dim])

    
    mu_logstd_z = model.pz_x(hs.mean(dim=1))    
    
    pz_mu = mu_logstd_z[:,model.z_dim:]
    pz_sigma = mu_logstd_z[:,:model.z_dim].exp()
    
    pz = dist.Normal(pz_mu,pz_sigma)
    z = pz.rsample()  

    z_mean = torch.cat([z,data[:,:,0,:].mean(1)], dim=1)        
    mu_logstd_y = model.py_z(z_mean)    

    py_mu = mu_logstd_y[:,n_channels*n_timesteps:].view([1,n_channels,n_timesteps])
    #py_sigma = mu_logstd_y[:,:7*n_timesteps].exp().view([model.batch_size,7,n_timesteps])
    return py_mu




def shift_template(template, shift_chan, n_channels):
    for chan in np.arange(n_channels):
        shift = shift_chan[chan]
        if shift == 0:
            continue
        if shift > 0:
            template[:, chan] = np.concatenate((template[:, chan], np.zeros(shift)), axis = 0)[shift:]
        else:
            template[:, chan] =  np.concatenate((np.zeros(-shift), template[:, chan]), axis = 0)[:(template.shape[0])]
    return template

def shift_wfs(array, shifts, in_channels):
    return_array = np.zeros_like(array)
    for channel in in_channels:
        if len(return_array.shape) == 3:
            return_array[:,:, channel] = np.roll(array[:, :, channel], shifts[channel], 1)
        else:
            return_array[:,channel] = np.roll(array[:,channel], shifts[channel], 0)
    return return_array


def gather_spikes(spike_dict,unit, batch, needed_spikes, template_update_time):
    batch = batch - 1
    total_batches = len(spike_dict)
    return_wfs = []
    batch_idx = np.arange(0, len(spike_dict))
    batch_idx = batch_idx[np.argsort(np.abs(batch_idx - batch))[1:]]
    
    for idx in batch_idx:
        
        if len(return_wfs) + len(spike_dict[idx]) > needed_spikes:
            choice_idx = np.random.choice(spike_dict[idx].shape[0], needed_spikes - len(return_wfs))
            for choice in choice_idx:
                return_wfs.append(spike_dict[idx][choice])
            break
        else:
            return_wfs.extend(spike_dict[idx])
        
        if len(return_wfs) >= needed_spikes:
            break
    
    if len(return_wfs) >= needed_spikes:
       
        return np.asarray(return_wfs)
    else:
        return np.asarray([])
        

def get_wf(unit, spike_dict, sps, shift_chan, len_wf, min_time, max_time, n_channels, reader,vis_chans, template_update_time = 100, model = None, save = None, batch = 0, smooth = True):

    shift_chan = shift_chan[unit]
    spike_times = sps[:, 0]
    unit_bool = sps[:, 1] == unit
    time_bool = np.logical_and(spike_times + len_wf <= max_time, spike_times - len_wf > min_time)
    size = np.sum(np.logical_and(unit_bool, time_bool))
    idx4 = np.where(np.logical_and(unit_bool, time_bool))[0]
    spikes = spike_times[idx4] #- len_wf//2                    
    wfs = np.zeros((size, len_wf, n_channels))

    
    wfs = reader.read_waveforms(spikes)[0]
    
    if smooth:
        filter_idx = np.where(wfs[:, 30:-30, np.where(vis_chans[unit])[0]].ptp(1).max(1) > 3)[0]
        wfs = wfs[filter_idx]
        spikes = spikes[filter_idx]
    else:
        new_chans = np.where((wfs[:, 30:-30,:].ptp(1) > 2).all(0))[0]
        filter_idx = np.where(wfs[:, 30:-30, new_chans].ptp(1).max(1) > 3)[0]
        wfs = wfs[filter_idx]
        spikes = spikes[filter_idx]
    
    if wfs.shape[0] == 0:
        return wfs.mean(0), spikes, filter_idx.shape[0], np.asarray([]), np.asarray([])

    wfs_mean = wfs.mean(0)
    in_channels = np.where(wfs_mean.ptp(0) > 1)[0]
    shifts = 44 - wfs.mean(0).argmin(0)

    n_data, n_times, n_chans = wfs.shape
    wfs_shifted = shift_wfs(wfs, shifts, in_channels)    
    idx = np.random.choice(wfs.shape[0], np.min([wfs.shape[0], 300]))
    top_chan = wfs_shifted.mean(0).ptp(0).argsort()[::-1][0]

    
    choice_idx = np.random.choice(wfs.shape[0], np.min([15, wfs.shape[0]]))
    sample_wfs = wfs[choice_idx].copy()
    
    if wfs.shape[0] < 15 and not spike_dict is None:
        needed_spikes = 15 - wfs.shape[0]
        extra_wfs = gather_spikes(spike_dict, unit, batch, needed_spikes, template_update_time)
        if extra_wfs.shape[0] > 0:
            wfs = np.append(wfs, extra_wfs, axis = 0)
    if spike_dict is None:
        extra_wfs = None
    else:
        extra_wfs = np.asarray([])
   
    if not model is None and wfs.shape[0] >= 1:
        if sqrt(wfs.shape[0])*wfs_mean.ptp(0).max(0) <= 50*sqrt(template_update_time/100):
            wfs_shifted = shift_wfs(wfs, shifts, in_channels)
            wfs_denoised = predict0(wfs_shifted, model, in_channels)
            reshifted_denoised = shift_wfs(wfs_denoised, -shifts, in_channels)
            return shift_template(shift_wfs(wfs_denoised, -shifts, in_channels), shift_chan, n_channels), spikes, filter_idx.shape[0], extra_wfs, sample_wfs
        elif smooth:
            return shift_template(wfs.mean(0), shift_chan, n_channels), spikes, filter_idx.shape[0], extra_wfs, sample_wfs
        else:
            return wfs.mean(0), spikes, filter_idx.shape[0], extra_wfs, sample_wfs
    elif smooth:
        return shift_template(wfs.mean(0), shift_chan, n_channels), spikes, filter_idx.shape[0], extra_wfs, sample_wfs
    else:
        return wfs.mean(0), spikes, filter_idx.shape[0], extra_wfs, sample_wfs 


def full_rank_update(save_dir, update_object, batch_list, sps, tmps,  soft_assign = None, template_soft = None, backwards = False):
    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not backwards:
        updated_templates, batch = update_object.get_updated_templates(batch_list, sps , tmps, soft_assign, template_soft)
    else:
        updated_templates, batch = update_object.get_updated_templates_backwards(batch_list, sps , tmps, soft_assign, template_soft)
    np.save(os.path.join(save_dir, "templates.npy"), updated_templates)
    np.save(os.path.join(update_object.dir, "templates_{}.npy".format(str(batch))), updated_templates)
    return os.path.join(save_dir, "templates.npy")

class RegressionTemplates:
    def __init__(self, reader, CONFIG, ir_dir, smooth = True,  denoise = False, denoiser_params = None, lambda_pen = .2, 
                 num_iter = 15, num_basis_functions = 5):
        """
        standardized data is .npy 
        geom_array is .txt 
        n_chan = 385
        n_unit = 245
        len_wf = 61 length of individual waveforms
        lambda_pen penalization for the regression (= 0.5?)
        Parameters should be in the pipeline already
        """
        self.smooth = smooth #smooth
        self.denoise = denoise
        self.dir = os.path.join(ir_dir, "template_track_ir")
        self.forward_dir = ir_dir
        if not os.path.exists(os.path.join(ir_dir, "template_track_ir")):
            os.makedirs(os.path.join(ir_dir, "template_track_ir"))
        
        self.CONFIG = CONFIG
        self.reader = reader
        #self.spike_trains = np.load(spike_train_path)
        #self.spike_trains = self.spike_trains[np.where(self.spike_trains[:, 0] > 100)[0], :]
        
        self.geom_array = CONFIG.geom
        self.n_channels = CONFIG.recordings.n_channels
        self.sampling_rate = CONFIG.recordings.sampling_rate
        self.template_update_time = CONFIG.deconvolution.template_update_time
        self.num_chunks = int(np.ceil(reader.rec_len/(CONFIG.deconvolution.template_update_time*self.sampling_rate)))
        # Probably no need for this in the pipeline (number of basis functions already determined)
        self.num_basis_functions = num_basis_functions
        # Can fix model_rank to 5 (rank of the regression model)
        self.model_rank = 5
        self.num_iter = num_iter #number of iterations per batch in the regression (10?)
        # Only need one?
        
        self.lambda_pen = lambda_pen
        self.first = True
        self.max_time = int(reader.rec_len/self.sampling_rate)
        self.last_chunk = self.max_time/self.CONFIG.deconvolution.template_update_time - 1
    
        if denoiser_params is None:
            self.params_denoiser = params_denoiser = params = {
                                            'model_name': 'spike_denoiser',  # model name    # data shape 
                                            'n_timesteps': 121,  # width of each spike
                                            'n_channels': 14,  # number of local channels/units    # ResNet encoder: parameters for spike_encoder.py 
                                            'resnet_blocks': [1,1,1,1],
                                            'resnet_planes': 32,    # number of data points for training, N ~ unif(Nmin, Nmax)
                                            'Nmin': 5,
                                            'Nmax': 30,     # fraction of total templates that have collisions
                                            'Cmin': .7,
                                            'Cmax': .9,     # neural net architecture
                                            'z_dim': 1024,
                                            'h_dim': 1024,
                                            'H_dim': 256,
                                        }
        else:
            self.params_denoiser = denoiser_params
         
        if denoise:
            self.define_model()
            
    def define_model(self):
        self.params_denoiser['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        load_from_file = True
        self.model = IB_Denoiser(self.params_denoiser).to(self.params_denoiser['device'])
        self.model.params_denoiser = self.params_denoiser
        checkpoint = torch.load("/media/cat/julien/allen_75chan/IB2_jitter2_beta_zero_2000000.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def continuous_visible_channels(self, templates, threshold=.1, neighb_threshold=.5, spatial_neighbor_dist=70):
    ## Should be in the pipeline already
        """
        inputs:
        -------
        templates: np.ndarray with shape (# time points, # channels, #units)
        geom: np.ndarray with shape (# channel, 2)
        threshold: float
            Weaker channels threshold
        neighb_threshold: float
            Strong channel threshold
        spatial_neighbor_dist: float
            neighboring channel threshold (70 for 512 channels retinal probe)
        """
        geom = self.geom_array #Replace here 
        ptps_ = templates.ptp(0)
        pdist = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(geom))
        vis_chan = (ptps_ >= neighb_threshold).astype(np.int32)
        neighbs = np.logical_and(
            pdist > 0,
            pdist < spatial_neighbor_dist).astype(np.int32)
        
        vis_chans = np.logical_or(
            np.logical_and(
                np.matmul(neighbs, vis_chan) > 0,
                ptps_ >= threshold),
            ptps_ >= neighb_threshold)
        return vis_chans
    def visible_channels(self):
        #Can rewrite function or class so no reshape here
        
        templates_reshaped = np.transpose(self.templates, (1, 2, 0))
        #self.templates.reshape(self.len_wf, self.n_channels, self.n_unit)
        visible_chans = self.continuous_visible_channels(
            templates_reshaped, threshold=.5, neighb_threshold = 3., spatial_neighbor_dist=70)
        return(visible_chans)
    
    def find_shift(self):
        """
        The four following functions to create the shifted templates should not be needed in pipeline
        Create an array that stores the shift needed for each channel
        """
        visible_chans = self.visible_channels()
        self.unit_shifts = np.zeros((self.n_unit, self.n_channels))
        for unit in range(self.n_unit):
            n_channels = self.n_channels
            shift_chan = np.zeros(n_channels)
            templates = self.templates
            if np.sum(visible_chans[:, unit]) > 4:
                idx_chan_vis = np.where(visible_chans[:, unit])[0][templates[unit, :, visible_chans[:, unit]].ptp(1).argsort()[::-1]]
            else:
                idx_chan_vis = self.templates[unit].ptp(0).argsort()[::-1][:5]
            locmin_maxchan = templates[unit, :, idx_chan_vis[0]].argmin()
            shift_chan[idx_chan_vis] = templates[unit, :,idx_chan_vis].argmin(1) - locmin_maxchan
            self.unit_shifts[unit] = shift_chan
        self.unit_shifts = self.unit_shifts.astype(int)
        np.save(os.path.join(self.dir, "unit_shifts.npy"), self.unit_shifts)
    def make_wf_shift_batch(self, unit, batch):
        min_time = batch[0]*self.sampling_rate
        max_time = batch[1]*self.sampling_rate
        shift_chan = self.unit_shifts[unit]
        self.shift_chan = shift_chan
        unit_bool = self.spike_clusters == unit
        time_bool = np.logical_and(self.spike_times + self.len_wf <= max_time, self.spike_times - self.len_wf > min_time)
        idx4 = np.where(np.logical_and(unit_bool, time_bool))[0]
                                   
        spikes = self.spike_times[idx4] - self.len_wf//2                    
        self.u_spikes = spikes
        wfs = np.zeros((len(idx4), self.len_wf, self.n_channels))
        
        data_start = min_time - self.len_wf*2
        data_end = max_time  + 4*self.len_wf
        dat = self.reader.read_data(np.max([data_start,0]), data_end)
        offset = np.max([0, data_start])
        for i in range(len(idx4)):
            time = int(spikes[i])
            wf = np.zeros((self.len_wf, self.n_channels))
            for c in range(self.n_channels):
                wf[:, c] = dat[(-offset + time+shift_chan[c]):(-offset +time+ self.len_wf +shift_chan[c]), c]
            wfs[i, :, :] = wf
                                   
                                   
        return wfs, spikes
    
    
    def shift_templates_batch(self, unit, batch):
        n_unit = self.n_unit
        n_channels = self.n_channels
        num_chunks = self.num_chunks
        len_wf = self.len_wf
        #templates_reshifted = np.zeros((len_wf, n_channels, n_unit))
        
        u = unit
        wf, spikes_u = self.make_wf_shift_batch(u, batch)
        self.wf = wf
        num_spikes_u = len(spikes_u)
        picks = np.random.choice(num_spikes_u, np.min([num_spikes_u, 100]))
        
        np.save(os.path.join(self.dir, "raw_wf_{}_{}.npy".format(str(unit), str(batch[0]))), wf[picks])
        templates_reshifted = wf.mean(0)
        np.save(os.path.join(self.dir, "wf_{}.npy".format(unit)), templates_reshifted)
        return(templates_reshifted)

    def update_U(self, templates_unit_chunk, U, F, V, idx_chan_vis, num_chan_visible, num_spike_chunk, chunk, backwards):
        """
        The two update() functions might be faster with reshape/product instead of loops
        """
        b_bool = self.check_regression(chunk, U, True)
        f_bool = self.check_regression(chunk, U, False)
        lambda_pen = self.lambda_pen
        X = np.matmul(F, V[:, :, chunk]) #same shape as T
        if (num_spike_chunk>0):
            Y = templates_unit_chunk
            for i in idx_chan_vis[self.model_rank:]: # U = 1 for the channels of higher rank
                if (np.dot(X[:, i], X[:, i])>0):
                    if chunk != 0:
                        U[i, chunk] = (np.dot(X[:, i], Y[:, i])+ f_bool*lambda_pen*U[i, chunk-1] + b_bool*lambda_pen*U[i, chunk+1])/ (np.dot(X[:, i], X[:, i])+ (b_bool or f_bool)*lambda_pen)
                    else:
                        U[i, chunk] = (np.dot(X[:, i], Y[:, i]) + b_bool*lambda_pen*U[i, chunk+1])/ (np.dot(X[:, i], X[:, i])+ (b_bool)*lambda_pen)

        return(U)
    


    def update_V(self, templates_unit_chunk, U, F, V, idx_chan_vis, num_spike_chunk, chunk, backwards):
        """
        TODO : Add penalized regression
        V = (lambda+XtX)^-1*XtY + lambda*(lambda+XtX)^-1 V_prev
        """
        len_wf = self.len_wf
        lambda_pen = self.lambda_pen
        num_chunks = self.num_chunks
        num_basis_functions = self.num_basis_functions
        len_wf = self.len_wf
        W = np.ones(len_wf)
        if (num_spike_chunk > 0):
            W[:] = 1/num_spike_chunk
        
        b_bool = self.check_regression(chunk, U, True)
        f_bool = self.check_regression(chunk, U, False)
        for i in idx_chan_vis[:self.model_rank] : 
            Y = templates_unit_chunk[:, i]
            if not (f_bool or b_bool):
                V[:, i, chunk] = LinearRegression(fit_intercept=False).fit(F, y=Y).coef_
            elif chunk == 0:
                V[:, i, chunk] = Ridge(alpha = lambda_pen, fit_intercept=False).fit(F, Y).coef_ 
                mat_xx = np.matmul(F.transpose(), F)
                mat = np.linalg.pinv(lambda_pen * (np.eye(mat_xx.shape[0])) + mat_xx)
                V[:, i, chunk] = V[:, i, chunk] + lambda_pen*(b_bool*np.matmul(mat, V[:, i, chunk+1]))
                
            else:
                V[:, i, chunk] = Ridge(alpha = lambda_pen, fit_intercept=False).fit(F, Y).coef_ 
                mat_xx = np.matmul(F.transpose(), F)
                mat = np.linalg.pinv(lambda_pen * (np.eye(mat_xx.shape[0])) + mat_xx)
                V[:, i, chunk] = V[:, i, chunk] + lambda_pen*(f_bool*np.matmul(mat, V[:, i, chunk-1]) + b_bool*np.matmul(mat, V[:, i, chunk+1]))
        
        
        X = np.zeros((len_wf, num_basis_functions))                
        for i in idx_chan_vis[self.model_rank:] : 
            X = F * U[i, chunk]
            Y = templates_unit_chunk[:, i]

            if not (f_bool or b_bool):
                V[:, i, chunk] = LinearRegression(fit_intercept=False).fit(X, Y, sample_weight=W).coef_ 
            elif chunk == 0:
                V[:, i, chunk] = Ridge(alpha = lambda_pen, fit_intercept=False).fit(X, Y, sample_weight=W).coef_ 
                mat_xx = np.matmul(X.transpose(), X)
                mat = np.linalg.pinv(lambda_pen * (np.eye(mat_xx.shape[0])) + mat_xx)
                V[:, i, chunk] = V[:, i, chunk] + lambda_pen*( b_bool*np.matmul(mat, V[:, i, chunk+1]))
            else:
                V[:, i, chunk] = Ridge(alpha = lambda_pen, fit_intercept=False).fit(X, Y, sample_weight=W).coef_ 
                mat_xx = np.matmul(X.transpose(), X)
                mat = np.linalg.pinv(lambda_pen * (np.eye(mat_xx.shape[0])) + mat_xx)
                V[:, i, chunk] = V[:, i, chunk] + lambda_pen*(f_bool*np.matmul(mat, V[:, i, chunk-1]) + b_bool*np.matmul(mat, V[:, i, chunk+1]))
        return(V)
    
    def check_regression(self, batch,U, backwards = False):
        if not backwards:
            if np.sum(U[:, batch -1] - 1) != 0:
                return True
            else:
                return False
        else:
            if np.sum(U[:, batch + 1] - 1) != 0:
                return True
            else:
                return False

    def initialize_regression_params(self, templates_unit, num_chan_vis):
        """
        Initialize F using SVD on FIRST batch (?)
        """
        num_basis_functions = self.num_basis_functions
        num_chunks = self.num_chunks
        F = np.zeros((self.len_wf, num_basis_functions))
        u_mat, s, vh = np.linalg.svd(templates_unit, full_matrices=False)
        F = u_mat[:, :num_basis_functions]
        V = np.zeros((num_basis_functions, num_chan_vis, num_chunks + 1)) 
        
        for j in range(num_chunks):
            fill = (vh[:num_basis_functions, :].T * s[:num_basis_functions]).T
            V[:num_basis_functions, :, j]  = (vh[:num_basis_functions, :].T * s[:num_basis_functions]).T
        return V, F

    def batch_regression(self, unit, batch_times, templates_aligned, ptp_min = 3, backwards = False):
        batch = int(batch_times[0]/self.CONFIG.deconvolution.template_update_time)
        vis_chans = self.visible_chans_dict[unit]
        visible_chans_unit = np.where(vis_chans)[0]
        
        if os.path.exists(os.path.join(self.dir, "F_{}.npy".format(str(unit)))):
            U = np.load(os.path.join(self.dir, "U_{}.npy".format(str(unit))))
            if np.sum(U[:, batch] - 1) != 0:               
                V = np.load(os.path.join(self.dir, "V_{}.npy".format(str(unit))))
                F = np.load(os.path.join(self.dir, "F_{}.npy".format(str(unit))))
                self.templates_approx[unit, :, visible_chans_unit] = (np.matmul(F, V[:, :, batch])* U[:, batch]).T[:len(visible_chans_unit), :]
                return self.back_shift_template(unit, self.templates_approx[unit], visible_chans_unit)

        #to keep consistency with previous code
        len_wf = self.len_wf
        num_basis_functions = self.num_basis_functions
        num_chunks = self.num_chunks
        num_iter = self.num_iter
        
        #set mininum and maximum limits
        min_time_chunk = batch_times[0]*self.sampling_rate
        max_time_chunk = batch_times[1]*self.sampling_rate
        self.batch = batch
        
        unit_bool = self.spike_clusters == unit
        time_bool = np.logical_and(self.spike_times <= max_time_chunk, self.spike_times - self.len_wf//2 > min_time_chunk)
        idx4 = np.where(np.logical_and(unit_bool, time_bool))[0]
        #if not enough spikes we use the previous batches spikes
        if templates_aligned[2] < 2:
            return self.templates[unit]
        if self.templates[unit].ptp(0).max(0) < ptp_min:
            return self.templates[unit]

        idx_chan_vis = self.templates[unit, :, vis_chans].ptp(1).argsort()[::-1] #Needed to get top 5 channels
        
        '''
        templates_aligned = self.shift_templates_batch(unit, batch_times) #Not needed in the pipeline / faster if unit per unit
        '''
        
        self.vis = visible_chans_unit
        templates_unit, self.u_spikes = templates_aligned[0][:, visible_chans_unit],templates_aligned[1]
        self.templates_unit = templates_unit
        
        ## NUM SPIKE FOR EACH CHUNK 
        num_chan_vis = len(visible_chans_unit)
        
                
        if os.path.exists(os.path.join(self.dir, "F_{}.npy".format(str(unit)))):
            U = np.load(os.path.join(self.dir, "U_{}.npy".format(str(unit))))
            V = np.load(os.path.join(self.dir, "V_{}.npy".format(str(unit))))
            F = np.load(os.path.join(self.dir, "F_{}.npy".format(str(unit))))
        # Initialization         
        else:
            V, F = self.initialize_regression_params(templates_unit, num_chan_vis)
            U = np.ones((num_chan_vis,  num_chunks + 1))
        
        # Online Regression 
        for j in range(num_iter):
            U = self.update_U(templates_unit, U, F, V, idx_chan_vis, num_chan_vis, self.u_spikes.shape[0], batch, backwards)
            V = self.update_V(templates_unit, U, F, V, idx_chan_vis, self.u_spikes.shape[0], batch, backwards)
        
        self.templates_approx[unit, :, visible_chans_unit] = (np.matmul(F, V[:, :, batch])* U[:, batch]).T[:len(visible_chans_unit), :]
        
        np.save(os.path.join(self.dir, "U_{}.npy".format(str(unit))), U)
        np.save(os.path.join(self.dir, "V_{}.npy".format(str(unit))), V)
        np.save(os.path.join(self.dir, "F_{}.npy".format(str(unit))), F)
        return self.back_shift_template(unit, self.templates_approx[unit], visible_chans_unit)
    def back_shift_template(self,unit, template, visible_channels):
        shift_chan = self.unit_shifts[unit]
        for chan in visible_channels:
            shift = -shift_chan[chan]
            if shift == 0:
                continue
            if shift > 0:
                template[:, chan] = np.concatenate((template[:, chan], np.zeros(shift)), axis = 0)[shift:]
            else:
                template[:, chan] =  np.concatenate((np.zeros(-shift), template[:, chan]), axis = 0)[:(self.len_wf)]
        return template
                
    
    def reassign_spikes(self, probs, unit_assign):
        fix_idx = np.where(np.any(probs[:, 1:] > .8, axis = 1))[0]
        print(fix_idx.shape[0])
        for i, idx in enumerate(fix_idx):
            self.spike_trains[idx, 1] = unit_assign[idx, np.argmax(probs[idx,:])]
            probs[idx, 0] = probs[idx, np.argmax(probs[idx,:])]
        self.modified_spike_trains = self.spike_trains.copy()
        return probs
                         
    def get_updated_templates(self, batch_times, sps, tmps, soft_assign, template_soft, cross_batch = True):
        #load new template information 
        self.templates = np.load(tmps)
        self.n_unit = self.templates.shape[0]
        self.len_wf = self.templates.shape[1]
        
        
        #load new spike trains 
        self.spike_trains = np.load(sps)
        
        #Soft assignment based re-assignment and triaging
        if not soft_assign is None:
            soft_assign = np.load(soft_assign)
        template_assign = np.load(template_soft)['probs_templates']
        logprobs = np.load(template_soft)['logprobs_outliers']
        unit_assign = np.load(template_soft)['units_assignment']
        
        
        template_assign = self.reassign_spikes(template_assign, unit_assign)
        
        #need to un_hardcode this
        if not soft_assign is None:
            chi2_df = (2*(self.templates.shape[1] //2) + 1)*61
            cut_off = chi2(chi2_df).ppf(.995)
            include_idx = np.logical_and(soft_assign > .7, template_assign[:, 0] > .7)
            include_idx = np.logical_and(logprobs[:, 0] < cut_off, include_idx)                             
            self.spike_trains = self.spike_trains[include_idx]

        edge_idx = np.where(self.spike_trains[:, 0] > 100)[0]
        self.spike_trains = self.spike_trains[edge_idx, :]
        template_assign = template_assign[edge_idx, :]
        unit_assign = unit_assign[edge_idx, :]
                           
        self.spike_clusters = self.spike_trains[:, 1]
        self.spike_times = self.spike_trains[:, 0]

        #Find per channel shift
        self.find_shift()
        batch = int(batch_times[0]/self.CONFIG.deconvolution.template_update_time)
        self.batch = batch 
        
        #calculate data boundaries
        min_time = batch_times[0]*self.sampling_rate
        max_time = batch_times[1]*self.sampling_rate
        data_start = min_time - self.len_wf*2
        data_end = max_time  + 4*self.len_wf
        
        #number of batches 
        num_batches = int(self.max_time/self.CONFIG.deconvolution.template_update_time - 1)

        
        #Determine active channels of tracking Not required anymore
        if not os.path.exists(os.path.join(self.dir, "vis_chan.npy")):
            visible_chans = self.visible_channels()
            self.visible_chans_dict = np.zeros((self.templates.shape[0], self.templates.shape[2]), dtype = np.bool_)
            for unit in range(self.templates.shape[0]):
                if np.sum(visible_chans[:, unit]) > (self.num_basis_functions -1):
                    self.visible_chans_dict[unit]  =  visible_chans[:, unit]
                else:
                    vis_chans = np.zeros(self.n_channels, dtype = np.bool_)
                    vis_chans[self.templates[unit].ptp(0).argsort()[::-1][:7]] = True
                    self.visible_chans_dict[unit] = vis_chans
            np.save(os.path.join(self.dir, "vis_chan.npy"), self.visible_chans_dict)
        
                           
        #Load preveious templates and chanel informration 
        if not os.path.exists(os.path.join(self.dir, "spike_dict_0.npy")):
            #spike_dict = {unit : {in_batch: [] for in_batch in range(num_batches + 1)} for unit in range(self.templates.shape[0])}
            for i_unit in range(self.templates.shape[0]):
                np.save(os.path.join(self.dir, "spike_dict_{}.npy".format(i_unit)), {in_batch: [] for in_batch in range(num_batches + 1)})
            #np.save(os.path.join(self.dir, "spike_dict.npy"), spike_dict)
        
        if batch > 1:
            self.prev_templates = np.load(os.path.join(self.dir, "templates_{}.npy".format(str(int(batch -1)))))
        
        
        #find visible channels of new units:
        
        if cross_batch and batch > 1:
            #spike_dict = np.load(os.path.join(self.dir, "spike_dict.npy"), allow_pickle = True).item()
            for i_unit in range(self.templates.shape[0]):
                if os.path.exists(os.path.join(self.dir, "spike_dict_{}.npy".format(int(i_unit)))):
                    continue
                np.save(os.path.join(self.dir, "spike_dict_{}.npy".format(i_unit)), {in_batch: [] for in_batch in range(num_batches + 1)})
            '''
            for new_unit in range(self.prev_templates.shape[0], self.templates.shape[0]):            
                for in_batch in range(num_batches + 1):
                    spike_dict[new_unit] = {in_batch: [] for in_batch in range(num_batches + 1)}
                    np.save("spike_dict_{}.npy".format(new_unit), int(spike_dict[new_unit]))
            
            np.save(os.path.join(self.dir, "spike_dict.npy"), spike_dict)
            '''
        print("SMOOTHING STATUS________")
        print(self.smooth)
        if batch > 1 and not self.prev_templates.shape[0] == self.templates.shape[0] and self.smooth:
            ### need to update visible channels for 
            self.visible_chans_dict = np.load(os.path.join(self.dir, "vis_chan.npy"), allow_pickle = True)
              
            visible_chans = self.visible_channels()
            new_array = np.zeros((self.templates.shape[0], self.templates.shape[2]), dtype = np.bool_)
            print(self.visible_chans_dict.shape)
            new_array[:self.prev_templates.shape[0]] = self.visible_chans_dict
            self.visible_chans_dict = new_array
            for new_unit in range(self.prev_templates.shape[0], self.templates.shape[0]):
                if np.sum(visible_chans[:, new_unit]) > (self.num_basis_functions -1):
                    self.visible_chans_dict[new_unit]  =  visible_chans[:, new_unit]
                else:
                    vis_chans = np.zeros(self.n_channels, dtype = np.bool_)
                    vis_chans[self.templates[new_unit].ptp(0).argsort()[::-1][:7]] = True
                    self.visible_chans_dict[new_unit] = vis_chans
            
                for in_batch in range(num_batches + 1):
                    spike_dict[new_unit] = {in_batch: [] for in_batch in range(num_batches + 1)}
                
            np.save(os.path.join(self.dir, "vis_chan.npy"), self.visible_chans_dict)
            np.save(os.path.join(self.dir, "spike_dict.npy"), spike_dict)
           
        else:
            self.visible_chans_dict = np.load(os.path.join(self.dir, "vis_chan.npy"), allow_pickle = True)

        
        
        #Initiate it all 
        if self.denoise:
            model = self.model
        else:
            model = None

        #get mean/denoised waveforms from the section by section deconvolution
        
        '''
        wf_list = [get_wf( 
                             in_unit,
                             sps = self.spike_trains,
                             shift_chan = self.unit_shifts, 
                             len_wf = self.len_wf,
                             min_time = min_time,
                             max_time = max_time,
                             n_channels = self.n_channels,
                             reader = self.reader,
                             vis_chans = self.visible_chans_dict, 
                             template_update_time = self.template_update_time,
                             model = model,
                             ind_den = individual_denoiser,
                             save = self.dir,
                             batch = batch, 
                             smooth = self.smooth,
                             spike_dict = spike_dict) for in_unit in range(self.templates.shape[0])]
        '''
        if cross_batch:
            unit_dict_list = [(in_unit, np.load(os.path.join(self.dir, "spike_dict_{}.npy".format(in_unit)), allow_pickle = True).item()) for in_unit in range(self.templates.shape[0])]
        else:
            unit_dict_list = [(in_unit, None) for in_unit in range(self.templates.shape[0])]
        
        wf_list = parmap.starmap(get_wf, 
                             unit_dict_list,
                             sps = self.spike_trains,
                             shift_chan = self.unit_shifts, 
                             len_wf = self.len_wf,
                             min_time = min_time,
                             max_time = max_time,
                             n_channels = self.n_channels,
                             reader = self.reader,
                             vis_chans = self.visible_chans_dict, 
                             template_update_time = self.template_update_time,
                             model = model,
                             save = self.dir,
                             batch = batch, 
                             smooth = self.smooth,
                            pm_pbar=True, pm_processes = 1)
        if cross_batch:
            for i, element in enumerate(wf_list):
                unit_dict_list[i][1][batch] = element[4]
                np.save(os.path.join(self.dir, "spike_dict_{}.npy".format(i)), unit_dict_list[i][1])
            
            '''
            for i, element in enumerate(wf_list):
                spike_dict[i][batch] = element[4]
            np.save(os.path.join(self.dir, "spike_dict.npy"), spike_dict)
            '''
        np.save(os.path.join(self.dir, "wfs_format{}.npy".format(batch)), np.asarray(wf_list))
        if self.smooth:
            self.templates_approx = np.zeros_like(self.templates)
            tmps = np.zeros((self.n_unit, self.len_wf, self.n_channels))
            for unit in tqdm(range(self.n_unit)):
                tmps[unit, :, :] = self.batch_regression(unit, batch_times, wf_list[unit])
        else:
            tmps = np.zeros((self.n_unit, self.len_wf, self.n_channels))
            for unit in tqdm(range(self.n_unit)):
                if wf_list[unit][1].shape[0] < 2 or wf_list[unit][0].ptp(1).max(0) < 3:
                    tmps[unit, :, :] = self.templates[unit]
                else:
                    tmps[unit, :, :] = wf_list[unit][0]
                    tmps[unit, :, tmps[unit, :, :].ptp(0) < 1] = 0
            print(tmps.ptp(1).max(1).min(0))
        return tmps, batch
    
    #same things but backwards
    def get_updated_templates_backwards(self, batch_times, sps, tmps, soft_assign, template_soft, cross_batch = True):
        #load new template information 
        self.templates = np.load(tmps)
        self.n_unit = self.templates.shape[0]
        self.len_wf = self.templates.shape[1]
        self.find_shift()
        batch = int((batch_times[1]/self.CONFIG.deconvolution.template_update_time) -1) 
        self.batch = batch
        #load new spike trains 
        self.spike_trains = np.load(sps)
        
        #Spike train info
        if not soft_assign is None:
            soft_assign = np.load(soft_assign)
        template_assign = np.load(template_soft)['probs_templates']
        logprobs = np.load(template_soft)['logprobs_outliers']
        unit_assign = np.load(template_soft)['units_assignment']
        

        template_assign = self.reassign_spikes(template_assign, unit_assign)
        
        #need to un_hardcode this
        if not soft_assign is None:
            chi2_df = (2*(self.templates.shape[1] //2) + 1)*61
            cut_off = chi2(chi2_df).ppf(.995)
            include_idx = np.logical_and(soft_assign > .7, template_assign[:, 0] > .7)
            include_idx = np.logical_and(logprobs[:, 0] < cut_off, include_idx)                             
            self.spike_trains = self.spike_trains[include_idx]

        edge_idx = np.where(self.spike_trains[:, 0] > 100)[0]
        self.spike_trains = self.spike_trains[edge_idx, :]
        template_assign = template_assign[edge_idx, :]
        unit_assign = unit_assign[edge_idx, :]
                           
        self.spike_clusters = self.spike_trains[:, 1]
        self.spike_times = self.spike_trains[:, 0]
        
        
        #load batch information
        min_time = batch_times[0]*self.sampling_rate
        max_time = batch_times[1]*self.sampling_rate
        data_start = min_time - self.len_wf*2
        data_end = max_time  + 4*self.len_wf
        offset = np.max([0, data_start])
        
        #max number of batches
        num_batches = int(self.max_time/self.CONFIG.deconvolution.template_update_time - 1)
        
        
        # probably don't need but for testing purposes, create the visible channel dictionaries
        if not os.path.exists(os.path.join(self.dir, "vis_chan.npy")):
            visible_chans = self.visible_channels()
            self.visible_chans_dict = np.zeros((self.templates.shape[0], self.templates.shape[2]), dtype = np.bool_)
            for unit in range(self.templates.shape[0]):
                if np.sum(visible_chans[:, unit]) > (self.num_basis_functions -1):
                    self.visible_chans_dict[unit]  =  visible_chans[:, unit]
                else:
                    vis_chans = np.zeros(self.n_channels, dtype = np.bool_)
                    vis_chans[self.templates[unit].ptp(0).argsort()[::-1][:7]] = True
                    self.visible_chans_dict[unit] = vis_chans
            np.save(os.path.join(self.dir, "vis_chan.npy"), self.visible_chans_dict)
        
        
        
        if batch < num_batches:
            self.prev_templates = np.load(os.path.join(self.dir, "templates_{}.npy".format(str(int(batch +1)))))
        else:
            self.prev_templates = np.load(os.path.join(self.dir, "templates_{}.npy".format(str(int(batch -1)))))

        if cross_batch:
            #spike_dict = np.load(os.path.join(self.dir, "spike_dict.npy"), allow_pickle = True).item()
            for i_unit in range(self.templates.shape[0]):
                if os.path.exists(os.path.join(self.dir, "spike_dict_{}.npy".format(int(i_unit)))):
                    continue
                np.save(os.path.join(self.dir, "spike_dict_{}.npy".format(i_unit)), {in_batch: [] for in_batch in range(num_batches + 1)})
        
        if batch < num_batches and not self.prev_templates.shape[0] == self.templates.shape[0] and self.smooth:
            self.visible_chans_dict = np.load(os.path.join(self.dir, "vis_chan.npy"), allow_pickle = True)
            visible_chans = self.visible_channels()
            new_array = np.zeros((self.templates.shape[0], self.templates.shape[2]), dtype = np.bool_)
            new_array[:self.prev_templates.shape[0]] = self.visible_chans_dict
            self.visible_chans_dict = new_array
            for new_unit in range(self.prev_templates.shape[0], self.templates.shape[0]):
                if np.sum(visible_chans[:, new_unit]) > (self.num_basis_functions -1):
                    self.visible_chans_dict[new_unit]  =  visible_chans[:, new_unit]
                else:
                    vis_chans = np.zeros(self.n_channels, dtype = np.bool_)
                    vis_chans[self.templates[new_unit].ptp(0).argsort()[::-1][:7]] = True
                    self.visible_chans_dict[new_unit] = vis_chans
                
            np.save(os.path.join(self.dir, "vis_chan.npy"), self.visible_chans_dict)

        else:
            self.visible_chans_dict = np.load(os.path.join(self.dir, "vis_chan.npy"), allow_pickle = True)

        
        #Initiate it all 
        if self.denoise:
            model = self.model
        else:
            model = None
        '''
        wf_list = [get_wf( 
                             in_unit,
                             sps = self.spike_trains,
                             shift_chan = self.unit_shifts, 
                             len_wf = self.len_wf,
                             min_time = min_time,
                             max_time = max_time,
                             n_channels = self.n_channels,
                             reader = self.reader,
                             vis_chans = self.visible_chans_dict, 
                             template_update_time = self.template_update_time,
                             model = model,
                             ind_den = individual_denoiser,
                             save = self.dir,
                             batch = batch, 
                             smooth = self.smooth,
                             spike_dict = spike_dict) for in_unit in range(self.templates.shape[0])]
        '''
        if cross_batch:
            unit_dict_list = [(in_unit, np.load(os.path.join(self.dir, "spike_dict_{}.npy".format(in_unit)), allow_pickle = True).item()) for in_unit in range(self.templates.shape[0])]
        else:
            unit_dict_list = [(in_unit, None) for in_unit in range(self.templates.shape[0])]
        
        wf_list = parmap.starmap(get_wf, 
                             unit_dict_list,
                             sps = self.spike_trains,
                             shift_chan = self.unit_shifts, 
                             len_wf = self.len_wf,
                             min_time = min_time,
                             max_time = max_time,
                             n_channels = self.n_channels,
                             reader = self.reader,
                             vis_chans = self.visible_chans_dict, 
                             template_update_time = self.template_update_time,
                             model = model,
                             save = self.dir,
                             batch = batch, 
                             smooth = self.smooth,
                            pm_pbar=True, pm_processes = 1)
        
        if cross_batch:
            for i, element in enumerate(wf_list):
                unit_dict_list[i][1][batch] = element[4]
                np.save(os.path.join(self.dir, "spike_dict_{}.npy".format(i)), unit_dict_list[i][1])

        '''
        if cross_batch:
            for i, element in enumerate(wf_list):
                spike_dict[i][batch] = element[4]
            np.save(os.path.join(self.dir, "spike_dict.npy"), spike_dict)
        '''
        #np.save(os.path.join(self.dir, "wfs_format{}.npy".format(batch)), np.asarray(wf_list))
        
        if self.smooth:
            self.templates_approx = np.zeros_like(self.templates)
            tmps = np.zeros((self.n_unit, self.len_wf, self.n_channels))
            for unit in tqdm(range(self.n_unit)):
                tmps[unit, :, :] = self.batch_regression(unit, batch_times, wf_list[unit])
        else:
            tmps = np.zeros((self.n_unit, self.len_wf, self.n_channels))
            for unit in tqdm(range(self.n_unit)):
                if wf_list[unit][1].shape[0] < 2 or wf_list[unit][0].ptp(1).max(0) < 3:
                    tmps[unit, :, :] = self.templates[unit]
                else:
                    tmps[unit, :, :] = wf_list[unit][0]
                    tmps[unit, :, tmps[unit, :, :].ptp(0) < 1] = 0
       
        return tmps, batch

class IB_Denoiser(nn.Module):

    def __init__(self, params):
        super(IB_Denoiser, self).__init__()


        self.params = params

        H = params['H_dim']
        self.z_dim = params['z_dim']
        self.h_dim = params['h_dim']
        self.device = params['device']
        
        self.n_chan = 1#params['n_channels']
        self.n_steps = params['n_timesteps']


        self.h = torch.nn.Sequential(        
            nn.Conv1d(1, 24, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(24, 48, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        
        self.hlin = torch.nn.Sequential(
            torch.nn.Linear(48*29, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, self.h_dim),
            torch.nn.PReLU(),                        
            )
            
            
            

        self.pz_x = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 2*self.z_dim),
        )


        self.py_z = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim + self.n_steps, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 2*self.n_chan*self.n_steps),
        )


    def encode(self, data1):


        h0 = self.h(data1)
        h1 = h0.view(h0.size(0), -1)
        hs = self.hlin(h1)

        return hs 

    def forward(self, data, target, beta):

            # first pass
        batch_size = data.shape[0]
        N = data.shape[1]
        n_channels = 1#data.shape[2]
        n_timesteps = data.shape[3]
        
        
        data1 = data[:,:,0:1,:].view([N*batch_size,1,n_timesteps])
        target1 = target[:,0:1,:]
        

        hs = self.encode(data1).view([batch_size, N,self.h_dim])        
        #data = data.view( [batch_size*N, data.shape[2], data.shape[3]])
        
                

        mu_logstd_z = self.pz_x(hs.mean(dim=1))    
        
        pz_mu = mu_logstd_z[:,self.z_dim:]
        pz_sigma = mu_logstd_z[:,:self.z_dim].exp()
        
        pz = dist.Normal(pz_mu,pz_sigma)
        z = pz.rsample()  

        
        z_mean = torch.cat([z,data[:,:,0,:].mean(1)], dim=1)        
        mu_logstd_y = self.py_z(z_mean)                  
        
        py_mu = mu_logstd_y[:,n_channels*n_timesteps:].view([batch_size,n_channels,n_timesteps])
        py_sigma = mu_logstd_y[:,:n_channels*n_timesteps].exp().view([batch_size,n_channels,n_timesteps])
        
        py = dist.Normal(py_mu,py_sigma)
        
        
        loss1 = -py.log_prob(target1).sum(dim=[1,2])
        
        KL = - pz_sigma.log() + (pz_sigma.pow(2) + pz_mu.pow(2))/2
        KL = KL.sum(1)
        
        loss = loss1+ beta*KL
        
        
        return loss.mean(), KL.mean().item(), loss1.mean().item()
