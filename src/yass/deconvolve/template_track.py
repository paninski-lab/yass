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

def get_wf(unit,dat, sps, shift_chan, len_wf, min_time, max_time, n_channels,offset, end, reader,vis_chans, save = None, batch = 0):
    #dat = np.load(dat)
    #dat = np.memmap(dat, dtype = np.float32, shape = (end -offset, n_channels))
    shift_chan = shift_chan[unit]
    spike_times = sps[:, 0]
    unit_bool = sps[:, 1] == unit
    time_bool = np.logical_and(spike_times + len_wf <= max_time, spike_times - len_wf > min_time)
    size = np.sum(np.logical_and(unit_bool, time_bool))
    idx4 = np.where(np.logical_and(unit_bool, time_bool))[0]
    spikes = spike_times[idx4] #- len_wf//2                    
    wfs = np.zeros((size, len_wf, n_channels))
    #last_chunk = self.max_time/self.CONFIG.deconvolution.template_update_time - 1
    '''
    for i in range(size):
        time = int(spikes[i])
        wf = np.zeros((len_wf, n_channels))
        for c in range(n_channels):
            wf[:, c] = dat[(-offset + time+shift_chan[c]):(-offset +time+ len_wf +shift_chan[c]), c]
            #wf[:, c]= reader.read_waveforms(np.asarray([time + shift_chan[c]]))[0][0, :, c]
        wfs[i, :, :] = wf
    '''
    
    wfs = reader.read_waveforms(spikes)[0]
    if not save is None:
        idx = np.random.choice(wfs.shape[0], np.min([wfs.shape[0], 300]))
        np.save(os.path.join(save, "raw_wfs_{}_{}.npy".format(batch, unit)), wfs[idx])
    filter_idx = np.where(wfs[:, 30:-30, np.where(vis_chans[unit])[0]].ptp(1).max(1) > 3)[0]
    wfs = wfs[filter_idx]
    return shift_template(wfs.mean(0), shift_chan, n_channels), spikes, filter_idx.shape[0]




def full_rank_update(save_dir, update_object, batch_list, sps, tmps, soft_assign = None, template_soft = None, backwards = False):
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
    def __init__(self, reader, CONFIG, ir_dir, lambda_pen = .5, 
                 num_iter = 10, num_basis_functions = 5):
        """
        standardized data is .npy 
        geom_array is .txt 
        n_chan = 385
        n_unit = 245
        len_wf = 61 length of individual waveforms
        lambda_pen penalization for the regression (= 0.5?)
        Parameters should be in the pipeline already

        """
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
                    if (chunk == 0):
                        U[i, chunk] = (np.dot(X[:, i], Y[:, i])+b_bool*int(backwards)*lambda_pen*U[i, chunk+1])/ (np.dot(X[:, i], X[:, i])+ lambda_pen)
                    elif chunk == self.last_chunk:
                        U[i, chunk] = (np.dot(X[:, i], Y[:, i])/np.dot(X[:, i], X[:, i])  + f_bool*lambda_pen*U[i, chunk-1])/ (np.dot(X[:, i], X[:, i])+ lambda_pen)
                    else:
                        U[i, chunk] = (np.dot(X[:, i], Y[:, i])+ f_bool*lambda_pen*U[i, chunk-1] + b_bool*int(backwards)*lambda_pen*U[i, chunk+1])/ (np.dot(X[:, i], X[:, i])+ lambda_pen)
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

        for i in idx_chan_vis[:self.model_rank] : 
            Y = templates_unit_chunk[:, i]
            if (chunk == 0):
                V[:, i, chunk] = LinearRegression(fit_intercept=False).fit(F, y=Y).coef_
            else:
                V[:, i, chunk] = Ridge(alpha = lambda_pen, fit_intercept=False).fit(F, Y).coef_ 
                mat_xx = np.matmul(F.transpose(), F)
                mat = np.linalg.pinv(lambda_pen * (np.eye(mat_xx.shape[0])) + mat_xx)
                V[:, i, chunk] = V[:, i, chunk] + 1*lambda_pen*np.matmul(mat, V[:, i, chunk-1])
        
        b_bool = self.check_regression(chunk, U, True)
        f_bool = self.check_regression(chunk, U, False)
        
        X = np.zeros((len_wf, num_basis_functions))                
        for i in idx_chan_vis[self.model_rank:] : 
            X = F * U[i, chunk]
            Y = templates_unit_chunk[:, i]
            
            if (chunk == 0):
                V[:, i, chunk] = LinearRegression(fit_intercept=False).fit(X, Y, sample_weight=W).coef_ 
            elif chunk == self.last_chunk:
                V[:, i, chunk] + f_bool*1*lambda_pen*np.matmul(mat, V[:, i, chunk-1])
            else:
                V[:, i, chunk] = Ridge(alpha = lambda_pen, fit_intercept=False).fit(X, Y, sample_weight=W).coef_ 
                mat_xx = np.matmul(X.transpose(), X)
                mat = np.linalg.pinv(lambda_pen * (np.eye(mat_xx.shape[0])) + mat_xx)
                V[:, i, chunk] = V[:, i, chunk] + f_bool*1*lambda_pen*np.matmul(mat, V[:, i, chunk-1]) + b_bool*1*lambda_pen*np.matmul(mat, V[:, i, chunk+1])
        return(V)
    
    def check_regression(self, batch,U, backwards = False):
        if not backwards:
            if np.sum(U[:, batch] - 1) != 0 and batch != 0:
                return True
            else:
                return False
        else:
            if np.sum(U[:, batch] + 1) != 0 and batch != (U.shape[1] -1) :
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
        V = np.zeros((num_basis_functions, num_chan_vis, num_chunks)) 
        
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
        if templates_aligned[2] < 10:
            print(unit)
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
            U = np.ones((num_chan_vis,  num_chunks))
        
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
                

    def get_updated_templates(self, batch_times, sps, tmps, soft_assign, template_soft):
        #load new spike trains 
        self.spike_trains = np.load(sps)
        soft_assign = np.load(soft_assign)
        template_soft = np.load(template_soft)['probs_templates']
        if not soft_assign is None:
            self.spike_trains = self.spike_trains[np.logical_and(soft_assign > .6, template_soft[:, 0] > .6)]
        self.spike_trains = self.spike_trains[np.where(self.spike_trains[:, 0] > 100)[0], :]
        self.spike_clusters = self.spike_trains[:, 1]
        self.spike_times = self.spike_trains[:, 0]
        #load new template information 
        self.templates = np.load(tmps)
        self.n_unit = self.templates.shape[0]
        self.len_wf = self.templates.shape[1]
        self.find_shift()
        batch = int(batch_times[0]/self.CONFIG.deconvolution.template_update_time)
        self.batch = batch 
        
 
        min_time = batch_times[0]*self.sampling_rate
        max_time = batch_times[1]*self.sampling_rate
        data_start = min_time - self.len_wf*2
        data_end = max_time  + 4*self.len_wf
        #dat = self.reader.read_data(np.max([data_start,0]), data_end)
        #dat.astype(np.float32).tofile(os.path.join(self.dir, "seg.dat"))
        #np.save(os.path.join(self.dir, "seg.npy"), dat)
        offset = np.max([0, data_start])
        #get_wf(0, os.path.join(self.dir, "seg.dat"),  self.spike_trains, self.unit_shifts, self.len_wf, min_time, max_time, #self.n_channels,offset, data_end, self.reader)

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
        

        if batch > 1:
            self.prev_templates = np.load(os.path.join(self.dir, "templates_{}.npy".format(str(int(batch -1)))))
        if batch > 1:
            print(self.prev_templates.shape[0])
            print(np.load(os.path.join(self.dir, "vis_chan.npy"), allow_pickle = True).shape[0])

        if batch > 1 and not self.prev_templates.shape[0] == self.templates.shape[0] :
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
        
        wf_list = parmap.map(get_wf, range(self.templates.shape[0]),os.path.join(self.dir, "seg.dat"),  self.spike_trains, self.unit_shifts, self.len_wf, min_time, max_time, self.n_channels,offset, data_end, self.reader,self.visible_chans_dict, save = self.dir,batch = batch, pm_pbar=True, pm_processes = 5)
        
        np.save(os.path.join(self.dir, "wfs_format{}.npy".format(batch)), np.asarray(wf_list))

        self.templates_approx = np.zeros_like(self.templates)
        tmps = np.zeros((self.n_unit, self.len_wf, self.n_channels))
        for unit in tqdm(range(self.n_unit)):
            tmps[unit, :, :] = self.batch_regression(unit, batch_times, wf_list[unit])
        #np.save(os.path.join(self.dir, "templates_{}".format(str(int(batch)))), tmps)
        return tmps, batch
    
    def get_updated_templates_backwards(self, batch_times, sps, tmps, soft_assign, template_soft):
        #load new spike trains 
        self.spike_trains = np.load(sps)
        soft_assign = np.load(soft_assign)
        template_soft = np.load(template_soft)['probs_templates']
        if not soft_assign is None:
            self.spike_trains = self.spike_trains[np.logical_and(soft_assign > .6, template_soft[:, 0] > .6)]

        self.spike_trains = self.spike_trains[np.where(self.spike_trains[:, 0] > 100)[0], :]
        self.spike_clusters = self.spike_trains[:, 1]
        self.spike_times = self.spike_trains[:, 0]
        #load new template information 
        self.templates = np.load(tmps)
        self.n_unit = self.templates.shape[0]
        self.len_wf = self.templates.shape[1]
        self.find_shift()
        batch = int((batch_times[1]/self.CONFIG.deconvolution.template_update_time) -1)  #int((self.max_time - batch_times[1])/self.CONFIG.deconvolution.template_update_time -1)
        
        min_time = batch_times[0]*self.sampling_rate
        max_time = batch_times[1]*self.sampling_rate
        data_start = min_time - self.len_wf*2
        data_end = max_time  + 4*self.len_wf
        #dat = self.reader.read_data(np.max([data_start,0]), data_end)
        #dat.astype(np.float32).tofile(os.path.join(self.dir, "seg.dat"))
        #np.save(os.path.join(self.dir, "seg.npy"), dat)
        offset = np.max([0, data_start])

        
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
        
        if batch < self.max_time/self.CONFIG.deconvolution.template_update_time - 1:
            self.prev_templates = np.load(os.path.join(self.dir, "templates_{}.npy".format(str(int(batch +1)))))

        if batch > self.max_time/self.CONFIG.deconvolution.template_update_time - 1 and not self.prev_templates.shape[0] == self.templates.shape[0] :
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
            np.save(os.path.join(self.dir, "vis_chan_{}.npy".format(str(batch))), self.visible_chans_dict)
        else:
            self.visible_chans_dict = np.load(os.path.join(self.dir, "vis_chan.npy"), allow_pickle = True)
        wf_list = parmap.map(get_wf, range(self.templates.shape[0]),os.path.join(self.dir, "seg.dat"),  self.spike_trains, self.unit_shifts, self.len_wf, min_time, max_time, self.n_channels,offset, data_end, self.reader,self.visible_chans_dict, pm_pbar=True, pm_processes = 5)

        
        self.templates_approx = np.zeros_like(self.templates)
        tmps = np.zeros((self.n_unit, self.len_wf, self.n_channels))
        for unit in tqdm(range(self.n_unit)):
            tmps[unit, :, :] = self.batch_regression(unit, batch_times, wf_list[unit], True)
        #np.save(os.path.join(self.dir, "templates_{}".format(str(int(batch)))), self.templates)
        return tmps, batch
