import scipy.spatial.distance
import numpy as np
from tqdm import tqdm_notebook
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from yass.reader import READER
from yass.config import Config


class RegressionTemplates:
    def __init__(self,standardized_data_path,  reader, CONFIG, spike_train_path, templates_path, lambda_pen = .5, 
                 num_iter = 10, num_basis_functions = 5):
    self.CONFIG = CONFIG
    self.reader = reader

    self.num_chunks = np.ceil(CONFIG.rec_len/(CONFIG.resources.n_sec_chunk_gpu_deconv*CONFIG.recordings.sampling_rate)).astype(np.int16)
    self.run_chunk_sec = CONFIG.resources.n_sec_chunk_gpu_deconv*CONFIG.recordings.sampling_rate

    #self.std_data = np.load(standardized_data_path).reshape(-1, n_chan) #Not sure it is needed in the pipeline

    self.spike_trains = np.load(spike_train_path)
    self.spike_trains = self.spike_trains[np.where(self.spike_trains[:, 0] > 100)[0], :]
    self.templates = np.load(templates_path)
    self.spike_clusters = self.spike_trains[:, 1]
    self.spike_times = self.spike_trains[:, 0]
    self.geom_array = CONFIG.geom
    self.n_channels = CONFIG.recordings.n_channels
    self.n_unit = self.templates.shape[0]
    self.len_wf = self.templates.shape[1]
    self.sampling_rate = CONFIG.recordings.sampling_rate
    
    # Probably no need for this in the pipeline (number of basis functions already determined)
    self.num_basis_functions = num_basis_functions
    # Can fix model_rank to 5 (rank of the regression model)
    self.model_rank = 5
    self.num_iter = num_iter #number of iterations per batch in the regression (10?)
    # Only need one?
    
    self.lambda_pen = lambda_pen
    # Probably no need for this in the pipeline
    #self.max_time = max_time
    #self.min_time = min_time
    #self.templates_approx = np.zeros((self.n_unit,self.len_wf, self.n_channels, self.num_chunks))
    self.prev_templates =  np.zeros((self.n_unit,self.len_wf, self.n_channels))
    self.initialized = {unit: False for unit in range(self.n_unit)}

    def continuous_visible_channels(self, templates, threshold=.5, neighb_threshold=1., spatial_neighbor_dist=70):
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
        return np.logical_or(
            np.logical_and(
                np.matmul(neighbs, vis_chan) > 0,
                ptps_ >= threshold),
            ptps_ >= neighb_threshold)
    
    def visible_channels(self):
        #Can rewrite function or class so no reshape here
        
        templates_reshaped = np.transpose(self.templates, (1, 2, 0))
        #self.templates.reshape(self.len_wf, self.n_channels, self.n_unit)
        visible_chans = self.continuous_visible_channels(
            templates_reshaped, threshold=2., neighb_threshold = 4., spatial_neighbor_dist=70)
        return(visible_chans)

    