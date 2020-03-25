import yass.reordering.utils 
import yass.reordering.cluster
import yass.reordering.default_params
import yass.reordering
from yass import read_config
import numpy as np
from yass.config import Config
from yass.reordering.preprocess import get_good_channels
import os
import cupy as cp
#initialize object 


class PARAM:
    pass

class PROBE:
    pass

def run(save_fname, standardized_fname, CONFIG,n_sec_chunk, nPCs = 3,  nt0 = 61, reorder = True, dtype = np.float32 ):

    

    params = PARAM()
    probe = PROBE()

    params.sigmaMask = 30
    params.Nchan = CONFIG.recordings.n_channels
    params.nPCs = nPCs
    params.fs = CONFIG.recordings.sampling_rate
    
    #magic numbers from KS
    #params.fshigh = 150.
    #params.minfr_goodchannels = 0.1
    params.Th = [10, 4]

    #spkTh is the PCA threshold for detecting a spike
    params.spkTh = -6 
    params.ThPre = 8
    ##
    params.loc_range = [5, 4]
    params.long_range = [30, 6]

    probe.chanMap = np.arange(params.Nchan)
    probe.xc = CONFIG.geom[:, 0]
    probe.yc =  CONFIG.geom[:, 1]
    probe.kcoords = np.zeros(params.Nchan)
    probe.Nchan = params.Nchan
    shape = (params.Nchan, CONFIG.rec_len)
    standardized_mmemap = np.memmap(standardized_fname, order = "F", dtype = dtype)
    params.Nbatch = np.ceil(CONFIG.rec_len/(n_sec_chunk*CONFIG.recordings.sampling_rate)).astype(np.int16)
    params.reorder = reorder
    params.nt0min = np.ceil(20 * nt0 / 61).astype(np.int16)


    result = yass.reordering.cluster.clusterSingleBatches(proc = standardized_mmemap,
        params =  params, 
        probe =  probe,
        yass_batch = params.Nbatch, 
        n_chunk_sec = int(n_sec_chunk*CONFIG.recordings.sampling_rate),
        nt0 = nt0)
    np.save(save_fname, cp.asnumpy(result['iorig']))
    
