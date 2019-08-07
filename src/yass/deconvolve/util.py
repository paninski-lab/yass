from yass.empty import empty

def make_CONFIG2(CONFIG):
    ''' Makes a copy of several attributes of original config parameters
        to be sent into parmap function; original CONFIG can't be pickled;
    '''
    
    # make a copy of the original CONFIG object;
    # multiprocessing doesn't like the methods in original CONFIG        
    CONFIG2 = empty()
    CONFIG2.recordings=empty()
    CONFIG2.resources=empty()
    CONFIG2.deconvolution=empty()
    CONFIG2.data=empty()

    CONFIG2.recordings.sampling_rate = CONFIG.recordings.sampling_rate
    CONFIG2.recordings.n_channels = CONFIG.recordings.n_channels
    CONFIG2.recordings.spike_size_ms = CONFIG.recordings.spike_size_ms
    
    CONFIG2.resources.n_processors = CONFIG.resources.n_processors
    CONFIG2.resources.multi_processing = CONFIG.resources.multi_processing
    CONFIG2.resources.n_sec_chunk = CONFIG.resources.n_sec_chunk
    CONFIG2.resources.n_gpu_processors = CONFIG.resources.n_gpu_processors
    CONFIG2.resources.n_sec_chunk_gpu_deconv = CONFIG.resources.n_sec_chunk_gpu_deconv

    CONFIG2.data.root_folder = CONFIG.data.root_folder
    CONFIG2.data.geometry = CONFIG.data.geometry
    CONFIG2.geom = CONFIG.geom

    CONFIG2.neigh_channels = CONFIG.neigh_channels

    CONFIG2.spike_size = CONFIG.spike_size
    CONFIG2.spike_size_nn = CONFIG.spike_size_nn
    
    CONFIG2.deconvolution.threshold = CONFIG.deconvolution.threshold
    CONFIG2.deconvolution.deconv_gpu = CONFIG.deconvolution.deconv_gpu

    CONFIG2.rec_len = CONFIG.rec_len
    
    CONFIG2.torch_devices = CONFIG.torch_devices

    return CONFIG2
