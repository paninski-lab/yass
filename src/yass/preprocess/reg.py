"""
Preprocess pipeline
"""
import logging
import os
import numpy as np
import parmap
import yaml
from scipy.ndimage import shift
from scipy.signal import convolve
from tqdm import tqdm
from yass import read_config
from yass.preprocess.util import *
from yass.reader import READER
import torch

def run(output_directory):
    """Preprocess pipeline: filtering, standarization and whitening filter +  registration
    This step (optionally) performs filtering on the data, standarizes it
    and computes a whitening filter. Filtering and standardized data are
    processed in chunks and written to disk.
    Parameters
    ----------
    output_directory: str
        where results will be saved
    Returns
    -------
    standardized_path: str
        Path to standardized data binary file
    standardized_params: str
        Path to standardized data parameters
    channel_index: numpy.ndarray
        Channel indexes
    whiten_filter: numpy.ndarray
        Whiten matrix
    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:
    * ``filtered.bin`` - Filtered recordings
    * ``filtered.yaml`` - Filtered recordings metadata
    * ``standardized.bin`` - Standarized recordings
    * ``standardized.yaml`` - Standarized recordings metadata
    * ``whitening.npy`` - Whitening filter
    Everything is run on CPU.
    Examples
    --------
    .. literalinclude:: ../../examples/pipeline/preprocess.py
    """

    # **********************************************
    # *********** Initialize ***********************
    # **********************************************
    
    logger = logging.getLogger(__name__)

    # load config
    CONFIG = read_config()

    # raw data info
    filename_raw = os.path.join(CONFIG.data.root_folder,
                                CONFIG.data.recordings)
    dtype_raw = CONFIG.recordings.dtype
    n_channels = CONFIG.recordings.n_channels
    
    
    filename_standardized = os.path.join(output_directory, "standardized.bin")
    if os.path.exists(filename_standardized):
        return filename_standardized, CONFIG.preprocess.dtype, None
    
    hist_location = os.path.join(output_directory, "histograms")
    if not os.path.exists(hist_location):
        os.makedirs(hist_location)
        
    reader_raw = READER(filename_raw, 'int16', CONFIG, n_sec_chunk = 1)
    logger.info("# of chunks: {}".format(reader_raw.n_batches))

    #         # get necessary parameters
    low_frequency = CONFIG.preprocess.filter.low_pass_freq
    high_factor = CONFIG.preprocess.filter.high_factor
    order = CONFIG.preprocess.filter.order
    sampling_rate = CONFIG.recordings.sampling_rate

    # estimate std from a small chunk
    chunk_5sec = CONFIG.recordings.sampling_rate
    if CONFIG.rec_len < chunk_5sec:
        chunk_5sec = CONFIG.rec_len
    small_batch = reader_raw.read_data(
        data_start=CONFIG.rec_len//2 - chunk_5sec*5//2,
        data_end=CONFIG.rec_len//2 + chunk_5sec*5//2)


    fname_mean_sd = os.path.join(
        output_directory, 'mean_and_standard_dev_value.npz')
    if not os.path.exists(fname_mean_sd):
        get_std(small_batch, sampling_rate,
                fname_mean_sd, CONFIG.preprocess.apply_filter,
                low_frequency, high_factor, order)
        
    # turn it off
    small_batch = None
    
    output_dir_filtered_data = os.path.join(output_directory, "filtered_files")
    if not os.path.exists(output_dir_filtered_data):
        os.makedirs(output_dir_filtered_data)
    
    num_hist = reader_raw.n_batches
    # read config params
    multi_processing = CONFIG.resources.multi_processing
    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            filter_standardize_batch,
            [i for i in range(num_hist)],
            reader_raw,
            fname_mean_sd,
            CONFIG.preprocess.apply_filter,
            CONFIG.preprocess.dtype,
            output_dir_filtered_data,
            low_frequency,
            high_factor,
            order,
            sampling_rate,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader_raw.n_batches):
            filter_standardize_batch(
                batch_id, reader_raw, fname_mean_sd,
                CONFIG.preprocess.apply_filter,
                CONFIG.preprocess.dtype,
                output_dir_filtered_data,
                low_frequency,
                high_factor,
                order,
                sampling_rate,
                )

#     merge_filtered_files(output_dir_filtered_data, output_directory, delete=True)
#     return filename_standardized, CONFIG.preprocess.dtype, None
    
    geomarray = np.load(CONFIG.data.geometry)
    registration_params = dict(
        num_chan=n_channels, 
        time_templates=int(CONFIG.recordings.spike_size_ms*CONFIG.recordings.sampling_rate/1000)+1,
        voltage_threshold= 5,
        histogram_batch_len=1, ##Each histogram lasts one second 
        num_bins = 10, 
        quantile_comp = 20, #Each step between sliding window for computing quantile
        br_quantile = 0.90, # For sinkhorn background removal
        max_displacement = 50, 
        iter_quantile = 10,
        geomarray = geomarray  
        )

    registration_params['length_um'] = int(geomarray[:, 1].max()) #Get from geomarray
    registration_params['space_bw_elec'] = geomarray[2, 1] - geomarray[0, 1]
    registration_params['num_y_pos'] = int(registration_params['length_um']) 
    registration_params['neighboring_chan'] = 2*len(np.where(np.linalg.norm(geomarray, axis = 1)<=40)[0]) 
    registration_params['sigma'] =  5
#     20*registration_params['num_y_pos']/registration_params['length_um'] #here, num_y_pos/length_um converts to number of bins in histograms

    M = np.exp(-(geomarray[:,1:] - np.arange(registration_params['num_y_pos']+1))**2/(2*(registration_params['sigma']**2)))
    M = np.divide(M.T, M.sum(0, keepdims = True).T + 1e-6).T
    print("Electrode2Space created")
            
    output_spikes = os.path.join(output_directory, 'spike_times')
    if not os.path.exists(output_spikes):
        os.makedirs(output_spikes)

    if True:
        n_processors = CONFIG.resources.n_processors
        results = parmap.map(
            make_histograms_gpu,
            [i for i in range(reader_raw.n_batches)],
            output_dir_filtered_data,
            hist_location, 
            output_spikes, ## To save spike times for testing - delete after
            registration_params['num_chan'], 
            CONFIG.recordings.sampling_rate, 
            registration_params['time_templates'], 
            registration_params['voltage_threshold'], 
            registration_params['length_um'], 
            registration_params['num_bins'], 
            registration_params['num_y_pos']+1, 
            registration_params['quantile_comp'], 
            registration_params['br_quantile'], 
            registration_params['iter_quantile'],
            registration_params['neighboring_chan'], 
            M,
            processes=n_processors,
            pm_pbar=True)
        hist = np.concatenate(results, axis = 0)
    else:
        hist = np.zeros([num_hist, 10, registration_params['num_y_pos']+1])
        for batch_id in range(reader_registration.n_batches):
            hist[i] = make_histograms_gpu(
                batch_id,
                output_dir_filtered_data,
                hist_location, 
                output_spikes, ## To save spike times for testing - delete after
                registration_params['num_chan'], 
                CONFIG.recordings.sampling_rate, 
                registration_params['time_templates'], 
                registration_params['voltage_threshold'], 
                registration_params['length_um'], 
                registration_params['num_bins'], 
                registration_params['num_y_pos'], 
                registration_params['quantile_comp'], 
                registration_params['br_quantile'], 
                registration_params['iter_quantile'],
                registration_params['neighboring_chan'], 
                M)
            
    num_hist = reader_raw.n_batches
    correlation = np.eye(num_hist)
    displacement = np.zeros((num_hist, num_hist))

    ## HOW TO TAKE ADVANTAGE OF DOUBLE MATRIX TO FASTEN COMPUTATION???
#   prob = 4*np.log(num_hist)*num_hist / (num_hist*(num_hist-1))
    prob = 1
    subsample = np.random.choice([0, 1], size=(num_hist,num_hist), p=[1 - prob, prob])

    possible_displacement = np.arange(-registration_params['max_displacement'], registration_params['max_displacement'] + 1, 1) ### 
# 0.5um increments? 
#     possible displacement =
#     hist = np.zeros([num_hist, 20, registration_params['num_y_pos']+1])
    hist = torch.from_numpy(hist).cuda().float()
    c2d = torch.nn.Conv2d(in_channels = 1, out_channels = num_hist, kernel_size = [10, registration_params['num_y_pos']+1], stride = 1, padding = [0, possible_displacement.size//2], bias = False).cuda()
    c2d.weight[:,0] = hist
    batchsize = 200
    displacement = np.zeros([hist.shape[0], hist.shape[0]])
    for i in tqdm(range(hist.shape[0]//200)):
        displacement[i*batchsize:(i+1)*batchsize] = possible_displacement[c2d(hist[i*batchsize:(i+1)*batchsize,None])[:,:,0,:].argmax(2).cpu()]


    print("HISTOGRAMS CREATED")
    
    fname = os.path.join(output_directory, "displacement_matrix.npy")
    np.save(fname, displacement)

    fname = os.path.join(output_directory, "subsample_matrix.npy")
    np.save(fname, subsample)

    
#     vec_subsampled = np.reshape(subsample, num_hist*num_hist)*np.reshape(displacement, num_hist*num_hist)
#     estimated_displacement = np.dot(np.linalg.pinv((np.kron(np.eye(num_hist), np.ones(num_hist)) - np.kron(np.ones(num_hist), np.eye(num_hist)))).T, vec_subsampled)
    estimated_displacement = calc_displacement(displacement, niter = 100)
    estimated_displacement =  estimated_displacement - estimated_displacement[0]

    fname = os.path.join(output_directory, "estimated_displacement.npy")
    np.save(fname, estimated_displacement)

    print("Displacement Calculated")

    arr_conv = np.arange(-num_hist, num_hist +1)
    window = np.exp(-arr_conv**2/(2*36))/(np.sqrt(2*np.pi*36))
    smooth_estimate = convolve(estimated_displacement, window, mode = 'same')
# 
    fname = os.path.join(output_directory, "smoothed_displacement.npy")
    np.save(fname, smooth_estimate)
    
#     smooth_estimate = np.load(fname)
    print("Displacement Smoothed")
    
    smooth_estimate = estimated_displacement

    output_dir_registered_data = os.path.join(output_directory, "registered_files")
    if not os.path.exists(output_dir_registered_data):
        os.makedirs(output_dir_registered_data)
        
        
    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            register_data,
            [i for i in range(reader_raw.n_batches)],
            output_dir_filtered_data,
            output_dir_registered_data, 
            smooth_estimate,
            geomarray,
            CONFIG.preprocess.dtype,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader_registration.n_batches):
            register_data(
                batch_id, 
                output_dir_filtered_data,
                output_dir_registered_data, 
                smooth_estimate, 
                geomarray,
                CONFIG.preprocess.dtype)

    merge_filtered_files(output_dir_registered_data, output_directory, delete=False)
    
    
    return filename_standardized, CONFIG.preprocess.dtype, None
