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

    print("REGISTERING DATA")

    ####### REGISTER DATA #######

    ##### Standardized Data : standardized_path = 'tmp/preprocess/standardized.bin'
    ##### Make directory to create histograms 
    ##### Histograms :
    filename_standardized = os.path.join(output_directory, "standardized.bin")
    hist_location = os.path.join(output_directory, "histograms")
    if not os.path.exists(hist_location):
        os.makedirs(hist_location)


    ######## TODO : Add to config file / Ask Nishchal 
    geomarray = np.genfromtxt(fname=CONFIG.data.geometry)  


    registration_params = dict(
        num_chan=n_channels, 
        time_templates=int(CONFIG.recordings.spike_size_ms*CONFIG.recordings.sampling_rate/1000),
        voltage_threshold=10,
        histogram_batch_len=1, ##Each histogram lasts one second 
        num_bins = 20, 
        quantile_comp = 20, #Each step between sliding window for computing quantile
        br_quantile = 0.9, # For sinkhorn background removal
        max_displacement = 30, 
        iter_quantile = 10,
        geomarray = geomarray  
        )

    registration_params['length_um'] = int(geomarray[:, 1].max()) #Get from geomarray
    registration_params['space_bw_elec'] = geomarray[2, 1] - geomarray[0, 1]
    registration_params['num_y_pos'] = int(2*registration_params['length_um']) 
    registration_params['neighboring_chan'] = 2*len(np.where(np.linalg.norm(geomarray, axis = 1)<=CONFIG.recordings.spatial_radius)[0]) 
    registration_params['sigma'] = 25*registration_params['num_y_pos']/registration_params['length_um'] #here, num_y_pos/length_um converts to number of bins in histograms

    M = np.exp(-(geomarray[:,1:] - np.arange(registration_params['num_y_pos'])/2)**2/(2*(registration_params['sigma']**2)))
    M = np.divide(M.T, M.sum(0, keepdims = True).T).T

    print("Electrode2Space created")

    reader_registration = READER(filename_raw, dtype_raw, CONFIG, n_sec_chunk = registration_params['histogram_batch_len'])
    logger.info("# of chunks: {}".format(reader_registration.n_batches))

    ########    # parmap.map or loop to create histograms 

    ## To save spike times for testing - delete after
    output_spikes = os.path.join(output_directory, 'spike_times')
    if not os.path.exists(output_spikes):
        os.makedirs(output_spikes)

    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            make_histograms_gpu,
            [i for i in range(reader_registration.n_batches)],
            reader_registration,
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
            M,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader_registration.n_batches):
            make_histograms(
                batch_id, 
                reader_registration_gpu, 
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

    print("HISTOGRAMS CREATED")


    ######## From histograms get Registration 
    num_hist = reader_registration.n_batches
    print("Number of histograms / Batches " + str(num_hist))
    correlation = np.eye(num_hist)
    displacement = np.zeros((num_hist, num_hist))

    ### HOW TO TAKE ADVANTAGE OF DOUBLE MATRIX TO FASTEN COMPUTATION???
    prob = 4*np.log(num_hist)*num_hist / (num_hist*(num_hist-1))
    prob = 1
    subsample = np.random.choice([0, 1], size=(num_hist,num_hist), p=[1 - prob, prob])

    possible_displacement = np.arange(-registration_params['max_displacement'], registration_params['max_displacement'] + 0.5, 0.5) ### 0.5um increments? 
    hist = np.zeros([num_hist, 20, registration_params['num_y_pos']])
    for i in range(num_hist):
        hist[i] = np.load(os.path.join(hist_location,"histogram_{}.npy".format(str(i).zfill(6))))
    hist = torch.from_numpy(hist).cuda().float()
    c2d = torch.nn.Conv2d(in_channels = 1, out_channels = num_hist, kernel_size = [20, registration_params['num_y_pos']], stride = 1, padding = [0, possible_displacement.size//2], bias = False).cuda()
    c2d.weight[:,0] = hist
    displacement = possible_displacement[c2d(hist[:,None])[:,:,0,:].argmax(2).cpu()]
    

    fname = os.path.join(output_directory, "displacement_matrix.npy")
    np.save(fname, displacement)
    fname = os.path.join(output_directory, "subsample_matrix.npy")
    np.save(fname, subsample)

    print("Displacement Matrix Estimated")
    
    vec_subsampled = np.reshape(subsample, num_hist*num_hist)*np.reshape(displacement, num_hist*num_hist)
    estimated_displacement = np.dot(np.linalg.pinv(np.reshape(subsample, num_hist*num_hist)*(np.kron(np.eye(num_hist), np.ones(num_hist)) - np.kron(np.ones(num_hist), np.eye(num_hist)))).T, vec_subsampled)
    estimated_displacement = estimated_displacement - estimated_displacement[0]

    fname = os.path.join(output_directory, "estimated_displacement.npy")
    np.save(fname, estimated_displacement)

    print("Displacement Calculated")

    arr_conv = np.arange(-num_hist, num_hist +1)
    window = np.exp(-arr_conv**2/(2*36))/(np.sqrt(2*np.pi*36))
    convolve = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = [window.size], stride = 1, padding = window_size//2 )
    smooth_estimate = convolve(estimated_displacement)

    fname = os.path.join(output_directory, "smoothed_displacement.npy")
    np.save(fname, smooth_estimate)

    print("Displacement Smoothed")


    ######## Interpolation - save data 
    output_dir_registered_data = os.path.join(output_directory, "registered_files")
    if not os.path.exists(output_dir_registered_data):
        os.makedirs(output_dir_registered_data)

    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            register_data,
            [i for i in range(reader_registration.n_batches)],
            reader_registration,
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
                reader_registration, 
                output_dir_registered_data, 
                smooth_estimate, 
                geomarray,
                CONFIG.preprocess.dtype)

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(output_dir_registered_data, output_directory, delete=False)

    print("REGISTRATION DONE!!!")

    return standardized_path, standardized_params['dtype']