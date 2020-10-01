"""
Preprocess pipeline
"""
import logging
import os
import numpy as np
import parmap
import yaml
from scipy.ndimage import shift

from yass import read_config
from yass.preprocess.util import *
from yass.reader import READER


def run(output_directory):
    """Preprocess pipeline: filtering, standarization and whitening filter
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

    # if apply filter, get recording reader
    n_sec_chunk = CONFIG.resources.n_sec_chunk
    reader = READER(filename_raw, dtype_raw, CONFIG, n_sec_chunk)
    logger.info("# of chunks: {}".format(reader.n_batches))

    # make output directory
    if not os.path.exists(output_directory):
        logger.info('Creating temporary folder: {}'.format(output_directory))
        os.makedirs(output_directory)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(output_directory))

    # make output parameters
    standardized_path = os.path.join(output_directory, "standardized.bin")
    standardized_params = dict(
        dtype=CONFIG.preprocess.dtype,
        n_channels=n_channels)
    logger.info('Output dtype for transformed data will be {}'
            .format(CONFIG.preprocess.dtype))

    # Check if data already saved to disk and skip:
    if os.path.exists(standardized_path):
        return standardized_path, standardized_params['dtype']

    # **********************************************
    # *********** run filter & stdarize  ***********
    # **********************************************

    # get necessary parameters
    low_frequency = CONFIG.preprocess.filter.low_pass_freq
    high_factor = CONFIG.preprocess.filter.high_factor
    order = CONFIG.preprocess.filter.order
    sampling_rate = CONFIG.recordings.sampling_rate

    # estimate std from a small chunk
    chunk_5sec = 5*CONFIG.recordings.sampling_rate
    if CONFIG.rec_len < chunk_5sec:
        chunk_5sec = CONFIG.rec_len
    small_batch = reader.read_data(
        data_start=CONFIG.rec_len//2 - chunk_5sec//2,
        data_end=CONFIG.rec_len//2 + chunk_5sec//2)

    fname_mean_sd = os.path.join(
        output_directory, 'mean_and_standard_dev_value.npz')
    if not os.path.exists(fname_mean_sd):
        get_std(small_batch, sampling_rate,
                fname_mean_sd, CONFIG.preprocess.apply_filter,
                low_frequency, high_factor, order)
    # turn it off
    small_batch = None

    # Make directory to hold filtered batch files:
    filtered_location = os.path.join(output_directory, "filtered_files")
    if not os.path.exists(filtered_location):
        os.makedirs(filtered_location)

    # read config params
    multi_processing = CONFIG.resources.multi_processing
    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            filter_standardize_batch,
            [i for i in range(reader.n_batches)],
            reader,
            fname_mean_sd,
            CONFIG.preprocess.apply_filter,
            CONFIG.preprocess.dtype,
            filtered_location,
            low_frequency,
            high_factor,
            order,
            sampling_rate,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader.n_batches):
            filter_standardize_batch(
                batch_id, reader, fname_mean_sd,
                CONFIG.preprocess.apply_filter,
                CONFIG.preprocess.dtype,
                filtered_location,
                low_frequency,
                high_factor,
                order,
                sampling_rate,
                )

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(filtered_location, output_directory)

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
        voltage_threshold=50,
        sigma = 1,
        histogram_batch_len=1, ##Each histogram lasts one second 
        num_bins = 20, 
        quantile_comp = 5, #Each step between sliding window for computing quantile
        timepoints_bin = int(CONFIG.recordings.sampling_rate/5), #read from quantile_comp
        br_quantile = 0.9, # For sinkhorn background removal
        max_displacement = 30, 
        geomarray = geomarray  
        )

    registration_params['length_um'] = int(geomarray[:, 1].max()) #Get from geomarray
    registration_params['space_bw_elec'] = geomarray[2, 1] - geomarray[0, 1]
    registration_params['num_y_pos'] = int(registration_params['length_um']/registration_params['space_bw_elec']) 
    registration_params['neighboring_chan'] = len(np.where(np.linalg.norm(geomarray, axis = 1)<=CONFIG.recordings.spatial_radius)[0])

    M = np.zeros((registration_params['num_chan'], registration_params['length_um']))
    for i in range(registration_params['num_chan']):
        for j in range(registration_params['length_um']):
            M[i, j] = np.exp(-(geomarray[i, 1] - j)**2/(2*(registration_params['sigma']**2)))

    print("Electrode2Space created")

    reader_registration = READER(filename_standardized, CONFIG.preprocess.dtype, CONFIG, n_sec_chunk = registration_params['histogram_batch_len'])
    logger.info("# of chunks: {}".format(reader_registration.n_batches))

    ########    # parmap.map or loop to create histograms 

    ## To save spike times for testing - delete after
    output_spikes = os.path.join(output_directory, 'spike_times')
    if not os.path.exists(output_spikes):
        os.makedirs(output_spikes)

    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            make_histograms,
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
            registration_params['neighboring_chan'], 
            M,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader_registration.n_batches):
            make_histograms(
                batch_id, 
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
                registration_params['neighboring_chan'], 
                M)


    ######## From histograms get Registration 
    num_hist = reader_registration.n_batches
    print("Number of histograms / Batches " + str(num_hist))
    correlation = np.eye(num_hist)
    displacement = np.zeros((num_hist, num_hist))

    possible_displacement = np.arange(-registration_params['max_displacement'], registration_params['max_displacement'] + 0.5, 0.5) ### 0.5um increments? 

    for s in range(num_hist):
        for t in np.arange(s+1, num_hist):
            cor_cmp = 0
            dis = -registration_params['max_displacement']
            fname1 = os.path.join(hist_location,"histogram_{}.npy".format(str(s).zfill(6)))
            fname2 = os.path.join(hist_location,"histogram_{}.npy".format(str(t).zfill(6)))
            hist1 = np.load(fname1)
            hist2 = np.load(fname2)
            for d in possible_displacement:
                hist_2_shift = shift(hist2, (0, d/registration_params['space_bw_elec'] ))
                cor = np.mean((hist1 - hist1.mean()) * (hist_2_shift - hist_2_shift.mean()))
                stds = hist1.std() * hist_2_shift.std()
                if stds == 0:
                    cor = 0
                else:
                    cor /= stds
                if cor > cor_cmp:
                    dis = d
                    cor_cmp = cor
            correlation[s, t]=cor_cmp
            correlation[t, s]=cor_cmp
            displacement[s, t]=dis
            displacement[t, s]=-dis

    fname = os.path.join(output_directory, "displacement_matrix.npy")
    np.save(fname, displacement)


    estimated_dis_mat = displacement.T
    for i in range(num_hist):
        estimated_dis_mat[:, i] = estimated_dis_mat[:, i] - displacement.T[:, 0]

    estimated_displacement = estimated_dis_mat.mean(0)

    fname = os.path.join(output_directory, "estimated_displacement.npy")
    np.save(fname, estimated_displacement)

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
            estimated_displacement,
            CONFIG.preprocess.dtype,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader_registration.n_batches):
            register_data(
                batch_id, 
                reader_registration, 
                output_dir_registered_data, 
                estimated_displacement, 
                CONFIG.preprocess.dtype)

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(output_dir_registered_data, output_directory, delete=False)

    print("REGISTRATION DONE!!!")

    # save yaml file with params
    path_to_yaml = standardized_path.replace('.bin', '.yaml')
    with open(path_to_yaml, 'w') as f:
        logger.info('Saving params...')
        yaml.dump(standardized_params, f)

    return standardized_path, standardized_params['dtype']