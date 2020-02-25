import os
import logging
import numpy as np
import parmap
import scipy
import warnings
warnings.filterwarnings("ignore")
import datetime as dt
from tqdm import tqdm
import torch
import torch.multiprocessing as mp

from yass import read_config
from yass.reader import READER
from yass.deconvolve.match_pursuit_gpu_new import deconvGPU
from yass.deconvolve.util import make_CONFIG2

def run(fname_templates_in,
        output_directory,
        recordings_filename,
        recording_dtype,
        threshold=None,
        run_chunk_sec='full',
        save_up_data=True):
            
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standardized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    CONFIG = make_CONFIG2(CONFIG)

    #print("... deconv using GPU device: ", torch.cuda.current_device())
    
    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_templates = os.path.join(
        output_directory, 'templates.npy')
    fname_spike_train = os.path.join(
        output_directory, 'spike_train.npy')
    fname_shifts = os.path.join(
        output_directory, 'shifts.npy')
    fname_scales = os.path.join(
        output_directory, 'scales.npy')

    if (os.path.exists(fname_templates) and
        os.path.exists(fname_spike_train) and
        os.path.exists(fname_shifts) and
        os.path.exists(fname_scales)):
        return (fname_templates, fname_spike_train,
                fname_shifts, fname_scales)
    # parameters
    if threshold is None:
        threshold = CONFIG.deconvolution.threshold
    elif threshold == 'low_fp':
        threshold = 150

    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    # reader
    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    CONFIG.resources.n_sec_chunk_gpu_deconv,
                    chunk_sec=chunk_sec)
    # enforce broad buffer
    reader.buffer=1000
         
    deconv_ONgpu(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 CONFIG,
                 run_chunk_sec)

    return (fname_templates, fname_spike_train,
            fname_shifts, fname_scales)



def deconv_ONgpu(fname_templates_in,
                 output_directory,
                 reader,
                 threshold,
                 CONFIG,
                 run_chunk_sec):

    # **************** MAKE DECONV OBJECT *****************
    d_gpu = deconvGPU(CONFIG, fname_templates_in, output_directory)

    # Cat: TODO: read from CONFIG
    d_gpu.max_iter = 1000
    d_gpu.deconv_thresh = threshold

    # Cat: TODO read from CONFIG file 
    d_gpu.RANK = 5

    # fit height
    d_gpu.fit_height = True
    d_gpu.max_height_diff = 0.1
    d_gpu.fit_height_ptp = 20

    # debug/printout parameters
    d_gpu.verbose = False
    
    # Turn on refactoriness
    d_gpu.refractoriness = True
    
    # Stochastic gradient descent option
    # Cat: TODO: move these and other params to CONFIG
    d_gpu.scd = True
    if d_gpu.scd==False:
        print (" ICD TURNED OFFF.....")
    else:
        print (" ICD TUREND ON .....")

        # Cat: TODO: move to CONFIG; # of times to run scd inside the chunk
        # Cat: TODO: the number of stages need to be a fuction of # of channels; 
        #      around 1 stage per 20-30 channels seems to work; 
        #      but for 100s of chans this many need to be scaled further
        # d_gpu.n_scd_stages = self.CONFIG.recordings.n_channels // 24 
        d_gpu.n_scd_stages = 2
        # Cat: TODO move to CONFIG; # of addition steps each time
        d_gpu.n_scd_iterations = 10
    
    # this can turn off the superresolution alignemnt as an option
    d_gpu.superres_shift = True

    # add reader
    d_gpu.reader = reader

    # *********************************************************
    # *********************** RUN DECONV **********************
    # *********************************************************
    begin=dt.datetime.now().timestamp()
    print("woot")
    d_gpu = run_core_deconv(d_gpu, CONFIG)
    print("aloud")
    # save templates
    templates_post_deconv = d_gpu.temps.transpose(2, 1, 0)
    fname_templates = os.path.join(d_gpu.out_dir, 'templates.npy')
    np.save(fname_templates, templates_post_deconv)

    subtract_time = np.round((dt.datetime.now().timestamp()-begin),4)

    print ("-------------------------------------------")
    total_length_sec = int((d_gpu.reader.end - d_gpu.reader.start)/d_gpu.reader.sampling_rate)
    print ("Total Deconv Speed ", np.round(total_length_sec/(subtract_time),2), " x Realtime")

    # ****************************************************************
    # *********************** GATHER SPIKE TRAINS ********************
    # ****************************************************************

    # ************** SAVE SPIKES & SHIFTS **********************
    print ("  gathering spike trains and shifts from deconv")

    # get number of max spikes first
    n_spikes = 0
    for chunk_id in tqdm(range(reader.n_batches)):
        time_index = int((chunk_id+1)*reader.n_sec_chunk +
                         d_gpu.reader.start/d_gpu.reader.sampling_rate)
        fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')
        n_spikes += len(np.load(fname, allow_pickle=True)['spike_train'])

    # loop over chunks and add spikes;
    batch_size = d_gpu.reader.batch_size
    buffer_size = d_gpu.reader.buffer

    spike_train = np.zeros((n_spikes, 2), 'int32')
    shifts = np.zeros(n_spikes, 'float32')
    scales = np.zeros(n_spikes, 'float32')

    counter = 0
    for chunk_id in tqdm(range(reader.n_batches)):
        #fname = os.path.join(d_gpu.seg_dir,str(chunk_id).zfill(5)+'.npz')
        time_index = int((chunk_id+1)*reader.n_sec_chunk +
                         d_gpu.reader.start/d_gpu.reader.sampling_rate)
        fname = os.path.join(d_gpu.seg_dir,str(time_index).zfill(6)+'.npz')
        data = np.load(fname, allow_pickle=True)

        offset = data['offset']
        spike_train_chunk = data['spike_train']
        shifts_chunk = data['shifts']
        scales_chunk = data['heights']

        idx_keep = np.logical_and(
            spike_train_chunk[:, 0] >= buffer_size,
            spike_train_chunk[:, 0] < batch_size + buffer_size)
        idx_keep = np.where(idx_keep)[0]

        # add offset
        spike_train_chunk[:, 0] += offset

        # stack data
        idx = slice(counter, counter+len(idx_keep))
        spike_train[idx] = spike_train_chunk[idx_keep]
        shifts[idx] = shifts_chunk[idx_keep]
        scales[idx] = scales_chunk[idx_keep]

        counter += len(idx_keep)

    spike_train = spike_train[:counter]
    shifts = shifts[:counter]
    scales = scales[:counter]

    # sort spike train by time
    print ("   ordering spikes: ")
    idx = spike_train[:,0].argsort(0)
    spike_train = spike_train[idx]
    shifts = shifts[idx]
    scales = scales[idx]

    # save spike train
    print ("  saving spike_train: ", spike_train.shape)
    fname_spike_train = os.path.join(d_gpu.out_dir, 'spike_train.npy')
    np.save(fname_spike_train, spike_train)

    # save shifts
    fname_shifts = os.path.join(d_gpu.out_dir, 'shifts.npy')
    np.save(fname_shifts, shifts)

    # save scales
    fname_scales = os.path.join(d_gpu.out_dir, 'scales.npy')
    np.save(fname_scales, scales)


def run_core_deconv(d_gpu, CONFIG):

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(i) for i in range(torch.cuda.device_count())])
    chunk_ids = np.arange(d_gpu.reader.n_batches)
    print("still here")
    d_gpu.initialize(move_data_to_gpu=False)
    print("not here")
    start_sec = int(d_gpu.reader.start/d_gpu.reader.sampling_rate)
    end_sec = int(start_sec + d_gpu.reader.n_sec_chunk*d_gpu.reader.n_batches)
    print ("running deconv from {} to {} seconds".format(start_sec, end_sec))
    processes = []
    if len(CONFIG.torch_devices) == 1:
        run_core_deconv_parallel(d_gpu, chunk_ids, CONFIG.torch_devices[0].index)
    else:
        chunk_ids_split_gpu = np.array_split(
             chunk_ids, len(CONFIG.torch_devices))
        for ii, device in enumerate(CONFIG.torch_devices):
            p = mp.Process(target=run_core_deconv_parallel,
                           args=(d_gpu, chunk_ids_split_gpu[ii], device.index))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    return d_gpu


def run_core_deconv_parallel(d_gpu, chunk_ids, device):

    torch.cuda.set_device(device)
    d_gpu.data_to_gpu()

    for chunk_id in chunk_ids:
        time_index = int((chunk_id+1)*d_gpu.reader.n_sec_chunk + d_gpu.reader.start/d_gpu.reader.sampling_rate)
        fname = os.path.join(d_gpu.seg_dir, str(time_index).zfill(6)+'.npz')

        if not os.path.exists(fname):

            #print ("deconv: ", time_index, " sec, ", chunk_id, "/", d_gpu.reader.n_batches)

            # run deconv
            d_gpu.run(chunk_id)

            # save deconv results
            np.savez(fname,
                     spike_train = d_gpu.spike_train,
                     offset = d_gpu.offset,
                     shifts = d_gpu.shifts,
                     heights = d_gpu.heights)
