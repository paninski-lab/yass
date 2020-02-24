import os
import logging
import numpy as np
import torch

from yass import read_config
from yass.reader import READER
from yass.residual.residual import RESIDUAL
from yass.residual.residual_gpu import RESIDUAL_GPU2, RESIDUAL_GPU3


def run(fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        output_directory,
        recordings_filename,
        recording_dtype,
        dtype_out='float32',
        update_templates=False,
        run_chunk_sec='full',
        min_ptp_units=0,
        min_ptp_vis_chan=0
       ):
            
    """Compute residual

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

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Note: this uses spike times occuring at beginning of spike
    fname_out = os.path.join(output_directory, 'residual.bin')
    if os.path.exists(fname_out):
        return fname_out, dtype_out

    if (min_ptp_units > 0) or (min_ptp_vis_chan > 0):
        templates = np.load(fname_templates)
        spike_train = np.load(fname_spike_train)
        shifts = np.load(fname_shifts)
        scales = np.load(fname_scales)
        ptps = templates.ptp(1)
        ptps_max = ptps.max(1)

        units_in = np.where(ptps_max > min_ptp_units)[0]
        templates_new = templates[units_in]

        for ii, k in enumerate(units_in):
            templates_new[ii, :, ptps[k] <= min_ptp_vis_chan] = 0

        idx_in = np.in1d(spike_train[:, 1], units_in)
        spike_train_tmp = spike_train[idx_in]
        spike_train_new = np.copy(spike_train_tmp)
        shifts_new = shifts[idx_in]
        scales_new = scales[idx_in]
        
        for ii, k in enumerate(units_in):
            spike_train_new[spike_train_tmp[:, 1] == k, 1] = ii
        
        fname_templates = os.path.join(output_directory, 'templates.npy')
        fname_spike_train = os.path.join(output_directory, 'spike_train.npy')
        fname_shifts = os.path.join(output_directory, 'shifts.npy')
        fname_scales = os.path.join(output_directory, 'scales.npy')

        np.save(fname_templates, templates_new)
        np.save(fname_spike_train, spike_train_new)
        np.save(fname_shifts, shifts_new)
        np.save(fname_scales, scales_new)
        
        
    if CONFIG.deconvolution.deconv_gpu: 
        residual_ONgpu(recordings_filename,
                       recording_dtype,
                       CONFIG,
                       fname_shifts,
                       fname_scales,
                       fname_templates,
                       output_directory,
                       dtype_out,
                       fname_out,
                       fname_spike_train,
                       update_templates,
                       run_chunk_sec)
    
    else:
        residual_ONcpu(fname_templates,
                       fname_spike_train,
                       output_directory,
                       recordings_filename,
                       recording_dtype,
                       dtype_out,
                       fname_out,
                       run_chunk_sec,
                       CONFIG)
            
    return fname_out, dtype_out

    
def residual_ONgpu(recordings_filename,
                   recording_dtype,
                   CONFIG,
                   fname_shifts,
                   fname_scales,
                   fname_templates,
                   output_directory,
                   dtype_out,
                   fname_out,
                   fname_spike_train,
                   update_templates,
                   run_chunk_sec):
    print(np.load(fname_templates).shape)
    # get data reader
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    CONFIG.resources.n_sec_chunk_gpu_deconv,
                    buffer=(CONFIG.spike_size-1)*2,
                    chunk_sec=chunk_sec)
                    
    if False:
        RESIDUAL_GPU3(reader,
                  recordings_filename,
                  recording_dtype,
                  CONFIG,
                  fname_shifts,
                  fname_scales,
                  fname_templates,
                  output_directory,
                  dtype_out,
                  fname_out,
                  fname_spike_train,
                  update_templates)
    else:
        RESIDUAL_GPU2(reader,
                      CONFIG,
                      fname_shifts,
                      fname_scales,
                      fname_templates,
                      output_directory,
                      dtype_out,
                      fname_out,
                      fname_spike_train,
                      update_templates)
                  
    
def residual_ONcpu(fname_templates,
                   fname_spike_train,
                   output_directory,
                   recordings_filename,
                   recording_dtype,
                   dtype_out,
                   fname_out,
                   run_chunk_sec,
                   CONFIG):

    # get data reader
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    reader = READER(recordings_filename,
                    recording_dtype,
                    CONFIG,
                    CONFIG.resources.n_sec_chunk,
                    chunk_sec=chunk_sec)

    # get residual object
    residual_object = RESIDUAL(fname_templates,
                               fname_spike_train,
                               reader,
                               fname_out,
                               dtype_out)


    # compute residual
    seg_dir = os.path.join(output_directory, 'segs')
    residual_object.compute_residual(seg_dir,
                                     CONFIG.resources.multi_processing,
                                     CONFIG.resources.n_processors)

    # concatenate all segments
    residual_object.save_residual()
