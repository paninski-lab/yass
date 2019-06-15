"""
Detection pipeline
"""
import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from yass import read_config
from yass.reader import READER
from yass.augment.noise import noise_cov
from yass.augment.util import crop_and_align_templates, Detection_Training_Data, Denoising_Training_Data
from yass.template import run_template_computation

def run(fname_recording, recording_dtype, fname_spike_train,
        output_directory):
           
    """
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # make output directory if not exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # get reader
    reader = READER(fname_recording,
                    recording_dtype,
                    CONFIG)
    reader.spike_size = CONFIG.spike_size_small

    # get noise covariance
    logger.info('Compute Noise Covaraince')
    save_dir = os.path.join(output_directory, 'noise_cov')
    chunk = [0, np.min((5*60*reader.sampling_rate, reader.end))]
    fname_spatial_sig, fname_temporal_sig = get_noise_covariance(
        reader, save_dir, CONFIG, chunk)
    
    # get processed templates
    logger.info('Crop Templates')
    save_dir = os.path.join(output_directory, 'templates')
    fname_templates_snippets = get_templates_on_local_channels(reader,
                                                               save_dir,
                                                               fname_spike_train,
                                                               CONFIG)

    # make training data
    logger.info('Make Training Data')
    DetectTD = Detection_Training_Data(
        fname_templates_snippets,
        fname_spatial_sig,
        fname_temporal_sig)
    
    DenoTD = Denoising_Training_Data(
        fname_templates_snippets,
        fname_spatial_sig,
        fname_temporal_sig)
    
    return DetectTD, DenoTD
    

def get_noise_covariance(reader, save_dir, CONFIG, chunk=None):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname_spatial_sig = os.path.join(save_dir, 'spatial_sig.npy')
    fname_temporal_sig = os.path.join(save_dir, 'temporal_sig.npy')    
    
    if os.path.exists(fname_spatial_sig) and os.path.exists(fname_temporal_sig):
        return fname_spatial_sig, fname_temporal_sig

    # only need subset of channels
    n_neigh_channels = np.sum(CONFIG.neigh_channels, axis=1)
    c_chosen = np.where(n_neigh_channels == np.max(n_neigh_channels))[0][0]
    channels = np.where(CONFIG.neigh_channels[c_chosen])[0]

    if chunk is None:
        rec = reader.read_data(reader.start, reader.end, channels)
    else:
        rec = reader.read_data(chunk[0], chunk[1], channels)

    spatial_SIG, temporal_SIG = noise_cov(
        rec,
        temporal_size=reader.spike_size,
        window_size=reader.spike_size,
        sample_size=1000,
        threshold=3.0)
    
    np.save(fname_spatial_sig, spatial_SIG)
    np.save(fname_temporal_sig, temporal_SIG)
    
    return fname_spatial_sig, fname_temporal_sig


def get_templates_on_local_channels(reader, save_dir,
                                    fname_spike_train, CONFIG):

    # increase the size of templates
    reader.spike_size = (reader.spike_size-1)*4 + 1

    # first compute templates
    fname_templates = run_template_computation(
        fname_spike_train,
        reader,
        save_dir,
        max_channels=None,
        unit_ids=None,
        multi_processing=CONFIG.resources.multi_processing,
        n_processors=CONFIG.resources.n_processors)

    # cropping templates
    fname_templates_cropped = crop_and_align_templates(
        fname_templates, save_dir, CONFIG)

    return fname_templates_cropped

