import logging
import numpy as np
import os
from tqdm import tqdm
import parmap

from yass import read_config
from yass.reader import READER
from yass.cluster.cluster import Cluster
from yass.cluster.util import (make_CONFIG2,
                               make_spike_index_from_spike_train,
                               partition_input,
                               gather_clustering_result,
                               load_waveforms,
                               align_waveforms,
                               nn_denoise_wf, denoise_wf,
                               denoise_then_estimate_template)
from yass.cluster.ptp_split import run_split_on_ptp
from yass.cluster.sharpen import sharpen_templates
from yass.neuralnetwork import Denoise
from yass.template import run_template_computation, fix_template_edges_by_file

def run(output_directory,
        fname_recording,
        recording_dtype,
        fname_residual=None,
        residual_dtype=None,
        fname_spike_index=None,
        fname_templates=None,
        fname_spike_train=None,
        fname_shifts=None,
        fname_scales=None,
        raw_data=True,
        full_run=False):

    """Spike clustering

    Parameters
    ----------

    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    out_dir: str, optional
        Location to store/look for the generate spike train relative to
        config output directory

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for the
      spike_train_cluster.npy. file If 'overwrite' it replaces the files if
      exists, if 'abort' it raises a ValueError exception if exists,
      if 'skip' it skips the operation if the file exists (and returns the
      stored file)

    save_results: bool, optional
        Whether to save spike train to disk
        (in CONFIG.data.root_folder/relative_to/spike_train_cluster.npy),
        defaults to False

    Returns
    -------
    spike_train: (TODO add documentation)

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/cluster.py

    """
    logger = logging.getLogger(__name__)

    ########################
    ### INITIALIZE #########
    ########################

    CONFIG = read_config()
    # get CONFIG2 for clustering
    # Cat: TODO: Edu said the CONFIG file can be passed as a dictionary
    CONFIG2 = make_CONFIG2(CONFIG)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # data reader
    reader_raw = READER(fname_recording,
                        recording_dtype,
                        CONFIG,
                        CONFIG.resources.n_sec_chunk_gpu_deconv,
                        chunk_sec = CONFIG.clustering_chunk)
    if fname_residual is not None:
        reader_resid = READER(fname_residual,
                              residual_dtype,
                              CONFIG,
                              CONFIG.resources.n_sec_chunk_gpu_deconv,
                              chunk_sec = CONFIG.clustering_chunk
                             )
    else:
        reader_resid = None

    # nn denoiser
    if CONFIG.neuralnetwork.apply_nn:
        # load NN denoiser
        denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                           CONFIG.neuralnetwork.denoise.filter_sizes,
                           CONFIG.spike_size_nn, CONFIG)
        denoiser.load(CONFIG.neuralnetwork.denoise.filename)
        denoiser = denoiser.cuda()
    else:
        denoiser = None


    # if the output exists and want to skip, just finish
    fname_templates_out = os.path.join(output_directory, 'templates.npy')
    fname_spike_train_out = os.path.join(output_directory, 'spike_train.npy')
    if not os.path.exists(fname_templates_out):
    
        # if clustering on clean waveforms, spike train is given 
        # => make spike index and labels
        if fname_spike_index is None:
            savedir = os.path.join(output_directory, 'spike_index')
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            (fname_spike_index,
             fname_labels_input) = make_spike_index_from_spike_train(
                fname_spike_train, fname_templates, savedir)

        else:
            # if we have spike_index, then we have no initial labels
            fname_labels_input = None

        #################################
        #### STAGE 1: Cluster on PTP ####
        #################################

        # keep track of input label because this is the deconv label
        # and it is necessary when making cleaned spikes
        logger.info("Split on PTP")
        (fname_spike_index,
         fname_labels,
         fname_labels_input) = run_split_on_ptp(
            os.path.join(output_directory, 'ptp_split'),
            fname_spike_index,
            CONFIG2,
            raw_data,
            fname_labels_input,
            fname_templates,
            fname_shifts,
            fname_scales,
            reader_raw,
            reader_resid,
            denoiser)

        ############################################
        #### STAGE 2: LOCAL + DISTANT CLUSTERING ###
        ############################################

        # load and align waveforms
        logger.info("load waveforms on local channels")
        units, fnames_input = load_waveforms(
            os.path.join(output_directory, 'input'),
            raw_data,
            fname_labels,
            fname_spike_index,
            fname_labels_input,
            fname_templates,
            fname_shifts,
            fname_scales,
            reader_raw,
            reader_resid,
            CONFIG2)

        if CONFIG.neuralnetwork.apply_nn:
            logger.info("NN denoise")
            # denoise it
            nn_denoise_wf(fnames_input, denoiser, CONFIG.torch_devices, CONFIG)
        else:
            logger.info("denoise")
            denoise_wf(fnames_input)

        #if raw_data:
        # align if raw data
        # no need to align for clean waveforms
        # because input shift is already used for alignment
        logger.info("align waveforms on local channels")
        align_waveforms(fnames_input, CONFIG2)

        # save location for intermediate results
        tmp_save_dir = os.path.join(output_directory, 'cluster_result')
        if not os.path.exists(tmp_save_dir):
            os.makedirs(tmp_save_dir)

        # Cat: TODO: this parallelization may not be optimally asynchronous
        # make arg list first
        args_in = []
        for ctr, unit in enumerate(units):

            # check to see if chunk + channel already completed
            filename_postclustering = os.path.join(
                tmp_save_dir, "cluster_result_{}.npz".format(unit))

            # skip 
            if os.path.exists(filename_postclustering):
                continue
            args_in.append([raw_data,
                            full_run,
                            CONFIG2,
                            reader_raw,
                            reader_resid,
                            filename_postclustering,
                            fnames_input[ctr]])

        logger.info("starting clustering")
        if CONFIG.resources.multi_processing:
            parmap.map(Cluster, args_in, 
                       processes=CONFIG.resources.n_processors,
                       pm_pbar=True)

        else:
            with tqdm(total=len(args_in)) as pbar:
                for arg_in in args_in:
                    Cluster(arg_in)
                    pbar.update()


        # first gather clustering result
        fname_templates_out, fname_spike_train_out = gather_clustering_result(
            tmp_save_dir, output_directory)

        for fname in fnames_input:
            os.remove(fname)

    #check_long_temp = os.path.join(output_directory, 'long_template.npy')
    #if not os.path.exists(check_long_temp):
    #    logger.info("get longer templates")
    #    fname_templates_out = run_template_computation(
    #        output_directory,
    #        fname_spike_train_out,
    #        reader_raw,
    #        spike_size=CONFIG.spike_size,
    #        multi_processing=CONFIG.resources.multi_processing,
    #        n_processors=CONFIG.resources.n_processors)
    #    np.save(check_long_temp, None)

    #check_low_fr_temp = os.path.join(output_directory, 'check_low_fr_template.npy')
    #if not os.path.exists(check_low_fr_temp):
    #    if CONFIG.neuralnetwork.apply_nn:
    #        # denoise wfs before computing templates for low fr units
    #        logger.info("re-estimate templates of low firing rate units")
    #        fname_templates_out = denoise_then_estimate_template(
    #            fname_templates_out,
    #            fname_spike_train_out,
    #            reader_raw,
    #            denoiser,
    #            CONFIG,
    #            n_max_spikes=100)

    #    np.save(check_low_fr_temp, None)

    #check_sharpen = os.path.join(output_directory, 'check_sharpen.npy')
    #if not os.path.exists(check_sharpen):
    #fname_templates_aligned = os.path.join(output_directory, 'templates_aligned.npy')
    #if not os.path.exists(fname_templates_aligned):
    #    logger.info("subsample template alignment")
    #    fname_templates_out = sharpen_templates(fname_templates_out,
    #                                            fname_templates_aligned)

    # zero-out edges
    #check_zero_out = os.path.join(output_directory, 'check_zero_out.npy')
    #if not os.path.exists(check_zero_out):
    #    logger.info("zero out unnecessary parts")
    #    fix_template_edges_by_file(fname_templates_out,
    #                               CONFIG.center_spike_size)
    #    np.save(check_zero_out, None)

    return fname_templates_out, fname_spike_train_out
