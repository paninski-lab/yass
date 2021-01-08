import logging
import numpy as np
import os

from yass import read_config
from yass.reader import READER
from yass.geometry import n_steps_neigh_channels
from yass.postprocess.small_ptp import remove_small_units
from yass.postprocess.off_centered import remove_off_centered_units
from yass.postprocess.low_fr import remove_low_fr_units
from yass.postprocess.high_fr import remove_high_fr_units
from yass.postprocess.duplicate import remove_duplicates
from yass.postprocess.collision import remove_collision
from yass.postprocess.mad import remove_high_mad
from yass.postprocess.duplicate_l2 import duplicate_l2
from yass.postprocess.xcorr_peaks import remove_high_xcorr_peaks
from yass.postprocess.duplicate_soft_assignment import duplicate_soft_assignment
from yass.postprocess.util import get_weights

def run(methods = [],
        output_directory=None,
        fname_recording=None,
        recording_dtype=None,
        fname_templates=None,
        fname_spike_train=None,
        fname_template_soft_assignment=None,
        fname_noise_soft_assignment=None,
        fname_shifts=None,
        fname_scales=None,
        units_to_process=None):

    ''' Run a sequence of post processes
    
    methods: list of strings.
        Options are 'low_ptp', 'duplicate', 'collision',
        'high_mad', 'low_fr', 'high_fr'
        
    '''   

    logger = logging.getLogger(__name__)

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # get weights and number of units
    fname_weights, n_units = get_weights(
        output_directory,
        fname_templates,
        fname_spike_train,
        fname_noise_soft_assignment)

    # get output names
    fname_templates_out = os.path.join(output_directory, 'templates.npy')
    fname_spike_train_out = os.path.join(output_directory, 'spike_train.npy')
    fname_noise_soft_assignment_out = os.path.join(output_directory, 'noise_soft_assignment.npy')
    fname_scales_out = os.path.join(output_directory, 'scales.npy')
    fname_shifts_out = os.path.join(output_directory, 'shifts.npy') 
    if os.path.exists(fname_templates_out) and os.path.exists(fname_spike_train_out):
        return fname_templates_out, fname_spike_train_out, fname_noise_soft_assignment_out, fname_shifts_out, fname_scales_out

    # run each method
    units_survived = np.arange(n_units)
    if units_to_process is None:
        units_to_process = np.copy(units_survived)

    logger.info("{} units are in".format(n_units))
    for ctr, method in enumerate(methods):

        # save name
        fname_result = os.path.join(
            output_directory,
            'units_survived_{}_{}.npy'.format(ctr, method))

        if os.path.exists(fname_result):
            units_survived = np.load(fname_result)
            logger.info("{} is already done. {} units survived".format(
                method, len(units_survived)))

            continue

        units_survived = post_process(
            output_directory,
            fname_templates,
            fname_spike_train,
            fname_weights,
            fname_template_soft_assignment,
            fname_recording,
            recording_dtype,
            units_survived,
            units_to_process,
            method,
            ctr)

        # save result for record
        np.save(fname_result, units_survived)

    # load templates and spike train
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)
    if len(units_survived) < n_units:

        # update templates
        templates = templates[units_survived]

        # update spike train
        spike_train_new = np.copy(spike_train)
        idx_keep = np.in1d(spike_train[:,1], units_survived)
        spike_train_new = spike_train_new[idx_keep]
        dics = {unit: ii for ii, unit in enumerate(units_survived)}
        for j in range(spike_train_new.shape[0]):
            spike_train_new[j,1] = dics[spike_train_new[j,1]]

        if fname_noise_soft_assignment is not None:
            noise_soft_assignment = np.load(fname_noise_soft_assignment)
            noise_soft_assignment = noise_soft_assignment[idx_keep]
            np.save(fname_noise_soft_assignment_out, noise_soft_assignment)

        if fname_scales is not None:
            scales = np.load(fname_scales)
            scales = scales[idx_keep]
            np.save(fname_scales_out, scales)
            
        if fname_shifts is not None:
            shifts = np.load(fname_shifts)
            shifts = shifts[idx_keep]
            np.save(fname_shifts_out, shifts)

    else:
        spike_train_new = np.copy(spike_train)
        if fname_noise_soft_assignment is not None:
            np.save(fname_noise_soft_assignment_out, np.load(fname_noise_soft_assignment))
        if fname_scales is not None:
            np.save(fname_scales_out, np.load(fname_scales))
        if fname_shifts is not None:
            np.save(fname_shifts_out, np.load(fname_shifts))

    np.save(fname_templates_out, templates)
    np.save(fname_spike_train_out, spike_train_new)

    return (fname_templates_out, fname_spike_train_out, 
            fname_noise_soft_assignment_out, fname_shifts_out, fname_scales_out)

def post_process(output_directory,
                 fname_templates,
                 fname_spike_train,
                 fname_weights,
                 fname_template_soft_assignment,
                 fname_recording,
                 recording_dtype,
                 units_in,
                 units_to_process,
                 method,
                 ctr):

    ''' 
    Run a single post process
    method: strings.
        Options are 'low_ptp', 'duplicate', 'collision',
        'high_mad', 'low_fr', 'high_fr', 'off_center',
        'duplicate_l2'
    '''

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    if method == 'low_ptp':
        
        #

        # Cat: TODO: move parameter to CONFIG
        threshold = CONFIG.clean_up.min_ptp
        print("CONFIG threshoold: ", threshold)

        # load templates
        templates = np.load(fname_templates)

        # remove low ptp
        idx_process = np.in1d(units_in, units_to_process)
        units_in_ = units_in[idx_process]
        units_not_in_ = units_in[~idx_process]
        units_out_ = remove_small_units(
            templates, threshold, units_in_)
        units_out = np.sort(np.hstack((units_not_in_, units_out_)))

        logger.info("{} units after removing low ptp units".format(
            len(units_out)))

    elif method == 'off_center':

        threshold = CONFIG.clean_up.off_center

        # load templates
        templates = np.load(fname_templates)

        # remove off centered units
        idx_process = np.in1d(units_in, units_to_process)
        units_in_ = units_in[idx_process]
        units_not_in_ = units_in[~idx_process]
        units_out_ = remove_off_centered_units(
            templates, threshold, units_in_)
        units_out = np.sort(np.hstack((units_not_in_, units_out_)))

        logger.info("{} units after removing off centered units".format(
            len(units_out)))

    elif method == 'duplicate':

        # tmp saving dir
        save_dir = os.path.join(output_directory,
                                'duplicates_{}'.format(ctr))

        # remove duplicates
        units_out = remove_duplicates(
            fname_templates,
            fname_weights,
            save_dir,
            CONFIG,
            units_in,
            units_to_process,
            CONFIG.resources.multi_processing,
            CONFIG.resources.n_processors)

        logger.info("{} units after removing duplicate units".format(
            len(units_out)))

    elif method == 'duplicate_l2':

        # tmp saving dir
        save_dir = os.path.join(output_directory,
                                'duplicates_l2_{}'.format(ctr))

        # remove duplicates
        n_spikes_big = 100
        min_ptp = 2
        units_out = duplicate_l2(
            fname_templates,
            fname_spike_train,
            CONFIG.neigh_channels,
            save_dir,
            n_spikes_big,
            min_ptp,
            units_in)

        logger.info("{} units after removing L2 duplicate units".format(
            len(units_out)))

    elif method == 'collision':
        # save folder
        save_dir = os.path.join(output_directory,
                                'collision_{}'.format(ctr))

        # find collision units and remove
        units_out = remove_collision(
            fname_templates,
            save_dir,
            CONFIG,
            units_in,
            CONFIG.resources.multi_processing,
            CONFIG.resources.n_processors)

        logger.info("{} units after removing collision units".format(
            len(units_out)))     

    elif method == 'high_mad':

        # get data reader
        reader = READER(fname_recording,
                        recording_dtype,
                        CONFIG) 

        # save folder
        save_dir = os.path.join(output_directory,
                                'mad_{}'.format(ctr))

        # neighboring channels
        neigh_channels = n_steps_neigh_channels(
            CONFIG.neigh_channels, 2)

        max_violations = CONFIG.clean_up.mad.max_violations
        min_var_gap = CONFIG.clean_up.mad.min_var_gap

        # find high mad units and remove
        units_out = remove_high_mad(
            fname_templates,
            fname_spike_train,
            fname_weights,
            reader,
            neigh_channels,
            save_dir,
            min_var_gap,
            max_violations,
            units_in,
            CONFIG.resources.multi_processing,
            CONFIG.resources.n_processors)

        logger.info("{} units after removing high mad units".format(
            len(units_out)))
    
    elif method == 'low_fr':

        threshold = CONFIG.clean_up.min_fr
 
        # length of recording in seconds
        rec_len = np.load(fname_spike_train)[:, 0].ptp()
        rec_len_sec = float(rec_len)/CONFIG.recordings.sampling_rate

        # load templates
        weights = np.load(fname_weights)

        # remove low ptp
        idx_process = np.in1d(units_in, units_to_process)
        units_in_ = units_in[idx_process]
        units_not_in_ = units_in[~idx_process]
        units_out_ = remove_low_fr_units(
            weights, rec_len_sec, threshold, units_in_)
        units_out = np.sort(np.hstack((units_not_in_, units_out_)))

        logger.info("{} units after removing low fr units".format(
            len(units_out)))

    elif method == 'high_fr':

        # TODO: move parameter to config?
        threshold = 70

        # length of recording in seconds
        rec_len = np.load(fname_spike_train)[:, 0].ptp()
        rec_len_sec = float(rec_len)/CONFIG.recordings.sampling_rate

        # load templates
        weights = np.load(fname_weights)

        # remove low ptp
        units_out = remove_high_fr_units(
            weights, rec_len_sec, threshold, units_in)

        logger.info("{} units after removing high fr units".format(
            len(units_out)))

    elif method == 'high_xcorr':

        save_dir = os.path.join(output_directory,
            'high_xcorr_{}'.format(ctr))
        threshold=5
        units_out = remove_high_xcorr_peaks(
            save_dir,
            fname_spike_train,
            fname_templates,
            CONFIG.recordings.sampling_rate,
            threshold,
            units_in)

        logger.info("{} units after removing high xcorr units".format(
            len(units_out)))


    elif method == 'duplicate_soft_assignment':

        threshold = 0.7

        # remove low ptp
        idx_process = np.in1d(units_in, units_to_process)
        units_in_ = units_in[idx_process]
        units_not_in_ = units_in[~idx_process]
        units_out_ = duplicate_soft_assignment(
            fname_template_soft_assignment,
            threshold,
            units_in_)
        units_out = np.sort(np.hstack((units_not_in_, units_out_)))

        logger.info("{} units after removing duplicates using soft assignment".format(
            len(units_out)))
    else:
        units_out = np.copy(units_in)
        logger.info("Method not recognized. Nothing removed")

    return units_out
