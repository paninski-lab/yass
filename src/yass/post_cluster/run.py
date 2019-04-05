import logging
import numpy as np
import os

from yass import read_config
from yass.reader import READER
from yass.post_cluster.small_ptp import remove_small_units
from yass.post_cluster.duplicate import remove_duplicates
from yass.post_cluster.collision import remove_collision
from yass.post_cluster.mad import remove_high_mad
from yass.post_cluster.util import get_weights

def run(fname_templates,
        fname_spike_train,
        output_directory,
        small_units=True,
        duplicate=True,
        collision=True,
        high_mad=True,
        fname_recording=None,
        recording_dtype=None):

    ''' Function that cleans low spike count templates and merges the rest
    '''

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # get weights
    fname_weights = get_weights(fname_templates,
                                fname_spike_train,
                                output_directory)

    ##################################
    ##### Remove small ptp units #####
    ##################################

    if small_units:
        fname_result = os.path.join(output_directory,
                                    'units_post_ptp_kill.npy')
        if os.path.exists(fname_result):
            units_post_ptp_kill = np.load(fname_result)
        else:
            # TODO: move parameter to config?
            threshold = 3
            # remove
            units_post_ptp_kill = remove_small_units(
                fname_templates, threshold)
            # save
            np.save(fname_result, units_post_ptp_kill)

        logger.info("{} units after removing small units".format(
            len(units_post_ptp_kill)))
    else:
        units_post_ptp_kill = None

    ############################
    ##### Remove Duplicate #####
    ############################

    if duplicate:
        fname_result = os.path.join(output_directory, 
                                    'units_post_duplicate_kill.npy')
        if os.path.exists(fname_result):
            units_post_duplicate_kill = np.load(fname_result)

        else:
            # save folder
            save_dir = os.path.join(output_directory, 'duplicates')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # remove duplicates
            units_post_duplicate_kill = remove_duplicates(
                fname_templates,
                fname_weights,
                save_dir,
                units_post_ptp_kill,
                CONFIG.resources.n_processors,
                CONFIG.resources.multi_processing)

            # save
            np.save(fname_result, units_post_duplicate_kill)

        logger.info("{} units after removing duplicate units".format(
            len(units_post_duplicate_kill)))
    else:
        units_post_duplicate_kill = units_post_ptp_kill

    ############################
    ##### Remove Collision #####
    ############################

    if collision:
        fname_result = os.path.join(output_directory, 
                                    'units_post_collision_kill.npy')
        if os.path.exists(fname_result):
            units_post_collision_kill = np.load(fname_result)

        else:
            # save folder
            save_dir = os.path.join(output_directory, 'collision')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # find collision units and remove
            units_post_collision_kill = remove_collision(
                fname_templates,
                save_dir,
                units_post_duplicate_kill,
                CONFIG.resources.n_processors,
                CONFIG.resources.multi_processing)

            # save
            np.save(fname_result, units_post_collision_kill)

        logger.info("{} units after removing collision units".format(
            len(units_post_collision_kill)))
    else:
        units_post_collision_kill = units_post_duplicate_kill

    ###########################
    ##### Remove High MAD #####
    ###########################

    if high_mad:
        fname_result = os.path.join(output_directory, 
                                    'units_post_mad_kill.npy')
        if os.path.exists(fname_result):
            units_post_mad_kill = np.load(fname_result)

        else:
            
            # get data reader
            reader = READER(fname_recording,
                            recording_dtype,
                            CONFIG) 

            # save folder
            save_dir = os.path.join(output_directory, 'mad')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # find high mad units and remove
            units_post_mad_kill = remove_high_mad(
                fname_templates,
                fname_spike_train,
                fname_weights,
                reader,
                save_dir,
                units_post_collision_kill,
                CONFIG.resources.n_processors,
                CONFIG.resources.multi_processing)

            # save
            np.save(fname_result, units_post_mad_kill)

        logger.info("{} units after removing high mad units".format(
            len(units_post_mad_kill)))
    else:
        units_post_mad_kill = units_post_collision_kill
    
    units_sruvived = units_post_mad_kill

    # load templates and spike train
    templates = np.load(fname_templates)
    spike_train = np.load(fname_spike_train)

    if units_sruvived is not None:
        # update templates
        templates = templates[units_sruvived]

        # update spike train
        spike_train_new = np.copy(spike_train)
        spike_train_new = spike_train_new[
            np.in1d(spike_train[:,1], units_sruvived)]
        dics = {unit: ii for ii, unit in enumerate(units_sruvived)}
        for j in range(spike_train_new.shape[0]):
            spike_train_new[j,1] = dics[spike_train_new[j,1]]
        spike_train = spike_train_new

    fname_templates = os.path.join(output_directory, 'templates.npy')
    fname_spike_train = os.path.join(output_directory, 'spike_train.npy')

    np.save(fname_templates, templates)
    np.save(fname_spike_train, spike_train)        

    return fname_templates, fname_spike_train
