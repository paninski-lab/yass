import warnings
import os
import datetime
import logging
import numpy as np

from .process.triage import triage
from .process.coreset import coreset
from .process.mask import getmask
from .process.cluster import runSorter
from .process.clean import clean_output
from .process.templates import get_templates
from .MFM import spikesort
from .geometry import order_channels_by_distance
from .util import deprecated


@deprecated('Use function in process module, see examples/process.py')
class Mainprocessor(object):

    def __init__(self, config, score, spike_index_clear, spike_index_collision):

        self.config = config
        self.score = score
        self.spike_index_clear = spike_index_clear
        self.spike_index_collision = spike_index_collision

        self.logger = logging.getLogger(__name__)

    def mainProcess(self):

        SPIKE_TIME, MAIN_CHANNEL = 0, 1
            
        startTime = datetime.datetime.now()
        Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}
        
        nG = len(self.config.channelGroups)
        nneigh = np.max(np.sum(self.config.neighChannels, 0))
        n_coreset = 0
        K = 0

        # first column: spike_time
        # second column: cluster id
        spike_train_clear = np.zeros((0,2), 'int32')

        # order of channels
        c_idx = np.ones((self.config.recordings.n_channels, nneigh), 'int32')*self.config.recordings.n_channels
        for c in range(self.config.recordings.n_channels):
            ch_idx, _ = order_channels_by_distance(c,
                                                   np.where(
                                                   self.config.neighChannels[c])[0],
                                                   self.config.geom)
            c_idx[c,:ch_idx.shape[0]] = ch_idx


        for g in range(nG):
            self.logger.info("Processing group {} in {} groups.".format(g+1, nG))
            self.logger.info("Processiing data (triage, coreset, masking) ...")
            channels = self.config.channelGroups[g]
            neigh_chans = np.where(np.sum(self.config.neighChannels[channels], axis=0) > 0)[0]

            score_group = np.zeros((0, self.config.spikes.temporal_features, neigh_chans.shape[0]))
            coreset_id_group = np.zeros((0), 'int32')
            mask_group = np.zeros((0, neigh_chans.shape[0]))
            spike_index_clear_group = np.zeros((0, 2), 'int32')
            for c in channels:

                # index of data whose main channel is c
                idx = self.spike_index_clear[:,MAIN_CHANNEL] == c
                if np.sum(idx) > 0:

                    # score whose main channel is c
                    score_c = self.score[idx]
                    # spike_index_clear whose main channel is c
                    spike_index_clear_c = self.spike_index_clear[idx]


                    ##########
                    # Triage #
                    ##########

                    # TODO: refactor this as CONFIG.doTriage was removed
                    doTriage = True

                    _b = datetime.datetime.now()
                    index_keep = triage(score_c, 0, self.config.triage.nearest_neighbors,
                                        self.config.triage.percent, doTriage)
                    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

                    # add untriaged spike index to spike_index_clear_group
                    # and triaged spike index to spike_index_collision
                    spike_index_clear_group = np.concatenate((
                        spike_index_clear_group, spike_index_clear_c[index_keep]),
                        axis = 0)
                    self.spike_index_collision = np.concatenate((
                        self.spike_index_collision, spike_index_clear_c[~index_keep]),
                        axis = 0)

                    # keep untriaged score only
                    score_c = score_c[index_keep]


                    ###########
                    # Coreset #
                    ###########

                    # TODO: refactor this as CONFIG.doCoreset was removed
                    doCoreset = True

                    _b = datetime.datetime.now()
                    coreset_id = coreset(score_c, self.config.coreset.clusters,
                        self.config.coreset.threshold, doCoreset)
                    Time['c'] += (datetime.datetime.now()-_b).total_seconds()


                    ###########
                    # Masking #
                    ###########

                    _b = datetime.datetime.now()
                    mask = getmask(score_c, coreset_id, self.config.clustering.masking_threshold, self.config.spikes.temporal_features)
                    Time['m'] += (datetime.datetime.now()-_b).total_seconds()


                    ################
                    # collect data #
                    ################

                    # restructure score_c and mask to have same number of channels
                    # as score_group
                    score_temp = np.zeros((score_c.shape[0],
                        self.config.spikes.temporal_features, neigh_chans.shape[0]))
                    mask_temp = np.zeros((mask.shape[0],neigh_chans.shape[0]))
                    nneigh_c = np.sum(c_idx[c] < self.config.recordings.n_channels)
                    for j in range(nneigh_c):
                        c_interest = np.where(neigh_chans == c_idx[c,j])[0][0]
                        score_temp[:,:,c_interest] = score_c[:,:,j]
                        mask_temp[:,c_interest] = mask[:,j]

                    # add score, coreset_id, mask to the groups
                    score_group = np.concatenate(
                        (score_group, score_temp), axis = 0)
                    mask_group = np.concatenate(
                        (mask_group, mask_temp), axis = 0)
                    coreset_id_group = np.concatenate(
                        (coreset_id_group, coreset_id + n_coreset + 1), axis = 0)
                    n_coreset += np.max(coreset_id) + 1


            if score_group.shape[0] > 0:

                ##############
                # Clustering #
                ##############

                _b = datetime.datetime.now()
                self.logger.info("Clustering...")
                coreset_id_group = coreset_id_group - 1
                n_coreset = 0
                cluster_id = spikesort(score_group, mask_group, coreset_id_group, self.config)
                Time['s'] += (datetime.datetime.now()-_b).total_seconds()


                ############
                # Cleaning #
                ############

                # model based triage
                idx_triage = (cluster_id == -1)

                # concatenate spike index with cluster id of untriaged ones
                # to create spike_train_clear
                si_clustered = spike_index_clear_group[~idx_triage]
                spt = si_clustered[:,[0]]
                cluster_id = cluster_id[~idx_triage][:, np.newaxis]

                spike_train_temp = np.concatenate((spt,cluster_id+K), axis=1)
                spike_train_clear = np.concatenate(
                    (spike_train_clear, spike_train_temp), axis=0)
                K += np.amax(cluster_id)+1

                # concatenate triaged spike_index_clear_group
                # into spike_index_collision
                self.spike_index_collision = np.concatenate((
                    self.spike_index_collision, spike_index_clear_group[idx_triage]),
                    axis = 0)

        #################
        # Get templates #
        #################

        _b = datetime.datetime.now()
        self.logger.info("Getting Templates...")
        path_to_whiten = os.path.join(self.config.data.root_folder, 'tmp/wrec.bin')
        spike_train_clear, templates = get_templates(spike_train_clear,
                                                     self.config.batch_size,
                                                     self.config.BUFF,
                                                     self.config.nBatches,
                                                     self.config.recordings.n_channels,
                                                     self.config.spikeSize,
                                                     self.config.templatesMaxShift,
                                                     self.config.scaleToSave,
                                                     self.config.neighChannels,
                                                     path_to_whiten,
                                                     self.config.templates.merge_threshold)
        self.templates = templates
        Time['e'] += (datetime.datetime.now()-_b).total_seconds()


        currentTime = datetime.datetime.now()

        self.logger.info("Mainprocess done in {0} seconds.".format(
            (currentTime-startTime).seconds))
        self.logger.info("\ttriage:\t{0} seconds".format(Time['t']))
        self.logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
        self.logger.info("\tmasking:\t{0} seconds".format(Time['m']))
        self.logger.info("\tclustering:\t{0} seconds".format(Time['s']))
        self.logger.info("\ttemplates:\t{0} seconds".format(Time['e']))

        return spike_train_clear, self.spike_index_collision
