import warnings
import os
import datetime as dt
import logging

from .process.triage import triage
from .process.coreset import coreset
from .process.mask import getmask
from .process.cluster import runSorter
from .process.clean import clean_output
from .process.templates import get_templates
from .util import deprecated


@deprecated('Use function in process module, see examples/process.py')
class Mainprocessor(object):

    def __init__(self, config, score, clr_idx, spt):

        self.config = config
        self.score = score
        self.clr_idx = clr_idx
        self.spt = spt

        self.logger = logging.getLogger(__name__)

    def mainProcess(self):

        startTime = dt.datetime.now()
        Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

        _b = dt.datetime.now()
        self.logger.info("Triaging...")
        self.score, self.clr_idx = triage(self.score, self.clr_idx,
                                          self.config.nChan, self.config.triageK,
                                          self.config.triagePercent,
                                          self.config.neighChannels,
                                          self.config.doTriage)
        Time['t'] += (dt.datetime.now()-_b).total_seconds()

        if self.config.doCoreset:
            _b = dt.datetime.now()
            self.logger.info("Coresettting...")
            self.group = coreset(self.score, self.config.nChan,
                                 self.config.coresetK, self.config.coresetTh)

            Time['c'] += (dt.datetime.now()-_b).total_seconds()

        _b = dt.datetime.now()
        self.logger.info("Masking...")
        self.mask = getmask(self.score, self.group, self.config.maskTh,
                            self.config.nFeat, self.config.nChan,
                            self.config.doCoreset)

        Time['m'] += (dt.datetime.now()-_b).total_seconds()

        _b = dt.datetime.now()
        self.logger.info("Clustering...")

        spike_train_clear = runSorter(self.score, self.mask,
                                      self.clr_idx, self.group,
                                      self.config.channelGroups,
                                      self.config.neighChannels,
                                      self.config.nFeat,
                                      self.config)
        Time['s'] += (dt.datetime.now()-_b).total_seconds()
        spike_train_clear, spt_left = clean_output(spike_train_clear, self.spt,
                                            self.clr_idx,
                                            self.config.batch_size,
                                            self.config.BUFF)
            
        _b = dt.datetime.now()

        self.logger.info("Getting Templates...")

        path_to_wrec = os.path.join(self.config.root, 'tmp/wrec.bin')
        spike_train_clear, templates = get_templates(spike_train_clear,
                                                     self.config.batch_size,
                                                     self.config.BUFF,
                                                     self.config.nBatches,
                                                     self.config.nChan,
                                                     self.config.spikeSize,
                                                     self.config.templatesMaxShift,
                                                     self.config.scaleToSave,
                                                     self.config.neighChannels,
                                                     path_to_wrec,
                                                     self.config.tMergeTh)

        # FIXME: code to avoid breaking legacy code (used in deconcvolution)
        self.templates = templates

        Time['e'] += (dt.datetime.now()-_b).total_seconds()



        currentTime = dt.datetime.now()
        self.logger.info("Mainprocess done in {0} seconds.".format(
            (currentTime-startTime).seconds))
        self.logger.info("\ttriage:\t{0} seconds".format(Time['t']))
        self.logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
        self.logger.info("\tmasking:\t{0} seconds".format(Time['m']))
        self.logger.info("\tclustering:\t{0} seconds".format(Time['s']))
        self.logger.info("\tmake templates:\t{0} seconds".format(Time['e']))

        return spike_train_clear, spt_left
