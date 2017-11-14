import logging
import os
import errno

import datetime as dt
import progressbar
import numpy as np

from .neuralnet import NeuralNetDetector, NeuralNetTriage
from .preprocess.detect import threshold_detection
from .preprocess.filter import whitening_matrix, whitening, butterworth
from .preprocess.score import get_score_pca, get_pca_suff_stat, get_pca_projection
from .preprocess.waveform import get_waveforms
from .preprocess.standarize import standarize, sd
from .util import deprecated


@deprecated('Use function in preprocess module, see examples/preprocess.py')
class Preprocessor(object):

    # root: the absolute path to the location of the file
    # filename: name of the recording file which is binary
    # r_type: type of recording (int16, float, etc.)
    # nChan: number of channels
    # memory_allowance: maximum main memory allowance
    def __init__(self, config):

        self.config = config

        # initialize file handler
        self.File = None
        self.WFile = None

        # make tmp directory if not exist
        try:
            os.makedirs(os.path.join(config.data.root_folder, 'tmp'))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.logger = logging.getLogger(__name__)

    def openWFile(self, opt):
        self.WFile = open(os.path.join(
            self.config.data.root_folder, 'tmp', 'wrec.bin'), opt)

    def closeWFile(self):
        if self.WFile == None:
            return
        else:
            self.WFile.close()
            self.WFile = None

    # opens the binary file to create a buffer
    def openFile(self):
        self.closeFile()
        self.File = open(os.path.join(
            self.config.data.root_folder, self.config.data.recordings), 'rb')

    def closeFile(self):
        if self.File == None:
            return
        else:
            self.File.close()
            self.File = None

    # loads a chunk of binary data into memory
    # offset should be in terms of timesamples
    def load(self, offset, length):
        dsize = self.config.dsize
        self.File.seek(offset*dsize*self.config.recordings.n_channels)
        rec = self.File.read(dsize*self.config.recordings.n_channels*length)
        rec = np.fromstring(rec, dtype=self.config.recordings.dtype)
        rec = rec.reshape(length, self.config.recordings.n_channels)
        return rec

    # chunck should be in C x T format
    # format: format of the recording to be saved
    #         's': standard flattened
    #         't': temporally readable
    def save(self, fid, chunk, _format='s'):
        if _format == 's':
            chunk = chunk.reshape(chunk.shape[0]*chunk.shape[1])
            chunk.astype(self.config.recordings.dtype).tofile(fid)
        else:
            chunk = chunk.transpose().reshape(chunk.shape[0]*chunk.shape[1])
            chunk.astype(self.config.recordings.dtype).tofile(fid)

    def addZeroBuffer(self, rec, buffSize, option):
        buff = np.zeros((buffSize, rec.shape[1]))
        if option == 0:
            return np.append(buff, rec, axis=0)
        elif option == 1:
            return np.append(rec, buff, axis=0)
        elif option == 2:
            return np.concatenate((buff, rec, buff), axis=0)

    def process(self):

        # r: reading, f: filtering, s: standardization, d: detection, p: pca
        # w: whitening, e: extracting and saving waveform
        startTime = dt.datetime.now()
        Time = {'r': 0, 'f': 0, 's': 0, 'd': 0, 'w': 0, 'b': 0, 'e': 0}

        # load nueral net detector if necessary:
        if self.config.methods.detection == 'nn':
            self.nnDetector = NeuralNetDetector(self.config)
            self.proj = self.nnDetector.load_w_ae()
            self.nnTriage = NeuralNetTriage(self.config)

        self.openFile()
        self.openWFile('wb')

        batch_size = self.config.batch_size
        BUFF = self.config.BUFF
        nBatches = self.config.nBatches
        nPortion = self.config.nPortion
        residual = self.config.residual
        
        # initialize output variables
        get_score = 1
        spike_index_clear = None
        spike_index_collision = None
        score = None
        pca_suff_stat = None
        spikes_per_channel = None

        self.logger.info("Preprocessing the data in progress...")
        bar = progressbar.ProgressBar(maxval=nBatches)
        for i in range(0, nBatches):

            # reading data
            _b = dt.datetime.now()
            if nBatches == 1:
                rec = self.load(0, batch_size)
                rec = self.addZeroBuffer(rec, BUFF, 2)
            elif i == 0:
                rec = self.load(i*batch_size, batch_size+BUFF)
                rec = self.addZeroBuffer(rec, BUFF, 0)
            elif i < nBatches-1:
                rec = self.load(i*batch_size-BUFF, batch_size+2*BUFF)
            elif residual == 0:
                rec = self.load(i*batch_size-BUFF, batch_size+BUFF)
                rec = self.addZeroBuffer(rec, BUFF, 1)
            else:
                rec = self.load(i*batch_size-BUFF, residual+BUFF)
                rec = self.addZeroBuffer(rec, BUFF+(batch_size-residual), 1)
            Time['r'] += (dt.datetime.now()-_b).total_seconds()

            if i > nPortion:
                get_score = 0

            (si_clr_batch, score_batch, 
             si_col_batch, pss_batch, 
             spc_batch, Time) = self.batch_process(rec, get_score, 
                                                   BUFF, Time)

            # spike time w.r.t. to the whole recording
            si_clr_batch[:,0] = si_clr_batch[:,0] + i*batch_size - BUFF
            si_col_batch[:,0] = si_col_batch[:,0] + i*batch_size - BUFF
            
            if i == 0:
                spike_index_clear = si_clr_batch
                spike_index_collision = si_col_batch
                score = score_batch

                pca_suff_stat = pss_batch
                spikes_per_channel = spc_batch
            else:
                spike_index_clear = np.vstack((spike_index_clear,
                    si_clr_batch))
                spike_index_collision = np.vstack((spike_index_collision,
                    si_col_batch))
                if get_score == 1:
                    score = np.concatenate((score, score_batch), axis = 0)
                pca_suff_stat += pss_batch
                spikes_per_channel += spc_batch

            bar.update(i+1)

        self.closeFile()
        self.closeWFile()

        if self.config.methods.detection != 'nn':
            _b = dt.datetime.now()
            rot = get_pca_projection(pca_suff_stat, spikes_per_channel,
                                 self.config.nFeat, self.config.neighChannels)

            score = get_score_pca(spike_index_clear, rot, 
                                  self.config.neighChannels,
                                  self.config.geom, 
                                  self.config.batch_size,
                                  self.config.BUFF,
                                  self.config.nBatches,
                                  os.path.join(self.config.data.root_folder, 'tmp', 'wrec.bin'),
                                  self.config.scaleToSave)
            Time['e'] += (dt.datetime.now()-_b).total_seconds()

        # timing
        currentTime = dt.datetime.now()
        self.logger.info("Preprocessing done in {0} seconds.".format(
                         (currentTime-startTime).seconds))
        self.logger.info("\treading data:\t{0} seconds".format(Time['r']))
        self.logger.info("\tfiltering:\t{0} seconds".format(Time['f']))
        self.logger.info("\tstandardization:\t{0} seconds".format(Time['s']))
        self.logger.info("\tdetection:\t{0} seconds".format(Time['d']))
        self.logger.info("\twhitening:\t{0} seconds".format(Time['w']))
        self.logger.info("\tsaving recording:\t{0} seconds".format(Time['b']))
        self.logger.info("\tgetting waveforms:\t{0} seconds".format(Time['e']))

        bar.finish()

        return score, spike_index_clear, spike_index_collision

    def batch_process(self, rec, get_score, BUFF, Time):
        # filter recording
        if self.config.doFilter == 1:
            _b = dt.datetime.now()
            rec = butterworth(rec, self.config.filter.low_pass_freq,
                              self.config.filter.high_factor,
                              self.config.filter.order,
                              self.config.recordings.sampling_rate)
            Time['f'] += (dt.datetime.now()-_b).total_seconds()

        # standardize recording
        _b = dt.datetime.now()
        if not hasattr(self, 'sd'):
            self.sd = sd(rec, self.config.recordings.sampling_rate)

        rec = standarize(rec, self.sd)

        Time['s'] += (dt.datetime.now()-_b).total_seconds()

        # detect spikes
        _b = dt.datetime.now()
        if self.config.methods.detection == 'nn':
            spike_index = self.nnDetector.get_spikes(rec)
        else:

            spike_index = threshold_detection(rec,
                                        self.config.neighChannels,
                                        self.config.spikeSize,
                                        self.config.stdFactor)

        # From Peter: When the recording is too long, I load them by
        # little chunk by chunk (chunk it time-wise). But I also add
        # some buffer. If the detected spike time is in the buffer,
        # i remove that because it will be detected in another chunk
        spike_index = spike_index[np.logical_and(spike_index[:, 0] > BUFF,
                              spike_index[:, 0] < (rec.shape[0] - BUFF))]

        Time['d'] += (dt.datetime.now()-_b).total_seconds()

        # get withening matrix per batch or onece in total
        _b = dt.datetime.now()

        if self.config.whitenBatchwise or not hasattr(self, 'Q'):
            self.Q = whitening_matrix(rec, self.config.neighChannels,
                                      self.config.spikeSize)

        rec = whitening(rec, self.Q)

        Time['w'] += (dt.datetime.now()-_b).total_seconds()

        _b = dt.datetime.now()

        # what is being saved here?
        self.save(self.WFile, rec*self.config.scaleToSave)

        Time['b'] += (dt.datetime.now()-_b).total_seconds()

        _b = dt.datetime.now()
        if get_score == 0:
            # if we are not calculating score for this minibatch, every spikes
            # are considered as collision and will be referred during deconvlution
            spike_index_clear = np.zeros((0,2), 'int32')
            score = None
            spike_index_collision = spike_index

            pca_suff_stat = 0
            spikes_per_channel = 0
            
        elif self.config.methods.detection == 'nn':
            
            # with nn, get scores and triage bad ones
            (spike_index_clear, score,
            spike_index_collision) = get_waveforms(rec,
                                                   spike_index,
                                                   self.proj,
                                                   self.config.neighChannels,
                                                   self.config.geom,
                                                   self.nnTriage,
                                                   self.config.neural_network_detector.threshold_collision)            

            # since we alread have scores, no need to calculated sufficient
            # statistics for pca
            pca_suff_stat = 0
            spikes_per_channel = 0
            
        elif self.config.methods.detection == 'threshold':
            # every spikes are considered as clear spikes as no triage is done
            spike_index_clear = spike_index
            score = None
            spike_index_collision = np.zeros((0,2), 'int32')

            # get sufficient statistics for pca if we don't have projection matrix
            pca_suff_stat, spikes_per_channel = get_pca_suff_stat(rec, spike_index,
                self.config.spikeSize)
        
        Time['e'] += (dt.datetime.now()-_b).total_seconds()

        return (spike_index_clear, score, spike_index_collision,
                pca_suff_stat, spikes_per_channel, Time)

    def getTemplates(self, spikeTrain, R):

        K = np.amax(spikeTrain[:, 1])+1

        batch_size = self.config.batch_size
        BUFF = np.max( (self.config.BUFF, R) )
        nBatches = self.config.nBatches
        nPortion = self.config.nPortion
        residual = self.config.residual
        self.openFile()

        summedTemplatesBig = np.zeros((K, 2*R+1, self.config.recordings.n_channels))
        ndata = np.zeros(K)

        for i in range(0, nBatches):

            spt = spikeTrain[np.logical_and(
                spikeTrain[:, 0] >= i*batch_size, spikeTrain[:, 0] < (i+1)*batch_size)]

            # reading data
            if nBatches == 1:
                rec = self.load(0, batch_size)
                rec = self.addZeroBuffer(rec, BUFF, 2)
                spt[:, 0] = spt[:, 0] + BUFF

            elif i == 0:
                rec = self.load(i*batch_size, batch_size+BUFF)
                rec = self.addZeroBuffer(rec, BUFF, 0)
                spt[:, 0] = spt[:, 0] - i*batch_size + BUFF

            elif i < nBatches-1:
                rec = self.load(i*batch_size-BUFF, batch_size+2*BUFF)
                spt[:, 0] = spt[:, 0] - i*batch_size

            elif residual == 0:
                rec = self.load(i*batch_size-BUFF, batch_size+BUFF)
                rec = self.addZeroBuffer(rec, BUFF, 1)
                spt[:, 0] = spt[:, 0] - i*batch_size

            else:
                rec = self.load(i*batch_size-BUFF, residual+BUFF)
                rec = self.addZeroBuffer(rec, BUFF+(batch_size-residual), 1)
                spt[:, 0] = spt[:, 0] - i*batch_size

            # filter recording
            if self.config.doFilter == 1:
                rec = butterworth(rec, self.config.filter.low_pass_freq,
                                  self.config.filter.high_factor,
                                  self.config.filter.order,
                                  self.config.recordings.sampling_rate)

            # standardize recording
            if not hasattr(self, 'sd'):
                small_t = int(np.min((int(self.config.recordings.sampling_rate*5), rec.shape[0]))/2)
                mid_T = int(np.ceil(rec.shape[0]/2))
                rec_temp = rec[np.arange(mid_T-small_t, mid_T+small_t)]
                self.sd = np.median(np.abs(rec_temp), 0)/0.6745
            rec = np.divide(rec, self.sd)

            for j in range(spt.shape[0]):
                summedTemplatesBig[
                    spt[j, 1]] += rec[spt[j, 0]+np.arange(-R, R+1)]
                ndata[spt[j, 1]] += 1
        
        self.closeFile()

        return summedTemplatesBig/ndata[:, np.newaxis, np.newaxis]
