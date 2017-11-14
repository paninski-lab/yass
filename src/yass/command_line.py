import os
import argparse

import numpy as np
import logging

import yass
from yass.preprocessing import Preprocessor
from yass.neuralnet import NeuralNetDetector
from yass.mainprocess import Mainprocessor
from yass.deconvolute import Deconvolution

#from . import set_config
#from . import preprocess
#from . import process
#from . import deconvolute
#from . import read_config

def main():
    # removed reference to this file in config, it is not necessary
    # TODO: let the user specify the name through an option
    output_file = 'spike_train.csv'

    # configure logging module to get useful information
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Run YASS.')
    parser.add_argument('config', type=str,
                        help='Path to configuration file')

    args = parser.parse_args()

    cfg = yass.Config.from_yaml(args.config)

    pp = Preprocessor(cfg)
    score, spike_index_clear, spike_index_collision = pp.process()

    mp = Mainprocessor(cfg, score, spike_index_clear, spike_index_collision)
    spikeTrain_clear, spike_index_collision = mp.mainProcess()

    dc = Deconvolution(cfg, np.transpose(mp.templates,[1,0,2]), spike_index_collision)
    spikeTrain_col = dc.fullMPMU()

    spikeTrain = np.concatenate((spikeTrain_col, spikeTrain_clear))
    idx_sort = np.argsort(spikeTrain[:, 0])
    spikeTrain = spikeTrain[idx_sort]
    
    idx_keep = np.zeros(spikeTrain.shape[0],'bool')
    for k in range(mp.templates.shape[2]):
        idx_c = np.where(spikeTrain[:,1] == k)[0]
        idx_keep[idx_c[np.concatenate(([True], np.diff(spikeTrain[idx_c,0]) > 1))]] = 1
    spikeTrain = spikeTrain[idx_keep]

    path_to_file = os.path.join(cfg.root_folder, output_file)

    np.savetxt(path_to_file, spikeTrain, fmt='%i, %i')
    print('Done, spike train saved in: {}'.format(path_to_file))

    
    # set yass configuration parameters
    #set_config(args.config)
    #CONFIG = read_config()

    # run preprocessor
    #score, spike_index_clear, spike_index_collision = preprocess.run()

    # run processor
    #spike_train_clear, templates, spike_index_collision = process.run(score,
    #    spike_index_clear, spike_index_collision)

    # run deconvolution
    #spike_train = deconvolute.run(spike_train_clear, templates,
    #    spike_index_collision)

    # path_to_file = os.path.join(cfg.root_folder, output_file)
    #np.savetxt(path_to_file, spike_train, fmt='%i, %i')
    #logger.info('Done, spike train saved in: {}'.format(path_to_file))