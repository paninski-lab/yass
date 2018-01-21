import os
import argparse

import numpy as np
import logging

import yass
from yass.preprocessing import Preprocessor
from yass.mainprocess import Mainprocessor
from yass.deconvolute import Deconvolution


def _run_pipeline(config, output_file):
    """Run the entire pipeline given a config and output file
    """

    # configure logging module to get useful information
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cfg = yass.Config.from_yaml(config)

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

    path_to_file = os.path.join(cfg.data.root_folder, 'tmp/', output_file)

    np.save(path_to_file, spikeTrain)
    print('Done, spike train saved in: {}'.format(path_to_file))


def main():
    """Entry point for the command line utility
    """
    parser = argparse.ArgumentParser(description='Run YASS.')
    parser.add_argument('config', type=str,
                        help='Path to configuration file')

    args = parser.parse_args()

    return _run_pipeline(args.config, 'spike_train.npy')
