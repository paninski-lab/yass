import os
import argparse

import numpy as np

import yass
from yass.preprocessing import Preprocessor
from yass.neuralnet import NeuralNetDetector
from yass.mainprocess import Mainprocessor
from yass.deconvolution import Deconvolution


def main():
    parser = argparse.ArgumentParser(description='Run YASS.')
    parser.add_argument('config', type=str,
                        help='Path to configuration file')

    args = parser.parse_args()

    cfg = yass.Config.from_yaml(args.config)

    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()

    mp = Mainprocessor(cfg, score, clr_idx, spt)
    spikeTrain_clear, spt_left = mp.mainProcess()

    dc = Deconvolution(cfg, np.transpose(mp.templates,[1,0,2]), spt_left)
    spikeTrain_col = dc.fullMPMU()

    spikeTrain = np.concatenate((spikeTrain_col, spikeTrain_clear))
    idx_sort = np.argsort(spikeTrain[:, 0])
    spikeTrain = spikeTrain[idx_sort]
    
    idx_keep = np.zeros(spikeTrain.shape[0],'bool')
    for k in range(mp.templates.shape[2]):
        idx_c = np.where(spikeTrain[:,1] == k)[0]
        idx_keep[idx_c[np.concatenate(([True], np.diff(spikeTrain[idx_c,0]) > 1))]] = 1
    spikeTrain = spikeTrain[idx_keep]

    np.savetxt(os.path.join(cfg.root, cfg.spikeTrainName),
               spikeTrain, fmt='%i, %i')
