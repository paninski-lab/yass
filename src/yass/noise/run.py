import os
import logging
import numpy as np

from yass import read_config
from yass.reader import READER
from yass.neuralnetwork.model_detector import Detect
from yass.noise.soft_assignment import SOFTNOISEASSIGNMENT

def run(template_fname,
        spike_train_fname,
        shifts_fname,
        output_directory,
        residual_fname,
        residual_dtype):

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    #
    fname_out = os.path.join(output_directory, 'soft_assignment.npy')
    if os.path.exists(fname_out):
        return fname_out

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # reader for residual
    reader_resid = READER(residual_fname,
                          residual_dtype,
                          CONFIG,
                          CONFIG.resources.n_sec_chunk_gpu_deconv)

    # load NN detector
    detector = Detect(CONFIG.neuralnetwork.detect.n_filters,
                      CONFIG.spike_size_nn,
                      CONFIG.channel_index)
    detector.load(CONFIG.neuralnetwork.detect.filename)
    detector = detector.cuda()

    # initialize soft assignment calculator
    threshold = CONFIG.deconvolution.threshold/0.1
    sna = SOFTNOISEASSIGNMENT(spike_train_fname, template_fname, shifts_fname,
                              reader_resid, detector, CONFIG.channel_index, threshold)

    # compuate soft assignment
    probs = sna.compute_soft_assignment()
    np.save(fname_out, probs)
    
    return fname_out
    