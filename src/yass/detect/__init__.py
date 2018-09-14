"""
The :mod:`yass.detect` module implements spike detectors, nnet.py contains a
Neural Network detector and threshold.py contains the threshold detector,
both of them use the YASS batch processor (:mod:`yass.batch`)
"""

from yass.detect.run import run


__all__ = ['run']
