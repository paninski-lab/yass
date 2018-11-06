"""
This module contains functions for preprocessing data (filtering and
standarization), the batch/ folder contains code to preprocess data using
the YASS batch processor (:mod:`yass.batch`), experimental/ has a faster
(but unstable implementation). The bottleneck in the stable implementation
is due to the BatchProcessor, a re-implementation of the binary reader
used there should fix the performance issues.
"""

from yass.preprocess.run import run

__all__ = ['run']
