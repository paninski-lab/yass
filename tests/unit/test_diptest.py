from yass.cluster.diptest.diptest import dip
import numpy as np


def can_run_diptest():
    assert dip(np.random.rand(1000))
