from yass.cluster.diptest.diptest import dip
import numpy as np


def test_can_run_diptest():
    assert dip(np.random.rand(1000))
