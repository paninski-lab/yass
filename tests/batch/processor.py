import logging
import os.path
import pytest
import numpy as np

from yass.batch import BatchProcessor


logging.basicConfig(level=logging.INFO)

dtype = 'int16'
dsize = np.dtype(dtype).itemsize
obs_total = 10000
n_channels = 10
obs_size = dsize * n_channels


# TODO: fix this file

@pytest.fixture
def path():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel.bin')
    return path


@pytest.fixture
def bp():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel.bin')
    bp = BatchProcessor(path, dtype=dtype, n_channels=7, max_memory=14*5,
                        buffer_size=2)
    return bp


def test_max_memory_cannot_be_smaller_than_one_observation_per_channel(path):
    with pytest.raises(ValueError):
        BatchProcessor(path, dtype=dtype, n_channels=7, max_memory=13,
                       buffer_size=1)


def test_buffer_cannot_be_larger_than_batch_size(path):
    with pytest.raises(ValueError):
        BatchProcessor(path, dtype=dtype, n_channels=7, max_memory=14*2,
                       buffer_size=3)


def test_produces_the_right_number_of_batches(path):
    obs_per_batch = 5

    bp = BatchProcessor(path, dtype=dtype, n_channels=n_channels,
                        max_memory=obs_size*obs_per_batch, buffer_size=2)
    assert len([batch for batch in bp]) == obs_total/obs_per_batch


def test_produces_the_right_number_of_batches_with_residual(path):
    # this will generate ceil(n_observations/obs_per_batch) batches
    # ceil(10,000/3) = ceil(3,333.3) = 3,334
    obs_per_batch = 3

    bp = BatchProcessor(path, dtype=dtype, n_channels=n_channels,
                        max_memory=obs_size*obs_per_batch, buffer_size=2)
    batches = [batch for batch in bp]
    assert len(batches) == 3334


def test_produces_the_right_shape_for_batches_with_residual(path):
    obs_per_batch = 3

    # max memory: obs_size * obs_per_batch = (2 * 10) * 3
    # this means we can load at most 3 complete observations
    # if we add a buffer of size 2 (each sides) we will load
    # 3 + 2 + 2 = 7 observations on each batch. Last bacth
    # with residual is also completed to have 7 observations

    bp = BatchProcessor(path, dtype=dtype, n_channels=n_channels,
                        max_memory=obs_size * obs_per_batch, buffer_size=2)

    correct_size = [(7, 10) == b.shape for b in bp]
    assert all(correct_size)
