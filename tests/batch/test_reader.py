import itertools
from operator import itemgetter
import os
import tempfile

import numpy as np
import pytest

from yass.batch import RecordingsReader


@pytest.fixture
def path_to_long(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    pairs = itertools.product(range(10), range(10000))
    long_pairs = sorted(pairs, key=itemgetter(1))
    long_values = [float('{}.{}'.format(channel, obs))
                   for channel, obs in long_pairs]
    long_data = np.array(long_values).reshape((10000, 10))
    long_data.tofile(temp)
    temp.close()

    return path


@pytest.fixture
def path_to_wide(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    pairs = itertools.product(range(10), range(10000))
    wide_pairs = sorted(pairs, key=itemgetter(0))
    wide_values = [float('{}.{}'.format(channel, obs))
                   for channel, obs in wide_pairs]
    wide_data = np.array(wide_values).reshape((10, 10000))
    wide_data.tofile(temp)
    temp.close()

    return path


# TODO: re generate test data
def test_can_read_in_long_format(path_to_long, path_to_tests):
    indexer = RecordingsReader(path_to_long, n_channels=10,
                               data_format='long', dtype='float64',
                               loader='array')
    res = indexer[1000:1020, [1, 5]]
    # reader always returns data in wide format
    expected = np.load(os.path.join(path_to_tests,
                       'data/test_indexer/wide.npy')).T

    np.testing.assert_equal(res, expected)


def test_can_read_in_wide_format(path_to_wide, path_to_tests):
    indexer = RecordingsReader(path_to_wide, n_channels=10,
                               data_format='wide', dtype='float64',
                               loader='array')
    res = indexer[1000:1020, [1, 5]]
    expected = np.load(os.path.join(path_to_tests,
                       'data/test_indexer/wide.npy')).T

    np.testing.assert_equal(res, expected)
