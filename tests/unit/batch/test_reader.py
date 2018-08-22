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


expected = np.array(
    [[1.1, 1.1001, 1.1002, 1.1003, 1.1004, 1.1005, 1.1006, 1.1007,
      1.1008, 1.1009, 1.101, 1.1011, 1.1012, 1.1013, 1.1014, 1.1015,
      1.1016, 1.1017, 1.1018, 1.1019],
     [5.1, 5.1001, 5.1002, 5.1003, 5.1004, 5.1005, 5.1006, 5.1007,
      5.1008, 5.1009, 5.101, 5.1011, 5.1012, 5.1013, 5.1014, 5.1015,
      5.1016, 5.1017, 5.1018, 5.1019]]).T


def test_can_read_in_long_format(path_to_long, path_to_tests):
    indexer = RecordingsReader(path_to_long, n_channels=10,
                               data_order='samples', dtype='float64',
                               loader='array')
    res = indexer[1000:1020, [1, 5]]

    np.testing.assert_equal(res, expected)


def test_can_read_in_wide_format(path_to_wide, path_to_tests):
    indexer = RecordingsReader(path_to_wide, n_channels=10,
                               data_order='channels', dtype='float64',
                               loader='array')
    res = indexer[1000:1020, [1, 5]]

    np.testing.assert_equal(res, expected)
