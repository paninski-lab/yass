import os
import tempfile
import pytest
import numpy as np
from yass.batch import BinaryReader


@pytest.fixture
def long_data(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    # this generates data whose contiguous bytes contain the nth observation
    # of all channels
    data_long = np.array(np.arange(10000)).reshape(10, 1000).T
    data_long.tofile(path)

    return data_long, path


@pytest.fixture
def wide_data(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    # this generates data whose contiguous bytes contain observations from the
    # same channel
    data_wide = np.array(np.arange(10000)).reshape(10, 1000)
    data_wide.tofile(path)

    return data_wide, path


def test_can_read_data_in_wide_format(wide_data):
    data, path = wide_data
    wide = BinaryReader(path, 'int64', 10, 'wide')
    print(wide[1:10, 2:3], data[1:10, 2:3])
    np.testing.assert_equal(wide[1:10, 2:3], data[1:10, 2:3])


def test_can_read_data_in_long_format(long_data):
    data, path = long_data
    long = BinaryReader(path, 'int64', 10, 'long')
    print(long[1:10, 2:3], data[1:10, 2:3])
    np.testing.assert_equal(long[1:10, 2:3], data[1:10, 2:3])
