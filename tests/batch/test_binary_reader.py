import os
import tempfile
import pytest
import numpy as np
from yass.batch import BinaryReader


@pytest.fixture
def data_C(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    # row-major order
    data = np.array(np.arange(1000)).reshape(10, 100)
    data.tofile(path)

    return data, path


@pytest.fixture
def data_F(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    # column-major order
    data = np.array(np.arange(1000)).reshape(10, 100).T
    data.tofile(path)

    return data, path


def test_can_read_data_in_C_order(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')
    np.testing.assert_equal(c[1:10, 2:3], data[1:10, 2:3])


def test_can_read_data_in_F_order(data_F):
    data, path = data_F
    f = BinaryReader(path, data.dtype, data.shape, 'F')
    np.testing.assert_equal(f[1:10, 2:3], data[1:10, 2:3])
