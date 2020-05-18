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
    data = np.array(np.arange(1000)).reshape(10, 100)
    data.T.tofile(path)

    return data, path


def test_can_read_data_in_C_order(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')
    np.testing.assert_equal(c[1:10, 2:3], data[1:10, 2:3])


def test_can_read_data_in_F_order(data_F):
    data, path = data_F
    f = BinaryReader(path, data.dtype, data.shape, 'F')
    np.testing.assert_equal(f[1:10, 2:3], data[1:10, 2:3])


def test_can_read_data_in_C_order_empty_start(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')
    np.testing.assert_equal(c[:10, :3], data[:10, :3])


def test_can_read_data_in_F_order_empty_start(data_F):
    data, path = data_F
    f = BinaryReader(path, data.dtype, data.shape, 'F')
    np.testing.assert_equal(f[:10, :3], data[:10, :3])


def test_can_read_data_in_C_order_empty_end(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')
    np.testing.assert_equal(c[:10, :3], data[:10, :3])


def test_can_read_data_in_F_order_empty_end(data_F):
    data, path = data_F
    f = BinaryReader(path, data.dtype, data.shape, 'F')
    np.testing.assert_equal(f[3:, 8:], data[3:, 8:])


def test_can_read_rows_in_C_with_iterable(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')
    np.testing.assert_equal(c[[0, 1, 2], :3], data[[0, 1, 2], :3])


def test_can_read_columns_in_F_with_iterable(data_F):
    data, path = data_F
    f = BinaryReader(path, data.dtype, data.shape, 'F')
    np.testing.assert_equal(f[3:, [6, 7, 8]], data[3:, [6, 7, 8]])


def test_error_raised_when_trying_to_slice_with_one_slice(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')

    with pytest.raises(ValueError):
        c[:]


def test_error_raised_when_trying_to_slice_with_iterators_in_rows_F(data_F):
    data, path = data_F
    c = BinaryReader(path, data.dtype, data.shape, 'F')

    with pytest.raises(NotImplementedError):
        c[[0, 1, 2], :]


def test_error_raised_when_trying_to_slice_with_iterators_in_cols_C(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')

    with pytest.raises(NotImplementedError):
        c[:, [0, 1, 2]]


def test_error_raised_when_trying_to_slice_with_three_slices(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')

    with pytest.raises(ValueError):
        c[:, :, :]


def test_error_1d_array_when_indexing_with_ints_in_cols(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')

    assert c[:10, 0].ndim == 1


def test_error_1d_array_when_indexing_with_ints_in_rows(data_C):
    data, path = data_C
    c = BinaryReader(path, data.dtype, data.shape, 'C')

    assert c[0, :10].ndim == 1
