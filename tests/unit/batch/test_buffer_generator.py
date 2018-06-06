import numpy as np

from yass.batch.buffer import BufferGenerator


def test_updates_slice_when_enough_obs_start_end():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    obs_slice = slice(10, 20, None)
    ch_slice = slice(None, None, None)
    key = (obs_slice, ch_slice)

    ((obs_slice_new, ch_slice_new),
     (buff_start, buff_end)) = bg.update_key_with_buffer(key)

    assert obs_slice_new == slice(0, 30, None)
    assert ch_slice_new == ch_slice
    assert buff_start == 0 and buff_end == 0


def test_updates_slice_when_not_enough_obs_start():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    obs_slice = slice(5, 20, None)
    ch_slice = slice(None, None, None)
    key = (obs_slice, ch_slice)

    ((obs_slice_new, ch_slice_new),
     (buff_start, buff_end)) = bg.update_key_with_buffer(key)

    assert obs_slice_new == slice(0, 30, None)
    assert ch_slice_new == ch_slice
    assert buff_start == 5 and buff_end == 0


def test_updates_slice_when_not_enough_obs_end():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    obs_slice = slice(89, 99, None)
    ch_slice = slice(None, None, None)
    key = (obs_slice, ch_slice)

    ((obs_slice_new, ch_slice_new),
     (buff_start, buff_end)) = bg.update_key_with_buffer(key)

    assert obs_slice_new == slice(79, 100, None)
    assert ch_slice_new == ch_slice
    assert buff_start == 0 and buff_end == 9


def test_updates_slice_when_not_enough_obs_start_end():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    obs_slice = slice(5, 98, None)
    ch_slice = slice(None, None, None)
    key = (obs_slice, ch_slice)

    ((obs_slice_new, ch_slice_new),
     (buff_start, buff_end)) = bg.update_key_with_buffer(key)

    assert obs_slice_new == slice(0, 100, None)
    assert ch_slice_new == ch_slice
    assert buff_start == 5 and buff_end == 8


def test_does_not_add_buffer_when_enough_obs_long_data():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    d = np.ones((100, 10))

    index = (slice(20, 30, None), slice(None))

    (index_new, (buff_start, buff_end)) = bg.update_key_with_buffer(index)

    subset = bg.add_buffer(d[index_new], buff_start, buff_end)

    assert subset.shape == (30, 10)
    assert subset.sum() == 300


def test_adds_start_buffer_long():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    d = np.ones((100, 10))

    index = (slice(0, 10, None), slice(None))

    (index_new, (buff_start, buff_end)) = bg.update_key_with_buffer(index)

    subset = bg.add_buffer(d[index_new], buff_start, buff_end)

    assert subset.shape == (30, 10)
    assert subset.sum() == 200
    assert subset[:10, :].sum() == 0
    assert subset[10:, :].sum() == 200


def test_adds_end_buffer_long():
    bg = BufferGenerator(n_observations=100, data_shape='long',
                         buffer_size=10)

    d = np.ones((100, 10))

    index = (slice(90, 100, None), slice(None))

    (index_new, (buff_start, buff_end)) = bg.update_key_with_buffer(index)

    subset = bg.add_buffer(d[index_new], buff_start, buff_end)

    assert subset.shape == (30, 10)
    assert subset.sum() == 200
    assert subset[:20, :].sum() == 200
    assert subset[20:, :].sum() == 0


def test_does_not_add_buffer_when_enough_obs_wide_data():
    bg = BufferGenerator(n_observations=100, data_shape='wide',
                         buffer_size=10)

    d = np.ones((10, 100))

    # index is in observations, channels format
    index = (slice(20, 30, None), slice(None))

    (index_new, (buff_start, buff_end)) = bg.update_key_with_buffer(index)

    # reverse to match 'wide' data
    index_new = index_new[::-1]

    subset = bg.add_buffer(d[index_new], buff_start, buff_end)

    assert subset.shape == (10, 30)
    assert subset.sum() == 300


def test_adds_start_buffer_wide():
    bg = BufferGenerator(n_observations=100, data_shape='wide',
                         buffer_size=10)

    d = np.ones((10, 100))

    index = (slice(0, 10, None), slice(None))

    (index_new, (buff_start, buff_end)) = bg.update_key_with_buffer(index)

    # reverse to match 'wide' data
    index_new = index_new[::-1]

    subset = bg.add_buffer(d[index_new], buff_start, buff_end)

    assert subset.shape == (10, 30)
    assert subset.sum() == 200
    assert subset[:, :10].sum() == 0
    assert subset[:, 10:].sum() == 200


def test_adds_end_buffer_wide():
    bg = BufferGenerator(n_observations=100, data_shape='wide',
                         buffer_size=10)

    d = np.ones((10, 100))

    index = (slice(90, 100, None), slice(None))

    (index_new, (buff_start, buff_end)) = bg.update_key_with_buffer(index)

    # reverse to match 'wide' data
    index_new = index_new[::-1]

    subset = bg.add_buffer(d[index_new], buff_start, buff_end)

    assert subset.shape == (10, 30)
    assert subset.sum() == 200
    assert subset[:, :20].sum() == 200
    assert subset[:, 20:].sum() == 0
