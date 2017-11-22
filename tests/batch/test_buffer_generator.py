from yass.batch.reader import BufferGenerator


def test_updates_slice_when_enough_obs_start_end():
    bg = BufferGenerator(n_observations=100, data_format='long',
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
    bg = BufferGenerator(n_observations=100, data_format='long',
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
    bg = BufferGenerator(n_observations=100, data_format='long',
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
    bg = BufferGenerator(n_observations=100, data_format='long',
                         buffer_size=10)

    obs_slice = slice(5, 98, None)
    ch_slice = slice(None, None, None)
    key = (obs_slice, ch_slice)

    ((obs_slice_new, ch_slice_new),
     (buff_start, buff_end)) = bg.update_key_with_buffer(key)

    assert obs_slice_new == slice(0, 100, None)
    assert ch_slice_new == ch_slice
    assert buff_start == 5 and buff_end == 8
