from yass.explore import RecordingExplorer


def test_returns_empty_if_cannot_get_complete_wf(path_to_standarized_data):
    e = RecordingExplorer(path_to_standarized_data,
                          spike_size=15, dtype='float32', n_channels=10,
                          data_order='channels', loader='array')

    assert len(e.read_waveform(time=0)) == 0


def test_can_read_waveform(path_to_standarized_data):
    spike_size = 15

    e = RecordingExplorer(path_to_standarized_data,
                          spike_size=spike_size, dtype='float32',
                          n_channels=10, data_order='channels', loader='array')

    assert len(e.read_waveform(time=100)) == 2 * spike_size + 1
