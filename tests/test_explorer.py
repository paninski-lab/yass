import os
import pytest
from yass.explore import RecordingExplorer


def test_can_read_waveform(path_to_data_folder):
    e = RecordingExplorer(os.path.join(path_to_data_folder, 'filtered.bin'),
                          spike_size=15, dtype='float32', n_channels=10,
                          data_format='wide', loader='array')

    with pytest.raises(ValueError):
        e.read_waveform(time=0)


def test_error_raised_if_cannot_read_complete_waveform(path_to_data_folder):
    spike_size = 15

    e = RecordingExplorer(os.path.join(path_to_data_folder, 'filtered.bin'),
                          spike_size=spike_size, dtype='float32',
                          n_channels=10, data_format='wide', loader='array')

    assert len(e.read_waveform(time=100)) == 2 * spike_size + 1
