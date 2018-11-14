import numpy as np

from yass.augment import noise
from yass.batch import RecordingsReader


def test_can_kill_signal(path_to_standardized_data, path_to_threshold_config):
    recordings = RecordingsReader(path_to_standardized_data,
                                  loader='array')._data

    noise.kill_signal(recordings,
                      threshold=3.0,
                      window_size=10)


def test_can_estimate_temporal_and_spatial_sig(path_to_standardized_data,
                                               path_to_threshold_config):

    recordings = RecordingsReader(path_to_standardized_data,
                                  loader='array')._data

    (spatial_SIG,
        temporal_SIG) = noise.noise_cov(recordings,
                                        temporal_size=40,
                                        sample_size=1000,
                                        threshold=3.0,
                                        window_size=10)

    # check no nans
    assert (~np.isnan(spatial_SIG)).all()
    assert (~np.isnan(temporal_SIG)).all()
