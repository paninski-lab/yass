import pytest
from yass.threshold.dimensionality_reduction import pca


@pytest.mark.xfail
def test_can_compute_pca(path_to_data, data_info):
    pca(path_to_data, data_info['dtype'], data_info['n_channels'],
        data_info['data_order'], max_memory='1GB')


def test_pca_is_not_run_if_files_already_exist():
    pass


def test_threshold_detector_is_not_run_if_files_already_exist():
    pass
