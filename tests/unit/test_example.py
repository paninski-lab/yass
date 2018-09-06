"""
Testing example in README.md
"""
import numpy as np
import yaml
from yass import pipeline
import pytest


def test_example_works_default_pipeline(path_to_config_sample,
                                        make_tmp_folder):
    np.random.seed(0)
    pipeline.run(path_to_config_sample, output_dir=make_tmp_folder)


@pytest.mark.xfail
def test_example_works_default_pipeline_nn_detect(path_to_nnet_config,
                                                  make_tmp_folder):
    np.random.seed(0)
    with open(path_to_nnet_config) as f:
        CONFIG = yaml.load(f)

    pipeline.run(CONFIG, output_dir=make_tmp_folder)


def test_example_works_pip_and_dict(path_to_config_sample, make_tmp_folder):
    np.random.seed(0)

    with open(path_to_config_sample) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg, output_dir=make_tmp_folder)
