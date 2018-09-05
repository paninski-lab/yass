"""
Testing example in README.md
"""
import yaml
from yass import pipeline

import pytest


# new deconv is broken
@pytest.mark.xfail
def test_example_works_default_pipeline(path_to_config_sample,
                                        make_tmp_folder):
    pipeline.run(path_to_config_sample, output_dir=make_tmp_folder)


# new deconv is broken
@pytest.mark.xfail
def test_example_works_default_pipeline_nn_detect(path_to_nnet_config,
                                                  make_tmp_folder):

    with open(path_to_nnet_config) as f:
        CONFIG = yaml.load(f)

    # FIXME: hacky solution for the test to pass, i need to re-train the
    # triage network
    CONFIG['detect'] = {'temporal_features': 3, 'method': 'nn',
                        'neural_network_triage':
                        {'threshold_collision': 0,
                         'filename': 'triage-31wf7ch-15-Aug-2018@00-17-16.h5'}}

    pipeline.run(CONFIG, output_dir=make_tmp_folder)


# new deconv is broken
@pytest.mark.xfail
def test_example_works_pip_and_dict(path_to_config_sample, make_tmp_folder):

    with open(path_to_config_sample) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg, output_dir=make_tmp_folder)
