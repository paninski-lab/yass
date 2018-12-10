"""
Testing pipeline.run
"""
import yaml
from yass import pipeline


def test_works_with_nnet_config(patch_triage_network, path_to_config,
                                make_tmp_folder):
    pipeline.run(path_to_config, output_dir=make_tmp_folder)


def test_works_with_threshold_config(path_to_config_threshold,
                                     make_tmp_folder):
    pipeline.run(path_to_config_threshold, output_dir=make_tmp_folder)


def test_works_with_sample_config_passing_dict(path_to_config,
                                               make_tmp_folder):
    with open(path_to_config) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg, output_dir=make_tmp_folder)
