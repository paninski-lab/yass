"""
Testing pipeline.run
"""
import yaml
from yass import pipeline
from yass.detect import nnet, threshold


def test_works_with_nnet_config(patch_triage_network, path_to_config,
                                make_tmp_folder):
    pipeline.run(path_to_config, output_dir=make_tmp_folder,
                 detector=nnet.run)


def test_works_with_threshold_config(path_to_config,
                                     make_tmp_folder):
    pipeline.run(path_to_config, output_dir=make_tmp_folder,
                 detector=threshold.run)


def test_works_with_sample_config_passing_dict(path_to_config,
                                               make_tmp_folder):
    from yass import pipeline

    with open(path_to_config) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg, output_dir=make_tmp_folder)
