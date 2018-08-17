"""
Testing example in README.md
"""
import yaml
from yass import pipeline
from util import clean_tmp


def test_example_works_default_pipeline(path_to_config_sample):
    pipeline.run(path_to_config_sample)
    clean_tmp()


def test_example_works_default_pipeline_nn_detect(path_to_nnet_config):
    pipeline.run(path_to_nnet_config)
    clean_tmp()


def test_example_works_pip_and_dict(path_to_config_sample):

    with open(path_to_config_sample) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg)
    clean_tmp()
