"""
Testing example in README.md
"""
import yaml
from yass import pipeline


def test_example_works_default_pipeline(path_to_config_sample,
                                        make_tmp_folder):
    pipeline.run(path_to_config_sample, output_dir=make_tmp_folder)


def test_example_works_default_pipeline_nn_detect(path_to_nnet_config,
                                                  make_tmp_folder):
    pipeline.run(path_to_nnet_config, output_dir=make_tmp_folder)


def test_example_works_pip_and_dict(path_to_config_sample, make_tmp_folder):

    with open(path_to_config_sample) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg, output_dir=make_tmp_folder)
