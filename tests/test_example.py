"""
Testing example in README.md
"""
import os

import pytest
import yaml

from yass import command_line as cli
from yass import pipeline

from util import clean_tmp


@pytest.fixture
def path_to_output():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/tmp/output.csv')
    return path


def test_example_works_with_cli(path_to_config_sample, path_to_output):
    cli._run_pipeline(path_to_config_sample, path_to_output)
    clean_tmp()


def test_example_works_default_pipeline(path_to_config_sample, path_to_output):
    pipeline.run(path_to_config_sample)
    clean_tmp()


def test_example_works_pip_and_dict(path_to_config_sample, path_to_output):

    with open(path_to_config_sample) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg)
    clean_tmp()
