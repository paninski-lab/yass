"""
Testing example in README.md
"""
import os

import pytest

from yass import command_line as cli


@pytest.fixture
def path_to_config_sample():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_sample.yaml')
    return path


@pytest.fixture
def path_to_output():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/tmp/output.csv')
    return path


def test_example_works(path_to_config_sample, path_to_output):
    cli._run_pipeline(path_to_config_sample, path_to_output)
