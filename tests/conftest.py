import os

import pytest


@pytest.fixture(scope='session')
def path_to_tests():
    path = os.path.dirname(os.path.realpath(__file__))
    return path
