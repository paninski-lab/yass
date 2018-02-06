import logging
from logging import NullHandler

import tensorflow as tf

from .config import Config

logging.getLogger(__name__).addHandler(NullHandler())

__version__ = '0.6'

CONFIG = None

# reduce tensorflow logger verbosity, ignore DEBUG and INFO
tf.logging.set_verbosity(tf.logging.WARN)


def read_config():
    if CONFIG is None:
        raise ValueError('Configuration has not been set')

    return CONFIG


def set_config(path_to_config):
    """Set configuration settings

    This function lets the user configure global settings used
    trhoughout the execution of the pipeline. The variables set here
    determine the behavior of some steps. For that reason once, the values
    are set, they cannot be changed, to avoid changing global configuration
    changes at runtime
    """
    # this variable is accesible in the preprocessa and process blocks to avoid
    # having to pass the configuration over and over, to prevent global state
    # issues, the config cannot be edited after loaded
    global CONFIG
    CONFIG = Config.from_yaml(path_to_config)


def reset_config():
    global CONFIG
    CONFIG = None
