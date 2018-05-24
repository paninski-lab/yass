import logging
from logging import NullHandler
from yass.util import running_on_gpu

try:
    import tensorflow as tf
except ImportError:
    message = ('YASS requires tensorflow to work. It is not installed '
               'automatically to avoid overwriting existing installations.'
               ' See this for instructions: '
               'https://www.tensorflow.org/install/')
    raise ImportError(message)

from yass.config import Config

logging.getLogger(__name__).addHandler(NullHandler())

logger = logging.getLogger(__name__)

__version__ = '0.9'

CONFIG = None

GPU_ENABLED = running_on_gpu()

if GPU_ENABLED:
    logger.debug('Tensorflow GPU configuration detected')
else:
    logger.debug('No Tensorflow GPU configuration detected')

# reduce tensorflow logger verbosity, ignore DEBUG and INFO
tf.logging.set_verbosity(tf.logging.WARN)


def read_config():
    """
    Read YASS config
    """
    if CONFIG is None:
        raise ValueError('Configuration has not been set')

    return CONFIG


def set_config(config):
    """Set configuration settings

    Parameters
    ----------
    config: str or mapping (such as dictionary)
        Path to a yaml file or mapping object

    Notes
    -----
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

    if isinstance(config, str):
        CONFIG = Config.from_yaml(config)
    else:
        CONFIG = Config(config)

    logger.debug('CONFIG set to: %s', CONFIG._data)


def reset_config():
    global CONFIG
    CONFIG = None
