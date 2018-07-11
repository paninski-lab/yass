"""
YASS init file, it contains setup for the package.

One of the functions provided here is the ability to setup global config for
tensorflow sessions, there are some oddities to this documented here:

https://github.com/tensorflow/tensorflow/issues/8021

Summary: the only way to set global configuration is to create a first session
and pass the desired config, also, make sure that list_devices is not executed
before creating the first session. So do not create sessions here or use
list_devices here, since it will break that feature
"""
import logging
from logging import NullHandler


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

__version__ = '0.10dev'

CONFIG = None


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


def set_tensorflow_config(config):
    """Mock tf.Session to always return a session with the given config,
    make sure you run this before any other yass/tensorflow code since this
    needs to run first to work

    Examples
    --------

    .. code-block:: python

    import yass

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    yass.set_tensorflow_config(config=config)
    """
    # create first session and pass config, this will set the config for all
    # sessions (that's how tensorflow works for now)
    sess = tf.Session(config=config)
    sess.close()
