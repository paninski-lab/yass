"""
YASS init file, it contains setup for the package.

Summary: the only way to set global configuration is to create a first session
and pass the desired config, also, make sure that list_devices is not executed
before creating the first session. So do not create sessions here or use
list_devices here, since it will break that feature
"""
import logging
from logging import NullHandler

from yass.config import Config

logging.getLogger(__name__).addHandler(NullHandler())

logger = logging.getLogger(__name__)

__version__ = '2.0'

CONFIG = None

def read_config():
    """
    Read YASS config
    """
    if CONFIG is None:
        raise ValueError('Configuration has not been set')

    return CONFIG


def set_config(config, output_directory):
    """Set configuration settings

    Parameters
    ----------
    config: str or mapping (such as dictionary)
        Path to a yaml file or mapping object

    output_directory: str of pathlib.Path
        output directory for the project, this makes
        Config.output_directory return
        onfig.data.root_folder / output_directory, which is a common path
        used through the pipeline. If the path is absolute, it is not modified

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
        CONFIG = Config.from_yaml(config, output_directory)
    else:
        CONFIG = Config(config, output_directory)

    logger.debug('CONFIG set to: %s', CONFIG._data)

    return CONFIG


def reset_config():
    global CONFIG
    CONFIG = None
