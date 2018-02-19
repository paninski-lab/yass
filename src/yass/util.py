"""
Utility functions
"""
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

try:
    # py3
    from inspect import signature
    from inspect import _empty
except ImportError:
    # py2
    from funcsigs import signature
    from funcsigs import _empty

from . import __version__

import logging
import datetime
import os
import functools
import inspect
import warnings
import collections
from copy import copy
from functools import wraps, reduce

import numpy as np
from dateutil.relativedelta import relativedelta

import yaml
from pkg_resources import resource_filename

string_types = (type(b''), type(u''))
logger = logging.getLogger(__name__)


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Source: https://stackoverflow.com/a/40301488
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "class {name} is deprecated: ({reason})"
            else:
                fmt1 = "function {name} is deprecated: {reason})"

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


def load_yaml_asset(path):
    """
    Load a yaml located in the assets folder
    by specifying a relative path to the assets/ folder
    """
    relative_path = os.path.join('assets', path)
    absolute_path = resource_filename('yass', relative_path)

    with open(absolute_path) as f:
        asset = yaml.load(f)

    return asset


def load_asset(path):
    """
    Load a file located in the assets folder
    by specifying a relative path to the assets/ folder
    """
    relative_path = os.path.join('assets', path)
    absolute_path = resource_filename('yass', relative_path)

    with open(absolute_path) as f:
        asset = f.read()

    return asset


def load_logging_config_file():
    content = load_yaml_asset(os.path.join('logger', 'config.yaml'))
    return content


def load_yaml(path):
    with open(path) as f:
        content = yaml.load(f)
    return content


def function_path(fn):
    """
    Returns the name of the function along with the module containing it:
    module.submodule.name
    """
    module = inspect.getmodule(fn).__name__
    return '{}.{}'.format(module, fn.__name__)


def map_parameters_in_fn_call(args, kwargs, func):
    """
    Based on function signature, parse args to to convert them to key-value
    pairs and merge them with kwargs

    Any parameter found in args that does not match the function signature
    is still passed.

    Missing parameters are filled with their default values
    """
    # Get missing parameters in kwargs to look for them in args
    args_spec = inspect.getargspec(func).args
    params_all = set(args_spec)
    params_missing = params_all - set(kwargs.keys())

    # Remove self parameter from params missing since it's not used
    if 'self' in args_spec:
        params_missing.remove('self')
        offset = 1
    else:
        offset = 0

    # Get indexes for those args
    idxs = [args_spec.index(name) for name in params_missing]

    # Parse args
    args_parsed = dict()

    for idx in idxs:
        key = args_spec[idx]

        try:
            value = args[idx-offset]
        except IndexError:
            pass
        else:
            args_parsed[key] = value

    parsed = copy(kwargs)
    parsed.update(args_parsed)

    # fill default values
    default = {k: v.default for k, v
               in signature(func).parameters.items()
               if v.default != _empty}

    to_add = set(default.keys()) - set(parsed.keys())

    default_to_add = {k: v for k, v in default.items() if k in to_add}
    parsed.update(default_to_add)

    return parsed


def _unwrap_mixed_iterator(mixed_iterator):
    """ [[1,2,3], 4 [5,6]] -> [1,2,3,4,5,6]
    """
    unwrapped = []
    for element in mixed_iterator:
        if _is_collection(element):
            unwrapped.extend(element)
        else:
            unwrapped.append(element)
    return unwrapped


def _is_collection(obj):
    """Determine wheter obj is an interable (excluding strings and mappings)
    """
    iterable = isinstance(obj, collections.Iterable)
    string = isinstance(obj, str)
    mapping = isinstance(obj, collections.Mapping)
    return iterable and not string and not mapping


def _wrap_in_list(obj):
    if _is_collection(obj):
        return obj
    else:
        return [obj]


def ensure_iterator(param):
    """Ensure a certain parameters is always an iterator
    """

    def _ensure_repeated(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            kwargs = map_parameters_in_fn_call(args, kwargs, func)
            kwargs[param] = _wrap_in_list(kwargs[param])
            return func(self, **kwargs)

        return wrapper

    return _ensure_repeated


def sample(data, percentage):
    """Sample array
    """
    return np.random.choice(data, size=int(percentage*len(data)),
                            replace=False)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def human_readable_time(seconds):
    """Return a human readable string for a given amount of seconds

    Notes
    -----
    Based on: https://stackoverflow.com/a/26165034
    """
    intervals = ['days', 'hours', 'minutes', 'seconds']
    delta = relativedelta(seconds=seconds)
    return (' '.join('{} {}'.format(getattr(delta, k), k) for k in intervals
            if getattr(delta, k)))


def save_metadata(path):
    timestamp = datetime.datetime.now().strftime('%c')
    metadata = dict(version=__version__, timestamp=timestamp)

    with open(path, 'w') as f:
        yaml.dump(metadata, f)


def change_extension(path, new_extension):
    elements = path.split('.')
    elements[-1] = new_extension
    return reduce(lambda x, y: x+'.'+y, elements)


def requires(condition, message):
    """
    Utilify function to raise exception when an optional requirement is
    not installed
    """

    def _requires(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):

            if not condition:
                raise ImportError(message)

            return func(self, *args, **kwargs)

        return wrapper

    return _requires


def save_numpy_object(obj, output_path, if_file_exists, name='file'):
    """Utility to save a numpy object

    Parameters
    ----------
    obj: numpy.ndarray
        Object to save

    output_path: str
        Where to save the file

    if_file_exists: str, optional
        One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces the
        file if it exists, if 'abort' if raise a ValueError exception if
        the file exists, if 'skip' if skips the operation if the file
        exists

    name: str, optional
        Name (just used for logging messages)
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)

    if output_path.exists() and if_file_exists == 'abort':
        raise ValueError('{} already exists'.format(output_path))
    elif output_path.exists() and if_file_exists == 'skip':
        logger.info('{} already exists, skipping...'.format(output_path))
    else:
        np.save(str(output_path), obj)
        logger.info('Saved {} in {}'.format(name, output_path))


class LoadFile(object):

    def __init__(self, param, new_extension=None):
        self.param = param
        self.new_extension = new_extension

    def __call__(self, _kwargs):

        if self.new_extension is not None:
            filename = change_extension(_kwargs[self.param],
                                        self.new_extension)
        else:
            filename = _kwargs[self.param]

        path = Path(_kwargs['output_path'], filename)

        if path.suffix == '.npy':
            return np.load(str(path))
        elif path.suffix == '.yaml':
            return load_yaml(str(path))
        else:
            raise ValueError('Do not know how to load file with extension '
                             '{}'.format(path.suffix))


class ExpandPath(object):

    def __init__(self, param):
        self.param = param

    def __call__(self, _kwargs):
        return Path(_kwargs['output_path'], _kwargs[self.param])


def check_for_files(parameters, if_skip):
    """
    Decorator used to change the behavior of functions that write to disk

    Parameters
    ----------
    parameters: list
        List of strings with the parameters containing the filenames to
        check for, the function should also contain a parameter named
        output_path. The path to the file is created relative to the
        output_path

    if_skip: list
        List with values to return in case `skip` is selected, can optionally
        use the `LoadFile` (relative to `output_path` decorator to load some
        files instead of returning the values
    """
    def _check_for_files(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            _kwargs = map_parameters_in_fn_call(args, kwargs, func)

            # if not output_path exists, just run the function
            if _kwargs.get('output_path') is None:
                logger.debug('No output path was passed, running the '
                             'function without checking for files...')
                return func(*args, **kwargs)

            if_file_exists = _kwargs['if_file_exists']

            paths = [Path(_kwargs['output_path']) / _kwargs[p]
                     for p in parameters]
            exists = [p.exists() for p in paths]

            if (if_file_exists == 'overwrite' or
               if_file_exists == 'abort' and not any(exists)
               or if_file_exists == 'skip' and not all(exists)):
                logger.debug('Running the function...')
                return func(*args, **kwargs)

            elif if_file_exists == 'abort' and any(exists):
                conflict = [p for p, e in zip(paths, exists) if e]
                message = reduce(lambda x, y: str(x)+', '+str(y), conflict)

                raise ValueError('if_file_exists was set to abort, the '
                                 'program halted since the following files '
                                 'already exist: {}'.format(message))
            elif if_file_exists == 'skip' and all(exists):

                def expand(element, _kwargs):
                    if not isinstance(element, str):
                        return element(_kwargs)
                    else:
                        return element

                logger.info('Skipped {} execution. All necessary files exist'
                            ', loading them...'.format(function_path(func)))

                res = [expand(e, _kwargs) for e in if_skip]

                return res[0] if len(res) == 1 else res

            else:
                raise ValueError('Invalid value for if_file_exists {}'
                                 'must be one of overwrite, abort or skip'
                                 .format(if_file_exists))

        return wrapper

    return _check_for_files
