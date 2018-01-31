"""
Utility functions
"""
from . import __version__

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
