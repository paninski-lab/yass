"""
Utility functions
"""
import os
import functools
import inspect
import warnings
import collections
from copy import copy
from functools import wraps

import numpy as np

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


# TODO: this should probably be in explore.util
def vectorize_parameter(name):
    def vectorize(func):

        @wraps(func)
        def method_func_wrapper(self, *args, **kwargs):
            params = map_parameters_in_fn_call(args, kwargs, func)
            value = params.pop(name)
            return [func(self=self, **merge_dicts(params, {name: o}))
                    for o in value]

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            params = map_parameters_in_fn_call(args, kwargs, func)
            value = params.pop(name)
            return [func(**merge_dicts(params, {name: o})) for o in value]

        return (method_func_wrapper if 'self' in inspect.getargspec(func).args
                else func_wrapper)

    return vectorize


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
