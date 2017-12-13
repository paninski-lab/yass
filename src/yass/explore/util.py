"""
Various utility functions
"""
import collections
import inspect
from copy import copy
from functools import wraps

import numpy as np


def _map_parameters_in_fn_call(args, kwargs, func):
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
    if 'self' in params_missing:
        params_missing.remove('self')
    else:
        raise ValueError(("self was not found in the fn signature, "
                          " this is intended to be used with instance "
                          "methods only"))

    # Get indexes for those args
    # sum one to the index to account for the self parameter
    idxs = [args_spec.index(name) for name in params_missing]

    # Parse args
    args_parsed = dict()
    for idx in idxs:
        key = args_spec[idx]

        try:
            value = args[idx-1]
        except IndexError:
            pass
        else:
            args_parsed[key] = value

    parsed = copy(kwargs)
    parsed.update(args_parsed)

    return parsed


def vectorize_parameter(name):
    def vectorize(func):

        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            params = _map_parameters_in_fn_call(args, kwargs, func)
            value = params.pop(name)
            print(params, value)
            if _is_collection(value):
                return [func(o, **params) for o in value]
            else:
                return func(value, **params)

        return func_wrapper

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
            kwargs = _map_parameters_in_fn_call(args, kwargs, func)
            kwargs[param] = _wrap_in_list(kwargs[param])
            return func(self, **kwargs)

        return wrapper

    return _ensure_repeated


def sample(data, percentage):
    """Sample array
    """
    return np.random.choice(data, size=int(percentage*len(data)),
                            replace=False)
