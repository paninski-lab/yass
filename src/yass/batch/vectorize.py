"""
Vectorizing function execution, a lot of functions in YASS need to be executed
on different inputs over and over again, to simplify that workflow, some
utility functions are provided. Implementation is naive now (for loop) but
parallel execution will be implemented soon
"""
from functools import wraps
import inspect

from ..util import merge_dicts, map_parameters_in_fn_call


# TODO: this should probably be in explore.util
# TODO: maybe support non-strict mode so the function works with iterable
# and non-iterable, strict mode should only work with iterable (as it is now)
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
