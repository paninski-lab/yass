"""
Vectorizing function execution, a lot of functions in YASS need to be executed
on different inputs over and over again, to simplify that workflow, some
utility functions are provided. Implementation is naive now (for loop) but
parallel execution will be implemented soon
"""
from copy import copy
from functools import wraps
import inspect

from ..util import merge_dicts, map_parameters_in_fn_call


# TODO: this should probably be in explore.util
# TODO: maybe support non-strict mode so the function works with iterable
# and non-iterable, strict mode should only work with iterable (as it is now)
def vectorize_parameter(name):
    """Vectorize function

    Parameters
    ----------
    name: str
        Parameter to vectorize

    Returns
    -------
    function
        A function whose parameter named 'name' is vectorized

    Examples
    --------

    .. literalinclude:: ../../examples/batch/vectorize_parameter.py
    """
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


def split_parameters(params, to_split):
    """Distribute kwarg parameters in several dictionaries
    """
    params = copy(params)

    def multi_pop(d, to_pop):
        dic = {}
        for key in to_pop:
            dic[key] = d.pop(key)
        return dic

    params_to_split = multi_pop(params, to_split)

    keys = params_to_split.keys()
    values_tuples = zip(*params_to_split.values())

    distributed = [{k: v for k, v in zip(keys, values)} for values
                   in values_tuples]

    return [merge_dicts(dist, params) for dist in distributed]

# d = dict(a=[0, 10, 20], b=[1, 11, 21], c=1)
# to_pop = ['a', 'b']


# split_parameters(d, ['a', 'b'])
# split_parameters(d, ['a'])
