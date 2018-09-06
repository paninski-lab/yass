import os
import random
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path
import numpy as np
from six import with_metaclass

PATH_TO_TESTS = os.path.dirname(os.path.realpath(__file__))


class BaseClass(object):
    pass


class TestingType(type):
    def __getattr__(self, name):

        def wrapper(arr, path_to_reference, **kwargs):
            if os.environ.get('YASS_SAVE_OUTPUT_REFERENCE'):

                path_to_output_folder = Path(path_to_reference).parent

                if not path_to_output_folder.exists():
                    path_to_output_folder.mkdir()

                np.save(path_to_reference, arr)

            fn = getattr(np.testing, name)
            arr_reference = np.load(path_to_reference)
            fn(arr, arr_reference, **kwargs)

        return wrapper


class ReferenceTesting(with_metaclass(TestingType, BaseClass)):
    """
    Utility class for testing reference files

    Examples
    --------
    >>> ref_test = ReferenceTesting()
    >>> arr = np.zeros((10, 10))
    >>> # this will test arr to the contents of path/to/reference/file.npy,
    >>> # which is obviously different, so it will fail
    >>> ref_test.assert_array_equal(arr, 'path/to/reference/file.npy')

    Notes
    -----
    To re-generate the testing files change the SAVE_BEFORE_TESTING to True
    in TestingType and run the tests once
    """
    pass


def seed(i):
    random.seed(i)
    np.random.seed(i)


def dummy_predict_with_threshold(self, x, threshold, **kwargs):
    n = int(x.shape[0]/2)
    r = int(n % 2)
    ones = np.ones(n + r).astype(bool)
    zeros = np.zeros(n).astype(bool)
    return np.concatenate((ones, zeros))
