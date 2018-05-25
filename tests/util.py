import os
import shutil
import numpy as np
from six import with_metaclass


class BaseClass(object):
    pass


class TestingType(type):
    SAVE_BEFORE_TESTING = False

    def __getattr__(self, name):

        def wrapper(arr, path_to_reference, **kwargs):
            if self.SAVE_BEFORE_TESTING:
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


def clean_tmp():
    path_to_tests = os.path.dirname(os.path.realpath(__file__))
    TMP = os.path.join(path_to_tests, 'data/tmp/')

    if os.path.exists(TMP):
        shutil.rmtree(TMP)


def make_tmp():
    path_to_tests = os.path.dirname(os.path.realpath(__file__))
    TMP = os.path.join(path_to_tests, 'data/tmp/')

    os.mkdir(TMP)
