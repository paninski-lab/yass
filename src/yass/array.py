import numpy as np


class ArrayWithMetadata(np.ndarray):
    """
    Notes
    -----
    https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    Some functions don't work so we need a manual implementation:
    https://github.com/numpy/numpy/issues/11671
    """

    def __new__(cls, input_array, metadata=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        obj.metadata = metadata

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.metadata = getattr(obj, 'metadata', None)


def concatenate(arrs, axis=0, out=None, **unused_kwargs):
    metadata_all = []

    for a in arrs:
        if hasattr(a, 'metadata'):
            metadata_all.append(a.metadata)
        else:
            metadata_all.append(None)

    res = np.concatenate(arrs, axis=axis, out=out, **unused_kwargs)

    if any(metadata_all):
        ArrayWithMetadata(res, metadata_all)
    else:
        res
