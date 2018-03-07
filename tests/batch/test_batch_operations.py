import os
import tempfile
import pytest
import numpy as np
from yass.batch import BatchProcessor


@pytest.fixture
def path_to_data(request):
    temp = tempfile.NamedTemporaryFile(delete=False)
    path = temp.name

    def delete_file():
        os.unlink(path)

    request.addfinalizer(delete_file)

    array = (np.array([np.arange(100), np.arange(100)]).astype('int64')
             .reshape((2, 100)).T)

    array.tofile(temp)
    temp.close()

    return path


def test_can_pass_information_between_batches(path_to_data):
    bp = BatchProcessor(path_to_data, dtype='int64', n_channels=2,
                        data_format='long', max_memory='160B')

    def col_sums(data, previous_batch):
        current = np.sum(data, axis=0)

        if previous_batch is None:
            return current
        else:
            return np.array([previous_batch[0] + current[0],
                             previous_batch[1] + current[1]])

    res = bp.multi_channel_apply(col_sums,
                                 mode='memory',
                                 channels='all',
                                 pass_batch_results=True)

    assert res[0] == 4950 and res[1] == 4950
