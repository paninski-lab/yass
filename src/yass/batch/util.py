import logging
try:
    from pathlib2 import Path
except Exception:
    from pathlib import Path
import yaml
import numbers


def batch_runner(element, function, reader, pass_batch_info, cast_dtype,
                 kwargs, cleanup_function, buffer_size, save_chunks,
                 output_path=None):
    i, idx = element

    logger = logging.getLogger(__name__)
    logger.debug('Processing batch {}...'.format(i))

    if callable(reader):
        _reader = reader()
    else:
        _reader = reader

    # read chunk and run function
    logger.debug('Applying function in batch {}...'.format(i))

    subset, idx_local = _reader[idx]

    kwargs_other = dict()

    if pass_batch_info:
        kwargs_other['idx_local'] = idx_local
        kwargs_other['idx'] = idx

    kwargs.update(kwargs_other)

    res = function(subset, **kwargs)

    logger.debug('Done Applying function in batch {}...'.format(i))

    if cast_dtype is not None:
        res = res.astype(cast_dtype)

    if cleanup_function:
        res = cleanup_function(res, idx_local, idx, buffer_size)

    if save_chunks:
        # save chunk to disk
        chunk_path = make_chunk_path(output_path, i)

        with open(chunk_path, 'wb') as f:
            res.tofile(f)

    return res


def make_chunk_path(output_path, i):
    name, ext = output_path.parts[-1].split('.')
    filename = name+str(i)+'.'+ext

    parts = list(output_path.parts[:-1])
    parts.append(filename)
    chunk_path = Path(*parts)

    return str(chunk_path)


def make_metadata(channels, n_channels, dtype, output_path):
    """Make and save metadata for a binary file

    Parameters
    ----------
    chahnels: str or int
        The value of the channels parameter
    n_channels: int
        Number of channels in the whole dataset (not necessarily match the
        number of channels in the subset)
    dtype: str
        dtype
    output_path: str
        Where to save the file
    """
    if channels == 'all':
        _n_channels = n_channels
    elif isinstance(channels, numbers.Integral):
        _n_channels = 1
    else:
        _n_channels = len(channels)

    # save yaml file with params
    path_to_yaml = str(output_path).replace('.bin', '.yaml')

    params = dict(dtype=dtype, n_channels=_n_channels, data_order='samples')

    with open(path_to_yaml, 'w') as f:
        yaml.dump(params, f)

    return params
