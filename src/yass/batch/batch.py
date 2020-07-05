from functools import partial
import time
import logging
import os.path
from copy import copy
import os

try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

from multiprocess import Pool, Manager
import yaml
from tqdm import tqdm

from yass.util import function_path, human_readable_time
from yass.batch import util
from yass.batch.generator import IndexGenerator
from yass.batch.reader import RecordingsReader


class BatchProcessor(object):
    """
    Batch processing for large numpy matrices

    Parameters
    ----------
    path_to_recordings: str
        Path to recordings file

    dtype: str
        Numpy dtype

    n_channels: int
        Number of channels

    data_order: str
        Recordings order, one of ('channels', 'samples'). In a dataset with k
        observations per channel and j channels: 'channels' means first k
        contiguous observations come from channel 0, then channel 1, and so
        on. 'sample' means first j contiguous data are the first observations
        from all channels, then the second observations from all channels and
        so on

    max_memory: int or str
        Max memory to use in each batch, interpreted as bytes if int,
        if string, it can be any of {N}KB, {N}MB or {N}GB

    buffer_size: int, optional
        Buffer size, defaults to 0. Only relevant when performing multi-channel
        operations

    loader: str ('memmap', 'array' or 'python'), optional
        How to load the data. memmap loads the data using a wrapper around
        np.memmap (see :class:`~yass.batch.MemoryMap` for details), 'array'
        using numpy.fromfile and 'python' loads it using a wrapper
        around Python file API. Defaults to 'python'. Beware that the Python
        loader has limited indexing capabilities, see
        :class:`~yass.batch.BinaryReader` for details

    show_progress_bar: bool, optional
        Show progress bar when running operations, defaults to True

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels
    """

    def __init__(self, path_to_recordings, dtype=None, n_channels=None,
                 data_order=None, max_memory='1GB', buffer_size=0,
                 loader='memmap', show_progress_bar=True):
        self.data_order = data_order
        self.buffer_size = buffer_size
        self.path_to_recordings = path_to_recordings
        self.dtype = dtype
        self.n_channels = n_channels
        self.data_order = data_order
        self.loader = loader
        self.show_progress_bar = show_progress_bar

        self.reader = RecordingsReader(self.path_to_recordings,
                                       self.dtype, self.n_channels,
                                       self.data_order,
                                       loader=self.loader,
                                       buffer_size=buffer_size,
                                       return_data_index=True)
        self.indexer = IndexGenerator(self.reader.observations,
                                      self.reader.channels,
                                      self.reader.dtype,
                                      max_memory)

        self.logger = logging.getLogger(__name__)

    def single_channel(self, force_complete_channel_batch=True, from_time=None,
                       to_time=None, channels='all'):
        """
        Generate batches where each index has observations from a single
        channel

        Returns
        -------
        A generator that yields batches, if force_complete_channel_batch is
        False, each generated value is a tuple with the batch and the
        channel for the index for the corresponding channel

        Examples
        --------

        .. literalinclude:: ../../examples/batch/single_channel.py
        """
        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        if force_complete_channel_batch:
            for idx in indexes:
                subset, _ = self.reader[idx]
                yield subset
        else:
            for idx in indexes:
                channel_idx = idx[1]
                subset, _ = self.reader[idx]
                yield subset, channel_idx

    def multi_channel(self, from_time=None, to_time=None, channels='all',
                      return_data=True):
        """
        Generate indexes where each index has observations from more than
        one channel

        Returns
        -------
        generator:
            A tuple of size three: the first element is the subset of the data
            for the ith batch, second element is the slice object with the
            limits of the data in [observations, channels] format (excluding
            the buffer), the last element is the absolute index of the data
            again in [observations, channels] format

        Examples
        --------
        .. literalinclude:: ../../examples/batch/multi_channel.py
        """
        indexes = self.indexer.multi_channel(from_time, to_time, channels)

        for idx in indexes:
            if return_data:
                subset, _ = self.reader[idx]
                yield subset
            else:
                yield idx

    def single_channel_apply(self, function, mode, output_path=None,
                             force_complete_channel_batch=True,
                             from_time=None, to_time=None, channels='all',
                             if_file_exists='overwrite', cast_dtype=None,
                             **kwargs):
        """
        Apply a transformation where each batch has observations from a
        single channel

        Parameters
        ----------
        function: callable
            Function to be applied, must accept a 1D numpy array as its first
            parameter
        mode: str
            'disk' or 'memory', if 'disk', a binary file is created at the
            beginning of the operation and each partial result is saved
            (ussing numpy.ndarray.tofile function), at the end of the
            operation two files are generated: the binary file and a yaml
            file with some file parameters (useful if you want to later use
            RecordingsReader to read the file). If 'memory', partial results
            are kept in memory and returned as a list
        output_path: str, optional
            Where to save the output, required if 'disk' mode
        force_complete_channel_batch: bool, optional
            If True, every index generated will correspond to all the
            observations in a single channel, hence
            n_batches = n_selected_channels, defaults to True. If True
            from_time and to_time must be None
        from_time: int, optional
            Starting time, defaults to None
        to_time: int, optional
            Ending time, defaults to None
        channels: int, tuple or str, optional
            A tuple with the channel indexes or 'all' to traverse all channels,
            defaults to 'all'
        if_file_exists: str, optional
            One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces the
            file if it exists, if 'abort' if raise a ValueError exception if
            the file exists, if 'skip' if skips the operation if the file
            exists. Only valid when mode = 'disk'
        cast_dtype: str, optional
            Output dtype, defaults to None which means no cast is done
        **kwargs
            kwargs to pass to function

        Examples
        --------

        .. literalinclude:: ../../examples/batch/single_channel_apply.py

        Notes
        -----
        When applying functions in 'disk' mode will incur in memory overhead,
        which depends on the function implementation, this is an important
        thing to consider if the transformation changes the data's dtype (e.g.
        converts int16 to float64), which means that a chunk of 1MB in int16
        will have a size of 4MB in float64. Take that into account when
        setting max_memory.

        For performance reasons in 'disk' mode, output data is in 'channels'
        order
        """
        if mode not in ['disk', 'memory']:
            raise ValueError('Mode should be disk or memory, received: {}'
                             .format(mode))

        if mode == 'disk' and output_path is None:
            raise ValueError('output_path is required in "disk" mode')

        if (mode == 'disk' and if_file_exists == 'abort' and
                os.path.exists(output_path)):
            raise ValueError('{} already exists'.format(output_path))

        if (mode == 'disk' and if_file_exists == 'skip' and
                os.path.exists(output_path)):
            # load params...
            path_to_yaml = output_path.replace('.bin', '.yaml')

            if not os.path.exists(path_to_yaml):
                raise ValueError("if_file_exists = 'skip', but {}"
                                 " is missing, aborting..."
                                 .format(path_to_yaml))

            with open(path_to_yaml) as f:
                params = yaml.load(f)

            self.logger.info('{} exists, skiping...'.format(output_path))

            return output_path, params

        self.logger.info('Applying function {}...'
                         .format(function_path(function)))

        if mode == 'disk':
            fn = self._single_channel_apply_disk

            start = time.time()
            res = fn(function, output_path,
                     force_complete_channel_batch, from_time,
                     to_time, channels, cast_dtype, **kwargs)
            elapsed = time.time() - start
            self.logger.info('{} took {}'
                             .format(function_path(function),
                                     human_readable_time(elapsed)))
            return res
        else:
            fn = self._single_channel_apply_memory

            start = time.time()
            res = fn(function, force_complete_channel_batch, from_time,
                     to_time, channels, cast_dtype, **kwargs)
            elapsed = time.time() - start
            self.logger.info('{} took {}'
                             .format(function_path(function),
                                     human_readable_time(elapsed)))
            return res

    def _single_channel_apply_disk(self, function, output_path,
                                   force_complete_channel_batch, from_time,
                                   to_time, channels, cast_dtype, **kwargs):
        f = open(output_path, 'wb')

        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        indexes = list(indexes)
        iterator = enumerate(indexes)

        if self.show_progress_bar:
            iterator = tqdm(iterator, total=len(indexes))

        for i, idx in iterator:
            self.logger.debug('Processing channel {}...'.format(i))
            self.logger.debug('Reading batch...')
            subset, _ = self.reader[idx]

            self.logger.debug('Executing function...')

            if cast_dtype is None:
                res = function(subset, **kwargs)
            else:
                res = function(subset, **kwargs).astype(cast_dtype)

            self.logger.debug('Writing to disk...')
            res.tofile(f)

        f.close()

        params = util.make_metadata(channels, self.n_channels, str(res.dtype),
                                    output_path)

        return output_path, params

    def _single_channel_apply_memory(self, function,
                                     force_complete_channel_batch, from_time,
                                     to_time, channels, cast_dtype, **kwargs):

        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        indexes = list(indexes)
        iterator = enumerate(indexes)

        if self.show_progress_bar:
            iterator = tqdm(iterator, total=len(indexes))

        results = []

        for i, idx in iterator:
            self.logger.debug('Processing channel {}...'.format(i))
            self.logger.debug('Reading batch...')
            subset, _ = self.reader[idx]

            if cast_dtype is None:
                res = function(subset, **kwargs)
            else:
                res = function(subset, **kwargs).astype(cast_dtype)

            self.logger.debug('Appending partial result...')
            results.append(res)

        return results

    def multi_channel_apply(self, function, mode, cleanup_function=None,
                            output_path=None, from_time=None, to_time=None,
                            channels='all', if_file_exists='overwrite',
                            cast_dtype=None, pass_batch_info=False,
                            pass_batch_results=False, processes=1,
                            **kwargs):
        """
        Apply a function where each batch has observations from more than
        one channel

        Parameters
        ----------
        function: callable
            Function to be applied, first parameter passed will be a 2D numpy
            array in 'long' shape (number of observations, number of
            channels). If pass_batch_info is True, another two keyword
            parameters will be passed to function: 'idx_local' is the slice
            object with the limits of the data in [observations, channels]
            format (excluding the buffer), 'idx' is the absolute index of
            the data again in [observations, channels] format

        mode: str
            'disk' or 'memory', if 'disk', a binary file is created at the
            beginning of the operation and each partial result is saved
            (ussing numpy.ndarray.tofile function), at the end of the
            operation two files are generated: the binary file and a yaml
            file with some file parameters (useful if you want to later use
            RecordingsReader to read the file). If 'memory', partial results
            are kept in memory and returned as a list

        cleanup_function: callable, optional
            A function to be executed after `function` and before adding the
            partial result to the list of results (if `memory` mode) or to the
            biinary file (if in `disk mode`). `cleanup_function` will be called
            with the following parameters (in that order): result from applying
            `function` to the batch, slice object with the idx where the data
            is located (exludes buffer), slice object with the absolute
            location of the data and buffer size

        output_path: str, optional
            Where to save the output, required if 'disk' mode

        force_complete_channel_batch: bool, optional
            If True, every index generated will correspond to all the
            observations in a single channel, hence
            n_batches = n_selected_channels, defaults to True. If True
            from_time and to_time must be None

        from_time: int, optional
            Starting time, defaults to None

        to_time: int, optional
            Ending time, defaults to None

        channels: int, tuple or str, optional
            A tuple with the channel indexes or 'all' to traverse all channels,
            defaults to 'all'

        if_file_exists: str, optional
            One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces the
            file if it exists, if 'abort' if raise a ValueError exception if
            the file exists, if 'skip' if skips the operation if the file
            exists. Only valid when mode = 'disk'

        cast_dtype: str, optional
            Output dtype, defaults to None which means no cast is done

        pass_batch_info: bool, optional
            Whether to call the function with batch info or just call it with
            the batch data (see description in the function) parameter

        pass_batch_results: bool, optional
            Whether to pass results from the previous batch to the next one,
            defaults to False. Only relevant when mode='memory'. If True,
            function will be called with the keyword parameter
            'previous_batch' which contains the computation for the last
            batch, it is set to None in the first batch

        **kwargs
            kwargs to pass to function

        Returns
        -------
        output_path, params (when mode is 'disk')
            Path to output binary file, Binary file params

        list (when mode is 'memory' and pass_batch_results is False)
            List where every element is the result of applying the function
            to one batch. When pass_batch_results is True, it returns the
            output of the function for the last batch

        Examples
        --------

        .. literalinclude:: ../../examples/batch/multi_channel_apply_disk.py

        .. literalinclude:: ../../examples/batch/multi_channel_apply_memory.py

        Notes
        -----
        Applying functions will incur in memory overhead, which depends
        on the function implementation, this is an important thing to consider
        if the transformation changes the data's dtype (e.g. converts int16 to
        float64), which means that a chunk of 1MB in int16 will have a size
        of 4MB in float64. Take that into account when setting max_memory

        For performance reasons, outputs data in 'samples' order.
        """
        if mode not in ['disk', 'memory']:
            raise ValueError('Mode should be disk or memory, received: {}'
                             .format(mode))

        if mode == 'disk' and output_path is None:
            raise ValueError('output_path is required in "disk" mode')

        if (mode == 'disk' and if_file_exists == 'abort' and
                os.path.exists(output_path)):
            raise ValueError('{} already exists'.format(output_path))

        self.logger.info('Applying function {}...'
                         .format(function_path(function)))

        if (mode == 'disk' and if_file_exists == 'skip' and
                os.path.exists(output_path)):
            # load params...
            path_to_yaml = output_path.replace('.bin', '.yaml')

            if not os.path.exists(path_to_yaml):
                raise ValueError("if_file_exists = 'skip', but {}"
                                 " is missing, aborting..."
                                 .format(path_to_yaml))

            with open(path_to_yaml) as f:
                params = yaml.load(f)

            self.logger.info('{} exists, skiping...'.format(output_path))

            return output_path, params

        if mode == 'disk':

            if processes == 1:
                fn = self._multi_channel_apply_disk
            else:
                fn = partial(self._multi_channel_apply_disk_parallel,
                             processes=processes)

            start = time.time()
            res = fn(function, cleanup_function, output_path, from_time,
                     to_time, channels, cast_dtype, pass_batch_info,
                     pass_batch_results, **kwargs)
            elapsed = time.time() - start
            self.logger.info('{} took {}'
                             .format(function_path(function),
                                     human_readable_time(elapsed)))
            return res
        else:
            fn = self._multi_channel_apply_memory

            start = time.time()
            res = fn(function, cleanup_function, from_time, to_time, channels,
                     cast_dtype, pass_batch_info, pass_batch_results,
                     **kwargs)
            elapsed = time.time() - start
            self.logger.info('{} took {}'
                             .format(function_path(function),
                                     human_readable_time(elapsed)))
            return res

    def _multi_channel_apply_disk(self, function, cleanup_function,
                                  output_path, from_time, to_time, channels,
                                  cast_dtype, pass_batch_info,
                                  pass_batch_results, **kwargs):

        if pass_batch_results:
            raise NotImplementedError("pass_batch_results is not "
                                      "implemented on 'disk' mode")

        f = open(output_path, 'wb')

        output_path = Path(output_path)

        data = self.multi_channel(from_time, to_time, channels,
                                  return_data=False)
        n_batches = self.indexer.n_batches(from_time, to_time, channels)

        iterator = enumerate(data)

        if self.show_progress_bar:
            iterator = tqdm(iterator, total=n_batches)

        for i, idx in iterator:
            res = util.batch_runner((i, idx), function, self.reader,
                                    pass_batch_info, cast_dtype,
                                    kwargs, cleanup_function, self.buffer_size,
                                    save_chunks=False)
            res.tofile(f)

        f.close()

        params = util.make_metadata(channels, self.n_channels, str(res.dtype),
                                    output_path)

        return output_path, params

    def _multi_channel_apply_disk_parallel(self, function, cleanup_function,
                                           output_path, from_time, to_time,
                                           channels, cast_dtype,
                                           pass_batch_info,
                                           pass_batch_results,
                                           processes, **kwargs):

        self.logger.debug('Starting parallel operation...')

        if pass_batch_results:
            raise NotImplementedError("pass_batch_results is not "
                                      "implemented on 'disk' mode")

        # need to convert to a list, oherwise cannot be pickled
        data = list(self.multi_channel(from_time, to_time, channels,
                    return_data=False))
        n_batches = self.indexer.n_batches(from_time, to_time, channels)

        self.logger.info('Data will be splitted in %s batches', n_batches)

        output_path = Path(output_path)

        # create local variables to avoid pickling problems
        _path_to_recordings = copy(self.path_to_recordings)
        _dtype = copy(self.dtype)
        _n_channels = copy(self.n_channels)
        _data_order = copy(self.data_order)
        _loader = copy(self.loader)
        _buffer_size = copy(self.buffer_size)

        reader = partial(RecordingsReader,
                         path_to_recordings=_path_to_recordings,
                         dtype=_dtype,
                         n_channels=_n_channels,
                         data_order=_data_order,
                         loader=_loader,
                         return_data_index=True)

        m = Manager()
        mapping = m.dict()
        next_to_write = m.Value('i', 0)

        def parallel_runner(element):
            i, _ = element

            res = util.batch_runner(element, function, reader,
                                    pass_batch_info, cast_dtype,
                                    kwargs, cleanup_function, _buffer_size,
                                    save_chunks=False, output_path=output_path)

            if i == 0:
                mapping['dtype'] = str(res.dtype)

            while True:
                if next_to_write.value == i:
                    with open(str(output_path), 'wb' if i == 0 else 'ab') as f:
                        res.tofile(f)

                    next_to_write.value += 1
                    break

        # run jobs
        self.logger.debug('Creating processes pool...')

        p = Pool(processes)
        res = p.map_async(parallel_runner, enumerate(data))

        finished = 0

        if self.show_progress_bar:
            pbar = tqdm(total=n_batches)

        if self.show_progress_bar:

            while True:
                if next_to_write.value > finished:
                    update = next_to_write.value - finished
                    pbar.update(update)
                    finished = next_to_write.value

                if next_to_write.value == n_batches:
                    break

            pbar.close()
        else:
            res.get()

        # save metadata
        params = util.make_metadata(channels, self.n_channels,
                                    mapping['dtype'], output_path)

        return output_path, params

    def _multi_channel_apply_memory(self, function, cleanup_function,
                                    from_time, to_time, channels, cast_dtype,
                                    pass_batch_info, pass_batch_results,
                                    **kwargs):
        data = self.multi_channel(from_time, to_time, channels,
                                  return_data=False)
        n_batches = self.indexer.n_batches(from_time, to_time, channels)

        results = []

        if pass_batch_results:
            kwargs['previous_batch'] = None

        iterator = enumerate(data)

        if self.show_progress_bar:
            iterator = tqdm(iterator, total=n_batches)

        for i, idx in iterator:
            res = util.batch_runner((i, idx), function, self.reader,
                                    pass_batch_info, cast_dtype,
                                    kwargs, cleanup_function, self.buffer_size,
                                    save_chunks=False)

            if pass_batch_results:
                kwargs['previous_batch'] = res
            else:
                results.append(res)

        return res if pass_batch_results else results
