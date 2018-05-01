from functools import partial
import time
import logging
import os.path
from pathlib import Path
from multiprocess import Pool, Manager, Value
from copy import copy
import os

import yaml

from yass.util import function_path, human_readable_time
from yass.batch import util
from yass.batch.generator import IndexGenerator
from yass.batch.reader import RecordingsReader
from yass.batch.buffer import BufferGenerator


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

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels
    """

    def __init__(self, path_to_recordings, dtype=None, n_channels=None,
                 data_order=None, max_memory='1GB', buffer_size=0,
                 loader='memmap'):
        self.data_order = data_order
        self.buffer_size = buffer_size
        self.path_to_recordings = path_to_recordings
        self.dtype = dtype
        self.n_channels = n_channels
        self.data_order = data_order
        self.loader = loader
        self.reader = RecordingsReader(self.path_to_recordings,
                                       self.dtype, self.n_channels,
                                       self.data_order,
                                       loader=self.loader)
        self.indexer = IndexGenerator(self.reader.observations,
                                      self.reader.channels,
                                      self.reader.dtype,
                                      max_memory)

        # data format is long since reader will return data in that format
        self.buffer_generator = BufferGenerator(self.reader.observations,
                                                data_shape='long',
                                                buffer_size=buffer_size)

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
                yield self.reader[idx]
        else:
            for idx in indexes:
                channel_idx = idx[1]
                yield self.reader[idx], channel_idx

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
            obs_idx = idx[0]
            data_idx = (slice(self.buffer_size,
                              obs_idx.stop - obs_idx.start + self.buffer_size,
                              obs_idx.step), slice(None, None, None))

            if return_data:

                if self.buffer_size:
                    (idx_new,
                     (buff_start, buff_end)) = (self.buffer_generator
                                                .update_key_with_buffer(idx))
                    subset = self.reader[idx_new]
                    subset_buff = self.buffer_generator.add_buffer(subset,
                                                                   buff_start,
                                                                   buff_end)
                    yield subset_buff, data_idx, idx
                else:
                    yield self.reader[idx], data_idx, idx

            else:
                yield data_idx, idx

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
        for i, idx in enumerate(indexes):
            self.logger.debug('Processing channel {}...'.format(i))
            self.logger.debug('Reading batch...')
            subset = self.reader[idx]

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
        results = []

        for i, idx in enumerate(indexes):
            self.logger.debug('Processing channel {}...'.format(i))
            self.logger.debug('Reading batch...')
            subset = self.reader[idx]

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

        for i, (idx_local, idx) in enumerate(data):
            _data = i, (idx_local, idx)
            res = util.batch_runner(_data, function, self.reader,
                                    pass_batch_info, cast_dtype,
                                    kwargs, cleanup_function, self.buffer_size,
                                    output_path, save_chunks=False)
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
                                           processes=1, **kwargs):

        self.logger.debug('Starting parallel operation...')

        if pass_batch_results:
            raise NotImplementedError("pass_batch_results is not "
                                      "implemented on 'disk' mode")

        data = list(self.multi_channel(from_time, to_time, channels,
                                       return_data=False))
        self.logger.info('Data will be splitted in %s batches', len(data))

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
                         loader=_loader)

        # the list will keep track of finished jobs so their output can be
        # written to disk
        m = Manager()
        done = m.list()
        mapping = m.dict()

        def parallel_runner(data):
            i, (idx_local, idx) = data

            res = util.batch_runner(data, function, reader,
                                    pass_batch_info, cast_dtype,
                                    kwargs, cleanup_function, _buffer_size,
                                    output_path, save_chunks=True)

            if i == 0:
                mapping['dtype'] = str(res.dtype)

            # let the master process know that this job is done
            done.append(i)

        # run jobs
        self.logger.debug('Creating processes pool...')
        p = Pool(processes)
        p.map_async(parallel_runner, enumerate(data))

        # since we need to write chunks in order, start this flag to know
        # which one is next
        next_to_write = 0

        if output_path.is_file():
            os.remove(str(output_path))

        f = open(str(output_path), 'ab')

        while True:

            # wait for the next job to write to be done
            if next_to_write in done:

                # read its chunk and append it to the main file
                chunk_path = util.make_chunk_path(output_path, next_to_write)

                with open(chunk_path, "rb") as f2:
                    f.write(f2.read())
                    self.logger.debug('Appending chunk %i...', next_to_write)

                # remove chunk
                os.remove(chunk_path)

                next_to_write += 1

                # finish when you've written all parts
                if next_to_write == len(data):
                    self.logger.debug('Done running parallel operation...')
                    break

        f.close()

        # save metadata
        params = util.make_metadata(channels, self.n_channels,
                                    mapping['dtype'], output_path)

        return output_path, params

    def _multi_channel_apply_memory(self, function, cleanup_function,
                                    from_time, to_time, channels, cast_dtype,
                                    pass_batch_info, pass_batch_results,
                                    **kwargs):

        data = self.multi_channel(from_time, to_time, channels)

        results = []
        previous_batch = None

        for i, (subset, idx_local, idx) in enumerate(data):

            self.logger.debug('Processing batch {}...'.format(i))

            kwargs_other = dict()

            if pass_batch_info:
                kwargs_other['idx_local'] = idx_local
                kwargs_other['idx'] = idx

            if pass_batch_results:
                kwargs_other['previous_batch'] = previous_batch

            kwargs.update(kwargs_other)

            res = function(subset, **kwargs)

            if cast_dtype is not None:
                res = res.astype(cast_dtype)

            if cleanup_function:
                res = cleanup_function(res, idx_local, idx, self.buffer_size)

            if pass_batch_results:
                previous_batch = res
            else:
                results.append(res)

        return previous_batch if pass_batch_results else results
