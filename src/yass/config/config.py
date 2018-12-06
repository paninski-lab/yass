"""
YASS configuration
"""
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path
import pprint
from os import path
from collections import Mapping, MutableSequence
import keyword
import logging

import multiprocess
import yaml
import numpy as np

from yass import geometry as geom
from yass.config.validate import validate


class FrozenJSON(object):
    """A facade for navigating a JSON-like object
    using attribute notation. Based on FrozenJSON from 'Fluent Python'
    """

    def __new__(cls, arg):
        if isinstance(arg, Mapping):
            return super(FrozenJSON, cls).__new__(cls)

        elif isinstance(arg, MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping):
        self._logger = logging.getLogger(__name__)
        self._path_to_file = None

        self._data = {}

        for key, value in mapping.items():

            if keyword.iskeyword(key):
                key += '_'

            self._data[key] = value

    def __getattr__(self, name):
        if hasattr(self._data, name):
            return getattr(self._data, name)
        else:
            try:
                return FrozenJSON(self._data[name])
            except KeyError:
                raise KeyError('Trying to access a key that does not exist, '
                               '({}) keys are: {}'.format(
                                   name, self._data.keys()))

    def __dir__(self):
        return self._data.keys()

    def __getitem__(self, key):
        value = self._data.get(key)

        if value is None:
            raise ValueError('No value was set in Config{}for key "{}", '
                             'available keys are: {}'.format(
                                 self._path_to_file, key, self._data.keys()))

        return value

    def __repr__(self):
        s = '********** start of YASS configuration file **********'

        if self._path_to_file is not None:
            s += '\nLoaded from: '+self._path_to_file

        if self._data is not None:
            s += '\nContent: '+pprint.pformat(self._data, indent=4)

        s += '\n********** end of YASS configuration file **********'

        return s


class Config:
    """
    A configuration object for the package, it is a read-only FrozenJSON that
    inits from a yaml file with some caching capbilities to avoid
    redundant and common computations. It also computes some matrices that are
    used in several functions through the pipeline, see attributes section
    for more info

    Parameters
    ----------
    mapping: collections.Mapping
        Data
    output_directory: str of pathlib.Path, optional
        output directory for the project, this is optional and makes
        Config.output_directory return
        onfig.data.root_folder / output_directory, which is a common path
        used through the pipeline


    Attributes
    ---------
    geom: numpy.ndarray, [n_channels, 2]
        Recordings geometry, every row contains an (x, y) pair

    neigh_channels: numpy.ndarray, [n_channels, n_channels]
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors

    channel_index: numpy.ndarray, [n_channels, n_channels]
        An array whose whose ith row contains the ordered (by distance)
        neighbors for the ith channel

    output_directory: str
        Returhs path to config.data.root_folder / output_directory

    Notes
    -----
    After initialization, attributes cannot be changed
    """

    @classmethod
    def from_yaml(cls, path_to_file, output_directory=None):
        # load config file
        with open(path_to_file) as file:
            mapping = yaml.load(file)

        obj = cls(mapping, output_directory)

        # save path for reference, helps debugging
        obj._path_to_file = path_to_file

        logger = logging.getLogger(__name__)
        logger.debug('Loaded from file: %s', obj._path_to_file)

        return obj

    def __init__(self, mapping, output_directory=None):
        self._logger = logging.getLogger(__name__)

        # FIXME: not raising errors due to schema validation for now
        mapping = validate(mapping, silent=True)

        _processes = mapping['resources']['processes']
        mapping['resources']['processes'] = (multiprocess.cpu_count()
                                             if _processes == 'max'
                                             else _processes)

        self._frozenjson = FrozenJSON(mapping)

        if output_directory is not None:
            if path.isabs(output_directory):
                self._path_to_output_directory = output_directory
            else:
                _ = Path(self.data.root_folder, output_directory)
                self._path_to_output_directory = str(_)
        else:
            self._path_to_output_directory = None

        # init the rest of the parameters, these parameters are used
        # througout the pipeline so we compute them once to avoid redudant
        # computations

        # GEOMETRY PARAMETERS
        path_to_geom = path.join(self.data.root_folder, self.data.geometry)

        self._set_param('geom',
                        geom.parse(path_to_geom, self.recordings.n_channels))

        # check dimensions of the geometry file
        n_channels_geom, _ = self.geom.shape

        if self.recordings.n_channels != n_channels_geom:
            raise ValueError('Channels in the geometry file ({}) does not '
                             'value in the configuration file ({})'
                             .format(n_channels_geom,
                                     self.recordings.n_channels))

        neigh_channels = geom.find_channel_neighbors(
            self.geom, self.recordings.spatial_radius)
        self._set_param('neigh_channels', neigh_channels)

        channel_groups = geom.make_channel_groups(
            self.recordings.n_channels, self.neigh_channels, self.geom)
        self._set_param('channel_groups', channel_groups)

        self._set_param(
            'spike_size',
            int(
                np.round(self.recordings.spike_size_ms *
                         self.recordings.sampling_rate / (2 * 1000))))

        channel_index = geom.make_channel_index(self.neigh_channels,
                                                self.geom, steps=2)

        self._set_param('channel_index', channel_index)

    @property
    def path_to_output_directory(self):
        if self._path_to_output_directory is not None:
            return self._path_to_output_directory
        else:
            raise ValueError('This is only available when output_directory'
                             'is passed when setting the configuration')

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError('Cannot set values once the object has '
                                 'been initialized')
        else:
            self.__dict__[name] = value

    def __getattr__(self, key):
        return getattr(self._frozenjson, key)

    def _set_param(self, name, value):
        """
        Internal setattr method to set new parameters, only used to fill the
        parameters that need to be computed *right after* initialization
        """
        self._frozenjson._data[name] = value
