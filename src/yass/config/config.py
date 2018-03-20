"""
YASS configuration
"""

from os import path
from collections import Mapping, MutableSequence
import keyword
import logging

import yaml
import numpy as np

from yass import geometry as geom
from yass.config.validate import validate


class FrozenJSON(object):
    """A facade for navigating a JSON-like object
    using attribute notation. Based on FrozenJSON from 'Fluent Python'
    """
    @classmethod
    def from_yaml(cls, path_to_file):
        # load config file
        with open(path_to_file) as file:
            mapping = yaml.load(file)

        obj = cls(mapping)

        # save path for reference, helps debugging
        obj._path_to_file = path_to_file

        logger = logging.getLogger(__name__)
        logger.debug('Loaded from file: %s', obj._path_to_file)

        return obj

    def __new__(cls, arg):
        if isinstance(arg, Mapping):
            return super(FrozenJSON, cls).__new__(cls)

        elif isinstance(arg, MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping):
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Loaded with params: %s ', mapping)
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
                               '({}) keys are: {}'
                               .format(name, self._data.keys()))

    def __dir__(self):
        return self._data.keys()

    def __getitem__(self, key):
        value = self._data.get(key)

        if value is None:
            raise ValueError('No value was set in Config{}for key "{}", '
                             'available keys are: {}'
                             .format(self._path_to_file, key,
                                     self._data.keys()))

        return value

    def __repr__(self):
        if self._path_to_file:
            return ('YASS config file loaded from: {}'
                    .format(self._path_to_file))

        return 'YASS config file loaded with: {}'.format(self._data)


class Config(FrozenJSON):
    """
    A configuration object for the package, it is a read-only FrozenJSON that
    inits from a yaml file with some caching capbilities to avoid
    redundant and common computations

    Notes
    -----
    After initialization, attributes cannot be changed
    """
    def __init__(self, mapping):
        mapping = validate(mapping)

        super(Config, self).__init__(mapping)

        self._logger = logging.getLogger(__name__)

        # init the rest of the parameters, these parameters are used
        # througout the pipeline so we compute them once to avoid redudant
        # computations

        # GEOMETRY PARAMETERS
        path_to_geom = path.join(self.data.root_folder, self.data.geometry)
        self._set_param('geom', geom.parse(path_to_geom,
                                           self.recordings.n_channels))

        neigh_channels = geom.find_channel_neighbors(
            self.geom, self.recordings.spatial_radius)
        self._set_param('neigh_channels', neigh_channels)

        channel_groups = geom.make_channel_groups(self.recordings.n_channels,
                                                  self.neigh_channels,
                                                  self.geom)
        self._set_param('channel_groups', channel_groups)

        self._logger.debug('Geometry parameters. Geom: %s, neigh_channels: '
                           '%s, channel_groups %s', self.geom,
                           self.neigh_channels, self.channel_groups)

        self._set_param('spike_size',
                        int(np.round(self.recordings.spike_size_ms *
                                     self.recordings.sampling_rate /
                                     (2*1000))))
        self._set_param('templates_max_shift',
                        int(self.recordings.sampling_rate/1000))

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError('Cannot set values once the object has '
                                 'been initialized')
        else:
            self.__dict__[name] = value

    def _set_param(self, name, value):
        """
        Internal setattr method to set new parameters, only used to fill the
        parameters that need to be computed *right after* initialization
        """
        self._data[name] = value
