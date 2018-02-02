"""
Functions for parsing geometry data
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


def _parse_txt(path, n_channels):
    """Parse a geometry file in txt format
    """
    f = open(path)
    lines = f.readlines()
    f.close()

    geom = np.zeros((0, 2))

    for i, line in zip(range(n_channels), lines):
        line = line.replace('\r', '')
        line = line.replace('\n', '')
        row = line.split(' ')
        geom = np.vstack((geom, row[:2])).astype('float')

    return geom


def parse(path, n_channels):
    """Parse a geometry txt or npy file

    path: str
        Path to geometry file
    n_channels: int
        Number of channels

    Returns
    -------
    numpy.ndarray
        2-dimensional numpy array where each row contains the x, y coordinates
        for a channel

    Examples
    --------

    .. code-block:: python

    from yass import geometry

    geom = geometry.parse('path/to/geom.npy', n_channels=500)
    geom = geometry.parse('path/to/geom.txt', n_channels=500)
    """
    # TODO: infer the number of channels by the number of lines
    extension = path.split('.')[-1]

    if extension == 'txt':
        geom = _parse_txt(path, n_channels)
    elif extension == 'npy':
        geom = np.load(path)
    else:
        raise ValueError('Invalid file: {} extension not supported'
                         .format(extension))

    read_channels, _ = geom.shape

    if read_channels != n_channels:
        raise ValueError('Expected {} channels, but read {}'
                         .format(n_channels, read_channels))

    return geom


def find_channel_neighbors(geom, radius):
    """Compute a channel neighbors matrix

    Parameters
    ----------
    geom: np.array
        Array with the cartesian coordinates for the channels
    radius: float
        Maximum radius for the channels to be considered neighbors

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    return (squareform(pdist(geom)) <= radius)


def n_steps_neigh_channels(neighbors, steps):
    """Compute a neighbors matrix by considering neighbors of neighbors

    Parameters
    ----------
    neighbors: numpy.ndarray
        Neighbors matrix
    steps: int
        Number of steps to still consider channels as neighbors

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    C = neighbors.shape[0]
    output = np.eye(C, dtype='bool')

    for j in range(steps):
        for c in range(C):
            output[c][np.sum(neighbors[output[c]], axis=0).astype('bool')] = 1

    return output


# TODO: add documentation
# TODO: remove n_channels, we can infer it from neighbors or geom
def make_channel_groups(n_channels, neighbors, geom):
    """[DESCRIPTION]

    Parameters
    ----------
    n_channels: int
        Number of channels
    neighbors: numpy.ndarray
        Neighbors matrix
    geom: numpy.ndarray
        geometry matrix

    Returns
    -------
    list
        List of channel groups based on [?]
    """
    channelGroups = list()
    c_left = np.array(range(n_channels))
    neighChan_temp = np.array(neighbors)

    while len(c_left) > 0:
        c_tops = c_left[geom[c_left, 1] == np.max(geom[c_left, 1])]
        c_topleft = c_tops[np.argmin(geom[c_tops, 0])]
        c_group = np.where(
            np.sum(neighChan_temp[neighChan_temp[c_topleft]], 0))[0]

        neighChan_temp[c_group, :] = 0
        neighChan_temp[:, c_group] = 0

        for c in c_group:
            c_left = np.delete(c_left, int(np.where(c_left == c)[0]))

        channelGroups.append(c_group)

    return channelGroups


def order_channels_by_distance(reference, channels, geom):
    """Order channels by distance using certain channel as reference

    Parameters
    ----------
    reference: int
        Reference channel
    channels: np.ndarray
        Channels to order
    geom
        Geometry matrix

    Returns
    -------
    numpy.ndarray
        1D array with the channels ordered by distance using the reference
        channels
    numpy.ndarray
        1D array with the indexes for the ordered channels
    """
    coord_main = geom[reference]
    coord_others = geom[channels]
    idx = np.argsort(np.sum(np.square(coord_others - coord_main), axis=1))

    return channels[idx], idx


def ordered_neighbors(geom, neighbors):
    """
    Compute a list of arrays whose ith element contains the ordered
    (by distance) neighbors for the ith channel

    Parameters
    ----------
    geom: numpy.ndarray
        geometry matrix
    neighbors: numpy.ndarray
        Neighbors matrix
    """
    n_channels, _ = neighbors.shape

    # determine the max number of neighbors
    max_neighbors = np.max(np.sum(neighbors, axis=0))

    # build matrix filled with n_channels
    channel_indexes = []

    for c in range(n_channels):
        # get neighbors for channel c
        c_neighs = np.where(neighbors[c])[0]

        # order neighbors by distance
        ch_idx, _ = order_channels_by_distance(c, c_neighs, geom)

        # set the row for channel c as their ordered neighbors
        channel_indexes.append(ch_idx)

    return channel_indexes, max_neighbors
