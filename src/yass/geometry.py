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
    """
    Parse a geometry txt (one x, y pair per line, separated
    by spaces) or a npy file with shape (n_channels, 2), where every row
    contains a x, y pair

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
    """Compute a neighbors matrix by using a radius

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


def n_steps_neigh_channels(neighbors_matrix, steps):
    """Compute a neighbors matrix by considering neighbors of neighbors

    Parameters
    ----------
    neighbors_matrix: numpy.ndarray
        Neighbors matrix
    steps: int
        Number of steps to still consider channels as neighbors

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    C = neighbors_matrix.shape[0]

    # each channel is its own neighbor (diagonal of trues)
    output = np.eye(C, dtype='bool')

    # for every step
    for _ in range(steps):

        # go trough every channel
        for current in range(C):

            # neighbors of the current channel
            neighbors_current = output[current]

            # get the neighbors of all the neighbors of the current channel
            neighbors_of_neighbors = neighbors_matrix[neighbors_current]

            # sub over rows and convert to bool, this will turn to true entries
            # where at least one of the neighbors has each channel as its
            # neighbor
            is_neighbor_of_neighbor = np.sum(neighbors_of_neighbors,
                                             axis=0).astype('bool')

            # set the channels that are neighbors to true
            output[current][is_neighbor_of_neighbor] = True

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
    channel_groups = list()
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

        channel_groups.append(c_group)

    return channel_groups


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


def make_channel_index(neighbors, channel_geometry, steps=1):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C, C2 = neighbors.shape

    if C != C2:
        raise ValueError('neighbors is not a square matrix, verify')

    # get neighbors matrix
    neighbors = n_steps_neigh_channels(neighbors, steps=steps)

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # initialize channel index, initially with a dummy C value (a channel)
    # that does not exists
    channel_index = np.ones((C, n_neighbors), 'int32') * C

    # fill every row in the matrix (one per channel)
    for current in range(C):

        # indexes of current channel neighbors
        neighbor_channels = np.where(neighbors[current])[0]

        # sort them by distance
        ch_idx, _ = order_channels_by_distance(current, neighbor_channels,
                                               channel_geometry)

        # fill entries with the sorted neighbor indexes
        channel_index[current, :ch_idx.shape[0]] = ch_idx

    return channel_index
