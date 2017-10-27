"""
Functions for parsing geometry data
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


def _parse_txt(path, n_channels):
    f = open(path)
    lines = f.readlines()
    f.close()

    geom = np.zeros((0, 2))

    for i, l in zip(range(n_channels), lines):
        l = l.replace('\r', '')
        l = l.replace('\n', '')
        row = l.split(' ')
        geom = np.vstack((geom, row[:2])).astype('float')

    return geom


def parse(path, n_channels):
    """Parse a geometry txt file

    path: str
        Path to geometry file
    n_channels: int
        Number of channels
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
    """Compute a channel neighrborhood matrix

    Parameters
    ----------
    geom: np.array
        Array with the cartesian coordinates for the channels

    radius: float
        Maximum radius for the channels to be considered neighbors
    """
    return (squareform(pdist(geom)) <= radius)


def n_steps_neigh_channels(neighbors, steps):
    """Compute a n-steps channel neighrborhood matrix
    """
    C = neighbors.shape[0]
    neighChan_output = np.eye(C, dtype='bool')

    for j in range(steps):
        for c in range(C):
            neighChan_output[c][
                np.sum(neighbors[neighChan_output[c]], axis=0).astype('bool')] = 1

    return neighChan_output


def make_channel_groups(n_channels, neighbors, geom):
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


def order_channels_by_distance(refc, channels, geom):
    coord_main = geom[refc]
    coord_others = geom[channels]
    idx = np.argsort(np.sum(np.square(coord_others - coord_main), axis=1))

    return channels[idx], idx
