"""
The :mod:`yass.cluster` module implements spike clustering algorithms,
the :mod:`yass.cluster.legacy` implements two old clustering methods
`location` and `neigh_channels` they are no longer used and will be removed
soon when we add the new clustering algorithm
"""
from yass.cluster.run import run

__all__ = ['run']
