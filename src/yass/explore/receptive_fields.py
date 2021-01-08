# Class to compute receptive fields based on sta computation

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy import signal
from scipy import stats
from scipy.signal import argrelmax
from scipy.spatial import cKDTree
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from diptest import diptest as dp
from sklearn.cluster import AgglomerativeClustering

from yass.explore.explorers import RecordingExplorer
from yass import mfm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

colors = [
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink']

class RF(object):
    """Class for doing clustering."""

    def __init__(self, data_in):
            
        """Sets up the cluster class for each core
        Parameters: ...
              
        """
        
        # load data and check if prev completed
        if self.load_data(data_in):  return
        
        # run generational clustering on channel
        self.cluster(self.starting_indexes, self.starting_gen, self.triageflag)
                         
        # save clusters and make plots
        self.finish_clustering()
        
        
    def cluster(self, current_indexes, gen, triage_flag,  
                active_chans=None):

        pass





