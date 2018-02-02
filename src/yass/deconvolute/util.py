from scipy.interpolate import interp1d
import numpy as np

def upsample_templates(template, n_shifts = 5):
    """
    get n_shifts number of shifted templates.
    The amount of shift is evenly distributed from -0.5 to 0.5 timebin.

    Parameters
    ----------

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        A 2D array for collided spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../examples/deconvolute.py
    """
    n_channels, n_times = template.shape
    x = np.linspace(0, n_times-1, num=n_times, endpoint=True)
    ff = interp1d(x, template, kind='cubic')
    
    shifts = np.linspace(-0.5, 0.5, n_shifts, endpoint=False)
    shifted_templates = np.zeros((n_shifts, n_channels, n_times))
    for j in range(n_shifts):
        xnew = x - shifts[j]
        idx_good = np.logical_and(xnew >= 0, xnew <= n_times-1)
        shifted_templates[j][:,idx_good] = ff(xnew[idx_good])
    
    return shifted_templates
    
def make_spt_list(spike_index, C):
    answer = [None]*C
    
    for c in range(C):
        answer[c] = spike_index[spike_index[:,1]==c,0] 
    
    return answer

def get_longer_spt_list(spt, n_explore):

    spt = np.sort(spt)
    all_spikes = np.reshape(np.add(spt[:,np.newaxis],np.arange(-n_explore,n_explore+1)[np.newaxis,:]),-1)
    spt_long = np.sort(np.unique(all_spikes))

    return spt_long


