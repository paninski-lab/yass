import numpy as np
import scipy

from yass.cluster.cluster import align_get_shifts_with_ref, shift_chans


def two_templates_dist_linear_align(templates1, templates2):
    
    templates = np.concatenate((templates1, templates2), axis=0)
    temps_max = template_on_max_chan(templates)

    ref_template = temps_max[temps_max.ptp(1).argmax()]

    best_shifts = align_get_shifts_with_ref(temps_max, ref_template)    
    templates_aligned = shift_chans(templates, best_shifts)

    idx1 = np.zeros(templates.shape[0], 'bool')
    idx1[:templates1.shape[0]] = 1
    templates_aligned1 = templates_aligned[idx1].reshape([templates1.shape[0], -1])
    templates_aligned2 = templates_aligned[~idx1].reshape([templates2.shape[0], -1])

    dist = scipy.spatial.distance.cdist(templates_aligned1, templates_aligned2)

    return dist

def template_on_max_chan(templates):
    
    max_chans = templates.ptp(1).argmax(1)
    temps = []
    for k in range(max_chans.shape[0]):
        temps.append(templates[k, :, max_chans[k]])
    temps = np.vstack(temps)

    return temps
    
def compute_neighbours2(templates1, templates2, n_neighbours=3):
    
    dist = two_templates_dist_linear_align(templates1, templates2)

    nearest_units1 = np.zeros((dist.shape[0], n_neighbours), 'int32')
    for k in range(dist.shape[0]):
        nearest_units1[k] = np.argsort(dist[k])[:n_neighbours]
        
    nearest_units2 = np.zeros((dist.shape[1], n_neighbours), 'int32')
    for k in range(dist.shape[1]):
        nearest_units2[k] = np.argsort(dist[:, k])[:n_neighbours]
        
    return nearest_units1, nearest_units2

def compute_neighbours_rf2(STAs1, STAs2, n_neighbours=3):
    STAs_th1 = pre_process_sta(STAs1)
    STAs_th2 = pre_process_sta(STAs2)

    norms1 = np.linalg.norm(STAs_th1.T, axis=0)[:, np.newaxis]
    norms2 = np.linalg.norm(STAs_th2.T, axis=0)[:, np.newaxis]
    cos = np.matmul(STAs_th1, STAs_th2.T)/np.matmul(norms1, norms2.T)
    nearest_units_rf1 = np.zeros((cos.shape[0], n_neighbours), 'int32')
    for k in range(cos.shape[0]):
        nearest_units_rf1[k] = np.argsort(cos[k])[-n_neighbours:][::-1]
        
    nearest_units_rf2 = np.zeros((cos.shape[1], n_neighbours), 'int32')
    for k in range(cos.shape[1]):
        nearest_units_rf2[k] = np.argsort(cos[:, k])[-n_neighbours:][::-1]

    return nearest_units_rf1, nearest_units_rf2
    
def pre_process_sta(STAs, th=0.05):

    STAs_th = np.copy(STAs)
    STAs_th[np.abs(STAs_th) < th] = 0
    STAs_th = STAs_th.reshape(STAs.shape[0], -1)

    return STAs_th

def combine_two_spike_train(templates1, templates2, spike_train1, spike_train2):
    
    K1 = templates1.shape[2]
    K2 = templates2.shape[2]
    
    templates = np.concatenate((templates1, templates2), axis=2)
    

    spike_train2_new_id = np.copy(spike_train2)    
    new_id2 = np.arange(K2) + K1
    for j in range(spike_train2.shape[0]):
        spike_train2_new_id[j,1] = new_id2[spike_train2[j,1]]
    
    spike_train = np.concatenate((spike_train1, spike_train2_new_id))
    spike_train = spike_train[np.argsort(spike_train[:,0])]
    
    return templates, spike_train


def combine_two_rf(STAs1, STAs2, Gaussian_params1, Gaussian_params2):

    STAs = np.concatenate((STAs1, STAs2), axis=0)
    Gaussian_params = np.concatenate((Gaussian_params1, Gaussian_params2), axis=0)
    
    return STAs, Gaussian_params

