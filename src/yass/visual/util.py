import numpy as np
import scipy

from yass.cluster.cluster import align_get_shifts_with_ref, shift_chans


def two_templates_dist_linear_align(templates1, templates2, max_shift=5, step=0.5):

    K1, R, C = templates1.shape
    K2 = templates2.shape[0]

    shifts = np.arange(-max_shift,max_shift+step,step)
    ptps1 = templates1.ptp(1)
    max_chans1 = np.argmax(ptps1, 1)
    ptps2 = templates2.ptp(1)
    max_chans2 = np.argmax(ptps2, 1)

    shifted_templates = np.zeros((len(shifts), K2, R, C))
    for ii, s in enumerate(shifts):
        shifted_templates[ii] = shift_chans(templates2, np.ones(K2)*s)

    distance = np.ones((K1, K2))*1e4
    for k in range(K1):
        candidates = np.abs(ptps2[:, max_chans1[k]] - ptps1[k, max_chans1[k]])/ptps1[k,max_chans1[k]] < 0.5
        
        dist = np.min(np.sum(np.square(
            templates1[k][np.newaxis, np.newaxis] - shifted_templates[:, candidates]),
                             axis=(2,3)), 0)
        dist = np.sqrt(dist)
        distance[k, candidates] = dist
        
    return distance
    
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
    cos[np.isnan(cos)] = 0

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


def ttest_notch(greater, smaller, data, sig = 0.05):
    N = data.sum()
    
    ngreater, nsmaller = data[greater], data[smaller]
    
    
    pg = ngreater/N
    pg = pg.sum()/pg.size
    siggsq = pg*(1-pg)/N
    siggsq = siggsq.sum()/siggsq.size**2    
    
    ps  = nsmaller/N
    ps = ps.sum()/ps.size
    siggsq = sigssq = ps*(1-ps)/N
    sigssq = sigssq.sum()/sigssq.size**2
    
    tstat = (pg - ps)/np.sqrt(siggsq + sigssq)
    df = (siggsq + sigssq)**2/((siggsq)**2/(N-1) + (sigssq)**2/(N-1) )
    
    if tstat > 0 and (1 - t.cdf(tstat, df)) < sig:
        return True
    else:
        return False

def findnotchpairs(correlograms, mc, CONFIG):
    
    nunits = correlograms.shape[0]
    
    baseline = np.arange(correlograms.shape[2]//2-29, correlograms.shape[2]//2-8)
    baseline = np.concatenate([baseline, np.arange(correlograms.shape[2]//2+8, correlograms.shape[2]//2+28)])
    centrebins = [correlograms.shape[2]//2]
    offbins = np.arange(correlograms.shape[2]//2-7, correlograms.shape[2]//2-2)
    offbins = np.concatenate([offbins, np.arange(correlograms.shape[2]//2+2, correlograms.shape[2]//2+7)])
    
    notchpairs = []
    
    
    for unit1 in range(nunits):
        idx = np.in1d(mc, np.arange(49)[CONFIG.neigh_channels[mc[unit1]]])
        closeunits = np.where(idx2)[0]
        notchpairs.append([])
        for unit2 in closeunits:
            if ttest_notch(baseline, centrebins, correlograms[unit1, unit2])[0]:
                if ttest_notch(offbins, baseline, correlograms[unit1, unit2])[0]:
                    notchpairs[unit1].append(unit2)
                    
    return notchpairs
                   
 
            
        
        
    
    
