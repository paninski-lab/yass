import numpy as np

from ..geometry import order_channels_by_distance

def noise_cov(path, dtype, batch_size, n_channels, neighbors, geom, temporal_size):
    wfile = open(path, 'rb')
    dsize = np.dtype(dtype).itemsize
    flattenedLength = dsize*batch_size*n_channels
    
    wfile.seek(0)
    rec = wfile.read(flattenedLength)
    rec = np.fromstring(rec, dtype=dtype)
    rec = np.reshape(rec, (-1, n_channels))

    c_ref = np.argmax(np.sum(neighbors,0))
    ch_idx = np.where(neighbors[c_ref])[0]
    ch_idx, temp = order_channels_by_distance(c_ref, ch_idx, geom)
    rec = rec[:,ch_idx]
    wfile.close()

    T,C = rec.shape
    idxNoise = np.zeros((T,C))

    R = int((temporal_size-1)/2)
    for c in range(C):
        idx_temp = np.where(rec[:,c] > 3)[0]
        for j in range(-R,R+1):
            idx_temp2 = idx_temp +j
            idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >=0, idx_temp2 <T)]
            rec[idx_temp2,c] = np.nan
        idxNoise_temp = (rec[:,c] == rec[:,c])
        rec[:,c] = rec[:,c]/np.nanstd(rec[:,c])

        rec[~idxNoise_temp,c] = 0
        idxNoise[idxNoise_temp,c] = 1

    spatial_cov = np.divide(np.matmul(rec.T,rec), np.matmul(idxNoise.T,idxNoise))

    w, v = np.linalg.eig(spatial_cov)
    spatial_SIG = np.matmul(np.matmul(v,np.diag(np.sqrt(w))),v.T)
    
    spatial_whitener = np.matmul(np.matmul(v,np.diag(1/np.sqrt(w))),v.T)
    rec = np.matmul(rec,spatial_whitener)

    noise_wf = np.zeros((1000,temporal_size))
    count = 0
    while count < 1000:
        tt = np.random.randint(T-temporal_size)
        cc = np.random.randint(C)
        temp = rec[tt:(tt+temporal_size),cc]
        temp_idxnoise = idxNoise[tt:(tt+temporal_size),cc]
        if np.sum(temp_idxnoise == 0) == 0:
            noise_wf[count] = temp
            count += 1

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v,np.diag(np.sqrt(w))),v.T)
    
    return spatial_SIG, temporal_SIG
    
def align_templates(t1,t2):
    temp = np.convolve(t1,np.flip(t2,0),'full')
    shift = np.argmax(temp)
    return shift - t1.shape[0] + 1        