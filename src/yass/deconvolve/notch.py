import numpy as np
import scipy
from scipy.stats import t

def ztest_notch(greater, smaller, data):
    ngreater, nsmaller = data[greater].sum(), data[smaller].sum()
    n_bins_greater, n_bins_smaller = len(greater), len(smaller)
    ntotal = ngreater + nsmaller

    p_hat = float(ngreater)/ntotal
    p_hat_var = p_hat*(1-p_hat)/ntotal

    p_null = float(n_bins_greater)/(n_bins_greater+n_bins_smaller)

    z = (p_hat - p_null)/np.sqrt(p_hat_var)

    pval = 1 - norm.cdf(z)
    
    return pval

def ttest_notch(greater, smaller, data):
    N = data.sum()
    
    ngreater, nsmaller = data[greater], data[smaller]
    
    
    pg = ngreater/N
    siggsq = pg*(1-pg)/N
    pg = pg.sum()/pg.size
    siggsq = siggsq.sum()/siggsq.size**2
    
#     print(pg, siggsq)
    
    ps  = nsmaller/N
    sigssq = ps*(1-ps)/N
    ps = ps.sum()/ps.size
    sigssq = sigssq.sum()/sigssq.size**2
    
    tstat = (pg - ps)/np.sqrt(siggsq + sigssq)
    df = (siggsq + sigssq)**2/((siggsq)**2/(N-1) + (sigssq)**2/(N-1) )
    
    
#     print(pg, ps, siggsq, sigssq)
    return (1 - t.cdf(tstat, df))

def findnotchpairs(correlograms, mc, CONFIG):
    
    nunits = correlograms.shape[0]
    numbins = correlograms.shape[2]
    
    baseline = np.arange(0, 20)
    baseline = np.concatenate([baseline, np.arange(numbins - 20, numbins)])
    centrebins = [numbins//2-1, numbins//2, numbins//2+1]
    offbins = np.arange(correlograms.shape[2]//2-13, correlograms.shape[2]//2-3)
    offbins = np.concatenate([offbins, np.arange(numbins//2+3, numbins//2+13)])

#     baseline = np.arange(correlograms.shape[2]//2-29, correlograms.shape[2]//2-8)
#     baseline = np.concatenate([baseline, np.arange(correlograms.shape[2]//2+8, correlograms.shape[2]//2+28)])
#     centrebins = [correlograms.shape[2]//2]
#     offbins = np.arange(correlograms.shape[2]//2-7, correlograms.shape[2]//2-2)
#     offbins = np.concatenate([offbins, np.arange(correlograms.shape[2]//2+2, correlograms.shape[2]//2+7)])
    
    notchpairs = []
    
    
    for unit1 in range(nunits):
        idx = np.in1d(mc, np.arange(49)[CONFIG.neigh_channels[mc[unit1]]])
        closeunits = np.where(idx)[0]
        notchpairs.append([])
        for unit2 in closeunits:
#             print(unit1, unit2)
            if unit2 <= unit1:
                continue
            if ttest_notch(baseline, centrebins, correlograms[unit1, unit2]):
                if ttest_notch(offbins, baseline, correlograms[unit1, unit2]):
                    notchpairs[unit1].append(unit2)
                    
    return notchpairs

    
def notch_finder(correlogram, sig=0.05):
    
    numbins = len(correlogram)
    
    baseline = np.arange(0, 20)
    baseline = np.concatenate([baseline, np.arange(numbins - 20, numbins)])
    centrebins = [numbins//2-1, numbins//2, numbins//2+1]
    offbins = np.arange(numbins//2-13, numbins//2-3)
    offbins = np.concatenate([offbins, np.arange(numbins//2+3, numbins//2+13)])


    pval1 = ttest_notch(baseline, centrebins, correlogram)
    pval2 = ttest_notch(offbins, baseline, correlogram)
    
    if pval1 < sig and pval2 < sig:
        notch = True
    else:
        notch = False

    return notch, pval1, pval2