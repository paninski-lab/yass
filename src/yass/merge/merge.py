import os
import numpy as np
import tqdm
import parmap
import scipy

from diptest import diptest as dp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from yass.template import shift_chans, align_get_shifts_with_ref
from yass.merge.correlograms_phy import compute_correlogram_v2
from yass.merge.notch import notch_finder

class TemplateMerge(object):

    def __init__(self, 
                 save_dir,
                 raw_data,
                 reader,
                 fname_templates,
                 fnames_input,
                 multi_processing=False,
                 n_processors=1):                               
        """
        parameters:
        -----------
        templates: numpy.ndarray shape (K, C, T)
            templates
        spike_train: numpy.ndarray shape (N, 2)
            First column is times and second is cluster id.
        """
        
        # 
        self.save_dir = save_dir
        self.raw_data = raw_data
        self.reader = reader

        self.fnames_input = fnames_input
        self.multi_processing = multi_processing
        self.n_processors = n_processors

        templates = np.load(fname_templates)
        self.n_unit, self.spike_size, _ = templates.shape

        # proposed merge pairs
        fname_candidates = os.path.join(self.save_dir,
                                        'merge_candidates.npy')
        if os.path.exists(fname_candidates):
            self.merge_candidates = np.load(fname_candidates)

        else:
            # distance among templates
            affinity_matrix = template_dist_linear_align(templates)

            # Distance metric with diagonal set to large numbers
            dist_mat = np.zeros_like(affinity_matrix)
            for i in range(self.n_unit):
                dist_mat[i, i] = 1e4
            dist_mat += affinity_matrix

            # Template norms
            norms = np.linalg.norm(
                templates.reshape(self.n_unit, -1), axis=1)

            # relative difference
            dist_norm_ratio = dist_mat / np.maximum(norms[np.newaxis],
                                                    norms[:, np.newaxis])

            # ptp of each unit
            ptps = templates.ptp(1).max(1)

            self.find_merge_candidates(dist_norm_ratio,
                                       ptps)
            np.save(fname_candidates, self.merge_candidates)

    def find_merge_candidates(self, dist_norm_ratio, ptps):

        # units need to be close to each other
        idx1 = dist_norm_ratio < 0.5

        # ptp of both units need to be bigger than 4
        smaller_ptps = np.minimum(ptps[np.newaxis],
                                  ptps[:, np.newaxis])
        idx2 = smaller_ptps > 4
        units_1, units_2 = np.where(
            np.logical_and(idx1, idx2))

        # get unique pairs only for comparison
        unique_pairs = []
        for x,y in zip(units_1, units_2):
            if [x, y] not in unique_pairs and [y, x] not in unique_pairs:
                unique_pairs.append([x, y])

        # if there is a dip, do further test
        self.merge_candidates = []
        for pair in unique_pairs:

            # get spike times
            unit1, unit2 = pair
            spt1 = np.load(self.fnames_input[unit1])['spike_times']
            spt2 = np.load(self.fnames_input[unit2])['spike_times']
            
            # compute xcorr
            xcor = compute_correlogram_v2(spt1, spt2)
            if xcor.shape[0] == 1:
                xcor = xcor[0,0]
            elif xcor.shape[0] > 1:
                xcor = xcor[1,0]        

            # check for notch and if there, add it
            notch, pval1, pval2 = notch_finder(xcor)
            if pval1 < 0.05:
                self.merge_candidates.append(pair)

    def get_merge_pairs(self):
        ''' Run all pairs of merge candidates through the l2 feature computation        
        '''

        fname = os.path.join(self.save_dir,
                             'merge_pairs.npy')

        # Cat: TODO: does python2 like list(zip())?
        if not os.path.exists(fname):
            if self.multi_processing:
                merge_list = parmap.map(self.merge_templates_parallel, 
                             self.merge_candidates,
                             processes=self.n_processors,
                             pm_pbar=True)
            # single core version
            else:
                merge_list = []
                for pair in self.merge_candidates:
                    merge_list.append(self.merge_templates_parallel(pair))

            merge_pairs = []
            for j in range(len(merge_list)):
                if merge_list[j]:
                    merge_pairs.append(self.merge_candidates[j])
            np.save(fname, merge_pairs)
        else:
            merge_pairs = np.load(fname)

        print(merge_pairs)
        return merge_pairs

    def merge_templates_parallel(self, pair):
        """Whether to merge two templates or not.
        """

        # Cat: TODO: read from CONFIG
        threshold=0.95

        unit1, unit2 = pair

        fname_out = os.path.join(
            self.save_dir, 
            'l2features_{}_{}.npz'.format(unit1, unit2))

        if os.path.exists(fname_out):
            return np.load(fname_out)['merge']

        else:
            # get l2 features
            l2_features, spike_ids = self.get_l2_features(unit1, unit2)
            # get p value for dip test
            dp_val, _ = test_unimodality(l2_features, spike_ids)
            merge = dp_val > threshold

            np.savez(fname_out,
                     merge=merge,
                     spike_ids=spike_ids,
                     l2_features=l2_features,
                     dp_val=dp_val)

            return merge

    def get_l2_features(self, unit1, unit2, n_samples=2000):

        # get necessary data
        unit1_data = np.load(self.fnames_input[unit1])
        unit2_data = np.load(self.fnames_input[unit2])

        # get spike times
        spt1 = unit1_data['spike_times']
        spt2 = unit2_data['spike_times']

        # subsample
        if len(spt1) + len(spt2) > n_samples:
            ratio = len(spt1)/float(len(spt1)+len(spt2))

            n_samples1 = int(n_samples*ratio)
            n_samples2 = n_samples - n_samples1

            spt1_idx = np.random.choice(
                np.arange(len(spt1)),
                n_samples1, False)
            spt2_idx = np.random.choice(
                np.arange(len(spt2)),
                n_samples2, False)

        else:
            spt1_idx = np.arange(len(spt1))
            spt2_idx = np.arange(len(spt2))

        spt1 = spt1[spt1_idx]
        spt2 = spt2[spt2_idx]

        # get waveforms
        if self.raw_data:
            wfs1, _ = self.reader.read_waveforms(
                spt1, self.spike_size)

            wfs2, _ = self.reader.read_waveforms(
                spt1, self.spike_size)
        else:
            up_ids1 = unit1_data['up_ids'][spt1_idx]
            up_templates1 = unit1_data['up_templates']
            wfs1, skipped_idx1 = self.reader.read_clean_waveforms(
                spt1, up_ids1, up_templates1, self.spike_size)
            if len(skipped_idx1) > 0:
                spt1 = np.delete(spt1, skipped_idx1)

            up_ids2 = unit2_data['up_ids'][spt2_idx]
            up_templates2 = unit2_data['up_templates']
            wfs2, skipped_idx2 = self.reader.read_clean_waveforms(
                spt2, up_ids2, up_templates2, self.spike_size)
            if len(skipped_idx2) > 0:
                spt2 = np.delete(spt2, skipped_idx2)

        # assignment
        spike_ids = np.append(
            np.zeros(len(spt1), 'int32'),
            np.ones(len(spt2), 'int32'),
            axis=0)

        # templates
        template1 = unit1_data['template'][np.newaxis]
        template2 = unit2_data['template'][np.newaxis]

        l2_features = template_spike_dist_linear_align(
            np.concatenate((template1, template2), axis=0),
            np.concatenate((wfs1, wfs2), axis=0))

        return l2_features.T, spike_ids

def test_unimodality(pca_wf, assignment, max_spikes = 10000):

    '''
    Parameters
    ----------
    pca_wf:  pca projected data
    ssignment:  spike assignments
    max_spikes: optional
    '''

    #n_samples = np.max(np.unique(assignment, return_counts=True)[1])

    # compute diptest metric on current assignment+LDA

    
    ## find indexes of data
    #idx1 = np.where(assignment==0)[0]
    #idx2 = np.where(assignment==1)[0]
    #min_spikes = min(idx1.shape, idx2.shape)[0]

    # limit size difference between clusters to maximum of 5 times
    #ratio = 1
    #idx1=idx1[:min_spikes*ratio][:max_spikes]
    #idx2=idx2[:min_spikes*ratio][:max_spikes]

    #idx_total = np.concatenate((idx1,idx2))
    ## run LDA on remaining data
    lda = LDA(n_components = 1)
    #print (pca_wf[idx_total].shape, assignment[idx_total].shape) 
    #trans = lda.fit_transform(pca_wf[idx_total], assignment[idx_total])
    trans = lda.fit_transform(pca_wf, assignment).ravel()
    _, n_spikes = np.unique(assignment, return_counts=True)
    id_big = np.argmax(n_spikes)
    id_small = np.argmin(n_spikes)
    n_diff = n_spikes[id_big] - n_spikes[id_small]

    if n_diff > 0:
        repeat = int(np.ceil(np.max(n_spikes)/np.min(n_spikes)))
        idx_big = np.where(assignment == id_big)[0]
        pvals = np.zeros(repeat)
        for j in range(repeat):
            idx_remove = np.random.choice(idx_big, n_diff, replace=False)
            pvals[j] = dp(np.delete(trans, idx_remove))[1]
    else:
        pvals = [dp(trans)[1]]
    ## also compute gaussanity of distributions
    ## first pick the number of bins; this metric is somewhat sensitive to this
    # Cat: TODO number of bins is dynamically set; need to work on this
    #n_bins = int(np.log(n_samples)*3)
    #y1 = np.histogram(trans, bins = n_bins)
    #normtest = stats.normaltest(y1[0])

    return np.median(pvals), trans#, assignment[idx_total]#, normtest[1]

def template_spike_dist_linear_align(templates, spikes, vis_ptp=2.):
    """compares the templates and spikes.

    parameters:
    -----------
    templates: numpy.array shape (K, T, C)
    spikes: numpy.array shape (M, T, C)
    jitter: int
        Align jitter amount between the templates and the spikes.
    upsample int
        Upsample rate of the templates and spikes.
    """

    # get ref template
    max_idx = templates.ptp(1).max(1).argmax(0)
    ref_template = templates[max_idx]
    max_chan = ref_template.ptp(0).argmax(0)
    ref_template = ref_template[:, max_chan]

    # align templates on max channel
    best_shifts = align_get_shifts_with_ref(
        templates[:, :, max_chan],
        ref_template, nshifts = 21)
    templates = shift_chans(templates, best_shifts)

    # align all spikes on max channel
    best_shifts = align_get_shifts_with_ref(
        spikes[:,:,max_chan],
        ref_template, nshifts = 21)
    spikes = shift_chans(spikes, best_shifts)
    
    n_unit = templates.shape[0]
    n_spikes = spikes.shape[0]

    # get visible channels
    vis_chan = np.where(templates.ptp(1).max(0) >= vis_ptp)[0]
    templates = templates[:, :, vis_chan].reshape(n_unit, -1)
    spikes = spikes[:, :, vis_chan].reshape(n_spikes, -1)

    # get a subset of locations with maximal difference
    if templates.shape[0] == 1:
        # single unit: all timepoints
        idx = np.arange(templates.shape[1])
    elif templates.shape[0] == 2:
        # two units:
        # get difference
        diffs = np.abs(np.diff(templates, axis=0)[0])
        # points with large difference
        idx = np.where(diffs > 1.5)[0]
        min_diff_points = 5
        if len(idx) < 5:
            idx = np.argsort(diffs)[-min_diff_points:]
    else:
        # more than two units:
        # diff is mean diff to the largest unit
        diffs = np.mean(np.abs(
            templates-templates[max_idx][None]), axis=0)
        
        idx = np.where(diffs > 1.5)[0]
        min_diff_points = 5
        if len(idx) < 5:
            idx = np.argsort(diffs)[-min_diff_points:]
                       
    templates = templates[:, idx]
    spikes = spikes[:, idx]

    dist = scipy.spatial.distance.cdist(templates, spikes)

    return dist

def template_dist_linear_align(templates, distance=None, units=None, max_shift=5, step=0.5):

    K, R, C = templates.shape

    shifts = np.arange(-max_shift,max_shift+step,step)
    ptps = templates.ptp(1)
    max_chans = np.argmax(ptps, 1)

    shifted_templates = np.zeros((len(shifts), K, R, C))
    for ii, s in enumerate(shifts):
        shifted_templates[ii] = shift_chans(templates, np.ones(K)*s)

    if distance is None:
        distance = np.ones((K, K))*1e4
    if units is None:
        units = np.arange(K)

    for k in units:
        candidates = np.abs(ptps[:, max_chans[k]] - ptps[k, max_chans[k]])/ptps[k,max_chans[k]] < 0.5
        
        dist = np.min(np.sum(np.square(
            templates[k][np.newaxis, np.newaxis] - shifted_templates[:, candidates]),
                             axis=(2,3)), 0)
        dist = np.sqrt(dist)
        distance[k, candidates] = dist
        distance[candidates, k] = dist
        
    return distance    
