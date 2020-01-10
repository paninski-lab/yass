import os
import numpy as np
import tqdm
import parmap
import scipy
import logging

from diptest import diptest as dp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from yass.template import shift_chans, align_get_shifts_with_ref
from yass.correlograms_phy import compute_correlogram_v2
from yass.merge.notch import notch_finder

class TemplateMerge(object):

    def __init__(self, 
                 save_dir,
                 reader_residual,
                 fname_templates,
                 fname_spike_train,
                 fname_shifts,
                 fname_scales,
                 fname_soft_assignment,
                 fname_spatial_cov,
                 fname_temporal_cov,
                 geom,
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
        
        logger = logging.getLogger(__name__)

        # 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.save_dir = save_dir
        self.reader_residual = reader_residual

        # templates
        self.templates = np.load(fname_templates)
        self.n_units, self.spike_size, self.n_channels = self.templates.shape

        # compute ptp
        self.ptps = self.templates.ptp(1)

        # spike train, shift, soft assignment
        self.spike_train = np.load(fname_spike_train)
        self.shifts = np.load(fname_shifts)
        self.scales = np.load(fname_scales)
        self.soft_assignment = np.load(fname_soft_assignment)
        
        # remove edge spikes
        idx_in = np.logical_and(
            self.spike_train[:, 0] > self.spike_size,
            self.spike_train[:, 0] < self.reader_residual.rec_len - self.spike_size)

        self.spike_train = self.spike_train[idx_in]
        self.shifts = self.shifts[idx_in]
        self.scales = self.scales[idx_in]
        self.soft_assignment = self.soft_assignment[idx_in]

        self.multi_processing = multi_processing
        self.n_processors = n_processors

        logger.info('{} units in'.format(self.n_units))

        # effective number of spikes
        self.compute_n_spikes_soft()

        # proposed merge pairs
        fname_candidates = os.path.join(self.save_dir,
                                        'merge_candidates.npy')
        if os.path.exists(fname_candidates):
            self.merge_candidates = np.load(fname_candidates)

        else:
            logger.info("finding candidates using ptp")
            self.find_merge_candidates()
            np.save(fname_candidates, self.merge_candidates)
            
        # load cov
        self.temporal_cov = np.load(fname_temporal_cov)
        self.spatial_cov = np.load(fname_spatial_cov)
        self.get_temproal_whitener()
        self.geom = geom

    def compute_n_spikes_soft(self):

        n_spikes_soft = np.zeros(self.n_units)
        for j in range(self.spike_train.shape[0]):
            n_spikes_soft[self.spike_train[j, 1]] += self.soft_assignment[j]
        self.n_spikes_soft = n_spikes_soft.astype('int32')

    def get_temproal_whitener(self):
        
        w, v = np.linalg.eig(self.temporal_cov)
        self.temporal_whitener = np.matmul(
            np.matmul(v, np.diag(1/np.sqrt(w))), v.T)
        
    def get_spatial_whitener(self, vis_chan):
        
        chan_dist = squareform(pdist(self.geom[vis_chan]))
        spat_cov = np.zeros((len(vis_chan), len(vis_chan)))
        for ii, c in enumerate(self.spatial_cov[:,1]):
            spat_cov[chan_dist == c] = self.spatial_cov[ii, 0]

        w, v = np.linalg.eig(spat_cov)
        w[w<=0] = 1E-10
        inv_half_spat_cov = np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)

        return inv_half_spat_cov

    def find_merge_candidates(self):

        # find candidates by comparing ptps
        # if distance of ptps are close, then they need to be checked.

        # mask out small ptp
        self.ptps[self.ptps < 1] = 0

        # distances of ptps
        dist_mat = np.square(pairwise_distances(self.ptps))

        # compute distance relative to the norm of ptps
        norms = np.square(np.linalg.norm(self.ptps, axis=1))

        # exclude comparing to self
        for i in range(self.n_units):
            dist_mat[i, i] = np.max(norms)*100

        dist_norm_ratio = dist_mat / np.maximum(
            norms[np.newaxis], norms[:, np.newaxis])
        # units need to be close to each other
        idx1 = dist_norm_ratio < 0.5

        # ptp of both units need to be bigger than 3
        ptp_max = self.ptps.max(1)
        smaller_ptps = np.minimum(ptp_max[np.newaxis],
                                  ptp_max[:, np.newaxis])
        idx2 = smaller_ptps > 3

        # expect to have at least 10 spikes
        smaller_n_spikes = np.minimum(self.n_spikes_soft[None],
                                      self.n_spikes_soft[:, None])
        idx3 = smaller_n_spikes > 10

        units_1, units_2 = np.where(np.logical_and(
            np.logical_and(idx1, idx2), idx3))
        
        # get unique pairs only for comparison
        unique_pairs = []
        for x, y in zip(units_1, units_2):
            if [x, y] not in unique_pairs and [y, x] not in unique_pairs:
                ratio = self.n_spikes_soft[x]/self.n_spikes_soft[y]
                # if the ratio is too bad, ignore it because it will
                # always try to merge
                if ((ratio > 1/20 and ratio < 20) or 
                    (ptp_max[x] > 10 and ptp_max[y] > 10)):
                    unique_pairs.append([x, y])

        self.merge_candidates = unique_pairs


    def xcor_notch_test(self, pairs, templates):

        # if there is a dip, do further test
        merge_candidates = []

        for pair in pairs:

            # get spike times
            unit1, unit2 = pair
            spt1 = np.load(self.fnames_input[unit1])['spike_times']
            spt2 = np.load(self.fnames_input[unit2])['spike_times']

            temps = templates[pair]
            mc = temps.ptp(1).max(0).argmax()
            shift = np.diff(temps[:, :, mc].argmin(1))[0]
            spt1 -= shift

            if len(spt1) > 1 and len(spt2) > 1:
                # compute xcorr
                xcor = compute_correlogram_v2(spt1, spt2)
                if xcor.shape[0] == 1:
                    xcor = xcor[0,0]
                elif xcor.shape[0] > 1:
                    xcor = xcor[1,0]

                # if there are not enough synchronous spike
                # activities between two units, add it
                centerbins = np.arange(len(xcor)//2-1, len(xcor)//2+2)
                if xcor[centerbins].min() < 10:
                    merge_candidates.append(pair)
                else:
                    # check for notch and if there, add it
                    notch, pval1 = notch_finder(xcor)
                    if notch:
                        merge_candidates.append(pair)

        return merge_candidates

    def find_merge_candidates_orig(self, dist_norm_ratio, ptps):

        # units need to be close to each other
        idx1 = dist_norm_ratio < 0.5

        # ptp of both units need to be bigger than 4
        smaller_ptps = np.minimum(ptps[np.newaxis],
                                  ptps[:, np.newaxis])
        idx2 = smaller_ptps > 5
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
            
            if len(spt1) > 1 and len(spt2) > 1:
                # compute xcorr
                xcor = compute_correlogram_v2(spt1, spt2)
                if xcor.shape[0] == 1:
                    xcor = xcor[0,0]
                elif xcor.shape[0] > 1:
                    xcor = xcor[1,0]

                # check for notch and if there, add it
                notch, pval1 = notch_finder(xcor)
                if notch:
                    self.merge_candidates.append(pair)

    def get_merge_pairs(self):
        ''' Run all pairs of merge candidates through the l2 feature computation        
        '''

        fname = os.path.join(self.save_dir,
                             'merge_pairs.npy')

        if not os.path.exists(fname):

            if self.multi_processing:
                # break the list of pairs
                merge_candidates_partition = []
                for j in range(self.n_processors):
                    merge_candidates_partition.append(
                        self.merge_candidates[
                            slice(j, len(self.merge_candidates), self.n_processors)])

                merge_pairs_ = parmap.map(
                    self.merge_templates_parallel, 
                    merge_candidates_partition,
                    pm_processes=self.n_processors,
                    pm_pbar=True)

                merge_pairs = []
                for pp in merge_pairs_:
                    if len(pp) > 0:
                        merge_pairs.append(pp)
                merge_pairs = np.concatenate(np.array(merge_pairs))
            # single core version
            else:
                merge_pairs = self.merge_templates_parallel(
                    self.merge_candidates)
                merge_pairs = np.array(merge_pairs)

            np.save(fname, merge_pairs)
        else:
            merge_pairs = np.load(fname)

        self.merge_pairs = merge_pairs

    def merge_templates_parallel(self, pairs):
        """Whether to merge two templates or not.
        """
        n_samples = 2000
        p_val_threshold = 0.9
        merge_pairs = []

        for pair in pairs:
            unit1, unit2 = pair

            fname_out = os.path.join(
                self.save_dir,
                'unit_{}_{}.npz'.format(unit1, unit2))

            if os.path.exists(fname_out):
                if np.load(fname_out)['merge']:
                    merge_pairs.append(pair)

            else:
                
                # get spikes times and soft assignment
                idx1 = self.spike_train[:, 1] == unit1
                spt1 = self.spike_train[idx1, 0]
                prob1 = self.soft_assignment[idx1]
                shift1 = self.shifts[idx1]
                scale1 = self.scales[idx1]
                n_spikes1 = self.n_spikes_soft[unit1]
                
                idx2 = self.spike_train[:, 1] == unit2
                spt2 = self.spike_train[idx2, 0]
                prob2 = self.soft_assignment[idx2]
                shift2 = self.shifts[idx2]
                scale2 = self.scales[idx2]
                n_spikes2 = self.n_spikes_soft[unit2]
                
                # randomly subsample
                if n_spikes1 + n_spikes2 > n_samples:
                    ratio1 = n_spikes1/float(n_spikes1+n_spikes2)
                    n_samples1 = np.min((int(n_samples*ratio1), n_spikes1))
                    n_samples2 = n_samples - n_samples1

                else:
                    n_samples1 = n_spikes1
                    n_samples2 = n_spikes2
                idx1_ = np.random.choice(len(spt1), n_samples1, replace=False,
                                         p=prob1/np.sum(prob1))
                idx2_ = np.random.choice(len(spt2), n_samples2, replace=False,
                                         p=prob2/np.sum(prob2))
                spt1 = spt1[idx1_]
                spt2 = spt2[idx2_]
                shift1 = shift1[idx1_]
                shift2 = shift2[idx2_]
                scale1 = scale1[idx1_]
                scale2 = scale2[idx2_]

                ptp_max = self.ptps[[unit1, unit2]].max(0)
                mc = ptp_max.argmax()
                vis_chan = np.where(ptp_max > 1)[0]

                # align two units
                shift_temp = (self.templates[unit2, :, mc].argmin() - 
                              self.templates[unit1, :, mc].argmin())
                spt2 += shift_temp
                
                # load residuals
                wfs1, skipped_idx1 = self.reader_residual.read_waveforms(
                    spt1, self.spike_size, vis_chan)
                spt1 = np.delete(spt1, skipped_idx1)
                shift1 = np.delete(shift1, skipped_idx1)
                scale1 = np.delete(scale1, skipped_idx1)
                
                wfs2, skipped_idx2 = self.reader_residual.read_waveforms(
                    spt2, self.spike_size, vis_chan)
                spt2 = np.delete(spt2, skipped_idx1)
                shift2 = np.delete(shift2, skipped_idx2)
                scale2 = np.delete(scale2, skipped_idx2)
                
                # align residuals
                wfs1 = shift_chans(wfs1, -shift1)
                wfs2 = shift_chans(wfs2, -shift2)

                # make clean waveforms
                wfs1 += scale1[:, None, None]*self.templates[[unit1], :, vis_chan].T
                if shift_temp > 0:
                    temp_2_shfted = self.templates[[unit2], shift_temp:, vis_chan].T
                    wfs2[:, :-shift_temp] += scale2[:, None, None]*temp_2_shfted
                elif shift_temp < 0:
                    temp_2_shfted = self.templates[[unit2], :shift_temp, vis_chan].T
                    wfs2[:, -shift_temp:] += scale2[:, None, None]*temp_2_shfted
                else:
                    wfs2 += scale2[:, None, None]*self.templates[[unit2],:,vis_chan].T

                
                # compute spatial covariance
                spatial_whitener = self.get_spatial_whitener(vis_chan)
                # whiten
                wfs1_w = np.matmul(wfs1, spatial_whitener)
                wfs2_w = np.matmul(wfs2, spatial_whitener)
                wfs1_w = np.matmul(wfs1_w.transpose(0,2,1),
                                  self.temporal_whitener).transpose(0,2,1)
                wfs2_w = np.matmul(wfs2_w.transpose(0,2,1),
                                  self.temporal_whitener).transpose(0,2,1)


                temp_diff_w = np.mean(wfs1_w, 0) - np.mean(wfs2_w,0)
                c_w = np.sum(0.5*(np.mean(wfs1_w, 0) + np.mean(wfs2_w,0))*temp_diff_w)
                dat1_w = np.sum(wfs1_w*temp_diff_w, (1,2))
                dat2_w = np.sum(wfs2_w*temp_diff_w, (1,2))
                dat_all = np.hstack((dat1_w, dat2_w))
                p_val = dp(dat_all)[1]

                if p_val > p_val_threshold:
                    merge = True
                else:
                    merge= False

                centers_dist = np.linalg.norm(temp_diff_w)


                if p_val > p_val_threshold:
                    merge = True
                else:
                    merge= False
                    
                centers_dist = np.linalg.norm(temp_diff_w)
                np.savez(fname_out,
                         merge=merge,
                         dat1_w=dat1_w,
                         dat2_w=dat2_w,
                         centers_dist=centers_dist,
                         p_val=p_val)

                if merge:
                    merge_pairs.append(pair)

        return merge_pairs

    def merge_templates_parallel_orig(self, pairs):
        """Whether to merge two templates or not.
        """

        merge_pairs = []

        for pair in pairs:
            unit1, unit2 = pair

            fname_out = os.path.join(
                self.save_dir, 
                'l2features_{}_{}.npz'.format(unit1, unit2))

            if os.path.exists(fname_out):
                if np.load(fname_out)['merge']:
                    merge_pairs.append(pair)

            else:
                # get l2 features
                l2_features, spike_ids = self.get_l2_features(unit1, unit2)

                # enough spikes from both need to present otherwise skip it
                if l2_features is not None:

                    if np.sum(np.abs(l2_features)) == 0:
                        print(unit1)
                        print(unit2)
                        raise ValueError("something is wrong")

                    # test if they need to be merged
                    (merge,
                     lda_prob,
                     dp_val) = test_merge(l2_features, spike_ids)

                    np.savez(fname_out,
                             merge=merge,
                             spike_ids=spike_ids,
                             l2_features=l2_features,
                             lda_prob=lda_prob,
                             dp_val=dp_val)
                             #lda_feat=lda_feat)
                else:
                    merge = False
                    np.savez(fname_out,
                             merge=merge,
                             spike_ids=spike_ids,
                             l2_features=l2_features,
                             lda_prob=None,
                             dp_val=None)
                             #lda_feat=None)

                if merge:
                    merge_pairs.append(pair)

        return merge_pairs


    def get_l2_features(self, unit1, unit2, n_samples=2000):

        idx1 = self.spike_train[:, 1] == unit1
        spt1 = self.spike_train[idx1, 0]
        prob1 = self.soft_assignment[idx1]

        idx2 = self.spike_train[:, 1] == unit2
        spt2 = self.spike_train[idx2, 0]
        prob2 = self.soft_assignment[idx2]

        if self.n_spikes_soft[unit1]+self.n_spikes_soft[unit2] > n_samples:
            ratio1 = self.n_spikes_soft[unit1]/float(self.n_spikes_soft[unit1]+self.n_spikes_soft[unit2])
            n_samples1 = np.min((int(n_samples*ratio1), self.n_spikes_soft[unit1]))
            n_samples2 = n_samples - n_samples1

        else:
            n_samples1 = self.n_spikes_soft[unit1]
            n_samples2 = self.n_spikes_soft[unit2]

        spt1 = np.random.choice(spt1, n_samples1, replace=False, p=prob1/np.sum(prob1))
        spt2 = np.random.choice(spt2, n_samples2, replace=False, p=prob2/np.sum(prob2))

        wfs1 = self.reader_residual.read_waveforms(spt1, self.spike_size)[0] + self.templates[unit1]
        wfs2 = self.reader_residual.read_waveforms(spt2, self.spike_size)[0] + self.templates[unit2]

        # if two templates are not aligned, get bigger window
        # and cut oneside to get shifted waveforms
        mc = self.templates[[unit1, unit2]].ptp(1).max(0).argmax()
        shift = wfs2[:,:,mc].mean(0).argmin() - wfs1[:,:,mc].mean(0).argmin()
        if np.abs(shift) > self.spike_size//4:
            return None, None

        if shift < 0:
            wfs1 = wfs1[:, -shift:]
            wfs2 = wfs2[:, :shift]
        elif shift > 0:
            wfs1 = wfs1[:, :-shift]
            wfs2 = wfs2[:, shift:]

        # assignment
        spike_ids = np.hstack((np.zeros(len(wfs1), 'int32'),
            np.ones(len(wfs2), 'int32')))

        # recompute templates using deconvolved spikes
        template1 = np.mean(wfs1, axis=0, keepdims=True)
        template2 = np.mean(wfs2, axis=0, keepdims=True)
        l2_features = template_spike_dist_linear_align(
            np.concatenate((template1, template2), axis=0),
            np.concatenate((wfs1, wfs2), axis=0))

        return l2_features, spike_ids
    
    
    def merge_units(self):
        
        # make connected components
        max_dist = 10000
        merge_matrix = np.zeros((self.n_units, self.n_units),'int32')
        dist_matrix = np.ones((self.n_units, self.n_units), 'float32')*max_dist
        for pair in self.merge_pairs:
            merge_matrix[pair[0], pair[1]] = 1
            merge_matrix[pair[1], pair[0]] = 1

            fname_out = os.path.join(
                        self.save_dir, 
                        'unit_{}_{}.npz'.format(pair[0], pair[1]))
            dist_ = np.load(fname_out)['centers_dist']
            dist_matrix[pair[0], pair[1]] = dist_
            dist_matrix[pair[1], pair[0]] = dist_


        G = nx.from_numpy_matrix(merge_matrix)
        merge_array=[]
        for cc in nx.connected_components(G):
            cc = list(cc)
            if len(cc) > 2:
                cc = np.array(cc)
                clustering = AgglomerativeClustering(n_clusters=None,
                                             affinity='precomputed',
                                             linkage='complete',
                                             distance_threshold=max_dist)
                dist_matrix_ = dist_matrix[cc][:, cc]
                labels_ = clustering.fit(dist_matrix_).labels_
                for k in np.unique(labels_):
                    merge_array.append(list(cc[labels_==k]))
            else:
                merge_array.append(cc)
        
        weights = self.n_spikes_soft

        spike_train_new = np.copy(self.spike_train)
        templates_new = np.zeros((len(merge_array), self.spike_size, self.n_channels),
                                 'float32')

        new_ids = np.zeros(self.n_units, 'int32')
        shifts = np.zeros(self.n_units, 'int32')

        for new_id, units in enumerate(merge_array):
            if len(units) > 1:

                # save only a unit with the highest weight
                id_keep = weights[units].argmax()
                templates_new[new_id] = self.templates[units[id_keep]]

                # update spike train
                # determine shifts
                mc = self.templates[units[id_keep]].ptp(0).argmax()
                min_points = self.templates[units,:,mc].argmin(1)
                shifts_ = min_points - min_points[id_keep]

                new_ids[units] = new_id
                shifts[units] = shifts_

            elif len(units) == 1:
                templates_new[new_id] = self.templates[units[0]]

                new_ids[units[0]] = new_id

        spike_train_new[:, 1] = new_ids[self.spike_train[:,1]]
        spike_train_new[:, 0] += shifts[self.spike_train[:, 1]]

        # sort them by spike times
        idx_sort = np.argsort(spike_train_new[:, 0])
        spike_train_new = spike_train_new[idx_sort]
        soft_assignment_new = self.soft_assignment[idx_sort]

        return (templates_new, spike_train_new,
                soft_assignment_new, merge_array)
    
    
def test_merge(features, assignment,
               lda_threshold=0.7,
               diptest_threshold=0.8):

    '''
    Parameters
    ----------
    features:  feataures
    ssignment:  spike assignments (must be either 0 or 1)
    '''

    # do lda test
    lda_prob = run_ldatest(features, assignment)
    dip_pval = run_diptest(features, assignment)
    if lda_prob < lda_threshold or dip_pval > diptest_threshold:
        merge = True
    else:
        merge = False

    return merge, lda_prob, dip_pval

def run_ldatest(features, assignment):

    # determine which one has more spikes
    _, n_spikes = np.unique(assignment, return_counts=True)
    id_big = np.argmax(n_spikes)
    id_small = np.argmin(n_spikes)
    n_diff = n_spikes[id_big] - n_spikes[id_small]

    if n_diff > 0:
        n_repeat = int(np.ceil(np.max(n_spikes)/np.min(n_spikes)))
        idx_big = np.where(assignment == id_big)[0]

        lda_probs = np.zeros(n_repeat)
        for j in range(n_repeat):
            idx_remove = np.random.choice(idx_big, n_diff, replace=False)
            idx_in = np.ones(len(assignment), 'bool')
            idx_in[idx_remove] = False

            # fit lda
            lda = LDA(n_components = 1)
            lda.fit(features[idx_in], assignment[idx_in])

            # check tp of lda
            lda_probs[j] = lda.score(features[idx_in],
                                     assignment[idx_in])
        lda_prob = np.median(lda_probs)
    else:
        lda = LDA(n_components = 1)
        lda.fit(features, assignment)
        lda_prob = lda.score(features,
                             assignment)

    return lda_prob


def run_diptest(features, assignment):

    '''
    Parameters
    ----------
    pca_wf:  pca projected data
    assignment:  spike assignments
    '''

    _, n_spikes = np.unique(assignment, return_counts=True)
    ratio = n_spikes[0]/n_spikes[1]
    if ratio < 1:
        ratio = 1/ratio

    if ratio > 3:
        pval = 0
    else:
        lda = LDA(n_components = 1)
        lda_feat = lda.fit_transform(features, assignment).ravel()
        pval = dp(lda_feat)[1]

    return pval


def run_diptest2(features, assignment):

    '''
    Parameters
    ----------
    pca_wf:  pca projected data
    assignment:  spike assignments
    '''

    _, n_spikes = np.unique(assignment, return_counts=True)
    id_big = np.argmax(n_spikes)
    id_small = np.argmin(n_spikes)
    n_diff = n_spikes[id_big] - n_spikes[id_small]

    if n_diff > 0:
        n_repeat = int(np.ceil(np.max(n_spikes)/np.min(n_spikes)))
        idx_big = np.where(assignment == id_big)[0]

        pvals = np.zeros(n_repeat)
        for j in range(n_repeat):
            idx_remove = np.random.choice(idx_big, n_diff, replace=False)
            idx_in = np.ones(len(assignment), 'bool')
            idx_in[idx_remove] = False

            # fit lda
            lda = LDA(n_components = 1)
            lda_feat = lda.fit_transform(features[idx_in], assignment[idx_in]).ravel()

            # check tp of lda
            pvals[j] = dp(lda_feat)[1]

        pval = np.median(pvals)
    else:
        lda = LDA(n_components = 1)
        lda_feat = lda.fit_transform(features, assignment).ravel()
        pval = dp(lda_feat)[1]

    return pval


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
        ref_template, nshifts = 7)
    templates = shift_chans(templates, best_shifts)

    # align all spikes on max channel
    best_shifts = align_get_shifts_with_ref(
        spikes[:,:,max_chan],
        ref_template, nshifts = 7)
    spikes = shift_chans(spikes, best_shifts)
    
    # if shifted, cut shifted parts
    # because it is contaminated by roll function
    cut = int(np.ceil(np.max(np.abs(best_shifts))))
    if cut > 0:
        templates = templates[:,cut:-cut]
        spikes = spikes[:,cut:-cut]

    # get visible channels
    vis_ptp = np.min((vis_ptp, np.max(templates.ptp(1))))
    vis_chan = np.where(templates.ptp(1).max(0) >= vis_ptp)[0]
    templates = templates[:, :, vis_chan].reshape(
        templates.shape[0], -1)
    spikes = spikes[:, :, vis_chan].reshape(
        spikes.shape[0], -1)

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

    return dist.T

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
        candidates = np.abs(ptps[:, max_chans[k]] - ptps[k, max_chans[k]])/ptps[k, max_chans[k]] < 0.5

        dist = np.min(np.sum(np.square(
            templates[k][np.newaxis, np.newaxis] - shifted_templates[:, candidates]),
                             axis=(2,3)), 0)
        distance[k, candidates] = dist
        distance[candidates, k] = dist
        
    return distance    
