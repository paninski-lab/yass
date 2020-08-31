#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:39:48 2019
Updated on Wed Jul 08 14:23:30 2020

@author: kevin, Nishchal
"""

import os
import numpy as np 
from matplotlib.gridspec import GridSpec as gridspec
from tqdm import tqdm
import scipy.spatial.distance as dist
import torch
import cudaSpline as deconv
from scipy.interpolate import splrep
from numpy.linalg import inv as inv
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy import linalg as la


# def nearestPD2(A):
#     if isPD(A.cpu()):
#         return A
#     else:
#         eigvals, eigvecs = torch.eig(A, eigenvectors=True)
#         eigvals = eigvals[:,0]
#         eigvals[eigvals < 0.0] = 1e-6
        
#         print( eigvals.max(), eigvals.min())
#         print( torch.eig((eigvals[:,None,None] * torch.matmul(eigvecs[:,:,None], eigvecs[:,None])).sum(0))[0][:,0].max())
#         print( torch.eig((eigvals[:,None,None] * torch.matmul(eigvecs[:,:,None], eigvecs[:,None])).sum(0))[0][:,0].min())
        
        
#         plt.imshow(A.cpu())
#         plt.show()
        
#         plt.imshow((eigvals[:,None,None] * torch.matmul(eigvecs[:,:,None], eigvecs[:,None])).sum(0).cpu())
#         plt.show()
        
        
#         return (eigvals[:,None,None] * torch.matmul(eigvecs[:,:,None], eigvecs[:,None])).sum(0)
        


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    A = A.cpu().numpy()
    
    if isPD(A):
        return torch.from_numpy(A).cuda()
    
    B = A
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return torch.from_numpy(A3).cuda()

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return torch.from_numpy(A3).cuda()

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def fit_spline(curve, knots=None, prepad=0, postpad=0, order=3):
    if knots is None:
        knots = np.arange(len(curve) + prepad + postpad)
    return splrep(knots, np.pad(curve, (prepad, postpad), mode='symmetric'), k=order)

def transform_template(template, knots=None, prepad=7, postpad=3, order=3):

    if knots is None:
        knots = np.arange(len(template.data[0]) + prepad + postpad)
    splines = [
        fit_spline(curve, knots=knots, prepad=prepad, postpad=postpad, order=order) 
        for curve in template.data.cpu().numpy()
    ]
    coefficients = np.array([spline[1][prepad-1:-1*(postpad+1)] for spline in splines], dtype='float32')
    return deconv.Template(torch.from_numpy(coefficients).cuda(), template.indices)

def get_cov_matrix(spat_cov, geom):
    posistion = geom
    dist_matrix = dist.squareform(dist.pdist(geom ))

    cov_matrix = np.zeros((posistion.shape[0], posistion.shape[0]))

    for i in range(posistion.shape[0]):
        for j in range(posistion.shape[0]):
            if dist_matrix[i, j] > np.max(spat_cov[:, 1]):
                cov_matrix[i, j] = 0
                continue
            idx = np.where(spat_cov[:, 1]  == dist_matrix[i, j])[0]
            if len(idx) == 0:
                cov_matrix[i, j] = 0
                continue
            cov_matrix[i, j] = spat_cov[idx, 0]
    return cov_matrix

#Soft assign object

class TEMPLATE_ASSIGN_OBJECT(object):
    def __init__(self, fname_spike_train, fname_templates, fname_shifts,
                 reader_residual,  spat_cov, temp_cov, channel_idx, geom,
                 large_unit_threshold = 5, n_chans = 5, rec_chans = 512,
                 sim_units = 3, similar_array = None, temp_thresh= np.inf, lik_window = 41,
                 update_templates=False, template_update_time=None):

        #get the variance of the residual:        
        self.temp_thresh = temp_thresh
        self.rec_chans = rec_chans
        self.sim_units = sim_units
        self.update_templates = update_templates
        if self.update_templates:
            self.templates_dir = fname_templates
            fname_templates = os.path.join(self.templates_dir, 'templates_0sec.npy')
            
            # at which chunk templates need to be updated
            n_chunks_update = int(template_update_time/reader_residual.n_sec_chunk)
            self.update_chunk = np.arange(0, reader_residual.n_batches, n_chunks_update)
            
        
        self.templates = np.load(fname_templates).astype('float32') 
        self.templates[self.templates == 0.0] += 1e-6 * np.random.rand(self.templates[self.templates == 0.0].shape[0])
        self.offset = int((self.templates.shape[1] - (2*(lik_window//2) +1) )/2)
        self.spike_train = np.load(fname_spike_train)
        self.spike_train_og = np.load(fname_spike_train)
        #self.spike_train = self.spike_train[self.spike_train[:, 0] > 40]
        self.idx_included = set([])
        self.units_in = set([])
        self.shifts = np.load(fname_shifts)
        self.reader_residual = reader_residual
        self.spat_cov = get_cov_matrix(spat_cov, geom)
        self.channel_index = channel_idx
        self.n_neigh_chans = self.channel_index.shape[1]
        self.n_chans = n_chans
        self.n_units, self.n_times, self.n_channels = self.templates.shape
        
        T = temp_cov.shape[0]
        for t in range(1,T):
            c = .5*temp_cov.diagonal(t).mean() + .5*temp_cov.diagonal(-t).mean()
            np.fill_diagonal(temp_cov[:,t:],c)
            np.fill_diagonal(temp_cov[t:,:],c)
        self.temp_cov = torch.from_numpy(temp_cov[(self.offset):(self.n_times -(self.offset))][:,(self.offset):(self.n_times -(self.offset))]).float().cuda()
        self.n_total_spikes = self.spike_train.shape[0]

        if similar_array is None:
            self.similar_array = get_similar_array(self.templates,
                                                   self.sim_units)
        else:
            self.similar_array = similar_array
        self.compute_units_in()
        self.exclude_large_units(large_unit_threshold)

        t_start_include = self.n_times//2 + reader_residual.offset 
        t_end_include = reader_residual.rec_len -  self.n_times//2 + reader_residual.offset
        self.idx_included = np.logical_and(
            np.logical_and(self.spike_train_og[:, 0] < t_end_include,
                           self.spike_train_og[:, 0] > t_start_include),
            np.in1d(self.spike_train_og[:, 1], np.asarray(list(self.units_in))))
        self.spike_train = self.spike_train_og[self.idx_included]
        self.shifts = self.shifts[self.idx_included]

        # get channels for each comparisons
        self.get_chans()
        #
        self.preprocess_spike_times()

        #get aligned templatess
        self.move_to_torch()

    def get_chans(self):
        
        ptps = self.templates.ptp(1)
        mcs = ptps.argmax(1)
        
        #
        self.chans = np.zeros((self.n_units, self.n_chans), 'int32')
        # first channel is always the max channel of the ref unit
        self.chans[:, 0] = mcs

        self.chans = np.argsort(ptps, axis = 1)[:,-1:-6:-1].copy()

#         # for the rest, pick the largest
        for unit in range(self.n_units):
            chans_sorted = np.argsort(ptps[self.similar_array[unit]].max(0))[::-1]
            chans_sorted = chans_sorted[chans_sorted != mcs[unit]]
            self.chans[unit, 1:] = chans_sorted[:(self.n_chans-1)]

    def get_template_data(self, unit, chans, active_ids):
        
        active_ids = torch.where(active_ids)[0]
        
        self.aligned_template_list = []
        self.coeff_list = []
        for i in range(self.templates_aligned.shape[0]):
            diff_array = self.subtract_template(unit, self.similar_array[unit, active_ids[i]], chans)
            self.coeff_list.append(self.get_bspline_coeffs(diff_array[None].astype("float32")))
            self.aligned_template_list.append(diff_array[None])
            
        self.aligned_template_list.append(self.templates_aligned[0:1])
        self.coeff_list.append(self.get_bspline_coeffs(self.templates_aligned[0:1].astype("float32")))
        self.templates_aligned = [torch.from_numpy(element).float().cuda() for element in self.aligned_template_list]
        
    def get_residual_variance(self):
        num = int(60/self.reader_residual.n_sec_chunk)
        var_array = np.zeros(num)
        for batch_id in range(num):
            var_array[batch_id] = np.var(self.reader_residual.read_data_batch(batch_id, add_buffer=True))
        self.resid_var = np.mean(var_array)
        
    def get_kronecker(self, chans):

        self.cov_list = []
        covar = np.kron(self.spat_cov[np.ix_(chans.cpu(), chans.cpu())], self.temp_cov.cpu())
        return covar
        
        
    def calculate_coeff_matrices(self, Tt, temp_cov, spat_cov, active_ids, c_idx, emp_cov, R, C):
        
        A = torch.zeros([3, R*C, R*C]).cuda()
        
        signal_cov = torch.matmul(Tt[:,:,None], Tt[:,None])
#         for i in range(signal_cov.shape[0]):
#             print("sig ",torch.eig(signal_cov[i])[0][:,0].max(), torch.eig(signal_cov[i])[0][:,0].min())
#             print("emp ",torch.eig(emp_cov[i])[0][:,0].max(), torch.eig(emp_cov[i])[0][:,0].min())
        
        for i in range(signal_cov.shape[0]):
#             plt.imshow(signal_cov[i].cpu())
#             plt.show()

            signal_cov[i] = nearestPD(signal_cov[i])
            emp_cov[i] = nearestPD(emp_cov[i])
#             plt.imshow(signal_cov[i].cpu())
#             plt.show()

            
#         for i in range(signal_cov.shape[0]):
#             print("sig ",torch.eig(signal_cov[i])[0][:,0].max(), torch.eig(signal_cov[i])[0][:,0].min())
#             print("emp ",torch.eig(emp_cov[i])[0][:,0].max(), torch.eig(emp_cov[i])[0][:,0].min())
            
            
        
#         print(torch.eig(temp_cov)[0][:,0].max(), torch.eig(temp_cov)[0][:,0].min()) 
            
#         print(signal_cov.shape, emp_cov.shape, temp_cov.shape, spat_cov.shape)
        signal_cov = signal_cov.reshape([Tt.shape[0],C,R,C,R]).permute([0,1,3,2,4])
        
        kronecker = torch.from_numpy(np.kron(np.eye(C), np.ones([R, 1]))).cuda()
        T = (Tt[:,:,None] * kronecker).permute([0,2,1]).float()
        
        emp_cov =emp_cov.reshape([signal_cov.shape[0], C,R,C,R]).permute([0,1,3,2, 4])
        X = torch.cat([signal_cov[:,:,:,None], temp_cov.repeat([emp_cov.shape[0], C, C, 1, 1, 1])], axis = 3)
        
        inv = torch.inverse((X[:,:,:,None] * X[:,:,:,None].permute([0,1,2,4,3,5, 6])).sum([5,6]))
#         for i in range(3):
#             for c1 in range(5):
#                 for c2 in range(5):
#                     print(torch.sort(torch.eig(inv[i,c1,c2])[0][:,0])[0])
        
        inv =  (inv[:,:,:,:,:,None, None]* X[:,:,:,None]).sum(3)
#         print(inv.shape)
        W = (inv * emp_cov[:,:,:,None]).sum([4,5])
        
#         for i in range(3):
#             print(torch.sort(torch.eig(W[i,:,:,0])[0][:,0])[0])
#             print(torch.sort(torch.eig(W[i,:,:,1])[0][:,0])[0])
#         print()
        invW_n = torch.inverse(W[:,:,:,1])
        invW_s = torch.inverse(W[:,:,:,0])
        
#         for i in range(3):
#             print(torch.sort(torch.eig(invW_s[i,:,:])[0][:,0])[0])
#             print(torch.sort(torch.eig(invW_n[i,:,:])[0][:,0])[0])
#         quit()
            
        inv_temp_cov = torch.inverse(temp_cov)
        temp1 = torch.from_numpy(np.kron(invW_n.cpu().numpy(), inv_temp_cov.cpu().numpy())).cuda().float()
        temp2 = torch.inverse(invW_s + torch.matmul(torch.matmul(T, temp1), T.permute([0,2,1])))
        temp2 = torch.matmul(temp1, torch.matmul(torch.matmul(torch.matmul(T.permute([0,2,1]), temp2), T), temp1))
        A[active_ids] = temp1 - temp2
        
        if W.shape[0] < 3:
            temp3 = torch.from_numpy(np.kron(np.linalg.inv(spat_cov[c_idx[:,None],c_idx]), np.linalg.inv(temp_cov.cpu().numpy()))).float().cuda()
            temp3 = [temp3[None]] * (3 - int(active_ids.sum()))
            A[~active_ids] = torch.cat(temp3, axis = 0)
            
        const = torch.zeros(3)
        for i in range(3):
            A[i] = nearestPD(A[i])
            eigs = torch.eig(A[i])[0][:,0]
            eigs[eigs<0] = 1e-10
            const[i]  = torch.log(eigs).sum()*0.5
         
        return A, const

    def compute_units_in(self):

        ptps = self.templates.ptp(1)
        ptps[ptps<0.5] = 0
        # only compare when similar units are similar enough
        for i in range(self.n_units):
            if ptps[i].max() > 0:
                norm = np.sqrt(np.sum(np.square(ptps[i])))
                dist = np.sqrt(np.sum(np.square(ptps[self.similar_array[i]] - ptps[i]), 1))
                if np.sort(dist)[1]/norm < self.temp_thresh:
                    self.units_in.add(i)

    '''
    def get_similar(self):
        max_time = self.templates.argmax(1).astype("int16")
        padded_templates = np.concatenate((np.zeros((self.n_units, 5, self.rec_chans)), self.templates, np.zeros((self.n_units, 5, self.rec_chans))), axis = 1)
        reduced = np.zeros((self.templates.shape[0], 5, self.rec_chans))
        for unit in range(self.n_units):
            for chan in range(self.rec_chans):
                reduced[unit, :,  chan] = padded_templates[unit, (5 +max_time[unit, chan] - 2):(5 + max_time[unit, chan] + 3), chan]

        see = dist.squareform(dist.pdist(reduced.reshape(self.n_units, self.n_channels*5)))
        self.similar_array = np.zeros((self.n_units, self.sim_units)).astype("int16")
        for i in range(self.n_units):
            sorted_see = np.sort(see[i])[0:self.sim_units]
            self.similar_array[i] = np.argsort(see[i])[0:self.sim_units]
            norm = np.sqrt(np.sum(np.square(reduced[i])))
            #if np.min(self.similar_array[i][1:])/norm < self.temp_thresh:
            #    self.units_in.add(i)
            if sorted_see[1]/norm < self.temp_thresh:
                self.units_in.add(i)

    def get_similar(self):
        see = dist.squareform(dist.pdist(self.templates.reshape(self.n_units, self.n_channels*self.n_times)))
        self.similar_array = np.zeros((self.n_units, self.sim_units)).astype("int16")
        units_in = []
        for i in range(self.n_units):
            self.similar_array[i] = np.argsort(see[i])[0:self.sim_units]
            norm = np.sqrt(np.sum(np.square(self.templates[i])))
            if np.min(self.similar_array[i][1:])/norm < self.temp_thresh:
                self.units_in.add(i)
                #self.idx_included.update(np.where(self.spike_train_og[:, 1] == i)[0])
        
        #units_in= np.asarray(units_in)
        #in_spikes = np.where(np.in1d(self.spike_train[:,1], units_in))[0]
        #self.idx_included.update(in_spikes)
    '''

    #shift secondary template
    def shift_template(self, template, shift):
        if shift == 0:
            return template
        if shift > 0:
            return np.concatenate((template, np.zeros((shift, template.shape[1]))), axis = 0)[shift:, :]
        else:
            return np.concatenate((np.zeros((-shift, template.shape[1])), template), axis = 0)[:(self.n_times), :]

    def preprocess_spike_times(self):
        
        # templates on neighboring channels
        self.mcs = self.templates.ptp(1).argmax(1)
        
        #template used for alignment defined on neighboring channels
        templates_neigh = np.zeros((self.n_units, self.n_times, self.n_neigh_chans))
        for k in range(self.n_units):
            neigh_chans = self.channel_index[self.mcs[k]]
            neigh_chans = neigh_chans[neigh_chans<self.n_channels]
            templates_neigh[k, :, :len(neigh_chans)] = self.templates[k,:,neigh_chans].T

        # get alignment shift
        min_points = templates_neigh[:,:,0].argmin(1)
        ref_min = int(np.median(min_points))
        self.temp_shifts = min_points - ref_min

        # shift spike times according to the alignment
        self.spike_train[:, 0] += self.temp_shifts[self.spike_train[:, 1]]

    def preprocess_templates(self, unit, channels, active_ids):

        #template returned for likilihood calculation- defined on channels with largest ptp
        return_templates = np.zeros((active_ids.sum(), self.n_times, channels.shape[0]))
        
        for i, neuron in enumerate(self.similar_array[unit][active_ids]):
            try:
                return_templates[i, :, :channels.shape[0]] = self.templates[neuron,:,channels].T
            except:
                return_templates[i, :, 0] = self.templates[neuron,:,channels].T
                
                
        
        # add buffer for alignment
        buffer_size  = np.max(np.abs(self.temp_shifts))
        
        if buffer_size > 0:
            buffer = np.zeros((active_ids.sum(), buffer_size, channels.shape[0]))
            return_templates = np.concatenate((buffer, return_templates, buffer), axis=1)

        # get ailgned templates
        t_in = np.arange(buffer_size, buffer_size + self.n_times)
        templates_aligned = np.zeros((active_ids.sum(),
                                      self.n_times,
                                      channels.shape[0]), 'float32')
        for i, neuron in enumerate(self.similar_array[unit][active_ids]):
            t_in_temp = t_in + self.temp_shifts[neuron]
            templates_aligned[i] = return_templates[i,t_in_temp]
        
        self.templates_aligned = templates_aligned

    #shifted neighboring template according to shift in primary template
    def subtract_template(self, primary_unit, neighbor_unit,chans):
        primary_unit_shift = self.temp_shifts[primary_unit]
        shifted = self.shift_template(self.templates[neighbor_unit], primary_unit_shift)
        add_shift = np.argmin(self.templates_aligned[0,:, 0]) - np.argmin(shifted[ :, chans[0]])
        shifted = self.shift_template(shifted, -add_shift)

        return self.templates_aligned[0] - shifted[:, chans]
            
    def exclude_large_units(self, threshold):
        
        norms = np.zeros(self.n_units)
        for j in range(self.n_units):
            temp = self.templates[j]
            vis_chan = np.where(temp.ptp(0) > 1)[0]
            norms[j] = np.sum(np.square(temp[:, vis_chan]))
        
        
        self.units_in = self.units_in.intersection(np.where(self.templates.ptp(1).max(1) < threshold)[0])

    def move_to_torch(self):
        
        self.spike_train = torch.from_numpy(self.spike_train).long().cuda()
        self.shifts = torch.from_numpy(self.shifts).float().cuda()
        self.chans = torch.from_numpy(self.chans)
        #self.mcs = torch.from_numpy(self.mcs)
        
    def get_bspline_coeffs(self,  template_aligned):

        n_data, n_times, n_channels = template_aligned.shape

        channels = torch.arange(n_channels).cuda()
        temps_torch = torch.from_numpy(-(template_aligned.transpose(0, 2, 1))/2).cuda()

        temp_cpp = deconv.BatchedTemplates([deconv.Template(temp, channels) for temp in temps_torch])
        coeffs = deconv.BatchedTemplates([transform_template(template) for template in temp_cpp])
        return coeffs

    def get_shifted_templates(self, temp_ids, shifts, iteration, channels):
        
        temp_ids -= temp_ids
        temp_ids = torch.from_numpy(temp_ids.cpu().numpy()).long().cuda()
        shifts = torch.from_numpy(shifts.cpu().numpy()).float().cuda()
        
        n_sample_run = 1000
        n_times = self.aligned_template_list[iteration].shape[1]
        
        idx_run = np.hstack((np.arange(0, len(shifts), n_sample_run), len(shifts)))
        
        shifted_templates = torch.cuda.FloatTensor(len(shifts), n_times, channels.shape[0]).fill_(0)
        for j in range(len(idx_run)-1):
            ii_start = idx_run[j]
            ii_end = idx_run[j+1]
            
            obj = torch.cuda.FloatTensor(channels.shape[0], (ii_end-ii_start)*n_times + 10).fill_(0)
            times = torch.arange(0, (ii_end-ii_start)*n_times, n_times).long().cuda() + 5
            deconv.subtract_splines(obj,
                                    times,
                                    shifts[ii_start:ii_end],
                                    temp_ids[ii_start:ii_end],
                                    self.coeff_list[iteration], 
                                    torch.full( (ii_end - ii_start, ), 2).cuda())
            obj = obj[:,5:-5].reshape((channels.shape[0], (ii_end-ii_start), n_times))
            shifted_templates[ii_start:ii_end] = obj.permute(1,2,0)
    
        return shifted_templates
    
    def get_liklihood(self, unit, snip):
        chans = self.chans[unit]
        chans = chans < self.rec_chans
        log_prob = np.ravel(snip[:, chans].T) @ self.cov_list[unit] @ np.ravel(snip[:, chans].T) 
        return log_prob
    
    

    def compute_soft_assignment(self):
        
        probs = torch.zeros([self.spike_train.shape[0], self.sim_units]).cuda()
        logs = torch.zeros([self.spike_train.shape[0], self.sim_units]).cuda()
        probs[:, 0] = 1.0
        outliers = torch.zeros([0, 2]).long().cuda()
        logprobs = -torch.ones([self.spike_train.shape[0], self.sim_units]).cuda()
        
        _, sizes_og = np.unique(self.spike_train[:,1].cpu().numpy(), return_counts = True)

        # batch offsets
        offsets = torch.from_numpy(self.reader_residual.idx_list[:, 0]
                                   - self.reader_residual.buffer).cuda().long()
        with tqdm(total=self.reader_residual.n_batches) as pbar:
            for batch_id in range(self.reader_residual.n_batches):
#             for batch_id in [0,1,2,3,4,5]:
                if self.update_templates and np.any(self.update_chunk == batch_id):
                    time_sec_start = batch_id*self.reader_residual.n_sec_chunk
                    fname_templates = os.path.join(
                        self.templates_dir,
                        'templates_{}sec.npy'.format(time_sec_start))
                    self.templates = np.load(fname_templates)
                    self.templates[self.templates == 0.0] += 1e-10 * np.random.normal(self.templates[self.templates == 0.0].shape[0])

                # load residual data
                resid_dat = self.reader_residual.read_data_batch(
                    batch_id, add_buffer=True)#/np.sqrt(self.resid_var)
                resid_dat = torch.from_numpy(resid_dat).cuda()
                resid_dat = torch.cat((resid_dat, torch.zeros((resid_dat.shape[0], 1)).cuda()), 1)
                
                
                # relevant idx
                idx_in = torch.nonzero(
                    (self.spike_train[:, 0] >= self.reader_residual.idx_list[batch_id][0]) & 
                    (self.spike_train[:, 0] < self.reader_residual.idx_list[batch_id][1]))[:,0]
                
                if len(idx_in) == 0:
                    continue

                spike_train_batch = self.spike_train[idx_in] 
                spike_train_batch[:, 0] -= offsets[batch_id]
                shift_batch = self.shifts[idx_in]
                
                for prim_unit in range(self.n_units):
#                 for prim_unit in [12]:
                
                    ids1 = spike_train_batch[:, 1] == prim_unit
                    if ids1.sum() <= 1:
                        probs[idx_in[ids1]] = 1.0
                        logprobs[idx_in[ids1], 0] = 25 + 1e-6
                        logprobs[idx_in[ids1], 1:] = 200  + 1e-6                      
                        continue
                    
                    active_ids = torch.zeros(self.sim_units).bool()
                    sizes = torch.zeros(self.sim_units)
                    
                    for j, unit in enumerate(self.similar_array[prim_unit]):
                        idx_temp = torch.where(spike_train_batch[:,1] == unit)[0]
                        sizes[j] = idx_temp.shape[0]
                        if sizes[j] > 1:
                            active_ids[j] = True
                    c_index = self.chans[prim_unit]
                    
                    T = torch.from_numpy(self.templates[self.similar_array[prim_unit][active_ids,None], self.offset:-self.offset,c_index].reshape([active_ids.sum(), -1])).cuda()
                    
                    n_times = self.n_times - 2 * self.offset
                    emp_cov = torch.zeros([active_ids.sum(), n_times * c_index.shape[0], n_times * c_index.shape[0]]).cuda()
                    for j, sec_unit in enumerate(self.similar_array[prim_unit][active_ids]):
                        ids2 = spike_train_batch[:, 1] == sec_unit
                        t_index = spike_train_batch[ids2, 0][:, None] + torch.arange(-(self.n_times//2)+self.offset, self.n_times//2 + 1 - self.offset ).cuda()
                        resid_snippets = resid_dat[t_index[:,:,None], c_index[None, None,:].long()]
                        if j == 0:
                            wf1 = resid_snippets
                        emp_cov[j] = cov(resid_snippets.permute([0,2,1]).reshape([resid_snippets.shape[0], -1]))
                    prec_est, const = self.calculate_coeff_matrices(T, self.temp_cov, self.spat_cov, active_ids, c_index, emp_cov, n_times, c_index.shape[0])
                    prec_est, const = prec_est.cuda(), const.cuda()
                    
                    self.preprocess_templates(prim_unit, c_index, torch.ones(self.sim_units).bool())
                    self.get_template_data(prim_unit, c_index, torch.ones(self.sim_units).bool())
                    
                    shifted_templates = [
                        self.get_shifted_templates(
                            spike_train_batch[ids1,1].float(), shift_batch[ids1].float(), i, c_index) for i in range(3 + 1)]
                    
                    shifted_templates = [shifted_templates[i] for i in range(len(shifted_templates))]
                    
                    clean_wfs = [(wf1 + shifted[:, (self.offset):(self.n_times -(self.offset)), :]).permute(0,2,1).reshape([wf1.shape[0], -1]) for shifted in shifted_templates]
                    
                    clean_wfs = torch.stack(clean_wfs, dim=1)
                    temp = clean_wfs[:,:3]
                    
                    logprobs[idx_in[ids1]] = torch.sum(torch.matmul(torch.matmul(temp[:,:,None], prec_est), temp[:,:,:,None]),[2,3])
                    ml = torch.max(const -  logprobs[idx_in[ids1]]/2, dim = 1, keepdim = True)[0]
                    unscaled_probs = const - logprobs[idx_in[ids1]]/2 - ml
                    scaled_probs = torch.from_numpy(sizes_og[self.similar_array[prim_unit]]).cuda() * torch.exp(unscaled_probs)
                    
                    probs[idx_in[ids1]] = scaled_probs/ scaled_probs.sum(dim = 1, keepdim = True)
#                 print(probs[idx_in[ids1]].sum(0))
                pbar.update()
        
        return probs.float().cpu().numpy(), logprobs.float().cpu().numpy()

    def clean_wave_forms(self, spike_idx, unit):
        return_wfs = torch.zeros((spike_idx.shape[0],self.templates.shape[1], self.n_chans))
        with tqdm(total=self.reader_residual.n_batches) as pbar:
            for batch_id in range(self.reader_residual.n_batches):

                # load residual data
                resid_dat = self.reader_residual.read_data_batch(batch_id, add_buffer=True)
                resid_dat = torch.from_numpy(resid_dat).cuda()

                # relevant idx
                s1 = self.spike_train[spike_idx, 0] >= self.reader_residual.idx_list[batch_id][0]
                s2 = self.spike_train[spike_idx, 0] < self.reader_residual.idx_list[batch_id][1]
                
                idx_in = torch.nonzero((s1 & s2))[:,0]

                spike_train_batch = self.spike_train[spike_idx]
                spike_train_batch = spike_train_batch[idx_in]
                spike_train_batch[:, 0] -= (self.reader_residual.idx_list[batch_id][0] - self.reader_residual.buffer)

                shift_batch = self.shifts[spike_idx]
                shift_batch = shift_batch[idx_in]
                # get residual snippets

                t_index = spike_train_batch[:, 0][:, None] + torch.arange(-(self.n_times//2), self.n_times//2+1).cuda()
                c_index = self.chans[spike_train_batch[:, 1]].long()
                resid_dat = torch.cat((resid_dat, torch.zeros((resid_dat.shape[0], 1)).cuda()), 1)
                resid_snippets = resid_dat[t_index[:,:,None], c_index[:,None]]
                # get shifted templates

                shifted_og = self.get_shifted_templates(spike_train_batch[:,1], shift_batch, self.sim_units)
                clean_wfs = resid_snippets + shifted_og
                return_wfs[idx_in] = clean_wfs.cpu()
                
        return return_wfs.cpu().numpy()

    def get_assign_probs(self, log_lik_array):
        fix = log_lik_array*-.5
        fix = fix - fix.max(1)[:, None]
        probs =  np.exp(fix)/np.exp(fix).sum(1)[:, None]
        self.probs = probs
        return probs
    
    def run(self):

        #construct array to identify soft assignment units
        unit_assignment = self.similar_array[self.spike_train_og[:, 1]]

        probs, log_probs = self.compute_soft_assignment()
        replace_probs = np.zeros((self.spike_train_og.shape[0], self.sim_units))
        replace_log = -np.ones((self.spike_train_og.shape[0],self.sim_units))
        
        replace_probs[:, 0] = 1
        replace_probs[self.idx_included, :] = probs
        
        replace_log[self.idx_included, :] = log_probs
        
        return replace_probs, replace_log, unit_assignment
    

def cov(x):
    x = x - x.mean(0)
    
    c = x.T.mm(x)/(x.shape[0]-1)
    return c


# def get_similar_array(templates, sim_units):
    
#     n_units = templates.shape[0]
#     mc = templates.ptp(1).argmax(1)
#     ptps = templates.ptp(1)
#     ptps[ptps<0.5] = 0
#     see = dist.squareform(dist.pdist(ptps))
    
#     for unit in range(n_units):
#         candidates = mc == mc[unit]
#         np.argsort(see[candidates])[:sim_units]
    
    
#     similar_array = np.zeros(

def get_similar_array(templates, sim_units):

    n_units = templates.shape[0]
    ptps = templates.ptp(1)
    ptps[ptps<0.5] = 0
    see = dist.squareform(dist.pdist(ptps))

    similar_array = np.zeros((n_units, sim_units)).astype("int16")
    for i in range(n_units):
        mc = ptps[i].argmax()
        if ptps[i, mc] > 0:
            candidates = np.where(ptps[:, mc] > 0)[0]
            if len(candidates) >= sim_units:
                similar_array[i] = candidates[
                    np.argsort(see[i, candidates])][:sim_units]
            elif len(candidates) > 0:
                similar_array[i, :len(candidates)] = candidates[np.argsort(see[i, candidates])]
                
#         print(i, mc, similar_array[i])

    return similar_array
