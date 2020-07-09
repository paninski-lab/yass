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
                 sim_units = 3, similar_array = None, temp_thresh= np.inf, lik_window = 50,
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
        self.temp_cov = torch.from_numpy(temp_cov[(self.offset)-5:(self.n_times -(self.offset))-20][:, (self.offset)-5:(self.n_times -(self.offset))-20]).float().cuda()
        self.n_total_spikes = self.spike_train.shape[0]
        
        #get residual variance 
        #self.get_residual_variance()

        if similar_array is None:
            self.similar_array = get_similar_array(self.templates,
                                                   self.sim_units)
        else:
            self.similar_array = similar_array
        self.compute_units_in()
        self.exclude_large_units(large_unit_threshold)

        t_start_include = self.n_times//2 + reader_residual.offset - 5
        t_end_include = reader_residual.rec_len -  self.n_times//2 + reader_residual.offset - 20
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

#         self.get_kronecker()
        
#         if not self.update_templates:
#             self.get_template_data()

    def get_chans(self):

        ptps = self.templates.ptp(1)
        mcs = ptps.argmax(1)

        #
        self.chans = np.zeros((self.n_units, self.n_chans), 'int32')
        # first channel is always the max channel of the ref unit
        self.chans[:, 0] = mcs

        # for the rest, pick the largest
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
#         inv_temp  = inv()
        
        covar = np.kron(self.spat_cov[np.ix_(chans.cpu(), chans.cpu())], self.temp_cov.cpu())
        return covar
        
        
    def calculate_coeff_matrices(self, Tt, temp_cov, emp_cov, R, C):
        
        signal_cov = torch.matmul(Tt[:,:,None], Tt[:,None])
        signal_cov = signal_cov.reshape([Tt.shape[0],C,R,C,R]).permute([0,1,3,2,4])
        
        kronecker = torch.from_numpy(np.kron(np.eye(C), np.ones([R, 1]))).cuda()
        T = (Tt[:,:,None] * kronecker).permute([0,2,1]).float()
        
        emp_cov =emp_cov.reshape([signal_cov.shape[0], C,R,C,R]).permute([0,1,3,2, 4])
        X = torch.cat([signal_cov[:,:,:,None], temp_cov.repeat([emp_cov.shape[0], C, C, 1, 1, 1])], axis = 3)
        
        inv = torch.inverse((X[:,:,:,None] * X[:,:,:,None].permute([0,1,2,4,3,5, 6])).sum([5,6]))
        
        inv =  (inv[:,:,:,:,:,None, None]* X[:,:,:,None]).sum(4)
        W = (inv * emp_cov[:,:,:,None]).sum([4,5])
        
        try:
            invW_n = torch.inverse(W[:,:,:,1])
            
        except:
            print("wah wah")
        invW_s = torch.inverse(W[:,:,:,0])
        inv_temp_cov = torch.inverse(temp_cov)
        
        A = torch.from_numpy(np.kron(invW_n.cpu().numpy(), inv_temp_cov.cpu().numpy())).cuda().float()
        temp = torch.inverse(invW_s + torch.matmul(torch.matmul(T, A), T.permute([0,2,1])))
        temp = torch.matmul(A, torch.matmul(torch.matmul(torch.matmul(T.permute([0,2,1]), temp), T), A))
        return A-temp

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

        # batch offsets
        offsets = torch.from_numpy(self.reader_residual.idx_list[:, 0]
                                   - self.reader_residual.buffer).cuda().long()
        with tqdm(total=self.reader_residual.n_batches) as pbar:
            for batch_id in range(self.reader_residual.n_batches):

                if self.update_templates and np.any(self.update_chunk == batch_id):
                    time_sec_start = batch_id*self.reader_residual.n_sec_chunk
                    fname_templates = os.path.join(
                        self.templates_dir,
                        'templates_{}sec.npy'.format(time_sec_start))
                    self.templates = np.load(fname_templates)

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
                
                for i, prim_unit in enumerate(range(self.n_units)):

                    ids1 = spike_train_batch[:, 1] == prim_unit
                    if ids1.sum() <= 1:
                        probs[idx_in][ids1] = 1.0
                        continue
                        
                    active_ids = torch.zeros(self.sim_units).bool()
                    sizes = torch.zeros(self.sim_units)
                    for j, unit in enumerate(self.similar_array[prim_unit]):
                        idx_temp = torch.where(spike_train_batch[:,1] == unit)[0]
                        sizes[j] = idx_temp.shape[0]
                        if sizes[j] > 1:
                            active_ids[j] = True
                    c_index = self.chans[prim_unit]
                    
                    c_idx = self.templates[self.similar_array[prim_unit][:,None], :, c_index].ptp(2) > 5.0                    
                    active_ids = active_ids & torch.from_numpy(np.any(c_idx, axis = 1)).bool()
                    
                    c_index = c_index[np.all(c_idx[active_ids], axis = 0)]
                    
                    if c_index.shape[0] <= 1 or active_ids.sum() <= 1 or not active_ids[0]:
                        continue
                        
                    T = torch.from_numpy(self.templates[self.similar_array[prim_unit][active_ids,None], self.offset-5:-self.offset-20,c_index].reshape([active_ids.sum(), -1])).cuda()
                    n_times = self.n_times - 2 * self.offset +5 -20
                    emp_cov = torch.zeros([active_ids.sum(), n_times * c_index.shape[0], n_times * c_index.shape[0]]).cuda()
                    for j, sec_unit in enumerate(self.similar_array[prim_unit][active_ids]):
                        ids2 = spike_train_batch[:, 1] == sec_unit
                        t_index = spike_train_batch[ids2, 0][:, None] + torch.arange(-(self.n_times//2)+self.offset-5, self.n_times//2 + 1 - self.offset - 20).cuda()
                        resid_snippets = resid_dat[t_index[:,:,None], c_index[None, None,:].long()]
                        if j == 0:
                            wf1 = resid_snippets
                        emp_cov[j] = cov(resid_snippets.permute([0,2,1]).reshape([resid_snippets.shape[0], -1]))
                        
                    prec_est = self.calculate_coeff_matrices(T, self.temp_cov, emp_cov, n_times, c_index.shape[0])
                    prec_est = prec_est.half().cuda()
                    self.preprocess_templates(prim_unit, c_index, active_ids)
                    self.get_template_data(prim_unit, c_index, active_ids)
                    shifted_templates = [
                        self.get_shifted_templates(
                            spike_train_batch[ids1,1].float(), shift_batch[ids1].float(), i, c_index) for i in range(active_ids.sum() + 1)]
                    shifted_templates = [shifted_templates[i] for i in range(len(shifted_templates))]
                    
                    clean_wfs = [(wf1 + shifted[:, (self.offset)-5:(self.n_times -(self.offset))-20, :]).permute(0,2,1).reshape([wf1.shape[0], -1]).half() for shifted in shifted_templates]
                    
                    clean_wfs = torch.stack(clean_wfs, dim=1)
                    temp = clean_wfs[:,:3][:,active_ids]
                    logprobs = -torch.sum(torch.matmul(torch.matmul(temp[:,:,None], prec_est), temp[:,:,:,None]),[2,3])/2
                    
                    const = torch.logdet(prec_est.float()) * 0.5
                    idx = ~torch.isnan(const) & ~torch.isinf(logprobs.sum(0))
                    
                    if not idx[0]:
                        continue
                    else:
                        const = const[idx]
                        logprobs = logprobs[:, idx]
                        active_ids = torch.nonzero(active_ids)[idx][:,0]                   
                    
                    ml = torch.max(const+logprobs, dim = 1, keepdim = True)[0]
                    unscaled_probs = const + logprobs - ml
                    logs[idx_in[ids1, None], active_ids] = unscaled_probs + torch.log(sizes[active_ids].cuda())
                    scaled_probs = sizes[active_ids].cuda() * torch.exp(unscaled_probs)
                    probs[idx_in[ids1, None], active_ids] = scaled_probs/ scaled_probs.sum(dim = 1, keepdim = True)
                    
                pbar.update()
                
                

        return probs.float().cpu().numpy(), logs.float().cpu().numpy()

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
        replace_log = np.zeros((self.spike_train_og.shape[0],self.sim_units))
        
        replace_probs[:, 0] = 1
        replace_probs[self.idx_included, :] = probs
        
        replace_log[self.idx_included, :] = log_probs
        
        return replace_probs, probs, replace_log, unit_assignment
    

def cov(x):
    x = x - x.mean(0)
    
    c = x.T.mm(x)/(x.shape[0]-1)
    return c

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

    return similar_array
