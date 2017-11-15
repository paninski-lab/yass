import logging

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

from yass.preprocessing import Preprocessor
from yass.mainprocess import Mainprocessor

from .preprocess.filter import butterworth
from .geometry import (n_steps_neigh_channels,
                       order_channels_by_distance)


def getCleanSpikeTrain(config):
    """
        Threshold detection for extracting clean templates from the raw recording if groundtruth is not available

        Parameters:
        -----------
        config: configuration object
            configuration object containing the parameters for making the training data.
            
        Returns:
        -----------
        spikeTrain: np.array
            [number of spikes, 2] first column corresponds to spike time; second column corresponds to cluster id.
 
    """

    config.detctionMethod = 'threshold'
    config.doWhitening = 0
    config.doDeconv = 0

    pp = Preprocessor(config)
    score, clr_idx, spt = pp.process()
    mp=Mainprocessor(config, score, clr_idx, spt)
    mp.mainProcess()

    return mp.spikeTrain


class AugmentedSpikes(object):
    """
        Class for making the training data for the neural network detector, the autoencoder and the triage network
        
        Attributes:
        -----------
        config: configuration object
            configuration object containing the parameters for making the training data.
          
        spikeTrain: np.array
            [number of spikes, 2] first column corresponds to spike time; second column corresponds to cluster id.
    """
    
    def __init__(self, *args):
        
        """
            Initializes the attributes for the class AugmentedSpikes.

            Parameters:
            -----------
            args: if no spiketrain is input, threshold detection will be performed to obtain a spiketrain from the recording
            specified by the configuration file.
        """                

        config = args[0]
        self.config = config
        
        if len(args) == 1:    
            self.spikeTrain = getCleanSpikeTrain(config)
        elif len(args) == 2:
            self.spikeTrain = args[1]
        else:
            print('at most two input arguments')

        self.logger = logging.getLogger(__name__)
                 
    def getBigTemplates(self,R):
        """
            Gets clean templates with large temporal radius

            Parameters:
            -----------
            R: int
                length of the templates to be returned.

            Returns:
            -----------
            templates: np.array
                [number of templates, temporal length, number of channels] returned templates.

        """

        pp = Preprocessor(self.config)
        return pp.getTemplates(self.spikeTrain,R)
    
    def determineNoiseCov(self, temporal_size, D):
        """
            Determines the spatial and temporal covariance of the noise 

            Parameters:
            -----------
            temporal_size: int
                row size of the temporal covariance matrix.
            D: int
                number of channels away from the mainchannel. D=1 if only look at the main channel; D=2 if look at the main
                channel and its neighboring channels; D=3 if look at the main channel, the neighboring channels and the
                neighbors of the neighboring channels.
            Returns:
            -----------
            spatial_SIG: np.array
                [d, d] spatial covariance matrix of the noise where d depends on D.

            temporal_SIG: np.array
                [temporal_size, temporal_size] temporal covariance matrix of the noise.

        """
       
        pp = Preprocessor(self.config)
        
        batch_size = self.config.batch_size
        BUFF = self.config.BUFF
        nBatches = self.config.nBatches
        nPortion = self.config.nPortion
        residual = self.config.residual
        
        R = pp.config.spikeSize
        # get recording in the middle
        pp.openFile()
        i = np.ceil(nBatches/2)

        self.logger.debug('Loading batch {}...'.format(i))

        # reading data
        if nBatches ==1:
            rec = pp.load(0, batch_size)
            
        elif i == 0:
            rec = pp.load(i*batch_size, batch_size+BUFF)
            
        elif i < nBatches-1:
            rec = pp.load(i*batch_size-BUFF, batch_size+2*BUFF)

        elif residual==0:
            rec = pp.load(i*batch_size-BUFF, batch_size+BUFF)

        else:
            rec = pp.load(i*batch_size-BUFF, residual+BUFF)
        
        neighChanBig = n_steps_neigh_channels(self.config.neighChannels, D)
        c_ref = np.argmax(np.sum(neighChanBig,0))
        ch_idx = np.where(neighChanBig[c_ref])[0]
        ch_idx, temp = order_channels_by_distance(c_ref,ch_idx, self.config.geom)
        rec = rec[:,ch_idx]
        
        # filter recording
        if pp.config.preprocess.filter == 1:
            rec = butterworth(rec, self.config.filter.low_pass_freq,
                              self.config.filter.high_factor,
                              self.config.filter.order,
                              self.config.recordings.sampling_rate)

        # standardize recording
        small_t = np.min((int(pp.config.recordings.sampling_rate*5),6000000))
        mid_T = int(np.ceil(rec.shape[0]/2))
        rec_temp = rec[np.arange(mid_T-small_t,mid_T+small_t)]
        sd = np.median(np.abs(rec),0)/0.6745;
        rec = np.divide(rec,sd)

        pp.closeFile()
        
        T,C = rec.shape
        idxNoise = np.zeros((T,C))
        
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
    
        
    def make_training_data(self,R,D,k_idx,min_amp,nspikes):
        """
            Makes the training data for the neural network detector, the autoencoder and the triage network 

            Parameters:
            -----------
            R: int
                temporal radius (half length) of the training data.
            D: int
                number of channels away from the mainchannel. D=1 if only look at the main channel; D=2 if look at the main 
                channel and its neighboring channels; D=3 if look at the main channel, the neighboring channels and the neighbors 
                of the neighboring channels.
            k_idx: np.array
                [ncluster,] indices of the clusters whose waveforms will be used for making the training data. k_idx could
                be a (real) subset of the whole cluster indices if some of the clusters have low energy and bad shapes and 
                should not enter the function.
            min_amp: float
                minimum number for the maximum amplitude of the spike on the main channel for all the generated augmented spikes.
                This could be defined by visual inspection. Empirically min_amp = 4 works well for most of the cases.
            nspikes: int
                number of spikes to be generated. Notice that this is not necessarily the total number of the output training
                data (see Returns for details).

            Returns:
            -----------
            x_detect: np.array
                [5*nspikes, 2*R+1, d] training data for the neural network detector where d depends on D. This training data
                contains isolated spikes, noise, colliding spikes and misaligned spikes. The channels are ordered with respect
                to their distance to the first channel (x_detect[:,:,0]).
            y_detect: np.array
                [5*nspikes,1] label for x_detect, 1 for isolated spikes and collisions and 0 otherwise. 
            x_triage: np.array
                [3*nspikes, 2*R+1, d] training data for the triage network. This training data contains isolated spikes and                       colliding spikes.
            y_triage: np.array
            [5*nspikes,1] label for x_triage, 1 for isolated spikes and 0 otherwise. 
            x_ae: np.array
                [3*nspikes, 2*R+1] training data for the autoencoder. This training data contains noisy isolated spikes.
            y_ae: np.array
                [3*nspikes, 2*R+1] training data for the autoencoder. This training data contains clean isolated spikes (isolated
                spikes before superimposed with the noise).
        """

        # get templates with big temporal size and align
        templatesBig = self.getBigTemplates(4*R)
        templatesBig = templatesBig[k_idx]
        k_idx2 = np.max(np.max(templatesBig,axis=1),axis=1) > 4
        templatesBig = templatesBig[k_idx2]
        K = templatesBig.shape[0]

        templatesBig2, geom, neighChannels = self.crop_templates(templatesBig,R,D+1)

        amps = np.max(np.abs(templatesBig2),axis=1)
        mainc = amps > np.expand_dims(np.max(amps,axis=1)*0.8,-1)

        # make clean augmented spikes
        nk = int(np.ceil(nspikes/K))
        max_amp = np.max(amps)*1.5
    
        nneigh = np.max(np.sum(self.config.neighChannels, 0))
        
        x_clean = np.zeros((nk*K,templatesBig2.shape[1],templatesBig2.shape[2]))
        for k in range(K):
            tt  = templatesBig2[k]    
            amp_now = np.max(np.abs(tt))
            amps_range = (np.arange(nk)*(max_amp-min_amp)/nk+min_amp)[:,np.newaxis,np.newaxis]                    
            x_clean[k*nk:(k+1)*nk] = (tt/amp_now)[np.newaxis,:,:]*amps_range
        x_clean = x_clean[:,:,:nneigh]

        x_collision = np.zeros(x_clean.shape)
        max_shift = 2*R

        temporal_shifts = np.random.randint(max_shift*2, size = nk*K) - max_shift
        temporal_shifts[temporal_shifts<0] = temporal_shifts[temporal_shifts<0]-5
        temporal_shifts[temporal_shifts>=0] = temporal_shifts[temporal_shifts>=0]+5

        amp_per_data = np.max(x_clean[:,:,0],axis=1)
        #random_match = np.random.randint(nk*K, size = nk*K)

        mid_point = (x_clean.shape[1]-1)/2

        for j in range(nk*K):
            shift = temporal_shifts[j]

            x_collision[j] = np.copy(x_clean[j])
            idx_candidate = np.where(np.logical_and(amp_per_data > amp_per_data[j]*0.5, amp_per_data < amp_per_data[j]*2))[0]
            idx_match = idx_candidate[np.random.randint(idx_candidate.shape[0], size = 1)[0]]
            #idx_match = random_match[j]
            x_clean2 = np.copy(x_clean[idx_match][:,np.random.choice(nneigh, nneigh, replace=False)])

            if shift > 0:
                x_collision[j,:(x_collision.shape[1]-shift)] += x_clean2[shift:]

            elif shift < 0:
                x_collision[j,(-shift):] += x_clean2[:(x_collision.shape[1]+shift)]
            else:
                x_collision[j] += x_clean2

        # collision
        x_collision2 = np.zeros(x_clean.shape)
        max_shift = 2*R

        temporal_shifts = np.random.randint(max_shift*2, size = nk*K) - max_shift
        temporal_shifts[temporal_shifts<0] = temporal_shifts[temporal_shifts<0]-5
        temporal_shifts[temporal_shifts>=0] = temporal_shifts[temporal_shifts>=0]+6

        mid_point = (x_clean.shape[1]-1)/2

        for j in range(nk*K):
            shift = temporal_shifts[j]
            x_clean2 = np.copy(x_clean[j][:,np.random.choice(nneigh, nneigh, replace=False)])

            if shift > 0:
                x_collision2[j,:(x_collision.shape[1]-shift)] += x_clean2[shift:]

            elif shift < 0:
                x_collision2[j,(-shift):] += x_clean2[:(x_collision.shape[1]+shift)]
            else:
                x_collision2[j] += x_clean2


        x_collision3 = np.zeros(x_clean.shape)
        max_shift = 2*R

        temporal_shifts = np.random.randint(max_shift*2, size = nk*K) - max_shift
        temporal_shifts[temporal_shifts<0] = temporal_shifts[temporal_shifts<0]-5
        temporal_shifts[temporal_shifts>=0] = temporal_shifts[temporal_shifts>=0]+5

        amp_per_data = np.max(x_clean[:,:,0],axis=1)
        #random_match = np.random.randint(nk*K, size = nk*K)

        mid_point = (x_clean.shape[1]-1)/2

        for j in range(nk*K):
            shift = temporal_shifts[j]

            x_collision3[j] = np.copy(x_clean[j])
            idx_candidate = np.where(np.logical_and(amp_per_data > amp_per_data[j]*0.5, amp_per_data < amp_per_data[j]*2))[0]
            idx_match = idx_candidate[np.random.randint(idx_candidate.shape[0], size = 1)[0]]
            #idx_match = random_match[j]
            x_clean2 = np.copy(x_clean[idx_match])

            if shift > 0:
                x_collision3[j,:(x_collision.shape[1]-shift)] += x_clean2[shift:]

            elif shift < 0:
                x_collision3[j,(-shift):] += x_clean2[:(x_collision.shape[1]+shift)]
            else:
                x_collision3[j] += x_clean2


        spatial_SIG, temporal_SIG = self.determineNoiseCov(x_clean.shape[1],1)
        # get noise
        noise = np.random.normal(size=x_clean.shape)
        for c in range(noise.shape[2]):
            noise[:,:,c] = np.matmul(noise[:,:,c],temporal_SIG)

        reshaped_noise = np.reshape(noise,(-1,noise.shape[2]))
        noise = np.reshape(np.matmul(reshaped_noise,spatial_SIG),[x_clean.shape[0],x_clean.shape[1],x_clean.shape[2]])

        #y_clean = np.zeros((x_clean.shape[0],3))
        #y_clean[:,1] = 1

        #y_col = np.zeros((x_clean.shape[0],3))
        #y_col[:,2] = 1

        #y_noise = np.zeros((x_clean.shape[0],3))
        #y_noise[:,0] = 1

        y_clean = np.ones((x_clean.shape[0]))
        y_col = np.ones((x_clean.shape[0]))
        y_col2 = np.zeros((x_clean.shape[0]))
        y_col3 = np.ones((x_clean.shape[0]))
        y_noise = np.zeros((x_clean.shape[0]))
        
        x = np.concatenate( (
                x_clean + noise, 
                x_collision + noise[np.random.permutation(noise.shape[0])], 
                x_collision2 + noise[np.random.permutation(noise.shape[0])],
                x_collision3 + noise[np.random.permutation(noise.shape[0])],
                noise
            ) )
        x_detect = x[:,(mid_point-R):(mid_point+R+1),:]
        y_detect = np.concatenate( (y_clean, y_col, y_col2, y_col3, y_noise) )


        x = np.concatenate( (
                x_clean + noise, 
                x_collision + noise[np.random.permutation(noise.shape[0])], 
                x_collision3 + noise[np.random.permutation(noise.shape[0])],
            ) )
        x_triage = x[:,(mid_point-R):(mid_point+R+1),:]
        y_triage = np.concatenate( (
                y_clean, 
                np.zeros((x_clean.shape[0])), 
                np.zeros((x_clean.shape[0])), 
                            ) )

        ae_shift_max = 5
        temporal_shifts_ae = np.random.randint(ae_shift_max*2+1, size = x_clean.shape[0]) - ae_shift_max
        t_mid = (x_clean.shape[1]-1)/2
        y_ae = np.zeros((x_clean.shape[0],2*R+1))
        x_ae = np.zeros((x_clean.shape[0],2*R+1))
        for j in range(x_ae.shape[0]):
            y_ae[j] = x_clean[j,(t_mid-R+temporal_shifts_ae[j]):(t_mid+R+1+temporal_shifts_ae[j]),0]
            x_ae[j] = x_clean[j,(t_mid-R+temporal_shifts_ae[j]):(t_mid+R+1+temporal_shifts_ae[j]),0]+noise[j,(t_mid-R+temporal_shifts_ae[j]):(t_mid+R+1+temporal_shifts_ae[j]),0]    


        return x_detect, y_detect, x_triage, y_triage, x_ae, y_ae
    
    def crop_templates(self,templatesBig,R,D):
        """
            Crops the big templates and keeps their main channels and neighboring channels. Main channel is defined by the 
            channels with the largest absolute value.

            Parameters:
            -----------
            templatesBig: np.array
                [number of templates, temporal length, number of channels]  Big templates. 
            R: temporal length of the Big templates.
            D: int
                number of channels away from the mainchannel.

            Returns:
            -----------
            templatesBig2: np.array 
                [number of templates, temporal length, number of cropped channels] cropped templates with fewer number of 
                channels. All the channels of a template are ordered with respect to their distance to the main channel of 
                that template.
            geom: np.array
                [number of total channels, 2] geometry file for all the electrode channels. The second dimension corresponds to 
                the coordinates.
            neighChannels: np.array
                [number of total channels,number of total channels] boolean matrix indicating the neighboring channels for each
                channel with the spatial radius specified by D.
        """
        
        
                

            
        K = templatesBig.shape[0]
        mainC = np.argmax(np.amax(np.abs(templatesBig),axis=1),axis=1)

        amps = np.amax(np.abs(templatesBig),axis=(1,2))

        K_big = np.argmax(amps)
        max_amp = max(amps)*1.2

        templates_mainc = np.zeros((K,templatesBig.shape[1]))
        t_rec = templatesBig[K_big,:,mainC[K_big]]
        t_rec = t_rec/np.sqrt(np.sum(np.square(t_rec)))
        for k in range(K):
            t1 = templatesBig[k,:,mainC[k]]
            t1 = t1/np.sqrt(np.sum(np.square(t1)))
            shift = align_templates(t1,t_rec)
            if shift > 0:
                templates_mainc[k,:(templatesBig.shape[1]-shift)] = t1[shift:]
                templatesBig[k,:(templatesBig.shape[1]-shift)] = templatesBig[k,shift:]

            elif shift < 0:
                templates_mainc[k,(-shift):] = t1[:(templatesBig.shape[1]+shift)]        
                templatesBig[k,(-shift):] = templatesBig[k,:(templatesBig.shape[1]+shift)]        

            else:
                templates_mainc[k] = t1

        R2 = R
        center = np.argmax(np.convolve(np.sum(np.square(templates_mainc),0),np.ones(2*R2+1),'valid')) + R2
        templatesBig = templatesBig[:,(center-3*R):(center+3*R+1)]

        neighChanBig = n_steps_neigh_channels(self.config.neighChannels, D)
        c_ref = np.argmax(np.sum(neighChanBig,0))
        nChannels = np.sum(neighChanBig[c_ref])

        geom = self.config.geom[neighChanBig[c_ref]]
        neighChannels     = (squareform(pdist(geom)) <= self.config.recordings.spatial_radius)
        
        
        templatesBig2 = np.zeros((templatesBig.shape[0],templatesBig.shape[1],nChannels))
        for k in range(K):
            ch_idx = np.where(neighChanBig[mainC[k]])[0]
            ch_idx, temp = order_channels_by_distance(mainC[k],ch_idx, self.config.geom)
            templatesBig2[k,:,:ch_idx.shape[0]] = templatesBig[k][:,ch_idx]

        return templatesBig2, geom, neighChannels
    
#align templates
def align_templates(t1,t2):
    temp = np.convolve(t1,np.flip(t2,0),'full')
    shift = np.argmax(temp)
    return shift - t1.shape[0] + 1    
