import yass.reordering.utils 
import yass.reordering.cluster
import yass.reordering.default_params
import yass.reordering
from yass import read_config
import numpy as np
from yass.config import Config
from yass.reordering.preprocess import get_good_channels
import os
import cupy as cp

class PARAM:
    pass

class PROBE:
    pass

class ReadReorder(object):
	''' Class that reorders the raw binary standardized file based on
		output of rastermap function.
	'''
	
	def __init__(self):
		pass
       
	def read(self, data_start, length_of_chunk, channels=None):
		with open(self.bin_file, "rb") as fin:
            # Seek position and read N bytes
			fin.seek(int(data_start*self.dtype.itemsize), os.SEEK_SET)
			data = np.fromfile(fin, dtype=self.dtype,
				count=length_of_chunk)
		fin.close()

		data = data.reshape(-1, self.n_chans)

		return data    

	def reorder(self):
        
		# fixed value to pad raw data;
		pad_len = 200   
		self.dtype = np.float32([0]).dtype
		zero_chunk = np.zeros((pad_len, self.n_chans),dtype=self.dtype)
		pad_len_samples = pad_len*self.n_chans

		# set chunking information
		#chunk_len= 2

		# Cat: TODO: this may miss small bits of data if irregular ended acquisition
		data_size = int(os.path.getsize(self.bin_file)/self.n_chans/4)
		n_chunks = data_size//(self.sample_rate*self.chunk_len)
		#print ("Data_size: ", data_size, " nchunks: ", n_chunks, " idx: ", chunk_idxs)
			
		# load data in different order
		new_file_name = self.bin_file[:-4]+"_reordered.bin"
		new_file = open(new_file_name, 'wb')
		for idx_ in self.chunk_idxs:
			data_start = idx_*self.sample_rate*self.chunk_len*self.n_chans
			length_of_chunk = self.sample_rate*self.chunk_len*self.n_chans

			# if neither last nor first grab chunk + padding
			if (idx_ != 0) and (idx_!=(n_chunks-1)):
				chunk = self.read(data_start-pad_len_samples, length_of_chunk+pad_len_samples*2)
				#print(" start: ", data_start-pad_len_samples, " , length: ", length_of_chunk+pad_len_samples*2)
			elif (idx_==0):
				chunk = self.read(data_start, length_of_chunk+pad_len_samples)
				#print(" start: ", data_start, " , length: ", length_of_chunk+pad_len_samples)
				chunk = np.hstack((zero_chunk.T,chunk.T)).T
			elif (idx_==(n_chunks-1)):
				chunk = self.read(data_start-pad_len_samples, length_of_chunk+pad_len_samples)
				#print(" start: ", data_start-pad_len_samples, " , length: ", length_of_chunk+pad_len_samples)
				chunk = np.hstack((chunk.T, zero_chunk.T)).T

			new_file.write(chunk)

		new_file.close()
        
        # delete old standardized file and overwrite with the new one.
		os.system('mv '+ self.bin_file + " " + self.bin_file[:-4]+"_original.bin")       
		os.system('mv '+ new_file_name + " " + self.bin_file)       
                
        
def run(save_fname, standardized_fname, CONFIG, nPCs = 3,  nt0 = 61, reorder = True, dtype = np.float32 ):
   
    params = PARAM()
    probe = PROBE()

    params.sigmaMask = 30
    params.Nchan = CONFIG.recordings.n_channels
    params.nPCs = nPCs
    params.fs = CONFIG.recordings.sampling_rate
    
    #magic numbers from KS
    #params.fshigh = 150.
    #params.minfr_goodchannels = 0.1
    params.Th = [10, 4]

    #spkTh is the PCA threshold for detecting a spike
    params.spkTh = -6 
    params.ThPre = 8
    ##
    params.loc_range = [5, 4]
    params.long_range = [30, 6]

    probe.chanMap = np.arange(params.Nchan)
    probe.xc = CONFIG.geom[:, 0]
    probe.yc =  CONFIG.geom[:, 1]
    probe.kcoords = np.zeros(params.Nchan)
    probe.Nchan = params.Nchan
    shape = (params.Nchan, CONFIG.rec_len)
    standardized_mmemap = np.memmap(standardized_fname, order = "F", dtype = dtype)
    params.Nbatch = np.ceil(CONFIG.rec_len/(CONFIG.resources.n_sec_drift_chunk*CONFIG.recordings.sampling_rate)).astype(np.int16)
    params.reorder = reorder
    params.nt0min = np.ceil(20 * nt0 / 61).astype(np.int16)


    result = yass.reordering.cluster.clusterSingleBatches(proc = standardized_mmemap,
        params =  params, 
        probe =  probe,
        yass_batch = params.Nbatch, 
        n_chunk_sec = int(CONFIG.resources.n_sec_drift_chunk*CONFIG.recordings.sampling_rate),
        nt0 = nt0)
    
    # save chunk order and reorder file
    print ("   saving chunk order: ", save_fname)
    chunk_ids = cp.asnumpy(result['iorig'])
    dir_path = os.path.dirname(os.path.realpath(save_fname))
    np.save(os.path.join(dir_path, "ccb0"), cp.asnumpy(result['ccb0']))
    np.save(save_fname, chunk_ids)


	# initialize READER
    RR = ReadReorder()

	# initialize data
    RR.sample_rate = CONFIG.recordings.sampling_rate
    RR.bin_file = os.path.join(CONFIG.data.root_folder, 
							'tmp/preprocess/standardized.bin')
    print (RR.bin_file)
    RR.chunk_len = CONFIG.resources.n_sec_drift_chunk
    RR.n_chans = CONFIG.recordings.n_channels

    RR.chunk_idxs = chunk_ids
    RR.reorder()


def reorder_spike_train(CONFIG, spike_train_fname):
	''' Re order the spike trains obtained from monotonic drift
		version of standardized file back to original temporal order
	'''

	spike_train = np.load(spike_train_fname)
    
	indexes = np.load(os.path.join(CONFIG.data.root_folder,
		'tmp/preprocess/reorder.npy'))
	sample_rate = CONFIG.recordings.sampling_rate
	pad_len = 200
	chunk_len = CONFIG.resources.n_sec_drift_chunk
    
	spike_train_reordered = np.zeros((0,2),'int32')
	for ctr, idx in enumerate(indexes):
		start = ctr*(sample_rate*chunk_len+pad_len*2)+pad_len
		end = (start + chunk_len*sample_rate)
		temp_idx = np.where(np.logical_and(spike_train[:,0]>=start,spike_train[:,0]<end))[0]

		local_spike_train = spike_train[temp_idx]
		# reset to zero index
		local_spike_train[:,0] -= ctr*(sample_rate*chunk_len+pad_len*2)+pad_len
		local_spike_train[:,0] += idx*sample_rate*chunk_len
				
		spike_train_reordered = np.concatenate((spike_train_reordered,
											   local_spike_train))

	# move old spike train
	os.system('mv '+ spike_train_fname + " " + 
				spike_train_fname[:-4]+"_driftorder.npy")       
	np.save(spike_train_fname, spike_train_reordered)

