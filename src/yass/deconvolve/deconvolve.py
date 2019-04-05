class Deconv(object):

    def __init__(self, fname_templates, reader, output_directory, CONFIG):

        self.fname_templates = fname_templates
        self.output_directory = output_directory
        self.reader = reader
        self.CONFIG = CONFIG
        
        # initialize 
        self.initialize()
        
    def initialize(self): 
        
        # Cat: TODO: read from CONFIG
        self.threshold = 20.
        self.conv_approx_rank = 10
        self.upsample_max_val = 32.
        self.deconv_gpu = False

        # fixed parameter
        self.max_iter = 5000

    def match_pursuit_function_cpu(self):

        # initialize match pursuit
        mp_object = MatchPursuit_objectiveUpsample(
                                  temps=self.templates,
                                  deconv_chunk_dir=self.deconv_chunk_dir,
                                  standardized_filename=self.standardized_filename,
                                  max_iter=self.max_iter,
                                  upsample=self.upsample_max_val,
                                  threshold=self.threshold,
                                  conv_approx_rank=self.conv_approx_rank,
                                  n_processors=self.CONFIG.resources.n_processors,
                                  multi_processing=self.CONFIG.resources.multi_processing)
        
        print ("  running Match Pursuit...")

        if not os.path.isdir(self.deconv_chunk_dir+'/segs'):
            os.makedirs(self.deconv_chunk_dir+'/segs')

        # collect segments not yet completed
        args_in = []
        for k in range(len(self.idx_list_local)):
            fname_out = (self.deconv_chunk_dir+'/segs'+
                         "/seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
            if os.path.exists(fname_out)==False:
                args_in.append([[self.idx_list_local[k], k],
                                self.buffer_size])

        if len(args_in)>0:
            if self.CONFIG.resources.multi_processing:
                parmap.map(mp_object.run,
                            args_in,
                            processes=self.CONFIG.resources.n_processors,
                            pm_pbar=True)

                #p = mp.Pool(processes = self.CONFIG.resources.n_processors)
                #p.map_async(mp_object.run, args_in).get(988895)
                #p.close()
            else:
                for k in range(len(args_in)):
                    mp_object.run(args_in[k])
        
        # collect spikes
        res = []
        for k in range(len(self.idx_list_local)):
            fname_out = (self.deconv_chunk_dir+
                         "/segs//seg_{}_deconv.npz".format(
                         str(k).zfill(6)))
                         
            data = np.load(fname_out)
            res.append(data['spike_train'])

        print ("  gathering spike trains")
        self.dec_spike_train = np.vstack(res)
        
        # corrected spike trains using shift due to deconv tricks
        #self.dec_spike_train = mp_object.correct_shift_deconv_spike_train(
        #                                            self.dec_spike_train)
        print ("  initial deconv spike train: ", 
                            self.dec_spike_train.shape)

        '''
        # ********************************************
        # * LOAD CORRECT TEMPLATES FOR RESIDUAL COMP *
        # ********************************************
        '''
        
        self.results_fname = os.path.join(self.deconv_chunk_dir, 
                                  "results_post_deconv_pre_merge.npz")
        
        if os.path.exists(self.results_fname)==False:
            # get upsampled templates and mapping for computing residual
            self.sparse_upsampled_templates, self.deconv_id_sparse_temp_map = (
                                    mp_object.get_sparse_upsampled_templates())


            # save original spike ids (before upsampling
            # Cat: TODO: get this value from global/CONFIG
            self.spike_train = self.dec_spike_train.copy()
            self.spike_train[:, 1] = np.int32(self.spike_train[:, 1]/
                                                    self.upsample_max_val)
            self.spike_train_upsampled = self.dec_spike_train.copy()
            self.spike_train_upsampled[:, 1] = self.deconv_id_sparse_temp_map[
                self.spike_train_upsampled[:, 1]]
            np.savez(self.results_fname,
                     spike_train=self.spike_train,
                     templates=self.templates,
                     spike_train_upsampled=self.spike_train_upsampled,
                     templates_upsampled=self.sparse_upsampled_templates)

            np.save(os.path.join(self.deconv_chunk_dir,
                                'templates_post_deconv_pre_merge'),
                                self.templates)
                                
            np.save(os.path.join(self.deconv_chunk_dir,
                                'spike_train_post_deconv_pre_merge'),
                                self.spike_train)

        else:
            print ("  reading sorted data from disk...")
            data = np.load(self.results_fname)
            self.spike_train = data['spike_train']
            self.templates = data['templates']
            self.spike_train_upsampled = data['spike_train_upsampled']
            self.sparse_upsampled_templates = data['templates_upsampled']

        print ("  ... done match pursuit...")