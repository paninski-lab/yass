import numpy as np
import yaml
import sys
#from ruamel.yaml import YAML
import os

import tkinter
from tkinter import *
from tkinter import filedialog

def X_is_running():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0
    
if X_is_running==False:
	print ("  DISPLAY IS NOT SETUP, Use command line only")
else:
	import matplotlib
	matplotlib.use('TkAgg')
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
	from matplotlib.figure import Figure

# widget that manages allthe plotting omdules
class plot_widget:
    def __init__(self,  window):
        self.window = window
	
		# voltage window
        self.fig2 = Figure(figsize=(6,3))
        self.a2 = self.fig2.add_subplot(111)
        self.a2.set_yticks([])
        self.a2.set_xticks([])
        self.a2.set_title("Voltage chunk example", fontsize=8)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.window)
        self.canvas2.get_tk_widget().place(x = 225, y = 300)#, relwidth = , relheight = 1)
        self.canvas2.draw()
	        
		# geometry window
        self.fig1 = Figure(figsize=(3.0,3))
        self.a = self.fig1.add_subplot(111)
        self.a.set_ylabel("um", fontsize=8)
        self.a.set_yticks([])
        self.a.set_xticks([])

        self.a.set_xlabel("um", fontsize=8)
        self.a.set_title("Geometry", fontsize=8)
        self.a.xaxis.set_tick_params(labelsize=8)
        self.a.yaxis.set_tick_params(labelsize=8)

        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.window)
        self.canvas1.get_tk_widget().place(x = 0, y = 300)#, relwidth = , relheight = 1)
        self.canvas1.draw()

    def display_single(self, label_txt, txt, x, y):
	
        labelText=StringVar()
        labelText.set(label_txt)
        labelDir=Label(self.window, textvariable=labelText)
        labelDir.place(x=0,y=y)

        directory=StringVar(None)
        box=Entry(self.window)
        box.insert(0,txt)
        box.place(x=x,y=y, width = 400)
    
        return box
	
	
    def set_filename_denoise(self):
        self.nn_denoise_txt = filedialog.askopenfilename(
				    initialdir = ".",
				    title = "Select detect NN",
				    filetypes = (("NN files","*.pt"),
				    ("All files","*.*")))
	
        self.nn_denoise_box.delete(0, 'end')
        self.nn_denoise_box.insert(0, self.nn_denoise_txt)
		
		
    def refresh(self):
	
		# yaml can't save tuples, so we save them as strings
        self.config_params['recordings']['clustering_chunk'] = \
				str(self.config_params['recordings']['clustering_chunk'])

        self.config_params['neuralnetwork']['detect']['n_filters'] = \
			 str(self.config_params['neuralnetwork']['detect']['n_filters'])
			
        self.config_params['neuralnetwork']['denoise']['n_filters'] = \
			 str(self.config_params['neuralnetwork']['denoise']['n_filters'])
			 
        self.config_params['neuralnetwork']['denoise']['filter_sizes'] = \
			 str(self.config_params['neuralnetwork']['denoise']['filter_sizes'])

			 	
		# update single entries from screen
        self.root_dir = self.root_dir_box.get()

        self.sample_rate = float(self.sample_rate_box.get())
        self.config_params['recordings']['sampling_rate'] = int(self.sample_rate)

        self.n_chan = int(self.n_chan_box.get())
        self.config_params['recordings']['n_channels'] = self.n_chan

        self.radius = float(self.radius_box.get())
        self.config_params['recordings']['spatial_radius'] = self.radius
		
        self.spike_size_ms = float(self.spike_size_ms_box.get())
        self.config_params['recordings']['spike_size_ms'] = self.spike_size_ms

		# redraw everything
		# plot geometry file
        self.window.filemenu.plot.plot_geom()

		# load snipit of data and visualize
        self.window.filemenu.plot.plot_voltage()

        with open(self.fname_config[:-5]+"_modified.yaml", 'w') as f:
        #yaml.dump(self.config_params, self.fname_config)
        #    ruamel.yaml.dump(self.config_params, Dumper=ruamel.yaml.RoundTripDumper)
            yaml.dump(self.config_params, f, default_flow_style=False)
    
		# remove the string quotation marks around the updated tuples
		# Read in the file
        with open(self.fname_config[:-5]+"_modified.yaml", 'r') as file :
            filedata = file.read()

		# Replace the target string
        filedata = filedata.replace("'[", "[")
        filedata = filedata.replace("]'", "]")
        filedata = filedata.replace("null", "")

		# Write the file out again
        with open(self.fname_config[:-5]+"_modified.yaml", 'w') as file: 
          file.write(filedata)
		
    def set_filename_detect(self):
        self.nn_detect_txt = filedialog.askopenfilename(
				    initialdir = ".",
				    title = "Select denoise NN",
				    filetypes = (("NN files","*.pt"),
				    ("All files","*.*")))
	
        self.nn_detect_box.delete(0, 'end')
        self.nn_detect_box.insert(0, self.nn_detect_txt) 


    def nn_detect_button(self, label_txt, txt, x, y):
	#self.filename_loaded = txt
        self.button_detect = Button(self.window,text=label_txt,command=lambda:self.set_filename_detect())
        self.button_detect.place(x=x, y=y)
	
        #directory=StringVar(None)
        self.nn_detect_box=Entry(self.window, width=50)
        self.nn_detect_box.insert(0,txt)
        self.nn_detect_box.place(x=x+95,y=y)


    def nn_denoise_button(self, label_txt, txt, x, y):
	#self.filename_loaded = txt
        button_denoise = Button(self.window,text=label_txt,command=lambda:self.set_filename_denoise())
        button_denoise.place(x=x, y=y)
	
        #directory=StringVar(None)
        self.nn_denoise_box=Entry(self.window, width=50)
        self.nn_denoise_box.insert(0,txt)
        self.nn_denoise_box.place(x=x+95,y=y)
	

    def refresh_button(self, txt, x, y):
	#self.filename_loaded = txt
        button_refresh = Button(self.window,text=txt,command=lambda:self.refresh())
        button_refresh.place(x=x, y=y)

	
    def display_train_run_button(self, label_txt, txt, x, y):
	#self.filename_loaded = txt
        b1 = Button(self.window,text=label_txt,command=lambda:self.set_filename())
        b1.place(x=x, y=y)
	
        directory=StringVar(None)
        dirname=Entry(self.window)
        dirname.insert(0,txt)
        dirname.place(x=x+95,y=y)

    
    def run(self):
		
		# first save the file
        self.refresh()
	
		# run yass
        cmd = "yass sort "+ self.fname_config[:-5]+"_modified.yaml"

        returned_value = os.system(cmd)  # returns the exit code in unix
        #print('returned value:', returned_value)

        print ("")
        print ("")
        print ("  ******** YASS RUN COMPLETE ******** ")
        print ("  see /tmp/spike_train.npy for spike train results ")
        print ("  see /tmp/templates/ for dynamic templates ")

    def nn_retrain(self):
	
        cmd = "yass train "+self.fname_config

        returned_value = os.system(cmd)  # returns the exit code in unix
        #print('returned value:', returned_value)
	
	
    def yass_run(self, txt, x, y):
	
        button_run = Button(self.window,text=txt,command=lambda:self.run())
        button_run.place(x=x, y=y)
		
    def yass_nn_train(self, txt, x, y):
	
        button_run = Button(self.window,text=txt,command=lambda:self.nn_retrain())
        button_run.place(x=x, y=y)
		
	
    def display_metadata_and_buttons(self):
	
        if self.config_params['data']['root_folder']=='./':
            self.data_root = os.path.split(self.fname_config)[0]+'/'
            self.config_params['data']['root_folder'] = self.data_root
        else:
            self.data_root = self.config_params['data']['root_folder']
	    

	# display root directory
        #text_ = self.config_params['data']['root_folder']+self.config_params['data']['recordings']
        text_ = self.data_root+self.config_params['data']['recordings']
        self.root_dir_box = self.display_single("Data loc ", text_, 64, 30)
        self.root_dir = text_
	
	# display sample rate
        text_ = self.config_params['recordings']['sampling_rate']
        self.sample_rate_box = self.display_single("Samp rate ", text_, 64, 60)
        self.sample_rate = int(text_)
	
	# display # of channels
        text_ = self.config_params['recordings']['n_channels']
        self.n_chan_box = self.display_single("# chans ", text_, 64, 90)
        self.n_chan = int(text_)

	# load radius of local chans:
        text_= self.config_params['recordings']['spatial_radius']
        self.radius_box = self.display_single("Neighbour dist ", text_,  70, 120)
        self.radius = float(text_)

	# load radius of local chans:
        text_= self.config_params['recordings']['spike_size_ms']
        self.spike_size_ms_box = self.display_single("Spk width ", text_,  70, 150)
        self.spike_size_ms = float(text_)

	
	# Refresh button
        #text_ = config_params['neuralnetwork']['detect']['filename']
        self.refresh_button("Update config", 500, 30)
	
	# display NN detect
        text_ = self.config_params['neuralnetwork']['detect']['filename']
        self.nn_detect_button("NN detect", text_, 0, 180)
        self.nn_detect = text_

	# display NN denoise
        text_ = self.config_params['neuralnetwork']['denoise']['filename']
        self.nn_denoise_button("NN detect", text_, 0, 210)
        self.nn_denoise = text_

	# run YASS button
        #text_ = config_params['neuralnetwork']['denoise']['filename']
        self.yass_run("Run YASS", 0, 240)
	
	# retrain NNs
        #text_ = config_params['neuralnetwork']['denoise']['filename']
        self.yass_nn_train("NN retrain", 0, 270)
	
	
    def closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
	
        return np.argmin(dist_2)

    def chans_within_radius(self, node, nodes, radius):
        nodes = np.asarray(nodes)
        dist_2 = np.sqrt(np.sum((nodes - node)**2, axis=1))
        #print (dist_2, radius)
	
        return np.where(dist_2<=radius)[0]
	  
    def plot_geom(self):
	
        self.a.cla()
        self.a.tick_params(axis='y', rotation=45)

        self.a.set_title("Geometry (partial)\n(centre chan + neigbhours)", fontsize=8)    

		# load geometry file
        geom_file = self.data_root + self.config_params['data']['geometry']
        #print (geom_file)
        geom = np.loadtxt(geom_file)
        #print ("Geom: ", geom)

        self.a.scatter(geom[:,0],geom[:,1],s=10, color='black')
		
		#find nearest chan to middle:
        middle_chan_x = np.mean(geom[:,0])
        middle_chan_y = np.mean(geom[:,1])
        mid_chan_id = self.closest_node([middle_chan_x, middle_chan_y], geom)
	
        self.a.scatter(geom[mid_chan_id,0],geom[mid_chan_id,1],s=150,color='red')
	
		# find chans within radius
        local_chans_id = self.chans_within_radius(geom[mid_chan_id], geom, self.radius)
	
        self.a.scatter(geom[local_chans_id,0],geom[local_chans_id,1],s=50, color='blue')
	
		# zoom on on arrays to show middle chan + neighbour chans as example
        spacer = 60
        if np.min(geom[local_chans_id,0])>0:
            min_ = np.min(geom[local_chans_id,0])-spacer
        else:
            min_ = np.min(geom[local_chans_id,0])-spacer
        
        if np.max(geom[local_chans_id,0])>0:
            max_ = np.max(geom[local_chans_id,0])+spacer
        else:
            max_ = np.max(geom[local_chans_id,0])+spacer

        self.a.set_xlim([min_, max_])

        if np.min(geom[local_chans_id,1])>0:
            min_ = np.min(geom[local_chans_id,1])-spacer
        else:
            min_ = np.min(geom[local_chans_id,1])-spacer
        
        if np.max(geom[local_chans_id,1])>0:
            max_ = np.max(geom[local_chans_id,1])+spacer
        else:
            max_ = np.max(geom[local_chans_id,1])+spacer
	    
        self.a.set_ylim([min_, max_])

        self.canvas1.draw()
        
        
    def plot_voltage(self): 
        #fname_raw_data = self.data_root + self.config_params['data']['recordings']
        fname_raw_data = self.root_dir#self.data_root + self.config_params['data']['recordings']

        #print ("fname raw data: ", fname_raw_data)
	
        dtype = self.config_params['recordings']['dtype']
        if dtype == 'int16':
            dtype_len = 2
        else:
            print ("  ERROR: Only int6 filetypes currently supported...")
            quit()
	
        self.sampling_rate = self.config_params['recordings']['sampling_rate']
        self.n_channels = self.config_params['recordings']['n_channels']
	
		# get length of data snipit
        length = os.path.getsize(fname_raw_data)
        n_samples = length/float(self.n_channels)/float(dtype_len)
        sec_duration = length/float(self.sampling_rate)/float(self.n_channels)/float(dtype_len)
	
    	# load first 10sec of data
        snipit_duration = 9.0
        rawdata = np.fromfile(fname_raw_data, dtype=dtype, 
                  count=int(self.sampling_rate*self.n_channels*snipit_duration))
		  
		# select random 50ms raw chunk
        rawdata2D = rawdata.reshape(-1, self.n_channels)
        start = int(np.random.choice(np.arange(int(self.sampling_rate*snipit_duration)))-0.050*self.sampling_rate)
        end = start + int(0.050*self.sampling_rate)
	
		# plot voltages
        self.a2.cla()
        self.a2.set_yticks([])
        self.a2.set_title("Voltage (random 10chan, 0.05 sec chunk)", fontsize=8)	
	
        start_ch = np.random.choice(np.arange(self.n_channels)-10)
        if start_ch<0:
            start_ch=0
	    
        x = np.arange(end-start)/float(self.sampling_rate)
        #print (x.shape, end-start)
        for c in range(start_ch,min(start_ch+10, start_ch+self.n_channels)):
            self.a2.plot(x, rawdata2D[start:end,c]+c*100,c='black')
	
        self.canvas2.draw()    
	                
    def load_config(self):
        self.fname_config = filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("Config files","*.yaml"),("All files","*.*")))
	
        self.parse_yaml(self.fname_config)
	

    def parse_yaml(self, filename):
	
        with open(filename, 'r') as f:
	    # The FullLoader parameter handles the conversion from YAML
	    # scalar values to Python the dictionary format
            self.config_params = yaml.load(f, Loader=yaml.FullLoader)
	    
        #print (self.config_params)
        # laod meta data
        self.window.filemenu.plot.display_metadata_and_buttons()
	    
        # plot geometry file
        self.window.filemenu.plot.plot_geom()

		# load snipit of data and visualize
        self.window.filemenu.plot.plot_voltage()

 
