"""Provides a set of standard plots for visualizing evaluations."""

import numpy as np
import matplotlib.pyplot as plt

from yass.evaluate.util import main_channels


class ChristmasPlot(object):
    """Standard figure for evaluation comparison vs. template properties."""

    def __init__(self, data_set_title, n_dataset=1, methods=['Yass'],
                 logit_y=True, eval_type='Accuracy'):
        """Setup pyplot figures.

        Parameters
        ----------
        data_set_title: str
            Title of the data set that will be displayed in the plots.
        n_dataset: int
            Total umber of data sets that evaluations are performed on.
        methods: list of str
            The spike sorting methods that evaluations are done for.
        logit_y: bool
            Logit transform the y-axis (metric axis) to emphasize near 1
            and near 0 values.
        eval_type: str
            Type of metric (for display purposes only) which appears in the
            plots.
        """
        self.new_colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                           '#bcbd22', '#17becf')
        self.method_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p',
                               '*', 'h', 'H', 'D', 'd', 'P', 'X')
        self.n_dataset = n_dataset
        self.methods = methods
        self.data_set_title = data_set_title
        self.eval_type = eval_type
        self.logit_y = logit_y
        self.data_set_title = data_set_title
        # Contains evaluation metrics for methods and datasets.
        self.metric_matrix = {}
        for method in self.methods:
            self.metric_matrix[method] = []
            for i in range(self.n_dataset):
                self.metric_matrix[method].append(None)

    def logit(self, x, inverse=False):
        """Logit transfors the array x.

        Parameters
        ----------
        x: numpy.ndarray
            List of values [0-1] only to be logit transformed.
        inverse: bool
            Inverse-logit transforms if True.
        """
        # Add apsilon to avoid boundary conditions for transform.
        x[x == 0] += 0.0001
        x[x == 1] -= 0.0001
        if inverse:
            return 1 / (1 + np.exp(-x))
        return np.log(x / (1 - x))

    def set_logit_labels(
            self, labs=np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])):
        """Logit transforms the y axis.

        Parameters
        ----------
        labs: numpy.ndarray
            List of values ([0-1] only) to be displayed as ticks on
            y axis.
        """
        for i in range(self.n_dataset):
            self.ax[i].set_yticks(self.logit(labs))
            self.ax[i].set_yticklabels(labs)

    def add_metric(self, snr_list, percent_list, dataset_number=0,
                   method_name='Yass'):
        """Adds accuracy percentages for clusters/units of a method.

        Parameters
        ----------
        snr_list: numpy.ndarray of shape(N,)
            List of SNR/PNR values for clusters/units of the corresponding
            dataset number and spike sorting method.
        percent_list: numpy.ndarray of shape(N,)
            List of SNR/PNR values for clusters/units of the corresponding
            dataset number and spike sorting method.
        dataset_number: int
            Value should be between 0 and self.n_dataset - 1. Indicates
            which dataset are the evaluations for.
        method: str
            Should be a member of self.methods. Indicates which of the
            spike sorting methods the evaluations correspond to.
        """
        if method_name not in self.methods:
            raise KeyError('Method name does not exist in methods list.')
        if np.any(percent_list < 0) or np.any(percent_list > 1):
            raise TypeError(
                    'Percent accuracy list should contain only [0-1] values.')
        eval_tup = (snr_list, percent_list)
        self.metric_matrix[method_name][dataset_number] = eval_tup

    def generate_snr_metric_plot(self, save_to=None, show_id=False):
        """Generate pdf plots of evaluations for the datasets and methods.

        Parameters:
        -----------
        save_to: str or None
            Absolute path to file where the figure is written to. If None,
            the resulting figure is displayed.
        show_id: bool
            Plot the cluster id of each unit right next to its metric.
        """
        self.fig, self.ax = plt.subplots(self.n_dataset, 1)
        if self.n_dataset == 1:
            self.ax = [self.ax]
        for i in range(self.n_dataset):
            self.ax[i].set_title(
                '{} Dataset {}'.format(self.data_set_title, i + 1))
            self.ax[i].set_ylabel('Percent {}'.format(self.eval_type))
        self.ax[i].set_xlabel('Log PNR')
        if self.logit_y:
            self.set_logit_labels()
        for method_idx, method in enumerate(self.methods):
            for i in range(self.n_dataset):
                try:
                    metric_tuple = self.metric_matrix[method][i]
                    metrics = metric_tuple[1]
                    if self.logit_y:
                        metrics = self.logit(metrics)
                    if show_id:
                        for j in range(len(metrics)):
                            self.ax[i].text(
                                metric_tuple[0][j], metrics[j], str(j))
                    self.ax[i].scatter(
                        metric_tuple[0], metrics,
                        color=self.new_colors[method_idx],
                        marker=self.method_markers[method_idx])
                except Exception as exception:
                    print(exception)
                    print("No metric found for {} for dataset {}".format(
                        method, i + 1))
        self.fig.set_size_inches(16, 6 * self.n_dataset)
        for i in range(self.n_dataset):
            self.ax[i].legend(self.methods)
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()

    def generate_curve_plots(self, save_to=None, min_eval=0.5):
        """Generate curve plots of evaluations for the datasets and methods.

        Parameters:
        -----------
        min_eval: float (0, 1)
            Minimum evaluation rate to be considered.
        save_to: str or None
            Absolute path to file where the figure is written to. If None,
            the resulting figure is displayed.
        """
        self.fig, self.ax = plt.subplots(self.n_dataset, 1)
        if self.n_dataset == 1:
            self.ax = [self.ax]
        for i in range(self.n_dataset):
            self.ax[i].set_title(
                '{} Dataset {}'.format(self.data_set_title, i + 1))
            self.ax[i].set_ylabel(
                '# Units Above x% {}'.format(self.eval_type))
        self.ax[i].set_xlabel(self.eval_type)
        x_ = np.arange(1, min_eval, - 0.01)
        for method_idx, method in enumerate(self.methods):
            for i in range(self.n_dataset):
                try:
                    metric_tuple = self.metric_matrix[method][i]
                    metrics = metric_tuple[1]
                    y_ = np.zeros(len(x_), dtype='int')
                    for j, eval_rate in enumerate(x_):
                        y_[j] = np.sum(metrics > eval_rate)
                    self.ax[i].plot(
                        x_, y_, color=self.new_colors[method_idx],
                        marker=self.method_markers[method_idx], markersize=4)
                except Exception as exception:
                    print (exception)
                    print("No metric found for {} for dataset {}".format(
                        method, i + 1))
        self.fig.set_size_inches(9, 6 * self.n_dataset)
        for i in range(self.n_dataset):
            self.ax[i].set_xlim(1, min_eval)
            self.ax[i].legend(self.methods)
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()


class WaveFormTrace(object):
    """Class for plotting spatial traces of waveforms."""

    def __init__(self, background_clr='black', x_width=20, y_width=10, fig = None, ax1=None):
        ''' It's more flexible to not do anything by default; or just initialize a plot for example
	    Then can load units and do other things on this plotting canvas;
	    This allows overlaying of multiple unit plots on top of each other if necessary;
        
        x_width and y_width are the fgireu sizes
        
	'''
        if fig==None:
            self.fig, self.ax1 = plt.subplots(figsize=(x_width, y_width))
            self.ax1.set_facecolor('xkcd:black')
        else: 
            self.fig = fig
            self.ax1= ax1

    def load_units(self, geometry):
        """Sets up the plotting descriptions for spatial trace.

        Parameters:
        -----------
        geometry: numpy.ndarray shape (C, 2)
            Incidates coordinates of the probes.
        templates: numpy.ndarray shape (T, C, K)
            Where T, C and K respectively indicate time samples, number of
            channels, and number of units.
        """
		
        self.geometry = geometry

    def generate_geometry(self, geom, size=10):
        ''' Generate a plot of electrode locations
        '''
        self.geometry = geom

        for k in range(len(self.geometry)):
            self.ax1.scatter(self.geometry[k, 0], self.geometry[k, 1], color='black',s=size)

    def plot_wave(self, units, n_vis_chans=1, scale=5):
        """Plot spatial trace of the units

        Parameters:
        -----------
        units: list of int
            The units for which the spatial trace will be dispalyed.
        n_vis_chans: int
            Number of channels for which each waveform should be displayed.
        scale: float
            Scale the spikes for display purposes.
        """
        fig, ax = plt.subplots()
        if self.n_channels < n_vis_chans:
            n_vis_chans = self.n_channels
        for unit in units:
            # Only plot the strongest channels based on the given size.
            channels = main_channels(self.templates[:, :, unit])[-n_vis_chans:]
            p = ax.scatter(
                    self.geometry[channels, 0], self.geometry[channels, 1])
            # Get the color of the recent scatter to plot the trace with the
            # same color..
            col = p.get_facecolor()[-1, :3]
            #print (col)
            for c in channels:
                x_ = np.arange(0, self.samples, 1.0)
                x_ += self.geometry[c, 0] - self.samples / 2
                y_ = (self.templates[:, c, unit]) * scale + self.geometry[c, 1]
                ax.plot(x_, y_, color=col, label='_nolegend_')
        #ax.legend(["Unit {}".format(unit) for unit in units])
        ax.set_xlabel('Probe x coordinate')
        ax.set_ylabel('Probe y coordinate')
        fig.set_size_inches(15, 15)
        plt.show()
        
    
    def generate_energy_space_plot(self, units, n_vis_chans=5, color='black'):
        """Plot spatial trace of the units

        Parameters:
        -----------
        units: list of int
            The units for which the spatial trace will be dispalyed.
        n_vis_chans: int
            Number of channels for which each waveform should be displayed.
        scale: float
            Scale the spikes for display purposes.
        """

        for unit in units:
            # Only plot the strongest channels based on the given size.
            channels = main_channels(self.templates[:, :, unit])[-n_vis_chans:]

            #Compute peak-to-peak at each channel
            max_channel = channels[-1]
            locx = 0
            locy = 0
            ptp_sum=0
            for ch in channels:
                ptp = np.max(self.templates[:, ch, unit])-np.min(self.templates[:, ch, unit])
            
                ptp_sum +=ptp
                locx = locx + self.geometry[ch, 0]*ptp
                locy = locy + self.geometry[ch, 1]*ptp
            
            self.ax1.scatter(locx/float(ptp_sum), locy/float(ptp_sum),s=(np.max(self.templates[:, max_channel, unit])-np.min(self.templates[:, max_channel, unit]))*10,color=color,edgecolors='b',alpha=0.5)
            
        self.ax1.set_xlabel('Probe x coordinate',fontsize=25)
        self.ax1.set_ylabel('Probe y coordinate',fontsize=25)
        self.ax1.tick_params(axis='both', which='both', labelsize=20)
        self.ax1.set_ylim(-500,500)
        self.ax1.set_xlim(-1000,1000)

        return

    def generate_template_spatial_location(self, templates, units, n_vis_chans=5, scale = 1.0, color='black'):
        """Plot spatial trace of the units

        Parameters:
        -----------
        units: list of int
            The units for which the spatial trace will be dispalyed.
        n_vis_chans: int
            Number of channels for which each waveform should be displayed.
        scale: float
            Scale the spikes for display purposes.
        """
        
        self.templates = templates
        self.samples = templates[:,:,units[0]].shape[1]
        self.n_channels = templates[:,:,units[0]].shape[0]


        self.ax1.set_facecolor('xkcd:black')
        #if self.n_channels < n_vis_chans:
        #    n_vis_chans = self.n_channels
        text_label = ''
        for unit in units:
            # Only plot the strongest channels based on the given size.
            channels = main_channels(self.templates[:, :, unit])[-n_vis_chans:]
            p = self.ax1.scatter(self.geometry[channels, 0], self.geometry[channels, 1])
            # Get the color of the recent scatter to plot the trace with the
            # same color..
#            col = p.get_facecolor()[-1, :3]
            col = color
            #print (col)
            for c in channels:
                x_ = np.arange(0, self.samples, 1.0)/2.
                x_ += self.geometry[c, 0] - self.samples / 2
                y_ = (self.templates[:, c, unit]) * scale + self.geometry[c, 1]
                self.ax1.plot(x_, y_, color=col, label='_nolegend_', linewidth=2)
            
            text_label+=" "+str(unit)+", max ch: "+str(channels[0])
        
        #ax.legend(["Unit {}".format(unit) for unit in units])
        self.ax1.set_title("Unit "+text_label,fontsize=15)
        self.ax1.set_xlabel('Probe x coordinate')
        self.ax1.set_ylabel('Probe y coordinate')
        #fig.set_size_inches(15, 15)
        #plt.show()


    def generate_traces_spatial_location(self, traces, vis_chans=[0], scale = 1.0, color='black',linewidth=1, alpha=0.1, x_width=None, y_width=None):
        """Plot spatial trace of the units

        Parameters:
        -----------
        units: list of int
            The units for which the spatial trace will be dispalyed.
        n_vis_chans: int
            Number of channels for which each waveform should be displayed.
        scale: float
            Scale the spikes for display purposes.
        """
        
        
        samples = traces.shape[1]
        channels = vis_chans

        self.ax1.set_facecolor('xkcd:white')
        # Only plot the strongest channels based on the given size.
        #channels = main_channels
        #print channels
        
        #Plot chan locations
        p = self.ax1.scatter(self.geometry[channels, 0], self.geometry[channels, 1])
        # Get the color of the recent scatter to plot the trace with the
        col = color

        for ctr,c in enumerate(channels):
            x_ = np.arange(0, samples, 1.0)/2.     #x indexes
            x_ += self.geometry[c, 0] - samples / 2    #Offset x indexes by location of channel in physical space
            #print x_.shape
            x_array = np.tile(x_,(traces.shape[0],1)).T
            #print x_array.shape
            
            y_array = (traces[:,:,c].T) * scale + self.geometry[c, 1]  #Load spike waveform and offset in space
            #print y_array.shape
            
            # only write text in plotting window otherwise looks bad
            if x_width is not None: 
                if (self.geometry[c, 0]>x_width[0]) and (self.geometry[c, 0]<x_width[1]):
                    if (self.geometry[c, 1]>y_width[0]) and (self.geometry[c, 1]<y_width[1]):
                        self.ax1.text(self.geometry[c, 0]+2, self.geometry[c, 1], str(c), fontsize=10)
            else: 
                self.ax1.text(self.geometry[c, 0]+2, self.geometry[c, 1], str(c), fontsize=10)
            
            self.ax1.plot(x_array, y_array, color=col, label='_nolegend_', linewidth=linewidth, alpha=alpha)
        
        #text_label+=" "+str(unit)+", max ch: "+str(channels[0])
        
        #ax.legend(["Unit {}".format(unit) for unit in units])
        #self.ax1.set_title("Unit "+text_label,fontsize=15)
        self.ax1.set_xlabel('Probe x coordinate')
        self.ax1.set_ylabel('Probe y coordinate')
        #fig.set_size_inches(15, 15)
        #plt.show()
     
     
     
    def generate_template_spatial_location_visible_chans(self, template, vis_chans=[0,1,2,3,4,5], scale = 1.0, color='black'):
        """Plot spatial trace of the units

        Parameters:
        -----------
        units: list of int
            The units for which the spatial trace will be dispalyed.
        n_vis_chans: int
            Number of channels for which each waveform should be displayed.
        scale: float
            Scale the spikes for display purposes.
        """
        
        self.template = template
        self.samples = template.shape[1]
        print ("self.samples", self.samples)

        #if self.n_channels < n_vis_chans:
        #    n_vis_chans = self.n_channels
        text_label = ''
        # Only plot the strongest channels based on the given size.
        #channels = main_channels(self.template[:, :, unit])[-n_vis_chans:]
        channels = vis_chans
        print ("vis_chans: ", vis_chans)
        p = self.ax1.scatter(self.geometry[channels, 0], self.geometry[channels, 1])
        
        # Get the color of the recent scatter to plot the trace with the
        # same color..
#            col = p.get_facecolor()[-1, :3]
        col = color
        #print (col)
        for c in range(len(channels)):
            x_ = np.arange(0, self.samples, 1.0)/2.
            x_ += self.geometry[channels[c], 0] - self.samples / 2
            y_ = (self.template[c,:]) * scale + self.geometry[channels[c], 1]
            self.ax1.plot(x_, y_, color=col, label='_nolegend_', linewidth=2)
        
        #text_label+=" "+str(unit)+", max ch: "+str(channels[0])
        #self.ax1.xlim(np.min(), np.max())
        #ax.legend(["Unit {}".format(unit) for unit in units])
        #self.ax1.set_title("Unit "+text_label,fontsize=15)
        self.ax1.set_xlabel('Probe x coordinate')
        self.ax1.set_ylabel('Probe y coordinate')
        #fig.set_size_inches(15, 15)
        #plt.show()   

    def generate_template_multi_channel(self, units, n_vis_chans=5, scale = 1.0, color='black'):
        """Plot spatial trace of the units

        Parameters:
        -----------
        units: list of int
            The units for which the spatial trace will be dispalyed.
        n_vis_chans: int
            Number of channels for which each waveform should be displayed.
        scale: float
            Scale the spikes for display purposes.
        """
        for unit in units:
            # Only plot the strongest channels based on the given size.
            channels = main_channels(self.templates[:, :, unit])[-n_vis_chans:]

	    #Find weighted-average location for each template
            max_channel = channels[-1]
            locx = 0
            locy = 0
            ptp_sum=0
            for ch in channels:
                ptp = np.max(self.templates[:, ch, unit])-np.min(self.templates[:, ch, unit])
	    
                ptp_sum +=ptp
                locx = locx + self.geometry[ch, 0]*ptp
                locy = locy + self.geometry[ch, 1]*ptp
            locx = locx/float(ptp_sum)
            locy = locy/float(ptp_sum)
	
            x_ = np.arange(0, self.samples, 1.0)/2.
            #print (x_)
            x_ = x_+locx - self.samples / 2
            #print (x_)
            y_ = (self.templates[:, max_channel, unit]) * scale + locy
            self.ax1.plot(x_, y_, color=color,alpha=0.5)
	    
        self.ax1.set_xlabel('Probe x coordinate',fontsize=25)
        self.ax1.set_ylabel('Probe y coordinate',fontsize=25)
        self.ax1.tick_params(axis='both', which='both', labelsize=20)
        self.ax1.set_ylim(50,500)
        self.ax1.set_xlim(-850,-325)

        return
	
	
	
    def make_legend(self, legend_list, legend_colors,legend_size=15):
        import matplotlib.patches as mpatches
        
        legend_items=[]
        for name,clr in zip(legend_list,legend_colors):
            legend_items.append(mpatches.Patch(color = clr, edgecolor='black',label = name))

        self.ax1.legend(legend_items, legend_list,loc=1, ncol=2, prop={'size':legend_size}) 

	#Plot second legend using twinx()
        ax2 = self.ax1.twinx()
        #ax2.set_xticks([])
        ax2.set_yticks([])
        plt.scatter([50], [50], c='k', alpha=0.3, s=50, label='MEA Location')
        for area in [10, 100]:
            plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area)+"uV (PTP/SNR)")
        
        plt.legend(scatterpoints=1, loc=4, ncol=3, labelspacing=1, handletextpad=0.0, prop={'size':legend_size})

    def show(self):
        self.fig










