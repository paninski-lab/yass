"""Provides a set of standard plots for visualizing evaluations."""

import numpy as np
import matplotlib.pyplot as plt


class ChristmasPlot(object):
    """Standard figure for evaluation comparison vs. template properties."""

    def __init__(self, data_set_title, n_dataset=1, methods=['Method'],
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
        self.new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                           '#bcbd22', '#17becf']
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
                   method_name='Method'):
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

    def generate_snr_metric_plot(self):
        """Generate pdf plots of evaluations for the datasets and methods."""
        self.fig, self.ax = plt.subplots(self.n_dataset, 1)
        for i in range(self.n_dataset):
            self.ax[i].set_title(
                    self.data_set_title + 'Dataset {}'.format(i + 1))
            self.ax[i].set_ylabel('Percent {}'.format(self.eval_type))
            self.ax[i].legend(self.methods)
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
                    self.ax[i].scatter(
                        metric_tuple[0], metrics,
                        color=self.new_colors[method_idx])
                except Exception:
                    print("No metric found for {} for dataset {}".format(
                        method, i + 1))
        self.fig.set_size_inches(12, 4 * self.n_dataset)
        plt.savefig('{}_{}.pdf'.format(self.data_set_title, self.eval_type))
