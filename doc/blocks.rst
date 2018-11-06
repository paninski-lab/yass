Using pre-built pipeline
========================

Note: this document is outdated. The default pipeline has changed but not
documented yet.

YASS provides with a pre-built pipeline for spike sorting, which consists of
five parts: preprocess, detect, cluster, make templates and deconvolute.


Preprocess
----------

.. blockdiag::
    :desctable:

    blockdiag {
       default_fontsize = 15;
       node_width = 120;
       node_height = 60;


       filter -> standarize -> whiten;

       filter [label="Butterworth filter", description="Apply filtering to the n_observations x n_channels data matrix (optional)"]
       standarize [label="Standarize", description="Standarize data matrix"]
       whiten [label="Whitening", description="Compute whitening filter"]
    }



See :doc:`api/preprocess` for details.


Detect
------

.. blockdiag::
    :desctable:

    blockdiag {
       default_fontsize = 15;
       node_width = 120;
       node_height = 60;

       threshold_detector -> pca -> whiten_scores;

       neural_network_detector -> autoencoder;


       threshold_detector [label="Threshold detector", description="Detect spikes using a threshold"]
       pca [label="PCA", description="Dimensionality reduction using PCA"]
       whiten_scores [label="Whiten scores", description="Apply whitening to PCA scores"]
       neural_network_detector [label="Neural Network detector", description="Detect spikes using a Neural Network"]
       autoencoder [label="Autoencoder", description="Dimensionality reduction using an autoencoder"]

    }


See :doc:`api/detect` for details.

Cluster
-------

See :doc:`api/cluster` for details.

Templates
---------

See :doc:`api/templates` for details.


Deconvolve
----------

.. blockdiag::
    :desctable:

    blockdiag {
       default_fontsize = 15;
       node_width = 120;
       node_height = 60;


       deconvolution -> merge

       deconvolution [label="Deconvolution", description="Deconvolute unclear spikes using the templates"]
       merge [label="Merge", description="Merge all spikes to produce the final ouput"]
    }


See :doc:`api/deconvolute` for details.
