Using pre-built pipeline
========================

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


       filter -> standarize -> spike_detection -> waveforms;
       standarize ->  whiten -> waveforms;

       filter [label="Butterworth filter", description="Apply filtering to the n_observations x n_channels data matrix"]
       standarize [label="Standarize", description="Standarize data matrix"]
       spike_detection [label="Spike detection", description="Detect spikes using threshold or neural network"]
       whiten [label="Whitening", description="Apply whitening to the data matrix"]
       waveforms [label="Waveforms", description="Extract waveforms around detected spikes"]

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

See :doc:`api/deconvolute` for details.

Processing
----------

.. blockdiag::
    :desctable:

    blockdiag {
       default_fontsize = 15;
       node_width = 120;
       node_height = 60;


       triage -> dim_reduction -> triage_2 -> coreset -> mask ->
       cluster -> clean -> templates;

       triage_2 -> coreset [folded];
       cluster -> clean [folded];

       triage [label="Triage waveforms", description="Triage waveforms in clear/unclear"]
       dim_reduction [label="Dimensionality reduction", description="Reduce waveforms dimensionality"]
       triage_2 [label="Outlifer triage", description="Remove some outliers from clear waveforms"]
       coreset [label="Coreset", description="Find coresets"]
       mask [label="Mask", description="Mask data"]
       cluster [label="Cluster", description="Cluster clear waveforms"]
       clean [label="Clean", description=""]
       templates [label="Templates", description="Find waveform templates"]

    }

Deconvolution
-------------

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
