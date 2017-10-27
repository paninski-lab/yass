Using pre-built pipelines
=========================


Preprocessing
-------------

.. blockdiag::
    :desctable:

    blockdiag {
       default_fontsize = 15;
       node_width = 120;
       node_height = 60;


       filter -> standarize -> spike_detection -> waveforms;
       standarize ->  whiten -> waveforms;


    }


.. automodule:: yass.preprocess.run
    :members:


Processing
----------

.. blockdiag::
    :desctable:

    blockdiag {
       default_fontsize = 15;
       node_width = 120;
       node_height = 60;


       triage -> coreset -> mask -> cluster -> templates -> clean;

       mask -> cluster [folded];

    }

.. automodule:: yass.process.run
    :members:


Deconvolution
-------------
