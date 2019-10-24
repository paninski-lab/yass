YASS: Yet Another Spike Sorter
================================


.. image:: https://travis-ci.org/paninski-lab/yass.svg?branch=master
    :target: https://travis-ci.org/paninski-lab/yass.svg?branch=master


.. image:: https://readthedocs.org/projects/yass/badge/?version=latest
    :target: http://yass.readthedocs.io/en/latest/?badge=latest


.. image:: https://badges.gitter.im/paninski-lab/yass.svg
    :target: https://gitter.im/paninski-lab/yass?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


[UPDATE Oct 2019] 
------------------
Yass is undergoing some fixes and improvements this month. If you have already installed yass and are pulling updates,
please note that the CUDA code needs to be recompiled (see instructions below).  This can be achieved using:

.. code-block:: shell

   /src/gpu_deconv4/python setup.py install --force

[UPDATE Sep 2019] 
------------------
Yass ver. 1.0-Beta is now available in the master branch. We are actively looking for beta-testers so feel free to send
feedback or requests for participation (see list below).

Yass has been developed as a spike-sorting platform for high-firing rate, high-collision, large-spatial extent spikes obtained from
retinal recordings.  It contains several components including: spike event detection using neural-networks, clustering 
using mixture-of-finite-mixtures approaches, merging and splitting of neurons, and deconvolution steps involving super-resolution 
and stochastic correction.  Finally, yass implements a simple neuron-template drift model to handle small chages in 
neuron shape over time that are present in retinal recordings.

Yass features in development:

* Phy visualization (alpha)
* Multi-channel drift tracking (alpha)


YASS - Hardware Recommendations For Linux (Ubuntu 18.04)
--------------------------------------------------

Yass depends on GPU-based algorithms at multiple stages of processing.  Accordingly, we currently do not 
support CPU-only hardware configurations (please see the list of hardware recommendations below).

We recommend having a workstation with a minimum of 32GB of CPU-RAM, and a GPU with at least 8GB
of GPU-RAM (e.g. 1080Ti, Titan-XP etc.).  Cloud services such as AWS are also viable (we have successfuly run tests on AWS). 
Workstation CPU-cores can speed up processing at several stages ~linearly with the # of cores (see config.yaml file). 

Yass has been tested with the following configurations:

.. code-block:: shell

    CPUs (single 6-core and dual processor 16-cores)
    Ubuntu 18.04
    NVIDIA driver: 410
    Cuda toolkit: 10.0
    Conda 
    Python: 3.6
    GPU: Titan XP (1050Ti also works; Note: yass requires at least 1 GPU installed)

In addition, yass undergoes many IO operations and we strongly recommend using solid-state-drives (SSDs) with at 
least 5-6 times of free space compared to the size of the dataset.  For example, a 100GB raw binary file (int16) 
will require an additional ~450GB-500GB of free space for saving metadata during processing.


NVIDIA - Driver and Support Files Installations For Linux (Ubuntu 18.04)
--------------------------------------------------

1. We strongly recommend installation of NVIDIA drivers 410+:

https://www.nvidia.com/Download/index.aspx


2. We recommend installation of cuda toolkit 10+:

https://developer.nvidia.com/cuda-toolkit



YASS - Installation Instructions For Linux (Ubuntu 18.04)
--------------------------------------------------

Installing the master branch:

1.1 [Optional] Download anaconda environment manager (recommended):

https://www.anaconda.com/distribution/


1.2 [Optional] Create a conda environment to run yass using python 3.6 (recommended):

.. code-block:: shell

    conda create -n yass python=3.6


1.3 [Optional] Activate conda environment (recommended):

.. code-block:: shell

    source activate yass


2.  Clone the master repository:

.. code-block:: shell

    git clone https://github.com/paninski-lab/yass


3.  Change directory:

.. code-block:: shell

    cd yass
    
4.  Pip install the python code with dependencies:

.. code-block:: shell

   pip install .


5.  Pip install pytorch master [Note conda is now used instead of pip]:

.. code-block:: shell

   conda install pytorch

   
6.  Change directory to CUDA code directory [Note updated CUDA code in Oct/2019]:
   
.. code-block:: shell

   cd src/gpu_deconv4
   
   
7.  Compile cuda code using default gcc:

.. code-block:: shell

   python setup.py install --force
   

Common installation issues involve incompatible gcc versions for pytorch installation and for
cuda code.  We recommend using gcc 5 and pytorch 1.10 as they have been tested.

   
Running Default Test
-------------------

Yass comes with a small neurophysiology recording data file (60 second; 10 channels) for testing the install. To run
this test:

1.  Change directory to main directory of dataset:

.. code-block:: shell

   cd samples/10chan
   
2.  Run test using default configuration:

.. code-block:: shell

   yass sort config.yaml
      
If yass runs successfully, several files will be generated in your root directory.

.. code-block:: shell
   
    ├── data.bin
    ├── config.yaml
    ├── geom.txt
    ├── tmp
    │   ├── block_1
    │   ├── block_2
    │   ├── final_deconv
    │   ├── spike_train.npy
    │   ├── templates.npy
    │   └── yass.log

The spike_train.npy file is a 2-column python numpy array containing spiketimes (first column)
and cluster/neuron ids (second column)

The templates.npy file is a python numpy array containing the neuron templates shapes.


Running Additional Datasets
---------------------------

1.  Make a directory that will hold your data:

.. code-block:: shell

   mkdir ../data

2.  Copy the config.yaml file to the new directory:

.. code-block:: shell

   cp config.yaml ../data
   
3.  Edit the config.yaml file (using any editor) and modify the file location parameters:

.. code-block:: shell

    data:
      root_folder: [insert folder location of install]
      
      # recordings filename (must be a binary file), details about the recordings
      # are specified in the recordings section
      recordings: [insert binary filename]
      
      # channel geometry filename , supports txt (one x, y pair per line,
      # separated by spaces) or a npy file with shape (n_channels, 2),
      # where every row contains a x, y pair. see yass.geometry.parse for details
      geometry: [insert name of geometry text file]

4.  Edit the config.yaml file (using any editor) and modify the recording parameters:

.. code-block:: shell

    recordings:
      # precision of the recording – must be a valid numpy dtype
      dtype: int16 [only int16 is supported currently]
      
      # recording rate (in Hz)
      sampling_rate: [sampling rate] 
      
      # number of channels
      n_channels: [number of channels]
      
      # channels spatial radius to consider them neighbors, see
      # yass.geometry.find_channel_neighbors for details
      spatial_radius: [distance between channels + 10]
      
      # temporal length of waveforms in ms. It must capture
      # the full shape of waveforms but longer means slower
      spike_size_ms: 3 [3ms is default]
      
      # chunks to run clustering on (in seconds)
      # leave blank to run clustering on entire dataset [not recommended]
      clustering_chunk: [0, 300]  # default clustering is run on first 5mins of data
      
      # chunks to run final deconv on (in seconds)
      # leave blank to run it on full
      final_deconv_chunk:         # default leave blank


5.  Modify GPU and CPU processing parameters as required (contact yass developers for additional assistance):

.. code-block:: shell

    resources:
      # CPU multi-processing flag: 1 = use multiple cores
      multi_processing: 1
      
      # Number of CPU cores to use; recommended to set to # of physical cores available on CPU
      n_processors: 16
      
      # Length of processing chunks; if memory issues arise, decrease value
      n_sec_chunk: 10
      
      # number of GPUs to use [multi-gpu options being currently implemented]
      n_gpu_processors: 1
      
      # n_sec_chunk for gpu detection; if memory issues arise, decrease value
      n_sec_chunk_gpu_detect: 0.5
      
      # n_sec_chunk for gpu deconvolution; if memory issues arise, decrease value
      n_sec_chunk_gpu_deconv: 5


Contributors
------------

`Peter Lee`_, `Eduardo Blancas`, `Nishchal Dethe`_, `Shenghao Wu`_,
`Hooshmand Shokri`_,  `Catalin Mitelut`_, `Colleen Rhoades`, `Nora Brackbill`, `Alexandra Kling`,
`David Carlson`, `Denis Turcu`,
`EJ Chichilnisky`, `Liam Paninski`

.. _Peter Lee: https://github.com/pjl4303
.. _Nishchal Dethe: https://github.com/nd2506
.. _Shenghao Wu: https://github.com/ShenghaoWu
.. _Hooshmand Shokri: https://github.com/hooshmandshr
.. _Calvin Tong: https://github.com/calvinytong
.. _Catalin Mitelut: https://github.com/catubc

Reference
---------

A new manuscript will be available shortly.  The older version can be found here:

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928

------------
