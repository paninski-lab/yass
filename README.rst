YASS: Yet Another Spike Sorter
================================


.. image:: https://travis-ci.org/paninski-lab/yass.svg?branch=master
    :target: https://travis-ci.org/paninski-lab/yass.svg?branch=master


.. image:: https://readthedocs.org/projects/yass/badge/?version=latest
    :target: http://yass.readthedocs.io/en/latest/?badge=latest


.. image:: https://badges.gitter.im/paninski-lab/yass.svg
    :target: https://gitter.im/paninski-lab/yass?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


[UPDATE Sep 2019] 
------------------
YASS ver. 1.0 is now in the master branch. We are actively looking fore beta-testers so feel free to send
feedback or requests for participation.


INSTALLATION INSTRUCTIONS FOR LINUX (UBUNTU 18.04)
--------------------------------------------------

Installing the master branch:

1.1 [Optional] Download anaconda environment manager (strongly recommended):

https://www.anaconda.com/distribution/

1.2 [Optional] Create a conda environment to run yass using python 3.6 (strongly recommended):

.. code-block:: shell
    conda create -n yass python=3.6

1.3 [Optional] Activate conda environment:

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
   
5.  Change directory to CUDA code directory:
   
.. code-block:: shell
   cd src/gpu_deconv3
   
6.  Compile cuda code using default gcc:

.. code-block:: shell
   python setupy.py install --force
   
   
RUNNING SAMPLE TEST
-------------------

Yass comes with a small neurophysiology recording data file (1 minute; 49 channels).

7.  Change directory to main directory of dataset:

.. code-block:: shell
   cd samples/49chan
   
8.  Run test using default configuration:

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


RUNNING ADDITIONAL TESTS/DATASETS
---------------------------------

9.  Make a directory that will hold your data:

.. code-block:: shell
   mkdir ../data

10.  Copy the config.yaml file to the new directory:

.. code-block:: shell
   cp config.yaml ../data
   
11.  Edit the config.yaml file (using any editor) and modify the file location parameters:

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

12.  Edit the config.yaml file (using any editor) and modify the recording parameters:

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


13.  Modify GPU and CPU processing parameters as required (contact yass developers for additional assistance):

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

`Peter Lee`_, `Eduardo Blancas`_, `Nishchal Dethe`_, `Shenghao Wu`_,
`Hooshmand Shokri`_, `Calvin Tong`_, `Catalin Mitelut`_

.. _Peter Lee: https://github.com/pjl4303
.. _Eduardo Blancas: https://blancas.io
.. _Nishchal Dethe: https://github.com/nd2506
.. _Shenghao Wu: https://github.com/ShenghaoWu
.. _Hooshmand Shokri: https://github.com/hooshmandshr
.. _Calvin Tong: https://github.com/calvinytong
.. _Catalin Mitelut: https://github.com/catubc

Reference
---------

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928

------------
