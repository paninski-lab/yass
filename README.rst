YASS: Yet Another Spike Sorter
================================

.. image:: https://travis-ci.org/paninski-lab/yass.svg?branch=master
    :target: https://travis-ci.org/paninski-lab/yass

.. image:: https://readthedocs.org/projects/yass/badge/?version=latest
    :target: http://yass.readthedocs.io/en/latest/?badge=latest

YASS is a state-of-the-art method for spike sorting.

Note: We are currently working on improving YASS to be usable in different types
of neural data. The neural network detector currently only works in reduced
use cases. We will add more models soon.

Reference
---------

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928


Installation
------------

Installing the last stable version:


.. code-block:: shell

   pip install git+https://github.com/paninski-lab/yass

Example
-------

Quick example of YASS using a sample of the neuropixel data from `Nick
Steinmetz`_:

.. _Nick Steinmetz: http://data.cortexlab.net/singlePhase3/

.. code-block:: shell

    # clone the repo and move to the main folder
    git clone https://github.com/paninski-lab/yass
    cd yass

    # install package (this is the same as running
    # pip install git+https://github.com/paninski-lab/yass)
    pip install .

    # move to the examples folder and run yass in the sample data
    cd examples
    yass config_sample.yaml

    # see the spike train
    cat data/spike_train.csv


You can also use YASS in Python scripts. See the documentation for details.


Documentation
-------------

Documentation hosted at `https://yass.readthedocs.io`_


.. _https://yass.readthedocs.io: https://yass.readthedocs.io

Running tests
-------------

.. code-block:: shell

    pytest


Building documentation
----------------------

You need to install graphviz to build the graphs included in the documentation. On macOS:


.. code-block:: shell

    brew install graphviz


.. code-block:: shell

    cd doc
    make [format]


Mantainers
----------

`Peter Lee`_, `Eduardo Blancas`_



.. _Peter Lee: https://github.com/pjl4303
.. _Eduardo Blancas: https://edublancas.github.io/
