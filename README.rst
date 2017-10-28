YASS: Yet Another Spike Sorter
================================


Installation
------------

Installing the last stable version:


.. code-block:: shell

   pip install git+https://github.com/paninski-lab/yass

Example
-------

Quick example of YASS, try it out:

.. code-block:: shell

    # clone the repo and move to the main folder
    git clone https://github.com/paninski-lab/yass
    cd yass

    # install package (this is the same as running
    # pip install git+https://github.com/paninski-lab/yass)
    pip install .

    # move to the examples folder and run yass in the sample data
    yass config_sample.yaml

    # see the spike train
    cat data/spike_train.csv


You can also use YASS in Python scripts. See the documentation for details.


Documentation
-------------

Documentation hosted at `https://yass.readthedocs.io/en/latest/`_


.. _https://yass.readthedocs.io/en/latest/: https://yass.readthedocs.io/en/latest/

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


Reference
---------

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928