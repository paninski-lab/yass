YASS: Yet Another Spike Sorter
================================


.. image:: https://travis-ci.org/paninski-lab/yass.svg?branch=master
    :target: https://travis-ci.org/paninski-lab/yass.svg?branch=master


.. image:: https://readthedocs.org/projects/yass/badge/?version=latest
    :target: http://yass.readthedocs.io/en/latest/?badge=latest


.. image:: https://badges.gitter.im/paninski-lab/yass.svg
    :target: https://gitter.im/paninski-lab/yass?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


[UPDATE May 2019] 
------------------
YASS ver. 1.0 is being prepared for release sometime in June 2019. The main branch has not been updated to the main
development branch in > 1 year and we do not recommend its use at this time. We are actively looking fore beta-testers so feel free to send
feedback or requests for participation.

Reference
---------

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928


Installation (Note: The installation instructions are for yass V0.5 which has been deprecated as of Feb 2018).
------------

Installing the last stable version:


.. code-block:: shell

   pip install yass-algorithm[tf]


The above command will install yass and all its dependencies (including)
tensorflow (CPU), for GPU do :code:`pip install yass-algorithm[tf-gpu]`.

If you have Tensorflow already installed, running :code:`pip install yass-algorithm`
will install yass and its dependencies except for Tensorflow. For more
information regarding Tensorflow installation see `this`_.

.. _this: https://www.tensorflow.org/install/pip


If you are feeling adventurous, you can install from the master branch:


.. code-block:: shell

    pip install git+git://github.com/paninski-lab/yass@master

Example
-------

We are currently updating our documentation, this section
will be updated once we have an example with publicly available
data.


Documentation
-------------

Documentation hosted at `https://yass.readthedocs.io`_


.. _https://yass.readthedocs.io: https://yass.readthedocs.io

Running tests
-------------

Note: this is indented only for YASS developers, our testing
data is not publicly available.

Before running the tests, download the testing data:


.. code-block:: shell

    export YASS_TESTING_DATA_URL=[URL-TO-TESTING-DATA]

    make download-test-data

    make test

To run tests and flake8 checks (from the root folder):

.. code-block:: shell

    pip install -r requirements.txt

    make test


Building documentation
----------------------

You need to install graphviz to build the graphs included in the
documentation. On macOS:


.. code-block:: shell

    brew install graphviz


To build the docs (from the root folder):

.. code-block:: shell

    pip install -r requirements.txt

    make docs


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
