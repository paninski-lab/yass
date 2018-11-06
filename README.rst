YASS: Yet Another Spike Sorter
================================


.. image:: https://travis-ci.org/paninski-lab/yass.svg?branch=master
    :target: https://travis-ci.org/paninski-lab/yass.svg?branch=master


.. image:: https://readthedocs.org/projects/yass/badge/?version=latest
    :target: http://yass.readthedocs.io/en/latest/?badge=latest


.. image:: https://badges.gitter.im/paninski-lab/yass.svg
    :target: https://gitter.im/paninski-lab/yass?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


**Note**: YASS is in an early stage of development. Although it is stable, it has only been tested
with the data in our lab, but we are working to make it more flexible. Feel free to send
feedback through `Gitter`_. Expect a lot of API changes in the near future.

.. _Gitter: https://gitter.im/paninski-lab/yass

Reference
---------

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928


Installation
------------

Installing the last stable version:


.. code-block:: shell

   pip install yass-algorithm


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
