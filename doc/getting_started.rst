Getting started
===============


Using YASS pre-built pipelines
------------------------------

YASS configuration file
***********************

YASS is configured using a YAML file, below is an example of such configuration:

.. literalinclude:: ../examples/config_sample.yaml


If you want to use a Neural Network as detector, you need to provide your
own Neural Network. YASS provides tools for easily training the model, see
this `tutorial`_ for details.

.. _tutorial: https://github.com/paninski-lab/yass-examples/blob/master/NN_training_tutorial.ipynb

If you do now want to use a Neural Network, you can use the `threshold`
detector instead.

For details regarding the configuration file see :doc:`config`.


Running YASS from the command line
**********************************

After installing `yass`, you can run it from the command line:


.. code-block:: shell

    yass path/to/config.yaml


Run the following command for more information:

.. code-block:: shell

    yass --help


Running YASS in a Python script
*******************************

.. literalinclude:: ../examples/deconvolute.py


Building your own pipeline
--------------------------

Soon.