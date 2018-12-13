Getting started
===============


Using YASS pre-built pipelines
------------------------------

YASS configuration file
***********************

YASS is configured using a YAML file, below is an example of such configuration:

.. literalinclude:: ../examples/config.yaml


If you want to use a Neural Network as detector, you need to provide your
own Neural Network. YASS provides tools for easily training the model, see
this `tutorial`_ for details.

.. _tutorial: https://github.com/paninski-lab/yass-examples/blob/master/NN_training_tutorial.ipynb

If you do not want to use a Neural Network, you can use the `threshold`
detector instead.

For details regarding the configuration file see :doc:`config`.


Running YASS from the command line
**********************************

After installing `yass`, you can sort spikes from the command line:


.. code-block:: shell

    yass sort path/to/config.yaml


Run the following command for more information:

.. code-block:: shell

    yass sort --help


Running YASS in a Python script
*******************************

.. literalinclude:: ../examples/pipeline/deconvolve.py


Advanced usage
**************

`yass sort` is a wrapper for the code in `yass.pipeline.run`, it provides
a pipeline implementation with some defaults but you cannot customize it,
if you want to use experimental features, the only way to do so is to
customize your pipeline:

.. literalinclude:: ../examples/pipeline/custom.py
