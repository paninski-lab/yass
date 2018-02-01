Installing YASS in development mode
===================================

First, we need to install YASS in develop mode.

Clone the repo:

::

    git clone https://github.com/paninski-lab/yass

Move to the folder containing the ``setup.py`` file and install the
package in development mode:

::

    cd yass
    pip install --editable .

If you install it that way, you can modify the source code and changes
will reflect whenever you import the modules (but you need to restart
the session).

Make sure you can import the package and that it's loaded from the
location where you ran ``git clone``. First open a Python intrepreter:

::

    python

And load the package you installed:

::

    import yass
    yass

You should see something like this:

::

    path/to/cloned/repository

Developing a package without restarting a session
-------------------------------------------------

If you use IPython/Jupyter run these at the start of the session to
reload your packages without having to restart your session:

.. code:: python

    %load_ext autoreload
    %autoreload 2
