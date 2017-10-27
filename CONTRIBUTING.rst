Contributing
============

::

    Good programmers write code that humans can understand. Martin Fowler, 2008.

This guide will walk you through the basics for contributing to
YASS.

Installing the package in develop mode
--------------------------------------

First, we need to install YASS in develop mode.

Clone the repo:

::

    git clone https://github.com/pjl4303/YASS

Move to the folder containing the ``setup.py`` file (usually in the root
folder) and install the package in development mode:

::

    pip install --editable .

If you install it that way, you can modify the source code and changes
will reflect wherever you import the modules (but you need to restart
the session).

Make sure you can import the package and that it's loaded from the
location where you ran ``git clone``. First open a Python intrepreter:

::

    python

And load the package you installed:

::

    import yass

You should see something like this:

::

    path/to/cloned/repository

Developing a package without restarting a session
-------------------------------------------------

If you use IPython/Jupyter run these to reload your package without
having to restart your session:

.. code:: python

    %load_ext autoreload
    %autoreload 2

Minimum expected documentation
------------------------------

Every function should contain *at least* a brief description of what it
does, as well as input and output description. However, complex
functions might require more to be understood.

We will use
`numpydoc <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__
style docstrings.

Function example:

.. code:: python


    def fibonacci(n):
        """Compute the nth fibonacci number

        Parameters
        ----------
        n: int
            The index in the fibonacci sequence whose value will be calculated

        Returns
        -------
        int
            The nth fibonacci number
        """
        # fibonacci needs seed values for 0 and 1
        if n == 0:
            return 0
        elif n == 1:
            return 1
        # for n > 1, the nth fibonnacci number is defined as follows
        else:
            return fibonacci(n-1) + fibonacci(n-2)

Object example:

.. code:: python

    class Square(object):
        def __init__(self, l):
            """Represent a square

            Parameters
            ----------
            l: float
                Side length
            """
            self.l = l

        def area(self):
            """Compute the area of the square

            Returns
            -------
            float
                The area of the square
            """
            return self.l**2

**A note about comments**: comments should explain *why* you are doing
some operation *not what* operation. The what can be infered from the
code itself but the why is harder to infer. You do not need to comment every
line, but add them when it may be hard for others to understand what's going on.

**A note about objects**: objects are meant to encapsulate mutable
state. Mutable objectsa are hard to debug. When writing scientific
software, we usually do not need mutable state, we only want to process
input in a stateless manner, so only use objects when absolutely
necessary.

Python 3
--------

Write Python 3 code. `Python 2 is
retiring... <https://pythonclock.org/>`__

In most cases, it's really easy to write Python 2 and 3 compliant code,
here's the `official porting
guide <https://docs.python.org/3/howto/pyporting.html>`__.

Using logger, not print
-----------------------

Print is *evil*. It does not respect anyone or anything, it just throws
stuff into stdout without control. The only case when print makes sense
is when developing command line applications. So use logging, it's much
better and easy to setup. More about logging
`here <http://docs.python-guide.org/en/latest/writing/logging/>`__.

Setting up logger in a script:

.. code:: python

    import logging

    logger = logging.getLogger(__name__)

    def my awesome_function(a):
        logger.info('This is an informative message')

        if something_happens(a):
            logger.debug('This is a debugging message: something happened,'
                         ' it is not an error but we want you to know about it')

        # do stuff...

If you want to log inside an object, you need to do something a bit
different:

.. code:: python

    import logging

    class MyObject(object):

        def __init__():
            self.logger = logging.getLogger(__name__)

        def do_stuff():
            self.logger.debug('Doing stuff...')

Code style
----------

::

    Beautiful is better than ugly. The Zen of Python

To make our code readable and maintanble, we need some standards, Python
has a style guide called
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__. We don't expect
you to memorize it, so here's a `nice guide with the
basics <https://gist.github.com/sloria/7001839>`__.

If you still skipped the guide, here are the fundamental rules:

1. Variables, functions, methods, packages and modules:
   ``lower_case_with_underscores``
2. Classes and Exceptions: ``CapWords``
3. Avoid one-letter variables, except for counters
4. Use 4 spaces, never tabs
5. Line length should be between 80-100 characters

However, there are tools to automatically check if your code complies
with the standard. ``flake8`` is one of such tools, and can check for
PEP8 compliance as well as other common errors:

.. code:: shell

    pip install flake8

To check a file:

::

    flake8 my_script.py

Most text editors and IDE have plugins to automatically run tools such
as ``flake8`` when you modify a file, `here's one for Sublime
Text <http://www.sublimelinter.com/en/latest/>`__.

If you want to know more about ``flake8`` and similar tools, `this is a
nice
resource <https://blog.sideci.com/about-style-guide-of-python-and-linter-tool-pep8-pyflakes-flake8-haking-pyling-7fdbe163079d>`__

Virtual environments
--------------------

Virtual environmnets help you keep your project dependencies organized
and avoid messing up your system Python installation. A virtual
environment is just another Python installation on your computer.

For example, if you have a project A, which depends on package P1
version 1.5 and project B which depends o package P1 version 2.0, you
can create two virtual environments, each one with its own packages (and
even different Python versions) and easily switch between them.

For scientific Python, the recommended way if to use
`miniconda <https://conda.io/miniconda.html>`__.

Testing
-------

Soon...

Learn more
----------

-  `The Hitchhiker’s Guide to
   Python! <http://docs.python-guide.org/en/latest/>`__
