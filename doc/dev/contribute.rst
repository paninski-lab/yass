Git workflow
------------

Internal contributors have write permissins to the repo, you can create
new branches, do your work and submit pull requests:

.. code:: shell

    # move to the repo
    cd path/to/repo

    # when you start working on something new, create a new branch from master
    git checkout -b new-feature

    # work on new feature and remember to keep in sync with the master branch
    # from time to time
    git merge master

    # remember to push you changes to the remote branch
    git push

    # when the new feature is done open a pull request to merge new-feature to master

    # once the pull request is accepted and merged to master, don't forget to remove
    # the branch if you no longer are going to use it
    # remove from the remote repository
    git push -d origin new-feature
    # remove from your local repository
    git branch -d new-feature

Minimum expected documentation
------------------------------

Every function should contain *at least* a brief description of what it
does, as well as input and output description. However, complex
functions might require more to be understood.

We use
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
code itself but the why is harder to infer. You do not need to comment
every line, but add them when it may be hard for others to understand
what's going on

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

.. code:: shell

    flake8 my_script.py

Most text editors and IDE have plugins to automatically run tools such
as ``flake8`` when you modify a file, `here's one for Sublime
Text <http://www.sublimelinter.com/en/latest/>`__.

If you want to know more about ``flake8`` and similar tools, `this is a
nice
resource <https://blog.sideci.com/about-style-guide-of-python-and-linter-tool-pep8-pyflakes-flake8-haking-pyling-7fdbe163079d>`__
