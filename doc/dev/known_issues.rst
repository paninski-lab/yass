Important information
=====================

If you are planning to dive into YASS codebase, these are some
important things to know (as of November, 2018)

* Documentation is good but some parts are outdated, most functions
have docstrings but beware that some of them need updates
* There are some examples and tools in https://github.com/paninski-lab/yass-examples. Some of the examples are outdated
* We are testing with date we cannot share, so every time
we run Travis, we have to download it. See scripts/ for details
* tests/ needs some work, most tests are in tests/unit/ which
are the ones that are run on Travis, there are "reference" tests
which check that some deterministic pieces of code still return
the same results (such as preprocessing), this was done since
the code was being optimized and we wanted to make sure we were
not breaking the implementation
* There is a `version` script in the root folder which automates
releases


Code architecture changes
-------------------------

BatchReader and "experimental" implementations
----------------------------------------------

YASS Configuration file
-----------------------

Testing coverage
----------------

Support for Python 2
--------------------


Towards YASS 1.0
----------------