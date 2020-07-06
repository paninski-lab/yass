Important information
=====================

If you are planning to dive into YASS codebase, these are some
important things to know (as of November, 2018)

* Documentation is good but some parts are outdated, most functions have docstrings but beware that some of them need updates
* There are some examples and tools in https://github.com/paninski-lab/yass-examples. Some of the examples are outdated
* We are testing with date we cannot share, so every time we run Travis, we have to download it. See scripts/ for details
* There is a `version` script in the root folder which automates releases
* We no longer have an example in the README, in the past we provided an example using a sample of neuropixel data but since we are not developing (yet) with that type of data, we removed it since YASS has not been tested with such data yet


Code architecture changes
-------------------------

We have been making some architecture changes, in the past
we only kept one implementation of every pipeline step and
merged to master when ready. This proved troublesome, in some
cases, the steps (e.g. clustering) changed completely but
since they are still experimental, we did not have any stable
implementation. We are trying out another alternative: keep
a stable implementation and a experimental one at the same
time. Each step in the pipeline then offers a paramter
to choose which implementation to use.

So now each step is just a wrapper for another function which
provides convenient features.

In the future, it would be good to separate this "wrapping"
logic completely. Right know, we pass the entire CONFIG
object to these functions, which makes the code hard to understand
(we do not know which parameters the function uses), it would
be better for each implementation to provide all their parameters
(with nice defaults) in the function signature so that it is
clear which parameters they use.


BatchReader and "experimental" implementations
----------------------------------------------

The preprocess and detector experimental implementations were
mostly done due to performance issues with the stable implementations. The stable implementation use the BatchReader,
which internally uses the RecordingsReader. This reader seems
to be slow and is the major bottleneck of the stable
implementations. Re-implementing this reader will probably put
the stable implementation at comparable speed with the
experimental implementations. This is an important step since
the stable implementations are much more readable and clean.

There are some features (especially in the detector) that are
missing from the stable implementations, but this are not
difficult to migrate to the stable code.

YASS Configuration file
-----------------------

Initially, we had the YASS configuration file to hold all 
parameters for all steps in the pipeline. This caused a lot
of trouble since we had to modify the schema.yaml file
to reflect changes in experimental code, whenever someone
changed any of the steps a bit and wanted to introduce a new
parameter we needed to update the schema.yaml and tests to
make sure the code still worked.

The new architecture takes out most of the parameters and only
keeps the essentials, this comes to a cost: end users will have
to write Pythonn code if they want to customize the default pipeline since most parameters are no longer part of the
configuration file but parameters in the functions that
implement the steps. However, given the modular architecture,
this is trivial to do (see examples/pipeline/custom.py)

The advantage of this is that we no longer have to maintain
a configuration file, each new implementation is responsible
for providing defaults for its parameters.

Testing
-------

Most of the tests are unit or smoke tests (all of them
located in tests/unit/, although we should probably split
them into two folders). Unit tests check functionality of
YASS internal utilities (such as BatchReader). Some tests
just make sure the code runs (without checking results).

There are some "reference" tests  which check that some deterministic pieces of code still return the same results (such as preprocessing), this was done since the code was being optimized and we wanted to make sure we were not breaking the implementation.

There also integration tests which run the files in examples/.

Towards YASS 1.0
----------------

While we have a stable implementation for most of the steps
in the pipeline, the latest and greatest is still in development
and lacks of a stable, clean implementation. Before making a major
release, the followning has to be addressed.

* Re-implement RecordingsReader to put the stable implementations on par with the experimental implementations (preprocess and detect)
* Refactor experimental implementations to use the RecordingsReader/BatchProcessor
* Clean up and test the experimental implementations
* Update documentation
