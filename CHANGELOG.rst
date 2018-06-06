Changelog
=========

0.10dev
-------
* Colored logs
* Improved testing coverage


0.9 (2018-05-24)
-----------------
* Added parallelization to batch processor
* Preprocess step now runs in parallel
* Filtering and standarization running in one step to avoid I/O overhead


0.8 (2018-04-19)
-----------------

* It is now possible to save results for every step to resume execution, see `save_results` option
* Fixed a bug that caused excessive logging when logger level was set to DEBUG
* General improvements to the sorting algorithm
* Fixes a bug that was causing and import error in the mfm module (thanks @neil-gallagher for reporting this issue)


0.7 (2018-04-06)
-----------------

* New CLI tool for training neural networks `yass train`
* New CLI tool for exporting results to phy `yass export`
* Separated logic in five steps: preprocess, detect, cluster templates and deconvolute
* Improved Neural Network detector speed
* Improved package organization
* Updated examples
* Added integration tests
* Increased testing coverage
* Some examples include links to Jupyter notebooks
* Errors in documentation building are now tested in Travis
* Improved batch processor
* Simplified configuration file
* Preprocessing speedups


0.6 (2018-02-05)
-----------------
* New stability metric
* New batch module
* Rewritten preprocessor
* A lot of functions were rewritten and documented
* More partial results are saved to improve debugging
* Removed a lot of legacy code
* Removed batching logic from old functions, they are now using the `batch` module
* Rewritten CLI interface `yass` command is now `yass sort`


0.5 (2018-01-31)
-----------------
* Improved logging
* Last release with old codebase


0.4 (2018-01-19)
-----------------
* Fixes bug in preprocessing (#38)
* Increased template size
* Updates deconvolution method


0.3 (2017-11-15)
-----------------
* Adds new neural network module


0.2 (2017-11-14)
-----------------
* Config module refactoring, configuration files are now much simpler
* Fixed bug that was causing spike times to be off due to the buffer
* Various bug fixes
* Updates to input/output structure
* Adds new module for augmented spikes
* Function names changes in score module
* Simplified parameters for score module functions


0.1.1 (2017-11-01)
-------------------
* Minor changes to setup.py for uploading to pypi


0.1 (2017-11-01)
-----------------
* First release
