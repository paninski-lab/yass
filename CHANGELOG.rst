Changelog
=========

0.7dev
------

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
