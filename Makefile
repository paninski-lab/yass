.DEFAULT_GOAL := help
.PHONY: test integration-test

install: ## Install Locally
	conda env update -f environment.yml
	cd src/gpu_bspline_interp && conda run -n yass python setup.py install --force
	cd src/gpu_rowshift && conda run -n yass python setup.py install --force

test: ## Run Tests
	@echo "--> Running Tests"
	pytest tests/unit --flake8 --cov=yass


output-test: ## Generate output for performance tests
	@echo "--> Generating output for performance tests"
	@echo "--> Running pipeline with threshold detector..."
	cd tests/; yass sort config_threshold_49.yaml --zero_seed --output_dir=threshold/
	@echo "--> Running pipeline with neural network detector..."
	cd tests/; yass sort config_nn_49.yaml --zero_seed --output_dir=nnet/


performance-test: ## Run performance tests
	@echo "--> Running performance tests"
	pytest tests/performance/ --log-cli-level=INFO


integration-test: ## Run Integration Tests
	bash integration-test/integration-test.sh

docs: ## Build docs
	make -C doc html

download-test-data: ## Download data for running tests
	./scripts/download_test_data

generate-testing-data: ## Generates testing data and files that are used as reference in some tests to check that the output is still the same

	# First step is to run the make sample data script, the parameter points
	# to a folder that should have a retina/ folder, inside the folder, there
	# must be a config.yaml, ej49_data1_set1.bin and ej49_geometry1.txt files.
	# Output will be at path/to/data/yass-testing-data/recordings.
	# Update the ~/data parameter if the data is on a different location
	./scripts/make_sample_data ~/data

	# once we generated the data, we have to re-generate the output reference
	# tests, this tests run some deterministic  yass functions and verify that
	# for a given input, we still get the same results.
	# We first have to re-run the tests using the latest generated data
	# (from the step above) and save the output (which will be used as
	# reference in later runs)

	# use the data generatde in the first step
	rm -rf tests/assets/recordings
	cp -r ~/data/yass-testing-data/recordings tests/assets/recordings

	# generate output reference, make sure the output folder matches
	./scripts/generate_output_reference ~/data

	# Once this is done, zip the yass-testing-data folder (located in)
	# ~/data (or whatever folder you are using to store the data) and upload
	# it. Then go to the Travis config and update the YASS_TESTING_DATA_URL
	# so it points to the right URL. Example, if using Dropbox, upload the zip
	# file, then obtain the URL
	# (e.g. https://www.dropbox.com/s/SOMECHARACTERS/yass-testing-data.zip?dl=0)
	# to make this URL a direct download URL, change it to be in the following
	# format:
	# https://dl.dropboxusercontent.com/s/SOMECHARACTERS/yass-testing-data.zip?dl=0
	# then put this URL in the YASS_TESTING_DATA_URL env variable on travis

	@echo 'Done generating sample data, Make sure you upload it and set the YASS_TESTING_DATA_URL variable before running make download-test-data'

# self-documenting makefile as described in http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Show this documentation
	echo "\nYass build/test tools:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "   \033[36m%-20s\033[0m %s\n", $$1, $$2}'
