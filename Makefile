.DEFAULT_GOAL := help
.PHONY: test integration-test


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
	./scripts/make_sample_data
	# FIXME: reference testing was deactivated
	# ./scripts/generate_output_reference

	# echo 'Done generating sample data, upload it '

# self-documenting makefile as described in http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Show this documentation
	echo "\nYass build/test tools:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "   \033[36m%-20s\033[0m %s\n", $$1, $$2}'
