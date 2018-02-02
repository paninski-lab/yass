.DEFAULT_GOAL := help


install: ## Install Yass
	@echo "--> Installing Python Requirements"
	pip install -e .


install-dev: ## Install Yass + Dev Requirements
	@echo "--> Installing Python Requirements"
	pip install -e .
	@echo "--> Install Dev Requirements"
	pip install -r requirements.txt	


test: ## Run Tests
	@echo "--> Running Tests"
	py.test . --flake8


# self-documenting makefile as described in http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Show this documentation
	echo "\nYass build/test tools:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "   \033[36m%-20s\033[0m %s\n", $$1, $$2}'
