.DEFAULT_GOAL := help


install: install-dev install-yass ## Install Yass + Dev Requirements
	@echo "--> Installing Yass and Dev Requirements"
	

install-dev: ## Dev Requirements
	@echo "--> Install Dev Requirements"
	pip install -r requirements.txt	


install-yass: ## Install Yass package
	@echo "--> Installing Yass Package"
	pip install -e .


test: ## Run Tests
	@echo "--> Running Tests"
	py.test . --flake8 --cov=yass


# self-documenting makefile as described in http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Show this documentation
	echo "\nYass build/test tools:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "   \033[36m%-20s\033[0m %s\n", $$1, $$2}'
