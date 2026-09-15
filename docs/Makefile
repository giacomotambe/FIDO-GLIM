.PHONY: help
help:
	@echo "make cpp|mkdocs|all|deploy"

.PHONY: cpp
cpp:
	@echo "Building C++ documentation..."
	doxygen Doxyfile

.PHONY: mkdocs
mkdocs:
	@echo "Building MkDocs documentation..."
	cd .. && mkdocs build

.PHONY: all
all: cpp mkdocs
	@echo "All documentation built."

.PHONY: deploy
deploy:
	@echo "Deploying documentation..."
	cd .. && mkdocs gh-deploy --force
