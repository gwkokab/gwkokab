PIP=pip
UV=uv
PIP_FLAGS=--upgrade
TARGET?=gwkokab

.PHONY: install uninstall cache_clean help

# Verify UV installation
UV_CHECK := $(shell command -v $(UV) 2> /dev/null)

help:
	@echo "Available targets:"
	@echo "  install      - Install package"
	@echo "  uninstall    - Remove package"
	@echo "  cache_clean  - Clean pip and uv cache"
	@echo "  docs		  - Generate documentation"

install: uninstall
ifndef UV_CHECK
	@echo "uv is not installed. Continuing without uv."
	$(PIP) install $(PIP_FLAGS) .
else
	$(UV) $(PIP) install $(PIP_FLAGS) .
endif

uninstall:
	$(UV) $(PIP) uninstall $(TARGET)

cache_clean: uninstall
	$(PIP) cache purge
ifdef UV_CHECK
	$(UV) cache clean
endif

docs: install
	cd docs && make html
	cd ..
