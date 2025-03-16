PIP=pip
UV=uv
PIP_FLAGS=--upgrade
TARGET?=gwkokab
PLATFORM?=

ifeq ($(PLATFORM),)
	_PLATFORM=.
else
	_PLATFORM=.[$(PLATFORM)]
endif

.PHONY: install uninstall cache_clean help

# Verify UV installation
UV_CHECK := $(shell command -v $(UV) 2> /dev/null)
ifndef UV_CHECK
	$(error "uv is not installed. Please install uv and try again.")
endif

help:
	@echo "Available targets:"
	@echo "  install PLATFORM=... - Install package"
	@echo "  uninstall            - Remove package"
	@echo "  cache_clean          - Clean pip and uv cache"
	@echo "  docs		          - Generate documentation"

install: uninstall
	$(UV) $(PIP) install $(PIP_FLAGS) $(_PLATFORM)


uninstall:
	$(UV) $(PIP) uninstall $(TARGET)

cache_clean: uninstall
	$(PIP) cache purge
	$(UV) cache clean

docs: install
	cd docs && make html
	cd ..
