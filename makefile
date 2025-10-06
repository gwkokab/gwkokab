PIP=pip
UV=uv
PIP_FLAGS?=
TARGET?=gwkokab
EXTRA?=

ifeq ($(EXTRA),)
	_EXTRA=.
else
	_EXTRA=.[$(EXTRA)]
endif

ifeq ($(PIP_FLAGS),)
	_PIP_FLAGS=
else
	_PIP_FLAGS=$(PIP_FLAGS)
endif

.PHONY: install uninstall cache_clean help

# Verify UV installation
UV_CHECK := $(shell command -v $(UV) 2> /dev/null)
ifndef UV_CHECK
	$(error "uv is not installed. Please install uv and try again.")
endif

help:
	@echo "Available targets:"
	@echo "  install EXTRA=... - Install package"
	@echo "  uninstall         - Remove package"
	@echo "  cache_clean       - Clean pip and uv cache"
	@echo "  docs		       - Generate documentation"

install: uninstall
	$(UV) $(PIP) install $(_PIP_FLAGS) $(_EXTRA)


uninstall:
	$(UV) $(PIP) uninstall $(TARGET)

cache_clean: uninstall
	$(PIP) cache purge
	$(UV) cache clean

docs: install
	cd docs && make html
	cd ..
