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

install: uninstall
ifndef UV_CHECK
	$(error "$(UV) is not installed. Please install it first")
endif
	$(UV) $(PIP) install $(PIP_FLAGS) .

uninstall:
	$(PIP) uninstall $(TARGET) -y

cache_clean: uninstall
	$(PIP) cache purge
	$(UV) cache clean
