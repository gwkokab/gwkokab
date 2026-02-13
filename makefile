PIP       := pip
UV        := uv
TARGET    := gwkokab
PIP_FLAGS ?=
EXTRA     ?=

INSTALL_TARGET := .$(if $(EXTRA),[$(EXTRA)])

.PHONY: all install uninstall cache_clean help doc

all: help

help:
	@echo "Usage: make [target] [EXTRA=feature1,feature2]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## [-a-zA-Z_]+: .*' $(MAKEFILE_LIST) | sed 's/^## //' | column -t -s ':'

install: uninstall check-uv
	GWKOKAB_NIGHTLY_BUILD=1 $(UV) $(PIP) install $(PIP_FLAGS) "$(INSTALL_TARGET)"

uninstall: check-uv
	$(UV) $(PIP) uninstall $(TARGET) || true

cache_clean: check-uv
	$(UV) cache clean

doc: install
	@mkdir -p docs/source
	cp -r examples docs/source/
	$(MAKE) -C docs html

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { echo >&2 "Error: $(UV) is not installed."; exit 1; }
