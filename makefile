PIP             := pip
UV              := uv
TARGET          := gwkokab
PIP_FLAGS       ?=
EXTRA           ?=
GROUP           ?=
INSTALL_TARGET  := .

comma := ,
empty :=
space := $(empty) $(empty)

EXTRA_FLAGS     := $(if $(EXTRA),$(addprefix --extra ,$(subst $(comma),$(space),$(EXTRA))))
GROUP_FLAGS     := $(if $(GROUP),$(addprefix --group ,$(subst $(comma),$(space),$(GROUP))))

.DEFAULT_GOAL   := help
.PHONY: all install uninstall cache_clean help doc check-uv

install: uninstall check-uv
	GWKOKAB_NIGHTLY_BUILD=1 $(UV) $(PIP) install $(PIP_FLAGS) \
		$(INSTALL_TARGET) -r pyproject.toml \
		$(EXTRA_FLAGS) $(GROUP_FLAGS)

uninstall: check-uv
	@$(UV) $(PIP) uninstall $(TARGET) 2>/dev/null || true

cache_clean: check-uv
	$(UV) cache clean

doc: install
	@mkdir -p docs/source
	cp -r examples docs/source/
	$(MAKE) -C docs html

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { echo >&2 "Error: $(UV) is not installed."; exit 1; }
