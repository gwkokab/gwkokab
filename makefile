PIP=pip
UV=uv
PIP_FLAGS=--upgrade
TARGET=gwkokab

install: uninstall
	$(UV) $(PIP) install $(PIP_FLAGS) .

uninstall:
	$(PIP) uninstall $(TARGET) -y

cache_clean: uninstall
	$(PIP) cache purge
	$(UV) cache clean