# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import pickle
import sys
from unittest import mock

from absl.testing import parameterized

from gwkokab.internal.lazy_loader import LazyLoader


@parameterized.parameters(
    ("constants",),
    ("cosmology",),
    ("errors",),
    ("inference",),
    ("logger",),
    ("models",),
    ("parameters",),
    ("poisson_mean",),
    ("population",),
    ("utils",),
    ("vts",),
)
class TestLazyLoader(parameterized.TestCase):
    def test_module_lazy_loaded(self, module_name: str) -> None:
        """Test that module is only loaded when accessed."""
        with mock.patch("importlib.import_module") as mock_import:
            # Configure mock to return a mock module
            mock_module = mock.MagicMock()
            mock_import.return_value = mock_module

            # Create a globals dictionary and LazyLoader
            test_globals = {}
            lazy_module = LazyLoader(
                module_name, test_globals, "gwkokab." + module_name
            )

            # Import should not be called until attribute access
            mock_import.assert_not_called()

            # Access an attribute to trigger loading
            _ = lazy_module.some_attribute

            # Import should be called once with correct module path
            mock_import.assert_called_once_with("gwkokab." + module_name)

            # Module should be in globals dictionary
            assert test_globals[module_name] is mock_module

    def test_on_first_access_callback(self, module_name: str) -> None:
        """Test that the on_first_access callback is called exactly once."""
        callback_mock = mock.MagicMock()

        with mock.patch("importlib.import_module") as mock_import:
            mock_import.return_value = mock.MagicMock()

            # Create LazyLoader with callback
            lazy_module = LazyLoader(
                module_name, {}, "gwkokab." + module_name, callback_mock
            )

            # Access attribute to trigger loading
            _ = lazy_module.some_attribute
            callback_mock.assert_called_once()

            # Access another attribute - callback should still be called only once
            _ = lazy_module.another_attribute
            callback_mock.assert_called_once()


def test_pickling() -> None:
    """Test that LazyLoader instances can be pickled and unpickled."""
    # Create LazyLoader for the sys module (which is always available)
    lazy_sys = LazyLoader("sys", None, "sys")

    # Pickle and unpickle
    pickled = pickle.dumps(lazy_sys)
    unpickled = pickle.loads(pickled)

    # The unpickled object should be the actual sys module
    assert isinstance(unpickled, type(sys))
    assert unpickled.__name__ == "sys"


def test_real_module_attribute_access():
    """Test with a real module to ensure attributes are correctly accessed."""
    # Create LazyLoader for sys module
    lazy_sys = LazyLoader("sys", {}, "sys")

    # Access a known attribute
    version = lazy_sys.version

    # Verify we got the correct attribute
    assert version == sys.version
    assert "version" in lazy_sys.__dict__
