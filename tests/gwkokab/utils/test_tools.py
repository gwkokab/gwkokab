# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import pytest

from gwkokab.utils.tools import error_if, warn_if


def test_error_if():
    with pytest.raises(ValueError):
        error_if(True)
    with pytest.raises(ValueError):
        error_if(True, ValueError)
    with pytest.raises(ValueError):
        error_if(True, ValueError, "message")
    with pytest.raises(TypeError):
        error_if(True, TypeError, "message")


def test_warn_if():
    with pytest.warns(UserWarning):
        warn_if(True)
    with pytest.warns(UserWarning):
        warn_if(True, UserWarning)
    with pytest.warns(UserWarning):
        warn_if(True, UserWarning, "message")
    with pytest.warns(UserWarning):
        warn_if(True, UserWarning, "message")
