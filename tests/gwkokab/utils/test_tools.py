# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
