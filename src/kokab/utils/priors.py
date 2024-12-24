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


# Copyright (c) 2024 Colm Talbot
# SPDX-License-Identifier: MIT

import numpyro.distributions
from numpyro.distributions import Distribution


class _Available:
    def __getitem__(self, name: str) -> Distribution:
        if not (ord("A") <= ord(name[0]) <= ord("Z")):
            raise AttributeError(f"module {__name__} is an invalid prior")
        if name in numpyro.distributions.__all__:
            return getattr(numpyro.distributions, name)
        raise AttributeError(f"module {__name__} has no attribute {name}")


available = _Available()
