# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


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
