#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from typing import Optional

from jax.random import beta
from jaxtyping import Array

from ..typing import Numeric
from ..utils.misc import get_key
from .abstractspinmodel import AbstractSpinModel


class Wysocki2019SpinModel(AbstractSpinModel):
    """Beta distribution for the spin magnitude

    Wysocki2019SpinModel is a subclass of ContinuousRV and implements
    the beta distribution for the spin magnitude as described in
    https://arxiv.org/abs/1805.06442
    """

    def __init__(
        self,
        alpha: Numeric,
        beta: Numeric,
        chimax: Numeric = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._chimax = chimax

    def samples(self, num_of_samples: int) -> Array:
        # return super().rvs(shape=(num_of_samples,), key=None)
        return beta(
            key=get_key(),
            a=self._alpha,
            b=self._beta,
            shape=(num_of_samples,),
        )

    def __repr__(self) -> str:
        string = f"Wysocki2019SpinModel(alpha_chi={self._alpha}, "
        string += f"beta_chi={self._beta}, chimax={self._scale}"
        if self._name is not None:
            string += f", {self._name}"
        string += ")"
        return string
