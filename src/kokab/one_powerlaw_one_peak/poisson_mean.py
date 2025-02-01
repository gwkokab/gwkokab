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


from collections.abc import Callable
from typing import Union

import equinox as eqx
from jax import lax, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import (
    CategoricalProbs,
    Distribution,
    DoublyTruncatedPowerLaw,
    MixtureGeneral,
    TransformedDistribution,
    TruncatedNormal,
    Uniform,
)

from gwkokab.models import SmoothedPowerlawAndPeak
from gwkokab.models.transformations import (
    PrimaryMassAndMassRatioToComponentMassesTransform,
)
from gwkokab.models.utils import JointDistribution
from gwkokab.poisson_mean import PoissonMeanABC
from gwkokab.utils.kernel import log_planck_taper_window


class ImportanceSamplingPoissonMean(PoissonMeanABC):
    logVT_fn: Callable[[Array], Array] = eqx.field(init=False)
    num_samples: int = eqx.field(init=False, static=True)
    key: PRNGKeyArray = eqx.field(init=False)
    mixing_dist: CategoricalProbs = eqx.field(init=False)

    def __init__(
        self,
        logVT_fn: Callable[[Array], Array],
        key: PRNGKeyArray,
        num_samples: int,
        scale: Union[int, float, Array] = 1.0,
    ) -> None:
        self.scale = scale
        self.key = key
        self.num_samples = num_samples
        self.logVT_fn = logVT_fn
        self.mixing_dist = CategoricalProbs(jnp.array([1.0 - 0.05, 0.05]))

    def __call__(self, model: SmoothedPowerlawAndPeak) -> Array:
        if isinstance(model, TransformedDistribution):
            model = model.base_dist
        delta_region_dist_m1m2 = Uniform(
            low=jnp.array([model.mmin, model.mmin]),
            high=jnp.array([model.mmin + model.delta, model.mmin + model.delta]),
            validate_args=model._validate_args,
        )

        m1_powerlaw = DoublyTruncatedPowerLaw(
            alpha=model.alpha,
            low=model.mmin,
            high=model.mmax,
            validate_args=model._validate_args,
        )
        m2_powerlaw = DoublyTruncatedPowerLaw(
            alpha=model.beta,
            low=model.mmin,
            high=model.mmax,
            validate_args=model._validate_args,
        )
        m1_gaussian = TruncatedNormal(
            loc=model.loc,
            scale=model.scale,
            low=model.mmin,
            high=model.mmax,
            validate_args=model._validate_args,
        )

        powerlaw_component: Distribution = MixtureGeneral(
            self.mixing_dist,
            [
                JointDistribution(
                    m1_powerlaw,
                    m2_powerlaw,
                    validate_args=model._validate_args,
                ),
                delta_region_dist_m1m2,
            ],
            validate_args=model._validate_args,
        )
        gaussian_component: Distribution = MixtureGeneral(
            self.mixing_dist,
            [
                JointDistribution(
                    m1_gaussian,
                    m2_powerlaw,
                    validate_args=model._validate_args,
                ),
                delta_region_dist_m1m2,
            ],
            validate_args=model._validate_args,
        )

        key1, key2 = jrd.split(self.key, num=2)

        powerlaw_samples = powerlaw_component.sample(key1, (self.num_samples,))
        gaussian_samples = gaussian_component.sample(key2, (self.num_samples,))

        transformation = PrimaryMassAndMassRatioToComponentMassesTransform()
        powerlaw_samples_q = transformation._inverse(powerlaw_samples)
        gaussian_samples_q = transformation._inverse(gaussian_samples)

        rate_powerlaw = jnp.mean(
            (
                model._powerlaw_prob(powerlaw_samples[..., 0])
                / lax.stop_gradient(model._Z_q(powerlaw_samples[..., 0]))
            )
            * jnp.exp(
                self.logVT_fn(powerlaw_samples)
                + model._log_prob_q(powerlaw_samples_q)
                - transformation.log_abs_det_jacobian(
                    powerlaw_samples_q, powerlaw_samples
                )
                + log_planck_taper_window(
                    (powerlaw_samples[..., 0] - model.mmin) / model.delta
                )
                - lax.stop_gradient(powerlaw_component.log_prob(powerlaw_samples)),
            )
        )
        rate_gaussian = jnp.mean(
            (
                model._gaussian_prob(gaussian_samples[..., 0])
                / lax.stop_gradient(model._Z_q(gaussian_samples[..., 0]))
            )
            * jnp.exp(
                self.logVT_fn(gaussian_samples)
                + model._log_prob_q(gaussian_samples_q)
                - transformation.log_abs_det_jacobian(
                    gaussian_samples_q, gaussian_samples
                )
                + log_planck_taper_window(
                    (gaussian_samples[..., 0] - model.mmin) / model.delta
                )
                - lax.stop_gradient(gaussian_component.log_prob(gaussian_samples))
            )
        )

        total_estimated_rate = (
            (1 - model.lambda_peak) * rate_powerlaw / model._Z_powerlaw
        )
        total_estimated_rate += (model.lambda_peak * rate_gaussian) / model._Z_gaussian

        total_estimated_rate = total_estimated_rate * jnp.exp(model.log_rate)

        return self.scale * total_estimated_rate
