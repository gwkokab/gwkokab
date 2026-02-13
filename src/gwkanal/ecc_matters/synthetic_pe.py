# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from jax import numpy as jnp
from numpyro import distributions as dist

from gwkanal.core.synthetic_pe import (
    ErrorFunctionRegistryType,
    synthetic_discrete_pe_parser,
    SyntheticDiscretePEBase,
)
from gwkanal.ecc_matters.common import EccentricityMattersCore
from gwkanal.utils.logger import log_info
from gwkokab.errors import banana_error_m1_m2
from gwkokab.parameters import Parameters as P


class EccentricityMattersFakeDiscretePE(
    SyntheticDiscretePEBase, EccentricityMattersCore
):
    @property
    def error_function_registry(self) -> ErrorFunctionRegistryType:
        def banana_error_fn(scale_Mc, scale_eta, **kwargs):
            m1 = kwargs[P.PRIMARY_MASS_SOURCE]
            m2 = kwargs[P.SECONDARY_MASS_SOURCE]
            x = jnp.stack([m1, m2], axis=-1)
            return banana_error_m1_m2(
                x,
                self.size,
                self.rng_key,
                scale_Mc=scale_Mc,
                scale_eta=scale_eta,
            )

        def ecc_error_fn(scale, low, high, **kwargs):
            x = kwargs[P.ECCENTRICITY]
            err_x = dist.TruncatedNormal(loc=x, scale=scale, low=low, high=high).sample(
                key=self.rng_key, sample_shape=(self.size,)
            )
            mask = err_x < 0.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

        return {
            (P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE): (
                ("scale_Mc", "scale_eta"),
                banana_error_fn,
            ),
            P.ECCENTRICITY: (("scale", "low", "high"), ecc_error_fn),
        }


def main() -> None:
    parser = synthetic_discrete_pe_parser()
    args = parser.parse_args()

    log_info(start=True)

    EccentricityMattersFakeDiscretePE.init_rng_seed(seed=args.seed)

    generator = EccentricityMattersFakeDiscretePE(
        filename=args.filename,
        error_params_filename=args.error_params,
        size=args.size,
        derive_parameters=args.derive_parameters,
    )

    generator.generate_parameter_estimates()
