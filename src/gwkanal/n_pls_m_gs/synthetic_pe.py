# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Optional

import numpy as np

from gwkanal.core.synthetic_pe import (
    ErrorFunctionRegistryType,
    fake_discrete_pe_parser,
    SyntheticDiscretePEBase,
)
from gwkanal.n_pls_m_gs.common import NPowerlawMGaussianCore
from gwkanal.utils.logger import log_info
from gwkokab.errors import banana_error_m1_m2, truncated_normal_error
from gwkokab.parameters import Parameters as P


class NPowerlawMGaussianFakeDiscretePE(SyntheticDiscretePEBase, NPowerlawMGaussianCore):
    @property
    def error_function_registry(self) -> ErrorFunctionRegistryType:
        def banana_error_fn(scale_Mc, scale_eta, **kwargs):
            m1 = kwargs[P.PRIMARY_MASS_SOURCE]
            m2 = kwargs[P.SECONDARY_MASS_SOURCE]
            x = np.stack([m1, m2], axis=-1)
            return banana_error_m1_m2(
                x,
                self.size,
                self.rng_key,
                scale_Mc=scale_Mc,
                scale_eta=scale_eta,
            )

        def generic_truncated_normal_error_fn(
            parameter: P,
            cut_low: Optional[float] = None,
            cut_high: Optional[float] = None,
        ) -> tuple[tuple[str, ...], Callable]:
            def error_fn(**kwargs):
                x = kwargs[parameter]
                scale = kwargs[parameter + "_scale"]
                low = kwargs.get(parameter + "_low")
                high = kwargs.get(parameter + "_high")

                if cut_low is None:
                    cut_low_value = kwargs.get(parameter + "_cut_low")
                else:
                    cut_low_value = cut_low

                if cut_high is None:
                    cut_high_value = kwargs.get(parameter + "_cut_high")
                else:
                    cut_high_value = cut_high

                return truncated_normal_error(
                    x=x,
                    size=self.size,
                    key=self.rng_key,
                    scale=scale,
                    low=low,
                    high=high,
                    cut_low=cut_low_value,
                    cut_high=cut_high_value,
                )

            error_parameters: tuple[str, ...] = (
                parameter + "_scale",
                parameter + "_low",
                parameter + "_high",
            )
            if cut_low is None:
                error_parameters += (parameter + "_cut_low",)
            if cut_high is None:
                error_parameters += (parameter + "_cut_high",)

            return error_parameters, error_fn

        error_fns: ErrorFunctionRegistryType = {
            (P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE): (
                ("scale_Mc", "scale_eta"),
                banana_error_fn,
            )
        }
        for param, cut_low, cut_high in [
            (P.PRIMARY_SPIN_MAGNITUDE, 0.0, 1.0),
            (P.SECONDARY_SPIN_MAGNITUDE, 0.0, 1.0),
            (P.PRIMARY_SPIN_X, -1.0, 1.0),
            (P.SECONDARY_SPIN_X, -1.0, 1.0),
            (P.PRIMARY_SPIN_Y, -1.0, 1.0),
            (P.SECONDARY_SPIN_Y, -1.0, 1.0),
            (P.PRIMARY_SPIN_Z, -1.0, 1.0),
            (P.SECONDARY_SPIN_Z, -1.0, 1.0),
            (P.EFFECTIVE_SPIN, None, None),
            (P.PRECESSING_SPIN, None, None),
            (P.COS_TILT_1, -1.0, 1.0),
            (P.COS_TILT_2, -1.0, 1.0),
            (P.ECCENTRICITY, 0.0, 1.0),
            (P.REDSHIFT, 1e-3, None),
            (P.SIN_DECLINATION, -1.0, 1.0),
            (P.COS_IOTA, -1.0, 1.0),
        ]:
            error_fns[param] = generic_truncated_normal_error_fn(
                param, cut_low=cut_low, cut_high=cut_high
            )

        return error_fns


def main() -> None:
    parser = fake_discrete_pe_parser()
    args = parser.parse_args()

    log_info(start=True)

    NPowerlawMGaussianFakeDiscretePE.set_rng_key(seed=args.seed)

    generator = NPowerlawMGaussianFakeDiscretePE(
        filename=args.filename,
        error_params_filename=args.error_params,
        size=args.size,
        derive_parameters=args.derive_parameters,
    )

    generator.generate_parameter_estimates()
