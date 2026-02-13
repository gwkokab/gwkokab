# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from gwkanal.core.synthetic_events import (
    injection_generator_parser,
    SyntheticEventsBase,
)
from gwkanal.ecc_matters.common import EccentricityMattersCore, EccentricityMattersModel
from gwkanal.utils.logger import log_info


def main() -> None:
    """Main function of the script."""
    parser = injection_generator_parser()
    args = parser.parse_args()

    log_info(start=True)

    class EccentricityMattersInjectionGenerator(
        EccentricityMattersCore, SyntheticEventsBase
    ):
        pass

    EccentricityMattersInjectionGenerator.init_rng_seed(seed=args.seed)

    generator = EccentricityMattersInjectionGenerator(
        filename=args.output_filename,
        model_fn=EccentricityMattersModel,
        model_params_filename=args.model_params,
        poisson_mean_filename=args.pmean_cfg,
        derive_parameters=args.derive_parameters,
    )

    generator.from_inverse_transform_sampling()
