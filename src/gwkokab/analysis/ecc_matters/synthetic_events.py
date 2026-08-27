# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``synthetic_events_ecc_matter`` console script.

Draws a synthetic population from the eccentricity-matters model, using the model's own
expected rate to decide how many events to draw and its selection function to decide
which survive. The first step of the end-to-end workflow.

See Also
--------
gwkokab.analysis.core.synthetic_events : The population drawing machinery.
gwkokab.analysis.core.synthetic_pe : The next step, which adds measurement error.
"""

from gwkokab.analysis.core.synthetic_events import (
    injection_generator_parser,
    SyntheticEventsBase,
)
from gwkokab.analysis.ecc_matters.common import (
    EccentricityMattersCore,
    EccentricityMattersModel,
)
from gwkokab.analysis.utils.logger import log_info


def main() -> None:
    """Console script entry point for ``synthetic_events_ecc_matter``.

    Parses the command line, assembles a generator by mixing
    :class:`~gwkokab.analysis.ecc_matters.common.EccentricityMattersCore` with
    :class:`~gwkokab.analysis.core.synthetic_events.SyntheticEventsBase`, seeds the PRNG,
    and writes the drawn population.
    """
    parser = injection_generator_parser()
    args = parser.parse_args()

    log_info(start=True)

    class EccentricityMattersInjectionGenerator(
        EccentricityMattersCore, SyntheticEventsBase
    ):
        """Population generator for the eccentricity-matters family.

        Defined inside :func:`main` because it exists only for the duration of one
        script run. This family has no component counts or optional parameters, so the
        mixin needs no body of its own.
        """

        pass

    EccentricityMattersInjectionGenerator.init_rng_seed(seed=args.seed)

    generator = EccentricityMattersInjectionGenerator(
        filename=args.output_filename,
        model_fn=EccentricityMattersModel,
        model_params_filename=args.model_params,
        poisson_mean_filename=args.pmean_cfg,
        derive_parameters=args.derive_parameters,
        n_buffer_events=args.n_buffer_events,
    )

    generator.from_inverse_transform_sampling()
