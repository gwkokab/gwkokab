# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Factory for fetching the correct likelihood function based on the specified sampler
backend and analysis type.
"""

from typing import Callable, Literal

from gwkokab.inference.flowMC_analytical_poisson_likelihood import (
    flowMC_analytical_poisson_likelihood as flowMC_analytical_poisson_likelihood,
)
from gwkokab.inference.flowMC_discrete_poisson_likelihood import (
    flowMC_discrete_poisson_likelihood as flowMC_discrete_poisson_likelihood,
)
from gwkokab.inference.numpyro_analytical_poisson_likelihood import (
    numpyro_analytical_poisson_likelihood as numpyro_analytical_poisson_likelihood,
)
from gwkokab.inference.numpyro_discrete_poisson_likelihood import (
    numpyro_discrete_poisson_likelihood as numpyro_discrete_poisson_likelihood,
)
from gwkokab.inference.flowMC_sampled_poisson_likelihood import (
    flowMC_sampled_poisson_likelihood as flowMC_sampled_poisson_likelihood,
)
from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
    numpyro_sampled_poisson_likelihood as numpyro_sampled_poisson_likelihood,
)
from gwkokab.utils.exceptions import LoggedValueError


def get_likelihood_fn(
    sampler_name: Literal["flowMC", "numpyro"],
    analysis_type: Literal["discrete", "analytical", "sampled"],
) -> Callable[..., Callable]:
    if sampler_name == "flowMC":
        if analysis_type == "analytical":
            return flowMC_analytical_poisson_likelihood
        if analysis_type == "sampled":
            return flowMC_sampled_poisson_likelihood
        return flowMC_discrete_poisson_likelihood
    if sampler_name == "numpyro":
        if analysis_type == "analytical":
            return numpyro_analytical_poisson_likelihood
        if analysis_type == "sampled":
            return numpyro_sampled_poisson_likelihood
        return numpyro_discrete_poisson_likelihood

    raise LoggedValueError(
        f"Unsupported sampler_name '{sampler_name}' or analysis_type '{analysis_type}'."
    )
