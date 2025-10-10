# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union

import jax
from numpyro import distributions as dist
from numpyro._typing import DistributionT

from gwkokab.utils.tools import error_if
from kokab.utils.regex import match_all


def _available_prior(name: str) -> DistributionT:
    """Get the available prior from numpyro distributions.

    Parameters
    ----------
    name : str
        name of the prior

    Returns
    -------
    DistributionT
        prior distribution

    Raises
    ------
    AttributeError
        If the prior is not found in numpyro distributions
    """
    error_if(
        name not in dist.__all__,
        AttributeError,
        f"Prior {name} not found. Available priors are: " + ", ".join(dist.__all__),
    )
    return getattr(dist, name)


def _is_lazy_prior(prior_dict: dict[str, Union[str, float]]) -> bool:
    """Check if the prior is a lazy prior.

    Parameters
    ----------
    prior_dict : dict[str, Union[str, float]]
        dictionary of prior

    Returns
    -------
    bool
        True if the prior is a lazy prior, False otherwise
    """
    for value in prior_dict.values():
        if isinstance(value, str):
            return True
    return False


def get_processed_priors(params: List[str], priors: dict) -> dict:
    """Get the processed priors from a list of parameters. A processed prior is either
    an instantiated prior or a tuple of :code:`(jax.tree_util.Partial, lazy_vars)` where
    :code:`lazy_vars` is a dictionary of lazy variables.

    Parameters
    ----------
    params : List[str]
        list of parameters
    priors : dict
        dictionary of priors

    Returns
    -------
    dict
        dictionary of processed priors

    Raises
    ------
    ValueError
        if the prior value is invalid
    """
    matched_prior_params = match_all(params, priors)
    for param in params:
        error_if(param not in matched_prior_params, msg=f"Missing prior for {param}")

    for key, value in matched_prior_params.items():
        # if the value is not a dict, means its a constant/duplicate value
        if not isinstance(value, dict):
            continue
        value_cpy = value.copy()
        dist_type = value_cpy.pop("dist", None)
        error_if(
            not isinstance(dist_type, str) or dist_type == "",
            msg=f"Prior for '{key}' must specify a 'dist' string field.",
        )
        prior = _available_prior(dist_type)

        # if there are no lazy variables, instantiate the prior
        if not _is_lazy_prior(value_cpy):
            matched_prior_params[key] = prior(**value_cpy, validate_args=True)
            continue

        # separate the lazy variables and the non-lazy variables
        lazy_vars = {k: v for k, v in value_cpy.items() if isinstance(v, str)}
        non_lazy_vars = {k: v for k, v in value_cpy.items() if not isinstance(v, str)}
        matched_prior_params[key] = (
            jax.tree_util.Partial(prior, **non_lazy_vars, validate_args=True),
            lazy_vars,
        )
    return matched_prior_params
