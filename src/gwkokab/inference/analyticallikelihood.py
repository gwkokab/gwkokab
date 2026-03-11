# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Dict

from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Distribution

from gwkokab.models.utils import JointDistribution, ScaledMixture


def mvn_samples(
    loc: Array, scale_tril: Array, n_samples: int, key: PRNGKeyArray
) -> Array:
    """Generate samples from a multivariate normal distribution using method from
    `numpyro.distributions.MultivariateNormal`.

    Parameters
    ----------
    loc : Array
        Mean vector of the multivariate normal distribution.
    scale_tril : Array
        Lower-triangular Cholesky factor of the covariance matrix of the multivariate normal distribution.
    n_samples : int
        Number of samples to generate.
    key : PRNGKeyArray
        JAX random key for sampling.

    Returns
    -------
    Array
        Samples drawn from the multivariate normal distribution.
    """
    num_events, dim = loc.shape
    eps = jrd.normal(key, shape=(n_samples, num_events, dim))
    return loc + jnp.einsum("eij,sej->sei", scale_tril, eps)


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    analytical_to_model_coord_fn: Callable[[Array], Array],
    log_abs_det_jacobian_analytical_to_model_coord_fn: Callable[[Array, Array], Array],
    n_samples: int = 500,
) -> Callable[[Array, Dict[str, Any]], Array]:
    r"""Compute the analytical likelihood function for a model given its parameters.

    .. math::

        \mathcal{L}_{\mathrm{analytical}}(\Lambda,\kappa)\propto
        \exp\left(-\mu(\Lambda,\kappa)\right)
        \prod_{i=1}^{N}
        \iint\mathcal{G}(\theta,z|\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)
        \rho(\theta,z\mid\Lambda,\kappa)d\theta dz

        \mathcal{L}_{\mathrm{analytical}}(\Lambda,\kappa)\propto
        \exp{\left(-\mu(\Lambda,\kappa)\right)}\prod_{i=1}^{N}\bigg<
        \rho(\theta_{i,j},z_{i,j}\mid\Lambda,\kappa)
        \bigg>_{\theta_{i,j},z_{i,j}\sim\mathcal{G}(\theta,z|\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)}
    """

    def likelihood_fn(x: Array, data: Dict[str, Any]) -> Array:
        mean_stack = data["mean_stack"]
        scale_tril_stack = data["scale_tril_stack"]
        lb = data["lower_bounds"]
        ub = data["upper_bounds"]
        scale_stack = data["scale_stack"]
        ln_offsets = data["ln_offsets"]
        pmean_kwargs = data["pmean_kwargs"]
        T_obs = pmean_kwargs["T_obs"]
        master_key = data["key"]

        n_events, _ = mean_stack.shape

        params = {name: x[i] for name, i in variables_index.items()}
        model_instance = dist_fn(**constant_params, **params)

        samples = (
            mvn_samples(mean_stack, scale_tril_stack, n_samples, master_key)
            / scale_stack
        )

        mask = jnp.all((lb <= samples) & (samples <= ub), axis=-1)
        safe_samples = jnp.where(mask[..., jnp.newaxis], samples, 0.5 * (lb + ub))

        transformed_samples = analytical_to_model_coord_fn(safe_samples)

        mask &= model_instance.support.check(transformed_samples)

        safe_transformed = jnp.where(
            mask[..., jnp.newaxis],
            transformed_samples,
            model_instance.support.feasible_like(transformed_samples),
        )

        log_abs_det_jacobian = log_abs_det_jacobian_analytical_to_model_coord_fn(
            safe_samples, safe_transformed
        )
        log_weights = jnp.where(
            mask,
            model_instance.log_prob(safe_transformed) + log_abs_det_jacobian,
            -jnp.inf,
        )

        log_est = (
            jnn.logsumexp(log_weights, axis=0, where=mask)
            + ln_offsets
            - jnp.log(n_samples)
        )

        total_ln_l = jnp.sum(log_est)

        expected_rates = poisson_mean_estimator(model_instance, **pmean_kwargs)

        ln_post = (
            priors.log_prob(x) + total_ln_l + n_events * jnp.log(T_obs) - expected_rates
        )

        return jnp.nan_to_num(ln_post, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf)

    return likelihood_fn
