# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import quadax
from jax import Array, numpy as jnp, random as jrd
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from jaxtyping import ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils import doubly_truncated_power_law_log_norm_constant


class PowerlawRedshift(Distribution):
    r"""Redshift distribution for compact binary mergers modeled as a power law modulated
    by the cosmological volume element.

    The probability density function is defined as:

    .. math::
        p(z) \propto \frac{dV_c/dz(z) \cdot (1 + z)^{\kappa - 1}}}, \qquad 0 \leq z \leq z_{max}

    where:
      - dV_c/dz is the differential comoving volume element,
      -  is the redshift evolution power-law index,
      - z_max is the upper redshift cutoff.

    This distribution is normalized numerically on a fixed redshift grid.

    Parameters
    ----------
    kappa : float
        The power-law exponent :math:`\kappa`.
    z_max : float
        The maximum redshift (upper limit of the support).
    zgrid : Array
        A 1D array of redshift values for numerical integration and interpolation.
    dVcdz : Array
        The differential comoving volume evaluated on :code:`zgrid`.
    """

    arg_constraints = {
        "z_max": constraints.positive,
        "kappa": constraints.real,
        "zgrid": constraints.real_vector,
        "dVcdz": constraints.real_vector,
    }
    reparametrized_params = ["z_max", "kappa", "zgrid", "dVcdz", "cdfgrid"]
    pytree_data_fields = ("_support", "dVcdz", "kappa", "z_max", "zgrid", "cdfgrid")

    def __init__(
        self,
        kappa: ArrayLike,
        z_max: ArrayLike,
        zgrid: Array,
        dVcdz: Array,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.z_max, self.kappa = promote_shapes(z_max, kappa)
        self.zgrid = zgrid
        self.dVcdz = dVcdz

        mask = self.zgrid <= self.z_max
        dVcdz_cut = jnp.where(mask, self.dVcdz, 0.0)

        pdfs = dVcdz_cut * jnp.power(1.0 + self.zgrid, self.kappa - 1.0)

        norm = trapezoid(pdfs, self.zgrid)
        pdfs /= norm

        self.cdfgrid: Array = quadax.cumulative_trapezoid(
            pdfs, x=self.zgrid, initial=0.0
        )
        self.cdfgrid = self.cdfgrid.at[-1].set(1.0)  # ensure total probability = 1

        self._support = constraints.interval(0.0, z_max)
        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(kappa))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value: Array, dVdc: Optional[Array] = None) -> Array:
        """Evaluate the log probability density function at a given redshift.

        Parameters
        ----------
        value : Array
            Redshift(s) to evaluate.
        dVdc : Array, optional
            Precomputed dV/dz values at `value`. If None, it will be interpolated.

        Returns
        -------
        Array
            Log-probability values.
        """
        if dVdc is None:
            dVdc_val = jnp.interp(value, self.zgrid, self.dVcdz)
        logpdf_unnorm = jnp.log(dVdc_val) + (self.kappa - 1.0) * jnp.log1p(value)
        pdfs = self.dVcdz * jnp.power(1.0 + self.zgrid, self.kappa - 1.0)
        norm = trapezoid(pdfs, self.zgrid)
        return logpdf_unnorm - jnp.log(norm)

    def sample(self, key, sample_shape=()):
        """Draw samples from the distribution using inverse transform sampling.

        Note: Sampling is only supported when lamb and z_max are static scalars.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A PRNG key for sampling.
        sample_shape : tuple
            Shape of the desired sample batch.

        Returns
        -------
        samples : Array
            Redshift samples.
        """
        u = jrd.uniform(key, shape=sample_shape)
        return jnp.interp(u, self.cdfgrid, self.zgrid)

    def cdf(self, value):
        """Evaluate the cumulative distribution function (CDF) at a given redshift."""
        return jnp.interp(value, self.zgrid, self.cdfgrid)

    def icdf(self, q):
        """Evaluate the inverse CDF (quantile function)."""
        return jnp.interp(q, self.cdfgrid, self.zgrid)


class SimpleRedshiftPowerlaw(Distribution):
    r"""Simple redshift distribution defined as,

    .. math::
        p(z) \propto (1 + z)^{\kappa}, \qquad 0 \leq z \leq z_{max}

    Parameters
    ----------
    kappa : ArrayLike
        Power-law exponent :math:`\kappa`.
    z_max : ArrayLike
        Maximum redshift (upper limit of the support).
    validate_args : Optional[bool], optional
        Whether to validate arguments, by default None

    Returns
    -------
    dist.TransformedDistribution
        A transformed distribution representing the redshift law.
    """

    arg_constraints = {
        "kappa": constraints.real,
        "z_max": constraints.positive,
    }
    reparametrized_params = ["kappa", "z_max"]
    pytree_data_fields = ("_support", "kappa", "z_max")

    def __init__(
        self,
        kappa: ArrayLike,
        z_max: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.kappa, self.z_max = promote_shapes(kappa, z_max)
        self._support = constraints.interval(0.0, z_max)
        batch_shape = broadcast_shapes(jnp.shape(kappa), jnp.shape(z_max))
        super(SimpleRedshiftPowerlaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """Evaluate the log probability density function at a given redshift.

        Parameters
        ----------
        value : ArrayLike
            Redshift(s) to evaluate.

        Returns
        -------
        ArrayLike
            Log-probability values.
        """
        logpdf_unnorm = self.kappa * jnp.log1p(value)
        log_norm = doubly_truncated_power_law_log_norm_constant(
            alpha=self.kappa, low=1.0, high=1.0 + self.z_max
        )
        # return logpdf_unnorm - jax.lax.stop_gradient(log_norm)
        return logpdf_unnorm - log_norm

    def sample(self, key, sample_shape=()):
        u = jrd.uniform(key, shape=sample_shape + self.batch_shape)
        kappa_eq_neg_one = jnp.equal(self.kappa, -1.0)
        safe_kappa = jnp.where(kappa_eq_neg_one, 1.0, self.kappa)
        norm = jnp.exp(
            doubly_truncated_power_law_log_norm_constant(
                alpha=self.kappa,
                low=1.0,
                high=1.0 + self.z_max,
            )
        )
        samples_kappa_neq_neg_one = (
            jnp.power(
                u * norm * (safe_kappa + 1.0) + 1.0, jnp.reciprocal(safe_kappa + 1.0)
            )
            - 1.0
        )
        samples_kappa_eq_neg_one = jnp.expm1(u * norm)
        return jnp.where(
            kappa_eq_neg_one,
            samples_kappa_eq_neg_one,
            samples_kappa_neq_neg_one,
        )
