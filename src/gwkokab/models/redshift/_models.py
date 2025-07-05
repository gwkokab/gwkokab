# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
import quadax
from jax import Array, numpy as jnp, random as jrd
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from jaxtyping import ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from ...cosmology import PLANCK_2015_Cosmology
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
    """

    arg_constraints = {
        "z_max": constraints.positive,
        "kappa": constraints.real,
    }
    reparametrized_params = ["z_max", "kappa"]
    pytree_data_fields = ("_support", "z_grid", "cdfgrid", "kappa", "z_max")

    def __init__(
        self, z_max: Array, kappa: Array, *, validate_args: Optional[bool] = None
    ):
        self.z_max, self.kappa = promote_shapes(z_max, kappa)
        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(kappa))
        self.z_grid = jnp.linspace(0.0, self.z_max, 2500)
        self._support = constraints.interval(0.0, z_max)
        cdfgrid: Array = quadax.cumulative_trapezoid(
            jnp.exp(
                self.log_differential_spacetime_volume(self.z_grid) - self.log_norm()
            ),
            x=self.z_grid,
            initial=0.0,
        )
        self.cdfgrid = cdfgrid / cdfgrid[-1]
        super(PowerlawRedshift, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """The support of the distribution, which is the interval [0, z_max]."""
        return self._support

    def log_differential_spacetime_volume(self, z: Array) -> Array:
        """Placeholder method for computing the differential spacetime volume."""
        log_differential_spacetime_volume_val = (
            PLANCK_2015_Cosmology.logdVcdz_Gpc3(z) - jnp.log1p(z) + self.log_prob(z)
        )
        return log_differential_spacetime_volume_val

    def log_norm(self) -> Array:
        """Placeholder method for computing the log normalization constant."""
        log_differential_spacetime_volume = self.log_differential_spacetime_volume(
            self.z_grid
        )
        pdfs = jnp.exp(log_differential_spacetime_volume)
        norm = trapezoid(pdfs, self.z_grid)
        return jnp.log(norm)

    def sample(self, key, sample_shape=()):
        """Draw samples from the distribution using inverse transform sampling.

        Note: Sampling is only supported when z_max is a static scalar.

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
        u = jrd.uniform(key, shape=sample_shape + self.batch_shape)
        return jnp.interp(u, self.cdfgrid, self.z_grid)

    @validate_sample
    def log_prob(self, value: Array) -> Array:
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
        return self.kappa * jnp.log1p(value)


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

    arg_constraints = {"kappa": constraints.real, "z_max": constraints.positive}
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

    def log_norm(self) -> Array:
        log_norm = doubly_truncated_power_law_log_norm_constant(
            alpha=self.kappa, low=1.0, high=1.0 + self.z_max
        )
        return jax.lax.stop_gradient(log_norm)

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
        return logpdf_unnorm

    def sample(self, key, sample_shape=()):
        u = jrd.uniform(key, shape=sample_shape + self.batch_shape)
        kappa_eq_neg_one = jnp.equal(self.kappa, -1.0)
        safe_kappa = jnp.where(kappa_eq_neg_one, 1.0, self.kappa)
        norm = jnp.exp(self.log_norm())
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
