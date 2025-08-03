# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import quadax
from jax import Array, numpy as jnp, random as jrd
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from ...cosmology import PLANCK_2015_Cosmology


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
    pytree_data_fields = ("_support", "kappa", "z_max")

    def __init__(
        self, z_max: Array, kappa: Array, *, validate_args: Optional[bool] = None
    ):
        self.z_max, self.kappa = promote_shapes(z_max, kappa)
        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(kappa))
        self._support = constraints.interval(0.0, z_max)
        super(PowerlawRedshift, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """The support of the distribution, which is the interval [0, z_max]."""
        return self._support

    def log_differential_spacetime_volume(self, z: Array) -> Array:
        """Placeholder method for computing the differential spacetime volume."""
        logdVcdz = PLANCK_2015_Cosmology.logdVcdz(z)
        log_time_dilation = -jnp.log1p(z)
        log_differential_spacetime_volume_val = (
            log_time_dilation + logdVcdz + self.log_psi_of_z(z)
        )
        return log_differential_spacetime_volume_val

    def log_norm(self) -> Array:
        """Placeholder method for computing the log normalization constant."""
        z_grid = jnp.linspace(0.0, self.z_max, 2500)
        log_differential_spacetime_volume = self.log_differential_spacetime_volume(
            z_grid
        )
        pdfs = jnp.exp(log_differential_spacetime_volume)
        norm = trapezoid(pdfs, z_grid)
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
        z_grid = jnp.linspace(0.0, self.z_max, 10_000)
        pdfgrid = jnp.exp(self.log_differential_spacetime_volume(z_grid))
        norm = trapezoid(pdfgrid, z_grid)
        pdfgrid /= norm
        cdfgrid: Array = quadax.cumulative_trapezoid(pdfgrid, x=z_grid, initial=0.0)
        cdfgrid = cdfgrid / cdfgrid[-1]
        return jnp.interp(u, cdfgrid, z_grid)

    def log_psi_of_z(self, z: Array) -> Array:
        r"""Evaluate the psi function at a given redshift.

        .. math::

            \ln\psi(z) = \kappa \log(1 + z)

        Parameters
        ----------
        z : ArrayLike
            Redshift(s) to evaluate.

        Returns
        -------
        ArrayLike
            Values of the psi function.
        """
        return self.kappa * jnp.log1p(z)

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
        return self.log_differential_spacetime_volume(value)
