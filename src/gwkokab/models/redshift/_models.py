# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax.numpy as jnp
import quadax
from jax import Array, random as jrd
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from jax.typing import ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample


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
    reparametrized_params = ["z_max", "kappa", "zgrid", "dVcdz"]
    pytree_data_fields = ("_support", "dVcdz", "kappa", "z_max", "zgrid", "norm")

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

        self.norm = trapezoid(pdfs, self.zgrid)
        pdfs /= self.norm

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

        return logpdf_unnorm - jnp.log(self.norm)

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
