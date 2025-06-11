# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax.numpy as jnp
import quadax
from jax import Array, random
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from jax.typing import ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample


class PowerlawRedshift(Distribution):
    """Redshift distribution for compact binary mergers modeled as a power law modulated
    by the cosmological volume element.

    The probability density function is defined as:

        p(z) ∝ (dV_c/dz)(z) * (1 + z)^(λ - 1),   for 0 ≤ z ≤ z_max

    where:
      - dV_c/dz is the differential comoving volume element,
      - λ is the redshift evolution power-law index,
      - z_max is the upper redshift cutoff.

    This distribution is normalized numerically on a fixed redshift grid.

    Parameters
    ----------
    lamb : float
        The power-law exponent λ.
    z_max : float
        The maximum redshift (upper limit of the support).
    zgrid : jax.Array
        A 1D array of redshift values for numerical integration and interpolation.
    dVcdz : jax.Array
        The differential comoving volume evaluated on zgrid.
    """

    arg_constraints = {
        "z_max": constraints.positive,
        "lamb": constraints.real,
        "zgrid": constraints.real_vector,
        "dVcdz": constraints.real_vector,
    }
    reparametrized_params = ["z_max", "lamb", "zgrid", "dVcdz"]
    pytree_data_fields = ("_support", "dVcdz", "lamb", "z_max", "zgrid")

    def __init__(
        self,
        lamb: ArrayLike,
        z_max: ArrayLike,
        zgrid: Array,
        dVcdz: Array,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        # Promote to same shape for broadcasting
        self.z_max, self.lamb = promote_shapes(z_max, lamb)
        self.zgrid = zgrid
        self.dVcdz = dVcdz
        self._support = constraints.interval(0.0, z_max)

        # Only precompute the CDF if lamb and z_max are constants
        if jnp.ndim(self.z_max) == 0 and jnp.ndim(self.lamb) == 0:
            # Mask out values above z_max but preserve shape
            mask = self.zgrid <= self.z_max
            zgrid_cut = self.zgrid
            dVcdz_cut = jnp.where(mask, self.dVcdz, 0.0)

            # Compute unnormalized PDF: p(z) ∝ dV/dz * (1 + z)^(λ - 1)
            pdfs = dVcdz_cut * jnp.power(1.0 + zgrid_cut, self.lamb - 1.0)

            # Normalize the PDF numerically using trapezoidal rule
            norm = trapezoid(pdfs, zgrid_cut)
            pdfs /= norm

            # Compute CDF for inverse transform sampling
            self.cdfgrid = quadax.cumulative_trapezoid(pdfs, x=zgrid_cut, initial=0.0)
            self.cdfgrid = self.cdfgrid.at[-1].set(1.0)  # ensure total probability = 1
            self.zgrid_cut = zgrid_cut
        else:
            self.cdfgrid = None
            self.zgrid_cut = None

        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(lamb))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value, dVdc=None):
        """Evaluate the log probability density function at a given redshift.

        Parameters
        ----------
        value : jax.Array
            Redshift(s) to evaluate.
        dVdc : jax.Array, optional
            Precomputed dV/dz values at `value`. If None, it will be interpolated.

        Returns
        -------
        logp : jax.Array
            Log-probability values.
        """
        dVdc_val = jnp.interp(value, self.zgrid, self.dVcdz) if dVdc is None else dVdc
        log_unnorm = jnp.log(dVdc_val) + (self.lamb - 1.0) * jnp.log1p(value)

        # Normalize over full zgrid (masking values above z_max)
        mask = self.zgrid <= self.z_max
        dVcdz_cut = jnp.where(mask, self.dVcdz, 0.0)
        pdfs = dVcdz_cut * jnp.power(1.0 + self.zgrid, self.lamb - 1.0)
        norm = trapezoid(pdfs, self.zgrid)

        return log_unnorm - jnp.log(norm)

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
        samples : jax.Array
            Redshift samples.
        """
        if self.cdfgrid is None:
            raise NotImplementedError(
                "Sampling not available when lamb/z_max are dynamic."
            )

        u = random.uniform(key, shape=sample_shape)
        return jnp.interp(u, self.cdfgrid, self.zgrid_cut)

    def cdf(self, value):
        """Evaluate the cumulative distribution function (CDF) at a given redshift."""
        if self.cdfgrid is None:
            raise NotImplementedError("CDF not available with dynamic parameters.")
        return jnp.interp(value, self.zgrid_cut, self.cdfgrid)

    def icdf(self, q):
        """Evaluate the inverse CDF (quantile function)."""
        if self.cdfgrid is None:
            raise NotImplementedError("ICDF not available with dynamic parameters.")
        return jnp.interp(q, self.cdfgrid, self.zgrid_cut)
