# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2023 Farr Out Lab
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional

import jax.numpy as jnp
import quadax
from jax import Array, random
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from jax.typing import ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


class PowerlawRedshift(Distribution):
    r"""A power-law redshift distribution. The probability density function is given by.

    .. math::

        p(z) \propto \frac{dV_c}{dz} (1 + z)^{\lambda - 1}

    .. code:: python

        >>> import jax.numpy as jnp
        >>> from astropy.cosmology import Planck15
        >>> z_grid = jnp.linspace(0.001, 1, 1000)
        >>> dVcdz_grid = (
        ...     Planck15.differential_comoving_volume(z_grid).value * 4.0 * jnp.pi
        ... )
        >>> d = PowerlawRedshift(lamb=0.0, z_max=1.0, zgrid=z_grid, dVcdz=dVcdz_grid)
        >>> lpdfs = d.log_prob(self.grid)
        >>> lpdfs.shape
        (1000,)
    """

    arg_constraints = {
        "z_max": constraints.positive,
        "lamb": constraints.real,
        "zgrid": constraints.real_vector,
        "dVcdz": constraints.real_vector,
    }
    reparametrized_params = ["z_max", "lamb", "zgrid", "dVcdz"]
    pytree_data_fields = (
        "_support",
        "cdfgrid",
        "dVcdz",
        "lamb",
        "norm",
        "z_max",
        "zgrid",
    )

    def __init__(
        self,
        lamb: ArrayLike,
        z_max: ArrayLike,
        zgrid: Array,
        dVcdz: Array,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        lamb : ArrayLike
            power-law index
        z_max : ArrayLike
            maximum redshift
        zgrid : Array
            grid of redshifts
        dVcdz : Array
            differential comoving volume upon the :code:`zgrid` grid
        validate_args : Optional[bool], optional
            whether to validate input, by default None
        """
        self.z_max, self.lamb = promote_shapes(z_max, lamb)
        self.zgrid = zgrid
        self.dVcdz = dVcdz
        self._zgrid = jnp.clip(zgrid, 1e-10, z_max)
        pdfs = dVcdz * jnp.power(1.0 + self.zgrid, self.lamb - 1.0)
        norm = trapezoid(pdfs, self.zgrid)
        pdfs /= norm
        self.log_norm = jnp.log(norm)
        self.cdfgrid: Array = quadax.cumulative_trapezoid(
            pdfs, x=self.zgrid, initial=0.0
        )
        self.cdfgrid = self.cdfgrid.at[-1].set(1)
        self._support = constraints.interval(0.0, z_max)
        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(lamb))
        super(PowerlawRedshift, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value, dVdc=None):
        if dVdc is None:
            dVdc = jnp.interp(value, self.zgrid, self.dVcdz)
        return jnp.log(dVdc) + (self.lamb - 1.0) * jnp.log1p(value) - self.log_norm

    def cdf(self, value):
        return jnp.interp(value, self.zgrid, self.cdfgrid)

    def icdf(self, q):
        return jnp.interp(q, self.cdfgrid, self.zgrid)
