# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (c) 2023 Farr Out Lab
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import jax.numpy as jnp
from jax import random, vmap
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from jaxtyping import Array
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


def cumtrapz(y: Array, x: Array) -> Array:
    @vmap
    def _area(i: Array, d: Array) -> Array:
        return d * (y[i] + y[i + 1]) * 0.5

    difs = jnp.diff(x)
    idxs = jnp.arange(1, len(y))
    res = jnp.cumsum(_area(idxs, difs))
    return jnp.concatenate([jnp.array([0]), res])


class PowerlawRedshift(Distribution):
    r"""A power-law redshift distribution. The probability density function is given
    by.

    .. math::

        p(z) \propto \frac{dV_c}{dz} (1 + z)^{\lambda - 1}

    .. doctest::

        >>> import jax.numpy as jnp
        >>> from astropy.cosmology import Planck15
        >>> z_grid = jnp.linspace(0.001, 1, 1000)
        >>> dVcdz_grid = Planck15.differential_comoving_volume(z_grid).value * 4.0 * jnp.pi
        >>> d = PowerlawRedshift(lamb=0.0, z_max=1.0, zgrid=z_grid, dVcdz=dVcdz_grid)
        >>> lpdfs = d.log_prob(self.grid)
        >>> lpdfs.shape
        (1000,)
    """

    arg_constraints = {"z_max": constraints.positive, "lamb": constraints.real}
    reparametrized_params = ["z_max", "lamb"]

    def __init__(
        self, lamb, z_max, zgrid, dVcdz, low=0.0, high=1000.0, validate_args=None
    ):
        self.z_max, self.lamb = promote_shapes(z_max, lamb)
        self._support = constraints.interval(low, high)
        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(lamb))
        super(PowerlawRedshift, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )
        self.zs = zgrid
        self.dVdc_ = dVcdz
        self.pdfs = self.dVdc_ * (1 + self.zs) ** (lamb - 1)
        self.norm = trapezoid(self.pdfs, self.zs)
        self.pdfs /= self.norm
        self.cdfgrid = cumtrapz(self.pdfs, self.zs)
        self.cdfgrid = self.cdfgrid.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value, dVdc=None):
        if dVdc is None:
            dVdc = jnp.interp(value, self.zs, self.dVdc_)
        return jnp.where(
            jnp.less_equal(value, self.z_max),
            jnp.log(dVdc) + (self.lamb - 1.0) * jnp.log1p(value) - jnp.log(self.norm),
            jnp.nan_to_num(-jnp.inf),
        )

    def cdf(self, value):
        return jnp.interp(value, self.zs, self.cdfgrid)

    def icdf(self, q):
        return jnp.interp(q, self.cdfgrid, self.zs)
