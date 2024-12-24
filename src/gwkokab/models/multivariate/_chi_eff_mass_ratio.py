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


import jax.numpy as jnp
from jax import lax
from jax.scipy.stats import truncnorm
from jaxtyping import Array
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from gwkokab.models.utils import doubly_truncated_power_law_log_prob
from gwkokab.utils.transformations import chieff, mass_ratio


class ChiEffMassRatioConstraint(constraints.Constraint):
    r"""The :math:`\frac{\mathrm{d}V_c}{\mathrm{d}z}` is removed from the redshift
    distribution, because the reference prior is from same distribution with
    :math:`\kappa=2.7`."""

    is_discrete = False
    event_dim = 1

    def __init__(self, mmin, mmax):
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x):
        m1, m2, a1, a2, z = (
            x[..., 0],
            x[..., 1],
            x[..., 2],
            x[..., 3],
            x[..., 4],
        )
        mask = m2 >= self.mmin
        mask &= m1 >= m2
        mask &= m1 <= self.mmax
        mask &= a1 >= 0.0
        mask &= a1 <= 1.0
        mask &= a2 >= 0.0
        mask &= a2 <= 1.0
        mask &= z >= 0.0
        return mask

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())


class ChiEffMassRatioCorrelated(Distribution):
    arg_constraints = {
        "lambda_peak": constraints.real,
        "lamb": constraints.real,
        "loc_m": constraints.real,
        "scale_m": constraints.positive,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
        "gamma": constraints.real,
        "alpha": constraints.real,
        "beta": constraints.real,
        "mu_eff_0": constraints.interval(-1, 1),
        "log10_sigma_eff_0": constraints.real,
        "kappa": constraints.real,
    }
    reparametrized_params = [
        "lambda_peak",
        "lamb",
        "loc_m",
        "scale_m",
        "mmin",
        "mmax",
        "gamma",
        "alpha",
        "beta",
        "mu_eff_0",
        "log10_sigma_eff_0",
        "kappa",
    ]
    pytree_data_fields = (
        # "_z_powerlaw",
        "_support",
    )

    def __init__(
        self,
        # powerlaw+peak parameters
        lambda_peak: Array,
        lamb: Array,
        loc_m: Array,
        scale_m: Array,
        mmin: Array,
        mmax: Array,
        # mass ratio parameters
        gamma: Array,
        # chieff parameters
        alpha: Array,
        beta: Array,
        mu_eff_0: Array,
        log10_sigma_eff_0: Array,
        # redshift parameters
        kappa: Array,
        *,
        validate_args=None,
    ) -> None:
        (
            self.lambda_peak,
            self.lamb,
            self.loc_m,
            self.scale_m,
            self.mmin,
            self.mmax,
            self.gamma,
            self.alpha,
            self.beta,
            self.mu_eff_0,
            self.log10_sigma_eff_0,
            self.kappa,
        ) = promote_shapes(
            lambda_peak,
            lamb,
            loc_m,
            scale_m,
            mmin,
            mmax,
            gamma,
            alpha,
            beta,
            mu_eff_0,
            log10_sigma_eff_0,
            kappa,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(lambda_peak),
            jnp.shape(lamb),
            jnp.shape(loc_m),
            jnp.shape(scale_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(gamma),
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(mu_eff_0),
            jnp.shape(log10_sigma_eff_0),
            jnp.shape(kappa),
        )
        # zgrid = jnp.linspace(0.001, 3.0 + 1e-6, 100)
        # self._z_powerlaw = PowerlawRedshift(
        #     lamb=self.kappa,
        #     z_max=3.0 + 1e-6,
        #     zgrid=zgrid,
        #     dVcdz=4.0 * jnp.pi * PLANCK_2015_Cosmology.dVcdz(zgrid),
        # )
        self._support = ChiEffMassRatioConstraint(mmin=self.mmin, mmax=self.mmax)
        super(ChiEffMassRatioCorrelated, self).__init__(
            event_shape=(5,), batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1, m2, a1, a2, z = (
            value[..., 0],
            value[..., 1],
            value[..., 2],
            value[..., 3],
            value[..., 4],
        )
        # log_prob(m1)
        prob_m1 = (1 - self.lambda_peak) * jnp.exp(
            doubly_truncated_power_law_log_prob(
                x=m1, alpha=self.lamb, low=self.mmin, high=self.mmax
            )
        )
        prob_m1 += self.lambda_peak * truncnorm.pdf(
            x=m1,
            a=(self.mmin - self.loc_m) / self.scale_m,
            b=(self.mmax - self.loc_m) / self.scale_m,
            loc=self.loc_m,
            scale=self.scale_m,
        )

        log_prob_m1 = jnp.log(prob_m1)

        # log_prob(m2)
        invalid_mass_mask = jnp.less_equal(m1, self.mmin)
        safe_m1 = jnp.where(invalid_mass_mask, self.mmin + 2.0, m1)
        safe_m2 = jnp.where(invalid_mass_mask, self.mmin + 2.0, m2)
        log_prob_m2 = jnp.where(
            invalid_mass_mask,
            -jnp.inf,
            doubly_truncated_power_law_log_prob(
                x=safe_m2, alpha=self.gamma, low=self.mmin, high=safe_m1
            ),
        )

        # log_prob(chi_eff)
        invalid_mass_mask = jnp.less(m1, m2)
        q = jnp.where(invalid_mass_mask, 1.0, mass_ratio(m1=m1, m2=m2))
        chi_eff = jnp.where(
            invalid_mass_mask, 1.0, chieff(m1=m1, m2=m2, chi1z=a1, chi2z=a2)
        )
        mu_eff = self.mu_eff_0 + self.alpha * (q - 1)
        sigma_eff = jnp.power(10, self.log10_sigma_eff_0 + self.beta * (q - 1))

        log_prob_chi_eff = jnp.where(
            invalid_mass_mask,
            -jnp.inf,
            truncnorm.logpdf(
                x=chi_eff,
                a=(-1.0 - mu_eff) / sigma_eff,
                b=(1.0 - mu_eff) / sigma_eff,
                loc=mu_eff,
                scale=sigma_eff,
            ),
        )

        # log_prob(z)
        # log_prob_z = self._z_powerlaw.log_prob(z)
        log_prob_z = (self.kappa - 1) * jnp.log1p(z)

        return log_prob_m1 + log_prob_m2 + log_prob_chi_eff + log_prob_z
