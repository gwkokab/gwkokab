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
    r"""Constraint for the :class:`ChiEffMassRatioCorrelated` model. It is defined as,

    .. math::

        \forall m_1,m_2,a_1,a_2,z,
        \left(m_{\text{min}} \leq m_2 \leq m_1 \leq m_{\text{max}}\right)
        \land \left( a_1, a_2 \in [0,1] \right) \land \left( z \geq 0 \right)
    """

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
    r"""This model was proposed in `Who Ordered That? Unequal-mass Binary Black Hole
    Mergers Have Larger Effective Spins
    <https://ui.adsabs.harvard.edu/abs/2021ApJ...922L...5C>`_. This is the
    implementation of correlated model of :math:`\chi_{\text{eff}}` and :math:`q`
    detailed in Appendix A of the paper. Mathematically it is defined as,

    .. math::

        \begin{align}
            p(m_1\mid \lambda_{\text{peak}}, \lambda, \mu_m, \sigma_m, m_{\text{min}}, m_{\text{max}}) &=
            \left((1-\lambda_{\text{peak}})m_1^{\lambda}+\lambda_{\text{peak}}\mathcal{N}(m_1\mid \mu_m, \sigma_m)\right)
            \Theta(m_1-m_{\text{min}})\Theta(m_{\text{max}}-m_1) \\
            p(m_2\mid \gamma, m_1, m_{\text{min}}) &= m_2^{\gamma}\Theta(m_2-m_{\text{min}})\Theta(m_1-m_2)\\
            p(z\mid \kappa) &\propto \frac{\mathrm{d}V_c}{\mathrm{d}z}(1+z)^{\kappa-1}
        \end{align}

    where :math:`\Theta` is the Heaviside step function, :math:`\mathcal{N}` is the normal
    distribution, :math:`\mathrm{d}V_c/\mathrm{d}z` is the comoving volume element, and
    :math:`\kappa` is the redshift distribution parameter. The :math:`\chi_{\text{eff}}`
    model is defined as,

    .. math::

        \begin{align}
            \mu_{\chi}(\mu_{\text{eff},0}, \alpha, q) &= \mu_{\text{eff},0} + \alpha(q-1)\\
            \log_{10}(\sigma_{\chi}(\log_{10}(\sigma_{\text{eff},0}), \beta, q)) &=
            \log_{10}(\sigma_{\text{eff},0}) + \beta(q-1)\\
            p(\chi_{\text{eff}}\mid \mu_{\chi}, \sigma_{\chi}) &=
            \mathcal{N}_{[-1,1]}(\chi_{\text{eff}}\mid \mu_{\chi}, \sigma_{\chi})\\
        \end{align}

    The joint distribution is defined as,

    .. math::

        p(m_1, m_2, \chi_{\text{eff}}, z\mid \lambda_{\text{peak}}, \lambda, \mu_m, \sigma_m,
        m_{\text{min}}, m_{\text{max}}, \gamma, \alpha, \beta, \mu_{\text{eff},0},
        \log_{10}(\sigma_{\text{eff},0}), \kappa) = p(m_1\mid \lambda_{\text{peak}}, \lambda, \mu_m,
        \sigma_m, m_{\text{min}}, m_{\text{max}}) p(m_2\mid \gamma, m_1, m_{\text{min}})
        p(\chi_{\text{eff}}\mid \mu_{\chi}, \sigma_{\chi}) p(z\mid \kappa)


    .. note::

        The :math:`\frac{\mathrm{d}V_c}{\mathrm{d}z}` is removed from the redshift
        distribution, because the reference prior is from same distribution with
        :math:`\kappa=2.7`."""

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
        r"""
        Parameters
        ----------
        lambda_peak : Array
            Weight of the peak in the mass distribution, :math:`\lambda_{\text{peak}}`.
        lamb : Array
            Power-law index of the mass distribution, :math:`\lambda`.
        loc_m : Array
            Location parameter of the peak in the mass distribution, :math:`\mu_m`.
        scale_m : Array
            Scale parameter of the peak in the mass distribution, :math:`\sigma_m`.
        mmin : Array
            Minimum mass of the mass, :math:`m_{\text{min}}`.
        mmax : Array
            Maximum mass of the mass, :math:`m_{\text{max}}`.
        gamma : Array
            Power-law index of the mass ratio distribution, :math:`\gamma`.
        alpha : Array
            parameter of the :math:`\chi_{\text{eff}}` distribution, :math:`\alpha`.
        beta : Array
            parameter of the :math:`\chi_{\text{eff}}` distribution, :math:`\beta`.
        mu_eff_0 : Array
            parameter of the :math:`\chi_{\text{eff}}` distribution, :math:`\mu_{\text{eff},0}`.
        log10_sigma_eff_0 : Array
            parameter of the :math:`\chi_{\text{eff}}` distribution, :math:`\log_{10}(\sigma_{\text{eff},0})`.
        kappa : Array
            parameter of the redshift distribution, :math:`\kappa`.
        validate_args : bool, optional
            Whether to validate input arguments, by default None.
        """
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
