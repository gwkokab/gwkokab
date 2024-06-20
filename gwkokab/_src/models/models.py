from __future__ import annotations

from functools import partial
from typing_extensions import Self

import numpy as np
from jax import jit, lax, numpy as jnp, random as jrd, tree as jtr, vmap
from jax.scipy.stats import norm
from jaxtyping import Array, Float, Int, PRNGKeyArray, Real
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key

from ..utils.transformations import m1_q_to_m2, mass_ratio
from .utils import JointDistribution, numerical_inverse_transform_sampling
from .utils.constraints import mass_ratio_mass_sandwich, mass_sandwich
from .utils.smoothing import smoothing_kernel


__all__ = [
    "BrokenPowerLawMassModel",
    "GaussianSpinModel",
    "IndependentSpinOrientationGaussianIsotropic",
    "MultiPeakMassModel",
    "NDistribution",
    "PowerLawPeakMassModel",
    "PowerLawPrimaryMassRatio",
    "TruncatedPowerLaw",
    "Wysocki2019MassModel",
]


class BrokenPowerLawMassModel(dist.Distribution):
    r"""See equation (B7) and (B6) in [Population Properties of Compact Objects
    from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog](https://arxiv.org/abs/2010.14533).

    $$
        \begin{align*}
            p(m_1) &\propto \begin{cases}
                m_1^{-\alpha_1}S(m_1\mid m_{\text{min}},\delta_m)
                & \text{if } m_{\min} \leq m_1 < m_{\text{break}} \\
                m_1^{-\alpha_2}S(m_1\mid m_{\text{min}},\delta_m)
                & \text{if } m_{\text{break}} < m_1 \leq m_{\max} \\
                0 & \text{otherwise}
            \end{cases} \\
            p(q\mid m_1) &\propto q^{\beta_q}S(m_1q\mid m_{\text{min}},\delta_m)
        \end{align*}
    $$
      
    Where $S(m\mid m_{\text{min}},\delta_m)$ is the smoothing kernel,
    defined in [`gwkokab.models.utils.smoothing.
    smoothing_kernel`]
    (utils.html#gwkokab.models.utils.smoothing.smoothing_kernel).
    """

    arg_constraints = {
        "alpha1": dist.constraints.real,
        "alpha2": dist.constraints.real,
        "beta_q": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
        "mbreak": dist.constraints.positive,
        "delta": dist.constraints.positive,
    }
    reparametrized_params = [
        "alpha1",
        "alpha2",
        "beta_q",
        "mmin",
        "mmax",
        "mbreak",
        "delta",
    ]
    pytree_aux_fields = ("_logZ", "_support")

    def __init__(
        self: Self,
        alpha1: Float,
        alpha2: Float,
        beta_q: Float,
        mmin: Float,
        mmax: Float,
        mbreak: Float,
        delta: Float,
        **kwargs,
    ):
        r"""
        :param alpha1: Power-law index for first component of primary mass model
        :param alpha2: Power-law index for second component of primary mass
            model
        :param beta_q: Power-law index for mass ratio model
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mbreak: Break mass
        :param delta: Smoothing parameter
        :param default_params: If `True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If `False`, the
            model will use primary mass and mass ratio.
        """
        (
            self.alpha1,
            self.alpha2,
            self.beta_q,
            self.mmin,
            self.mmax,
            self.mbreak,
            self.delta,
        ) = promote_shapes(alpha1, alpha2, beta_q, mmin, mmax, mbreak, delta)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(beta_q),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mbreak),
            jnp.shape(delta),
        )
        self._default_params = kwargs.get("default_params", True)
        if self._default_params:
            self._support = mass_sandwich(mmin, mmax)
        else:
            self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(BrokenPowerLawMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )
        self.alpha1_powerlaw = TruncatedPowerLaw(
            alpha=-self.alpha1,
            xmin=self.mmin,
            xmax=self.mbreak,
        )
        self.alpha2_powerlaw = TruncatedPowerLaw(
            alpha=-self.alpha2,
            xmin=self.mbreak,
            xmax=self.mmax,
        )
        self._normalization()

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self: Self):
        return self._support

    def _normalization(self: Self):
        """Precomputes the normalization constant for the primary mass model
        and mass ratio model using Monte Carlo integration.
        """
        num_samples = 20_000
        self._logZ = jnp.zeros_like(self.mmin)
        samples = self.sample(
            jrd.PRNGKey(np.random.randint(0, 2**32 - 1)), (num_samples,)
        )
        log_prob = self.log_prob(samples)
        prob = jnp.exp(log_prob)
        volume = jnp.prod(self.mmax - self.mmin)
        self._logZ = jnp.log(jnp.mean(prob, axis=-1)) + jnp.log(volume)

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(
        self: Self, m1: Array | Real
    ) -> Array | Real:
        r"""Log probability of primary mass model.
        
        .. math::

            \log p(m_1) = \begin{cases}
                \log S(m_1\mid m_{\text{min}},\delta_m) 
                - \log Z_{\text{primary}}
                - \alpha_1 \log m_1 
                & \text{if } m_{\min} \leq m_1 < m_{\text{break}} \\
                \log S(m_1\mid m_{\text{min}},\delta_m) 
                - \log Z_{\text{primary}}
                - \alpha_2 \log m_1 
                & \text{if } m_{\text{break}} < m_1 \leq m_{\max}
            \end{cases}
            
        :param m1: primary mass 
        :return: log probability of primary mass
        """
        conditions = [
            (self.mmin <= m1) & (m1 < self.mbreak),
            (self.mbreak <= m1) & (m1 <= self.mmax),
        ]
        log_smoothing_val = jnp.log(smoothing_kernel(m1, self.mmin, self.delta))
        log_probs = [
            log_smoothing_val + self.alpha1_powerlaw.log_prob(m1),
            log_smoothing_val + self.alpha2_powerlaw.log_prob(m1),
        ]
        return jnp.select(
            conditions, log_probs, default=jnp.full_like(m1, -jnp.inf)
        )

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self: Self, m1: Array | Real, q: Array | Real
    ) -> Array | Real:
        r"""Log probability of mass ratio model

        .. math::

            \log p(q\mid m_1) = \beta_q \log q +
            \log S(m_1q\mid m_{\text{min}},\delta_m)

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(
            smoothing_kernel(m1_q_to_m2(m1=m1, q=q), self.mmin, self.delta)
        )
        return self.beta_q * jnp.log(q) + log_smoothing_val

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        if self._default_params:
            m2 = value[..., 1]
            q = mass_ratio(m1=m1, m2=m2)
        else:
            q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return log_prob_m1 + log_prob_q - self._logZ

    def sample(self: Self, key: PRNGKeyArray, sample_shape: tuple = ()):
        assert is_prng_key(key)
        flattened_sample_shape = jtr.reduce(lambda x, y: x * y, sample_shape, 1)

        m1 = numerical_inverse_transform_sampling(
            logpdf=self._log_prob_primary_mass_model,
            limits=(self.mmin, self.mmax),
            sample_shape=(flattened_sample_shape,),
            key=key,
            batch_shape=self.batch_shape,
            n_grid_points=1000,
        )

        keys = jrd.split(key, m1.shape)

        q = vmap(
            lambda _m1, _k: numerical_inverse_transform_sampling(
                logpdf=partial(self._log_prob_mass_ratio_model, _m1),
                limits=(self.mmin / _m1, 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, keys)

        if self._default_params:
            return jnp.column_stack([m1, m1_q_to_m2(m1=m1, q=q)]).reshape(
                sample_shape + self.event_shape
            )
        return jnp.column_stack([m1, q]).reshape(
            sample_shape + self.event_shape
        )


def GaussianSpinModel(
    mu_eff: Float, sigma_eff: Float, mu_p: Float, sigma_p: Float, rho: Float
) -> dist.MultivariateNormal:
    r"""Bivariate normal distribution for the effective and precessing spins.
    See Eq. (D3) and (D4) in [Population Properties of Compact Objects from
    the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog](https://arxiv.org/abs/2010.14533).
    
    $$
        \left(\chi_{\text{eff}}, \chi_{p}\right) \sim \mathcal{N}\left(
            \begin{bmatrix}
                \mu_{\text{eff}} \\ \mu_{p}
            \end{bmatrix},
            \begin{bmatrix}
                \sigma_{\text{eff}}^2 & \rho \sigma_{\text{eff}} \sigma_{p} \\
                \rho \sigma_{\text{eff}} \sigma_{p} & \sigma_{p}^2
            \end{bmatrix}
        \right)
    $$
    
    where $\chi_{\text{eff}}$ is the effective spin and
    $\chi_{\text{eff}}\in[-1,1]$ and $\chi_{p}$ is the precessing spin and
    $\chi_{p}\in[0,1]$.

    :param mu_eff: mean of the effective spin
    :param sigma_eff: standard deviation of the effective spin
    :param mu_p: mean of the precessing spin
    :param sigma_p: standard deviation of the precessing spin
    :param rho: correlation coefficient between the effective and precessing
        spins
    :return: Multivariate normal distribution for the effective and precessing
        spins
    """
    return dist.MultivariateNormal(
        loc=[mu_eff, mu_p],
        covariance_matrix=[
            [lax.square(sigma_eff), rho * sigma_eff * sigma_p],
            [rho * sigma_eff * sigma_p, lax.square(sigma_p)],
        ],
        validate_args=True,
    )


def IndependentSpinOrientationGaussianIsotropic(
    zeta: Float, sigma1: Float, sigma2: Float
) -> dist.MixtureGeneral:
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components. See Eq. (4) of [Determining the population
    properties of spinning black holes](https://arxiv.org/abs/1704.08370).

    $$
        p(z_1,z_2\mid\zeta,\sigma_1,\sigma_2) = \frac{1-\zeta}{4} +
        \zeta\mathbb{I}_{[-1,1]}(z_1)\mathbb{I}_{[-1,1]}(z_2)
        \mathcal{N}(z_1\mid 1,\sigma_1)\mathcal{N}(z_2\mid 1,\sigma_2)
    $$

    where $\mathbb{I}(\cdot)$ is the indicator function.

    :param zeta: The mixing probability of the second component.
    :param sigma1: The standard deviation of the first component.
    :param sigma2: The standard deviation of the second component.
    :return: Mixture model of spin orientations.
    """
    mixing_probs = jnp.array([1 - zeta, zeta])
    component_0_dist = JointDistribution(
        dist.Uniform(low=-1, high=1, validate_args=True),
        dist.Uniform(low=-1, high=1, validate_args=True),
    )
    component_1_dist = JointDistribution(
        dist.TruncatedNormal(
            loc=1.0,
            scale=sigma1,
            low=-1,
            high=1,
            validate_args=True,
        ),
        dist.TruncatedNormal(
            loc=1.0,
            scale=sigma2,
            low=-1,
            high=1,
            validate_args=True,
        ),
    )

    return dist.MixtureGeneral(
        mixing_distribution=dist.Categorical(probs=mixing_probs),
        component_distributions=[component_0_dist, component_1_dist],
        support=dist.constraints.real,
        validate_args=True,
    )


class MultiPeakMassModel(dist.Distribution):
    r"""See equation (B9) and (B6) in [Population Properties of Compact
    Objects from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog](https://arxiv.org/abs/2010.14533).

    $$
        p(m_1\mid\lambda,\lambda_1,\alpha,\delta,m_{\text{min}},m_{\text{max}},
        \mu,\sigma)\propto \left[(1-\lambda)m_1^{-\alpha}
        \Theta(m_\text{max}-m_1)
        +\lambda\lambda_1\varphi\left(\frac{m_1-\mu_1}{\sigma_1}\right)
        +\lambda(1-\lambda_1)\varphi\left(\frac{m_1-\mu_2}{\sigma_2}\right)
        \right]S(m_1\mid m_{\text{min}},\delta)
    $$

    $$
        p(q\mid \beta, m_1,m_{\text{min}},\delta)\propto
        q^{\beta}S(m_1q\mid m_{\text{min}},\delta)
    $$

    Where,

    $$
        \varphi(x)=\frac{1}{\sigma\sqrt{2\pi}}
        \exp{\left(\displaystyle-\frac{x^{2}}{2}\right)}
    $$


    $S(m\mid m_{\text{min}},\delta_m)$ is the smoothing kernel,
    defined in [`gwkokab.models.utils.smoothing
    .smoothing_kernel`]
    (utils.html#gwkokab.models.utils.smoothing.smoothing_kernel),
    and $\Theta$ is the Heaviside step function.
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "lam": dist.constraints.interval(0, 1),
        "lam1": dist.constraints.interval(0, 1),
        "delta": dist.constraints.positive,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
        "mu1": dist.constraints.positive,
        "sigma1": dist.constraints.positive,
        "mu2": dist.constraints.positive,
        "sigma2": dist.constraints.positive,
    }
    reparametrized_params = [
        "alpha",
        "beta",
        "lam",
        "lam1",
        "delta",
        "mmin",
        "mmax",
        "mu1",
        "sigma1",
        "mu2",
        "sigma2",
    ]
    pytree_aux_fields = ("_logZ", "_support")

    def __init__(
        self: Self,
        alpha: Float,
        beta: Float,
        lam: Float,
        lam1: Float,
        delta: Float,
        mmin: Float,
        mmax: Float,
        mu1: Float,
        sigma1: Float,
        mu2: Float,
        sigma2: Float,
        **kwargs,
    ):
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta: Power-law index for mass ratio model
        :param lam: weight for power-law component
        :param lam1: weight for first Gaussian component
        :param delta: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu1: Mean of first Gaussian component
        :param sigma1: Standard deviation of first Gaussian component
        :param mu2: Mean of second Gaussian component
        :param sigma2: Standard deviation of second Gaussian component
        :param default_params: If `True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If `False`, the
            model will use primary mass and mass ratio.
        """
        (
            self.alpha,
            self.beta,
            self.lam,
            self.lam1,
            self.delta,
            self.mmin,
            self.mmax,
            self.mu1,
            self.sigma1,
            self.mu2,
            self.sigma2,
        ) = promote_shapes(
            alpha, beta, lam, lam1, delta, mmin, mmax, mu1, sigma1, mu2, sigma2
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(lam),
            jnp.shape(lam1),
            jnp.shape(delta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mu1),
            jnp.shape(sigma1),
            jnp.shape(mu2),
            jnp.shape(sigma2),
        )
        self._default_params = kwargs.get("default_params", True)
        if self._default_params:
            self._support = mass_sandwich(mmin, mmax)
        else:
            self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(MultiPeakMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

        self._normalization()

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self: Self):
        return self._support

    def _normalization(self: Self):
        """Precomputes the normalization constant for the primary mass model
        and mass ratio model using Monte Carlo integration.
        """
        num_samples = 20_000
        self._logZ = jnp.zeros_like(self.mmin)
        samples = self.sample(
            jrd.PRNGKey(np.random.randint(0, 2**32 - 1)), (num_samples,)
        )
        log_prob = self.log_prob(samples)
        prob = jnp.exp(log_prob)
        volume = jnp.prod(self.mmax - self.mmin)
        self._logZ = jnp.log(jnp.mean(prob, axis=-1)) + jnp.log(volume)

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(
        self: Self, m1: Array | Real
    ) -> Array | Real:
        r"""Log probability of primary mass model.

        $$
            \begin{multline*}
                p(m_1\mid\lambda,\lambda_1,\alpha,\delta,m_{\text{min}},
                m_{\text{max}},\mu,\sigma)\propto\\
                \left[(1-\lambda)m_1^{-\alpha}\Theta(m_\text{max}-m_1)
                +\lambda\lambda_1\varphi\left(\frac{m_1-\mu_1}{\sigma_1}\right)
                +\lambda(1-\lambda_1)\varphi\left(\frac{m_1-\mu_2}{\sigma_2}
                \right)\right]S(m_1\mid m_{\text{min}},\delta)
            \end{multline*}
        $$
        
        :param m1: primary mass
        :return: log probability of primary mass
        """
        gaussian_term1 = jnp.exp(
            jnp.log(self.lam)
            + jnp.log(self.lam1)
            + norm.logpdf(m1, self.mu1, self.sigma1)
        )
        gaussian_term2 = jnp.exp(
            jnp.log(self.lam)
            + jnp.log(1 - self.lam1)
            + norm.logpdf(m1, self.mu2, self.sigma2)
        )
        powerlaw_term = jnp.where(
            m1 < self.mmax,
            jnp.exp(jnp.log(1 - self.lam) - self.alpha * jnp.log(m1)),
            jnp.zeros_like(m1),
        )
        log_prob_val = jnp.log(powerlaw_term + gaussian_term1 + gaussian_term2)
        log_prob_val += jnp.log(smoothing_kernel(m1, self.mmin, self.delta))
        return log_prob_val

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self: Self, m1: Array | Real, q: Array | Real
    ) -> Array | Real:
        r"""Log probability of mass ratio model

        $$
            \log p(q\mid m_1) = \beta \log q +
            \log S(m_1q\mid m_{\text{min}},\delta_m)
        $$

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(
            smoothing_kernel(m1_q_to_m2(m1=m1, q=q), self.mmin, self.delta)
        )
        return self.beta * jnp.log(q) + log_smoothing_val

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        if self._default_params:
            m2 = value[..., 1]
            q = mass_ratio(m1=m1, m2=m2)
        else:
            q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return log_prob_m1 + log_prob_q - self._logZ

    def sample(self: Self, key: PRNGKeyArray, sample_shape: tuple = ()):
        assert is_prng_key(key)
        flattened_sample_shape = jtr.reduce(lambda x, y: x * y, sample_shape, 1)

        m1 = numerical_inverse_transform_sampling(
            logpdf=self._log_prob_primary_mass_model,
            limits=(self.mmin, self.mmax),
            sample_shape=(flattened_sample_shape,),
            key=key,
            batch_shape=self.batch_shape,
            n_grid_points=1000,
        )

        key = jrd.split(key, m1.shape)

        q = vmap(
            lambda _m1, _k: numerical_inverse_transform_sampling(
                logpdf=partial(self._log_prob_mass_ratio_model, _m1),
                limits=(self.mmin / _m1, 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, key)

        if self._default_params:
            return jnp.column_stack([m1, m1_q_to_m2(m1=m1, q=q)]).reshape(
                sample_shape + self.event_shape
            )
        return jnp.column_stack([m1, q]).reshape(
            sample_shape + self.event_shape
        )


def NDistribution(
    distribution: dist.Distribution, n: Int, **params
) -> dist.MixtureGeneral:
    """Mixture of any $n$ distributions.

    ```python
    >>> distribution = NDistribution(
    ...     distribution=dist.MultivariateNormal,
    ...     n=4,
    ...     loc_0=jnp.array([2.0, 2.0]),
    ...     covariance_matrix_0=jnp.eye(2),
    ...     loc_1=jnp.array([-2.0, -2.0]),
    ...     covariance_matrix_1=jnp.eye(2),
    ...     loc_2=jnp.array([-2.0, 2.0]),
    ...     covariance_matrix_2=jnp.eye(2),
    ...     loc_3=jnp.array([2.0, -2.0]),
    ...     covariance_matrix_3=jnp.eye(2),
    ... )
    ```

    :param distribution: distribution to mix
    :param n: number of components
    :return: Mixture of $n$ distributions
    """
    arg_names = distribution.arg_constraints.keys()
    mixing_dist = dist.Categorical(probs=jnp.ones(n) / n, validate_args=True)
    args_per_component = [
        {arg: params.get(f"{arg}_{i}") for arg in arg_names} for i in range(n)
    ]
    component_dists = jtr.map(
        lambda x: distribution(**x),
        args_per_component,
        is_leaf=lambda x: isinstance(x, dict),
    )
    return dist.MixtureGeneral(
        mixing_dist,
        component_dists,
        support=distribution.support,
        validate_args=True,
    )


class PowerLawPeakMassModel(dist.Distribution):
    r"""See equation (B3) and (B6) in [Population Properties of Compact
    Objects from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog](https://arxiv.org/abs/2010.14533).

    $$
        \begin{align*}
            p(m_1\mid\lambda,\alpha,\delta,m_{\text{min}},m_{\text{max}},
            \mu,\sigma)
            &\propto \left[(1-\lambda)m_1^{-\alpha}\Theta(m_\text{max}-m_1)
            +\frac{\lambda}{\sigma\sqrt{2\pi}}
            e^{-\frac{1}{2}\left(\frac{m_1-\mu}{\sigma}\right)^{2}}\right]
            S(m_1\mid m_{\text{min}},\delta)
            \\
            p(q\mid \beta, m_1,m_{\text{min}},\delta)
            &\propto q^{\beta}S(m_1q\mid m_{\text{min}},\delta)
        \end{align*}
    $$
        
    Where $S(m\mid m_{\text{min}},\delta_m)$ is the smoothing kernel, defined
    in [`gwkokab.models.utils.smoothing
    .smoothing_kernel`](utils.html#gwkokab.models.utils.smoothing.smoothing_kernel),
    and $\Theta$ is the Heaviside step function.
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "lam": dist.constraints.interval(0, 1),
        "delta": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
        "mu": dist.constraints.real,
        "sigma": dist.constraints.positive,
    }
    reparametrized_params = [
        "alpha",
        "beta",
        "lam",
        "delta",
        "mmin",
        "mmax",
        "mu",
        "sigma",
    ]
    pytree_aux_fields = ("_logZ", "_support")

    def __init__(
        self: Self,
        alpha: Float,
        beta: Float,
        lam: Float,
        delta: Float,
        mmin: Float,
        mmax: Float,
        mu: Float,
        sigma: Float,
        **kwargs,
    ) -> None:
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta: Power-law index for mass ratio model
        :param lam: Fraction of Gaussian component
        :param delta: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu: Mean of Gaussian component
        :param sigma: Standard deviation of Gaussian component
        :param default_params: If `True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If `False`, the
            model will use primary mass and mass ratio.
        """
        (
            self.alpha,
            self.beta,
            self.lam,
            self.delta,
            self.mmin,
            self.mmax,
            self.mu,
            self.sigma,
        ) = promote_shapes(alpha, beta, lam, delta, mmin, mmax, mu, sigma)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(lam),
            jnp.shape(delta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mu),
            jnp.shape(sigma),
        )
        self._default_params = kwargs.get("default_params", True)
        if self._default_params:
            self._support = mass_sandwich(mmin, mmax)
        else:
            self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(PowerLawPeakMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

        self._normalization()

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self: Self):
        return self._support

    def _normalization(self: Self):
        """Precomputes the normalization constant for the primary mass model
        and mass ratio model using Monte Carlo integration.
        """
        num_samples = 20_000
        self._logZ = jnp.zeros_like(self.mmin)
        samples = self.sample(
            jrd.PRNGKey(np.random.randint(0, 2**32 - 1)), (num_samples,)
        )
        log_prob = self.log_prob(samples)
        prob = jnp.exp(log_prob)
        volume = jnp.prod(self.mmax - self.mmin)
        self._logZ = jnp.log(jnp.mean(prob, axis=-1)) + jnp.log(volume)

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(
        self: Self, m1: Array | Real
    ) -> Array | Real:
        r"""Log probability of primary mass model.

        .. math::

            p(m_1\mid\lambda,\alpha,\delta,m_{\text{min}},m_{\text{max}},
            \mu,\sigma) \propto \left[(1-\lambda)m_1^{-\alpha}
            \Theta(m_\text{max}-m_1)
            +\frac{\lambda}{\sigma\sqrt{2\pi}}
            e^{-\frac{1}{2}\left(\frac{m_1-\mu}{\sigma}\right)^{2}}\right]
            S(m_1\mid m_{\text{min}},\delta)

        :param m1: primary mass
        :return: log probability of primary mass
        """
        gaussian_term = jnp.exp(
            jnp.log(self.lam) + norm.logpdf(m1, self.mu, self.sigma)
        )
        powerlaw_term = jnp.where(
            m1 < self.mmax,
            jnp.exp(jnp.log(1 - self.lam) - self.alpha * jnp.log(m1)),
            jnp.zeros_like(m1),
        )
        log_prob_val = jnp.log(powerlaw_term + gaussian_term) + jnp.log(
            smoothing_kernel(m1, self.mmin, self.delta)
        )
        return log_prob_val

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self: Self, m1: Array | Real, q: Array | Real
    ) -> Array | Real:
        r"""Log probability of mass ratio model

        .. math::

            \log p(q\mid m_1) = \beta \log q +
            \log S(m_1q\mid m_{\text{min}},\delta_m)

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(
            smoothing_kernel(m1_q_to_m2(m1=m1, q=q), self.mmin, self.delta)
        )
        return self.beta * jnp.log(q) + log_smoothing_val

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        if self._default_params:
            m2 = value[..., 1]
            q = mass_ratio(m1=m1, m2=m2)
        else:
            q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return log_prob_m1 + log_prob_q - self._logZ

    def sample(self: Self, key: PRNGKeyArray, sample_shape: tuple = ()):
        flattened_sample_shape = jtr.reduce(lambda x, y: x * y, sample_shape, 1)

        m1 = numerical_inverse_transform_sampling(
            logpdf=self._log_prob_primary_mass_model,
            limits=(self.mmin, self.mmax),
            sample_shape=(flattened_sample_shape,),
            key=key,
            batch_shape=self.batch_shape,
            n_grid_points=1000,
        )

        keys = jrd.split(key, m1.shape)

        q = vmap(
            lambda _m1, _k: numerical_inverse_transform_sampling(
                logpdf=partial(self._log_prob_mass_ratio_model, _m1),
                limits=(self.mmin / _m1, 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, keys)

        if self._default_params:
            return jnp.column_stack([m1, m1_q_to_m2(m1=m1, q=q)]).reshape(
                sample_shape + self.event_shape
            )
        return jnp.column_stack([m1, q]).reshape(
            sample_shape + self.event_shape
        )


class PowerLawPrimaryMassRatio(dist.Distribution):
    r"""Power law model for two-dimensional mass distribution,
    modelling primary mass and conditional mass ratio
    distribution.

    $$p(m_1,q\mid\alpha,\beta) = p(m_1\mid\alpha)p(q \mid m_1, \beta)$$
    
    $$
        \begin{align*}
            p(m_1\mid\alpha)&
            \propto m_1^{\alpha},\qquad m_{\text{min}}\leq m_1\leq m_{\max}\\
            p(q\mid m_1,\beta)&
            \propto q^{\beta},\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1
        \end{align*}
    $$
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
    }
    reparametrized_params = ["alpha", "beta", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(
        self: Self,
        alpha: Float,
        beta: Float,
        mmin: Float,
        mmax: Float,
        **kwargs,
    ) -> None:
        """
        :param alpha: Power law index for primary mass
        :param beta: Power law index for mass ratio
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param default_params: If `True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If `False`, the
            model will use primary mass and mass ratio.
        """
        self.alpha, self.beta, self.mmin, self.mmax = promote_shapes(
            alpha, beta, mmin, mmax
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha), jnp.shape(beta), jnp.shape(mmin), jnp.shape(mmax)
        )
        self._default_params = kwargs.get("default_params", True)
        if self._default_params:
            self._support = mass_sandwich(mmin, mmax)
        else:
            self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(PowerLawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @dist.constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self: Self):
        return self._support

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        if self._default_params:
            m2 = value[..., 1]
            q = mass_ratio(m1=m1, m2=m2)
        else:
            q = value[..., 1]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            xmin=self.mmin,
            xmax=self.mmax,
        ).log_prob(m1)
        log_prob_q = TruncatedPowerLaw(
            alpha=self.beta,
            xmin=lax.div(self.mmin, m1),
            xmax=1.0,
        ).log_prob(q)
        return log_prob_m1 + log_prob_q

    def sample(self: Self, key: PRNGKeyArray, sample_shape: tuple = ()):
        m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            xmin=self.mmin,
            xmax=self.mmax,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = jrd.split(key)[1]
        q = TruncatedPowerLaw(
            alpha=self.beta,
            xmin=lax.div(self.mmin, m1),
            xmax=1.0,
        ).sample(key=key, sample_shape=())

        if self._default_params:
            return jnp.column_stack((m1, m1_q_to_m2(m1=m1, q=q)))
        return jnp.column_stack((m1, q))


class TruncatedPowerLaw(dist.Distribution):
    r"""A generic double side truncated power law distribution.
    
    !!! note
        There are many different definition of Power Law that include
        exponential cut-offs and interval cut-offs.  They are just
        interchangeably. This class is the implementation of power law that has
        been restricted over a closed interval.

    $$  
        p(x\mid\alpha, x_{\text{min}}, x_{\text{max}}):=
        \begin{cases}
            \displaystyle\frac{x^{\alpha}}{\mathcal{Z}}
            & 0<x_{\text{min}}\leq x\leq x_{\text{max}}\\
            0 & \text{otherwise}
        \end{cases}
    $$

    where $\mathcal{Z}$ is the normalization constant and $\alpha$ is the power
    law index. $x_{\text{min}}$ and $x_{\text{max}}$ are the lower and upper
    truncation limits, respectively. The normalization constant is given by,
    
    $$
        \mathcal{Z}:=\begin{cases}
            \log{x_{\text{max}}}-\log{x_{\text{min}}} & \alpha = -1 \\
            \displaystyle
            \frac{x_{\text{max}}^{1+\alpha}-x_{\text{min}}^{1+\alpha}}{1+\alpha}
            & \text{otherwise}
        \end{cases}
    $$
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "xmin": dist.constraints.dependent,
        "xmax": dist.constraints.dependent,
    }
    reparametrized_params = ["alpha", "xmin", "xmax"]
    pytree_aux_fields = ("_support", "_logZ")

    def __init__(self: Self, alpha: Float, xmin: Float, xmax: Float) -> None:
        r"""
        :param alpha: Index of the power law
        :param xmin: Lower truncation limit
        :param xmax: Upper truncation limit
        """
        self.alpha, self.xmin, self.xmax = promote_shapes(alpha, xmin, xmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha), jnp.shape(xmin), jnp.shape(xmax)
        )
        super(TruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=True
        )
        self._support = dist.constraints.interval(xmin, xmax)
        self._logZ = self._log_Z()

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self: Self):
        return self._support

    @partial(jit, static_argnums=(0,))
    def _log_Z(self: Self) -> Array | Real:
        """Computes the logarithm of normalization constant.

        :return: The logarithm of normalization constant.
        """
        return jnp.where(
            self.alpha == -1.0,
            jnp.log(jnp.log(self.xmax) - jnp.log(self.xmin)),
            jnp.log(
                jnp.abs(
                    jnp.power(self.xmax, 1.0 + self.alpha)
                    - jnp.power(self.xmin, 1.0 + self.alpha)
                )
            )
            - jnp.log(jnp.abs(1.0 + self.alpha)),
        )

    @validate_sample
    def log_prob(self: Self, value: Array | Real) -> Array | Real:
        return self.alpha * jnp.log(value) - self._logZ

    def sample(self: Self, key: PRNGKeyArray, sample_shape: tuple = ()):
        U = jrd.uniform(key, sample_shape + self.batch_shape)
        return jnp.where(
            self.alpha == -1.0,
            jnp.exp(
                jnp.log(self.xmin)
                + U * (jnp.log(self.xmax) - jnp.log(self.xmin))
            ),
            jnp.power(
                jnp.power(self.xmin, 1.0 + self.alpha)
                + U
                * (
                    jnp.power(self.xmax, 1.0 + self.alpha)
                    - jnp.power(self.xmin, 1.0 + self.alpha)
                ),
                jnp.reciprocal(1.0 + self.alpha),
            ),
        )


class Wysocki2019MassModel(dist.Distribution):
    r"""It is a double side truncated power law distribution, as described in
    equation 7 of the [Reconstructing phenomenological distributions of compact
    binaries via gravitational wave observations](https://arxiv.org/abs/1805.06442).

    $$
        p(m_1,m_2\mid\alpha,m_{\text{min}},m_{\text{max}},M_{\text{max}})\propto
        \frac{m_1^{-\alpha}}{m_1-m_{\text{min}}}
    $$
    """

    arg_constraints = {
        "alpha_m": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
    }
    reparametrized_params = ["alpha_m", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self: Self, alpha_m: Float, mmin: Float, mmax: Float) -> None:
        r"""
        :param alpha_m: index of the power law distribution
        :param mmin: lower mass limit
        :param mmax: upper mass limit
        """
        self.alpha_m, self.mmin, self.mmax = promote_shapes(alpha_m, mmin, mmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
        )
        self._support = mass_sandwich(mmin, mmax)
        super(Wysocki2019MassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self: Self):
        return self._support

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=-self.alpha_m,
            xmin=self.mmin,
            xmax=self.mmax,
        ).log_prob(m1)
        log_prob_m2_given_m1 = -jnp.log(m1 - self.mmin)
        return log_prob_m1 + log_prob_m2_given_m1

    def sample(
        self: Self, key: PRNGKeyArray, sample_shape: tuple = ()
    ) -> Array:
        m2 = dist.Uniform(
            low=self.mmin,
            high=self.mmax,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = jrd.split(key)[-1]
        m1 = TruncatedPowerLaw(
            alpha=-self.alpha_m,
            xmin=m2,
            xmax=self.mmax,
        ).sample(key=key, sample_shape=())
        return jnp.column_stack((m1, m2))
