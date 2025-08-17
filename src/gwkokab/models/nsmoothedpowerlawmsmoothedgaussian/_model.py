# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Literal, Optional

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import constraints, Distribution

from ...models.spin import BetaFromMeanVar, IndependentSpinOrientationGaussianIsotropic
from ...models.transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from .._models import SmoothedGaussianPrimaryMassRatio, SmoothedPowerlawPrimaryMassRatio
from ..constraints import any_constraint
from ..redshift import PowerlawRedshift
from ..utils import (
    combine_distributions,
    create_beta_distributions,
    create_independent_spin_orientation_gaussian_isotropic,
    create_powerlaw_redshift,
    create_smoothed_gaussians,
    create_smoothed_powerlaws,
    create_truncated_normal_distributions,
    ExtendedSupportTransformedDistribution,
    JointDistribution,
    ScaledMixture,
)


build_powerlaw_distributions = create_smoothed_powerlaws
build_gaussian_distributions = create_smoothed_gaussians
build_spin_distributions = create_beta_distributions
build_tilt_distributions = create_independent_spin_orientation_gaussian_isotropic
build_eccentricity_distributions = create_truncated_normal_distributions
build_redshift_distributions = create_powerlaw_redshift


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["pl", "g"],
    mass_distributions: List[Distribution],
    use_spin: bool,
    use_tilt: bool,
    use_eccentricity: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build distributions for non-mass parameters.

    Parameters
    ----------
    N : int
        Number of components
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    mass_distributions : List[Distribution]
        list of mass distributions
    use_spin : bool
        whether to include spin
    use_tilt : bool
        whether to include tilt
    use_eccentricity : bool
        whether to include eccentricity
    use_redshift : bool
        whether to include redshift
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[Distribution]
        list of distributions
    """
    build_distributions = mass_distributions

    if use_spin:
        chi1_dists = build_spin_distributions(
            N=N,
            parameter_name="chi1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        chi2_dists = build_spin_distributions(
            N=N,
            parameter_name="chi2",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        build_distributions = combine_distributions(build_distributions, chi1_dists)
        build_distributions = combine_distributions(build_distributions, chi2_dists)

    if use_tilt:
        tilt_dists = build_tilt_distributions(
            N=N,
            parameter_name="cos_tilt1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        build_distributions = combine_distributions(build_distributions, tilt_dists)

    if use_eccentricity:
        ecc_dists = build_eccentricity_distributions(
            N=N,
            parameter_name="ecc",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        build_distributions = combine_distributions(build_distributions, ecc_dists)
    if use_redshift:
        redshift_dists = build_redshift_distributions(
            N=N,
            parameter_name="redshift",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        build_distributions = combine_distributions(build_distributions, redshift_dists)

    return build_distributions


def _build_pl_component_distributions(
    N: int,
    use_spin: bool,
    use_tilt: bool,
    use_eccentricity: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[JointDistribution]:
    """Build distributions for power-law components.

    Parameters
    ----------
    N : int
        Number of components
    use_spin : bool
        whether to include spin
    use_tilt : bool
        whether to include tilt
    use_eccentricity : bool
        whether to include eccentricity
    use_redshift : bool
        whether to include redshift
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[JointDistribution]
        list of JointDistribution
    """
    smoothed_powerlaws = build_powerlaw_distributions(
        N=N, params=params, validate_args=validate_args
    )

    mass_distributions = [[powerlaw] for powerlaw in smoothed_powerlaws]

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type="pl",
        mass_distributions=mass_distributions,
        use_spin=use_spin,
        use_tilt=use_tilt,
        use_eccentricity=use_eccentricity,
        use_redshift=use_redshift,
        params=params,
        validate_args=validate_args,
    )

    return [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def _build_g_component_distributions(
    N: int,
    use_spin: bool,
    use_tilt: bool,
    use_eccentricity: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[JointDistribution]:
    """Build distributions for Gaussian components.

    Parameters
    ----------
    N : int
        Number of components
    use_spin : bool
        whether to include spin
    use_tilt : bool
        whether to include tilt
    use_eccentricity : bool
        whether to include eccentricity
    use_redshift : bool
        whether to include redshift
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[JointDistribution]
        list of JointDistribution
    """
    m1_q_dists = build_gaussian_distributions(
        N=N,
        params=params,
        validate_args=validate_args,
    )

    mass_distributions = [[m1q] for m1q in m1_q_dists]

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type="g",
        mass_distributions=mass_distributions,
        use_spin=use_spin,
        use_tilt=use_tilt,
        use_eccentricity=use_eccentricity,
        use_redshift=use_redshift,
        params=params,
        validate_args=validate_args,
    )

    return [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def NSmoothedPowerlawMSmoothedGaussian(
    N_pl: int,
    N_g: int,
    use_spin: bool = False,
    use_tilt: bool = False,
    use_eccentricity: bool = False,
    use_redshift: bool = False,
    *,
    validate_args=None,
    **params,
) -> ScaledMixture:
    r"""Create a mixture of Smoothed Powerlaws and Smoothed Gaussian components.

    This model has a lot of parameters, we can not list all of them here. Therefore,
    we are providing the general description of the each sub model and their parameters.

    .. important::

        Important information about this models are as follows:

        * The first :code:`N_pl` components are Smoothed Powerlaws and the next :code:`N_g` components are Smoothed Gaussian.
        * Log rates are named as :code:`log_rate_{i}` where :code:`i` is the index of the component.
        * First :code:`N_pl` log rates are for power-law components and the next :code:`N_g` log rates are for Gaussian components.
        * All log rates are in terms of natural logarithm.


    .. note::

        **Mass distribution**: For powerlaw mass distribution is
        :class:`SmoothedPowerlawPrimaryMassRatio` and for Gaussian we have
        :class:`SmoothedGaussianPrimaryMassRatio` distribution.

        .. math:: (m_1, q) \sim \text{SmoothedPowerlawPrimaryMassRatio}(\alpha, \beta, m_{\text{min}}, m_{\text{max}}, \delta)

        .. math:: (m_1, q) \sim \text{SmoothedGaussianPrimaryMassRatio}(\mu, \sigma, \beta, m_{\text{min}}, m_{\text{max}}, \delta)

        **Spin distribution**: Spin is taken from a beta distribution. The beta distribution
        is parameterized by :math:`\mu` and :math:`\sigma^2` where :math:`\mu` is the mean
        and :math:`\sigma^2` is the variance of the beta distribution.

        .. math:: \chi_i \sim \mathrm{Beta}(\mu, \sigma^2)

        .. warning::
            Not every choice of :math:`\mu` and :math:`\sigma^2` will result in a valid beta
            distribution. The beta distribution is only valid when :math:`\mu \in (0, 1)`,
            :math:`\sigma^2 \in (0, 0.25)`, and :math:`\mu(1-\mu) > \sigma^2`. Refer to the
            `link <https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance#comment926602_12239>`_
            for more information.

        **Tilt distribution**: Tilt is taken from a
        :class:`~numpyro.distributions.truncated.TruncatedNormal` distribution, with
        fixed mean :math:`\mu=1` and fixed bounds, :math:`a=-1` and :math:`b=1`.

        .. math:: \cos(\theta_i) \sim \mathcal{N}_{[-1, 1]}(\sigma^2\mid\mu=1)

        **Eccentricity distribution**: Eccentricity is taken from
        :class:`~numpyro.distributions.truncated.TruncatedNormal` distribution.

        .. math:: \varepsilon_i \sim \mathcal{N}_{[a,b]}(\mu, \sigma^2)


    .. attention::

        Interestingly, in :class:`~numpyro.distributions.truncated.TruncatedNormal`
        distribution, if any of the bounds are not provided, it will be set to
        :math:`\pm\infty`. For example, if we set :math:`\mu=0` and do not provide the
        upper bound, then the resulting distribution would be a
        `half normal <https://en.wikipedia.org/wiki/Half-normal_distribution>`_
        distribution.


    The naming of the parameters follows the following convention:

    .. code:: bash

        <parameter name>_<model parameter>_<component type>_<component number>

    with an exception for the powerlaw mass distribution where the
    :code:`<parameter name>` is ignored. For example, spin is taken from a beta
    distribution whose parameters are :code:`mean` and :code:`variance`. The naming
    convention for the spin parameters would be:

    .. code:: text

        chi[1-2]_mean_(pl|g)_[0-N_pl+N_g]
        chi[1-2]_variance_(pl|g)_[0-N_pl+N_g]

    Parameters
    ----------

    N_pl : int
        Number of power-law components
    N_g : int
        Number of Gaussian components
    use_spin : bool
        whether to include spin, defaults to False
    use_tilt : bool
        whether to include tilt, defaults to False
    use_eccentricity : bool
        whether to include eccentricity, defaults to False
    use_redshift : bool
        whether to include redshift, defaults to False
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None

    Returns
    -------
    ScaledMixture
        Mixture of power-law and Gaussian components
    """
    if N_pl > 0:
        pl_component_dist = _build_pl_component_distributions(
            N=N_pl,
            use_spin=use_spin,
            use_tilt=use_tilt,
            use_eccentricity=use_eccentricity,
            use_redshift=use_redshift,
            params=params,
            validate_args=validate_args,
        )

    if N_g > 0:
        g_component_dist = _build_g_component_distributions(
            N=N_g,
            use_spin=use_spin,
            use_tilt=use_tilt,
            use_eccentricity=use_eccentricity,
            use_redshift=use_redshift,
            params=params,
            validate_args=validate_args,
        )

    if N_pl == 0 and N_g != 0:
        component_dists = g_component_dist
    elif N_g == 0 and N_pl != 0:
        component_dists = pl_component_dist
    else:
        component_dists = pl_component_dist + g_component_dist

    N = N_pl + N_g
    log_rates = jnp.stack(
        [params.get(f"log_rate_{i}", 0.0) for i in range(N)],
        axis=-1,
    )

    return ScaledMixture(
        log_rates,
        component_dists,
        support=constraints.real_vector,
        validate_args=validate_args,
    )


def SmoothedPowerlawAndPeak(
    use_spin: bool = False,
    use_redshift: bool = False,
    use_tilt: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    smoothed_powerlaw = ExtendedSupportTransformedDistribution(
        SmoothedPowerlawPrimaryMassRatio(
            alpha=params["alpha"],
            beta=params["beta"],
            mmin=params["mmin"],
            mmax=params["mmax"],
            delta=params["delta"],
            validate_args=validate_args,
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
        validate_args=validate_args,
    )
    smoothed_gaussian = ExtendedSupportTransformedDistribution(
        SmoothedGaussianPrimaryMassRatio(
            loc=params["loc"],
            scale=params["scale"],
            beta=params["beta"],
            mmin=params["mmin"],
            mmax=params["mmax"],
            delta=params["delta"],
            validate_args=validate_args,
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
        validate_args=validate_args,
    )

    component_distribution_pl = [smoothed_powerlaw]
    component_distribution_g = [smoothed_gaussian]

    if use_spin:
        chi1_dist_pl = BetaFromMeanVar(
            mean=params["chi1_mean_pl"],
            variance=params["chi1_variance_pl"],
            validate_args=validate_args,
        )

        chi2_dist_pl = BetaFromMeanVar(
            mean=params["chi2_mean_pl"],
            variance=params["chi2_variance_pl"],
            validate_args=validate_args,
        )

        chi1_dist_g = BetaFromMeanVar(
            mean=params["chi1_mean_g"],
            variance=params["chi1_variance_g"],
            validate_args=validate_args,
        )

        chi2_dist_g = BetaFromMeanVar(
            mean=params["chi2_mean_g"],
            variance=params["chi2_variance_g"],
            validate_args=validate_args,
        )

        component_distribution_pl.extend([chi1_dist_pl, chi2_dist_pl])
        component_distribution_g.extend([chi1_dist_g, chi2_dist_g])

    if use_tilt:
        tilt_dist_pl = IndependentSpinOrientationGaussianIsotropic(
            zeta=params["cos_tilt_zeta_pl"],
            scale1=params["cos_tilt1_scale_pl"],
            scale2=params["cos_tilt2_scale_pl"],
            validate_args=validate_args,
        )
        tilt_dist_g = IndependentSpinOrientationGaussianIsotropic(
            zeta=params["cos_tilt_zeta_g"],
            scale1=params["cos_tilt1_scale_g"],
            scale2=params["cos_tilt2_scale_g"],
            validate_args=validate_args,
        )

        component_distribution_pl.append(tilt_dist_pl)
        component_distribution_g.append(tilt_dist_g)

    if use_redshift:
        z_max = params["z_max"]
        kappa = params["kappa"]
        powerlaw_z = PowerlawRedshift(
            z_max=z_max, kappa=kappa, validate_args=validate_args
        )

        component_distribution_pl.append(powerlaw_z)
        component_distribution_g.append(powerlaw_z)

    if len(component_distribution_pl) == 1:
        component_distribution_pl = component_distribution_pl[0]
    else:
        component_distribution_pl = JointDistribution(
            *component_distribution_pl, validate_args=validate_args
        )

    if len(component_distribution_g) == 1:
        component_distribution_g = component_distribution_g[0]
    else:
        component_distribution_g = JointDistribution(
            *component_distribution_g, validate_args=validate_args
        )
    return ScaledMixture(
        log_scales=jnp.stack(
            [
                params["log_rate"] + jnp.log1p(-params["lambda_peak"]),  # type: ignore[arg-type, operator]
                params["log_rate"] + jnp.log(params["lambda_peak"]),  # type: ignore[arg-type]
            ],
            axis=-1,
        ),
        component_distributions=[component_distribution_pl, component_distribution_g],
        support=any_constraint(
            [
                component_distribution_pl.support,  # type: ignore[attr-defined]
                component_distribution_g.support,  # type: ignore[attr-defined]
            ]
        ),
        validate_args=validate_args,
    )
