# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, List, Literal, Optional, Tuple

from jax import numpy as jnp, tree as jtr
from jaxtyping import Array
from numpyro.distributions import Distribution

from ...parameters import Parameters as P
from ..constraints import any_constraint
from ..utils import (
    ExtendedSupportTransformedDistribution,
    JointDistribution,
    ScaledMixture,
)
from ._ncombination import (
    combine_distributions,
    create_beta_distributions,
    create_independent_spin_orientation_gaussian_isotropic,
    create_powerlaw_redshift,
    create_powerlaws,
    create_truncated_normal_distributions,
    create_uniform_distributions,
)


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["pl", "g"],
    mass_distributions: List[Distribution],
    use_beta_spin_magnitude: bool,
    use_truncated_normal_spin_magnitude: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_tilt: bool,
    use_eccentricity: bool,
    use_mean_anomaly: bool,
    use_redshift: bool,
    use_cos_iota: bool,
    use_polarization_angle: bool,
    use_right_ascension: bool,
    use_sin_declination: bool,
    use_detection_time: bool,
    use_phi_1: bool,
    use_phi_2: bool,
    use_phi_12: bool,
    use_phi_orb: bool,
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
    use_mean_anomaly : bool
        whether to include mean_anomaly
    use_redshift : bool
        whether to include redshift
    use_cos_iota : bool
        whether to include cos_iota
    use_polarization_angle : bool
        whether to include polarization_angle
    use_right_ascension : bool
        whether to include right_ascension
    use_sin_declination : bool
        whether to include sin_declination
    use_detection_time : bool
        whether to include detection_time
    use_phi_1 : bool
        whether to include phi_1
    use_phi_2 : bool
        whether to include phi_2
    use_phi_12 : bool
        whether to include phi_12
    use_phi_orb : bool
        whether to include phi_orb
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
    # fmt: off
    _info_collection: List[Tuple[bool, str, Callable[..., List[Distribution]]]] = [
        (use_beta_spin_magnitude, P.PRIMARY_SPIN_MAGNITUDE.value, create_beta_distributions),
        (use_beta_spin_magnitude, P.SECONDARY_SPIN_MAGNITUDE.value, create_beta_distributions),
        (use_truncated_normal_spin_magnitude, P.PRIMARY_SPIN_MAGNITUDE.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_magnitude, P.SECONDARY_SPIN_MAGNITUDE.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_x, P.PRIMARY_SPIN_X.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_x, P.SECONDARY_SPIN_X.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_y, P.PRIMARY_SPIN_Y.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_y, P.SECONDARY_SPIN_Y.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_z, P.PRIMARY_SPIN_Z.value, create_truncated_normal_distributions),
        (use_truncated_normal_spin_z, P.SECONDARY_SPIN_Z.value, create_truncated_normal_distributions),
        # combined tilt distribution
        (use_tilt, P.COS_TILT_1.value + "_" + P.COS_TILT_2.value, create_independent_spin_orientation_gaussian_isotropic),
        (use_phi_1, P.PHI_1.value, create_uniform_distributions),
        (use_phi_2, P.PHI_2.value, create_uniform_distributions),
        (use_phi_12, P.PHI_12.value, create_uniform_distributions),
        (use_eccentricity, P.ECCENTRICITY.value, create_truncated_normal_distributions),
        (use_mean_anomaly, P.MEAN_ANOMALY.value, create_uniform_distributions),
        (use_redshift, P.REDSHIFT.value, create_powerlaw_redshift),
        (use_right_ascension, P.RIGHT_ASCENSION.value, create_uniform_distributions),
        (use_sin_declination, P.SIN_DECLINATION.value, create_uniform_distributions),
        (use_detection_time, P.DETECTION_TIME.value, create_uniform_distributions),
        (use_cos_iota, P.COS_IOTA.value, create_uniform_distributions),
        (use_polarization_angle, P.POLARIZATION_ANGLE.value, create_uniform_distributions),
        (use_phi_orb, P.PHI_ORB.value, create_uniform_distributions),
    ]
    # fmt: on

    # Iterate over the list of tuples and build distributions
    for use, param_name, build_func in _info_collection:
        if use:
            distributions = build_func(
                N=N,
                parameter_name=param_name,
                component_type=component_type,
                params=params,
                validate_args=validate_args,
            )
            build_distributions = combine_distributions(
                build_distributions, distributions
            )

    return build_distributions


def _build_pl_component_distributions(
    N: int,
    use_beta_spin_magnitude: bool,
    use_truncated_normal_spin_magnitude: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_tilt: bool,
    use_eccentricity: bool,
    use_mean_anomaly: bool,
    use_redshift: bool,
    use_cos_iota: bool,
    use_polarization_angle: bool,
    use_right_ascension: bool,
    use_sin_declination: bool,
    use_detection_time: bool,
    use_phi_1: bool,
    use_phi_2: bool,
    use_phi_12: bool,
    use_phi_orb: bool,
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
    use_mean_anomaly : bool
        whether to include mean_anomaly
    use_redshift : bool
        whether to include redshift
    use_cos_iota : bool
        whether to include cos_iota
    use_polarization_angle : bool
        whether to include polarization_angle
    use_right_ascension : bool
        whether to include right_ascension
    use_sin_declination : bool
        whether to include sin_declination
    use_detection_time : bool
        whether to include detection_time
    use_phi_1 : bool
        whether to include phi_1
    use_phi_2 : bool
        whether to include phi_2
    use_phi_12 : bool
        whether to include phi_12
    use_phi_orb : bool
        whether to include phi_orb
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[JointDistribution]
        list of JointDistribution
    """
    powerlaws = create_powerlaws(N=N, params=params, validate_args=validate_args)

    mass_distributions = jtr.map(
        lambda powerlaw: [powerlaw],
        powerlaws,
        is_leaf=lambda x: isinstance(x, ExtendedSupportTransformedDistribution),
    )

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type="pl",
        mass_distributions=mass_distributions,
        use_beta_spin_magnitude=use_beta_spin_magnitude,
        use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
        use_truncated_normal_spin_x=use_truncated_normal_spin_x,
        use_truncated_normal_spin_y=use_truncated_normal_spin_y,
        use_truncated_normal_spin_z=use_truncated_normal_spin_z,
        use_tilt=use_tilt,
        use_eccentricity=use_eccentricity,
        use_redshift=use_redshift,
        use_cos_iota=use_cos_iota,
        use_phi_12=use_phi_12,
        use_polarization_angle=use_polarization_angle,
        use_right_ascension=use_right_ascension,
        use_sin_declination=use_sin_declination,
        use_detection_time=use_detection_time,
        use_phi_1=use_phi_1,
        use_phi_2=use_phi_2,
        use_phi_orb=use_phi_orb,
        use_mean_anomaly=use_mean_anomaly,
        params=params,
        validate_args=validate_args,
    )

    return [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def _build_g_component_distributions(
    N: int,
    use_beta_spin_magnitude: bool,
    use_truncated_normal_spin_magnitude: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_tilt: bool,
    use_eccentricity: bool,
    use_mean_anomaly: bool,
    use_redshift: bool,
    use_cos_iota: bool,
    use_polarization_angle: bool,
    use_right_ascension: bool,
    use_sin_declination: bool,
    use_detection_time: bool,
    use_phi_1: bool,
    use_phi_2: bool,
    use_phi_12: bool,
    use_phi_orb: bool,
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
    use_mean_anomaly : bool
        whether to include mean_anomaly
    use_redshift : bool
        whether to include redshift
    use_cos_iota : bool
        whether to include cos_iota
    use_polarization_angle : bool
        whether to include polarization_angle
    use_right_ascension : bool
        whether to include right_ascension
    use_sin_declination : bool
        whether to include sin_declination
    use_detection_time : bool
        whether to include detection_time
    use_phi_1 : bool
        whether to include phi_1
    use_phi_2 : bool
        whether to include phi_2
    use_phi_12 : bool
        whether to include phi_12
    use_phi_orb : bool
        whether to include phi_orb
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[JointDistribution]
        list of JointDistribution
    """
    m1_dists = create_truncated_normal_distributions(
        N=N,
        parameter_name="m1",
        component_type="g",
        params=params,
        validate_args=validate_args,
    )
    m2_dists = create_truncated_normal_distributions(
        N=N,
        parameter_name="m2",
        component_type="g",
        params=params,
        validate_args=validate_args,
    )

    mass_distributions = jtr.map(
        lambda m1, m2: [m1, m2],
        m1_dists,
        m2_dists,
        is_leaf=lambda x: isinstance(x, Distribution),
    )

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type="g",
        mass_distributions=mass_distributions,
        use_beta_spin_magnitude=use_beta_spin_magnitude,
        use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
        use_truncated_normal_spin_x=use_truncated_normal_spin_x,
        use_truncated_normal_spin_y=use_truncated_normal_spin_y,
        use_truncated_normal_spin_z=use_truncated_normal_spin_z,
        use_tilt=use_tilt,
        use_eccentricity=use_eccentricity,
        use_redshift=use_redshift,
        use_cos_iota=use_cos_iota,
        use_phi_12=use_phi_12,
        use_polarization_angle=use_polarization_angle,
        use_right_ascension=use_right_ascension,
        use_sin_declination=use_sin_declination,
        use_detection_time=use_detection_time,
        use_phi_1=use_phi_1,
        use_phi_2=use_phi_2,
        use_phi_orb=use_phi_orb,
        use_mean_anomaly=use_mean_anomaly,
        params=params,
        validate_args=validate_args,
    )

    return [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def NPowerlawMGaussian(
    N_pl: int,
    N_g: int,
    use_beta_spin_magnitude: bool = False,
    use_truncated_normal_spin_magnitude: bool = False,
    use_truncated_normal_spin_x: bool = False,
    use_truncated_normal_spin_y: bool = False,
    use_truncated_normal_spin_z: bool = False,
    use_tilt: bool = False,
    use_eccentricity: bool = False,
    use_redshift: bool = False,
    use_cos_iota: bool = False,
    use_phi_12: bool = False,
    use_polarization_angle: bool = False,
    use_right_ascension: bool = False,
    use_sin_declination: bool = False,
    use_detection_time: bool = False,
    use_phi_1: bool = False,
    use_phi_2: bool = False,
    use_phi_orb: bool = False,
    use_mean_anomaly: bool = False,
    *,
    validate_args=None,
    **params,
) -> ScaledMixture:
    r"""Create a mixture of power-law and Gaussian components.

    This model has a lot of parameters, we can not list all of them here. Therefore,
    we are providing the general description of the each sub model and their parameters.

    .. important::

        Important information about this models are as follows:

        * The first :code:`N_pl` components are power-law and the next :code:`N_g` components are Gaussian.
        * Log rates are named as :code:`log_rate_{i}` where :code:`i` is the index of the component.
        * First :code:`N_pl` log rates are for power-law components and the next :code:`N_g` log rates are for Gaussian components.
        * All log rates are in terms of natural logarithm.


    .. note::

        **Mass distribution**: For powerlaw mass distribution is
        :class:`PowerlawPrimaryMassRatio` and for Gaussian we have
        :class:`~numpyro.distributions.truncated.TruncatedNormal` distribution.

        .. math:: (m_1, m_2) \sim \text{PowerlawPrimaryMassRatio}(\alpha, \beta, m_{\text{min}}, m_{\text{max}})

        .. math:: (m_1, m_2) \sim \mathcal{N}_{[a,b]}(\mu, \sigma^2)

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
    use_mean_anomaly : bool
        whether to include mean_anomaly, defaults to False
    use_redshift : bool
        whether to include redshift, defaults to False
    use_cos_iota : bool
        whether to include cos_iota, defaults to False
    use_polarization_angle : bool
        whether to include polarization_angle, defaults to False
    use_right_ascension : bool
        whether to include right_ascension, defaults to False
    use_sin_declination : bool
        whether to include sin_declination, defaults to False
    use_detection_time : bool
        whether to include detection_time, defaults to False
    use_phi_1 : bool
        whether to include phi_1, defaults to False
    use_phi_2 : bool
        whether to include phi_2, defaults to False
    use_phi_12 : bool
        whether to include phi_12, defaults to False
    use_phi_orb : bool
        whether to include phi_orb, defaults to False
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None

    Returns
    -------
    ScaledMixture
        scaled mixture of distributions
    """
    if N_pl > 0:
        pl_component_dist = _build_pl_component_distributions(
            N=N_pl,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
            use_truncated_normal_spin_x=use_truncated_normal_spin_x,
            use_truncated_normal_spin_y=use_truncated_normal_spin_y,
            use_truncated_normal_spin_z=use_truncated_normal_spin_z,
            use_tilt=use_tilt,
            use_eccentricity=use_eccentricity,
            use_redshift=use_redshift,
            use_cos_iota=use_cos_iota,
            use_phi_12=use_phi_12,
            use_polarization_angle=use_polarization_angle,
            use_right_ascension=use_right_ascension,
            use_sin_declination=use_sin_declination,
            use_detection_time=use_detection_time,
            use_phi_1=use_phi_1,
            use_phi_2=use_phi_2,
            use_phi_orb=use_phi_orb,
            use_mean_anomaly=use_mean_anomaly,
            params=params,
            validate_args=validate_args,
        )

    if N_g > 0:
        g_component_dist = _build_g_component_distributions(
            N=N_g,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
            use_truncated_normal_spin_x=use_truncated_normal_spin_x,
            use_truncated_normal_spin_y=use_truncated_normal_spin_y,
            use_truncated_normal_spin_z=use_truncated_normal_spin_z,
            use_tilt=use_tilt,
            use_eccentricity=use_eccentricity,
            use_redshift=use_redshift,
            use_cos_iota=use_cos_iota,
            use_phi_12=use_phi_12,
            use_polarization_angle=use_polarization_angle,
            use_right_ascension=use_right_ascension,
            use_sin_declination=use_sin_declination,
            use_detection_time=use_detection_time,
            use_phi_1=use_phi_1,
            use_phi_2=use_phi_2,
            use_phi_orb=use_phi_orb,
            use_mean_anomaly=use_mean_anomaly,
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
    log_rates = jnp.stack([params[f"log_rate_{i}"] for i in range(N)], axis=-1)

    return ScaledMixture(
        log_rates,
        component_dists,
        support=any_constraint(
            [component_dists.support for component_dists in component_dists]
        ),
        validate_args=validate_args,
    )
