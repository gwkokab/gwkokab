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


from typing_extensions import Dict, List, Literal, Optional

from jax import numpy as jnp, tree as jtr
from jaxtyping import Array
from numpyro.distributions import constraints, Distribution

from .._models import SmoothedGaussianPrimaryMassRatio, SmoothedPowerlawPrimaryMassRatio
from ..utils import (
    combine_distributions,
    create_powerlaw_redshift,
    create_smoothed_gaussians,
    create_smoothed_powerlaws,
    create_truncated_normal_distributions,
    create_truncated_normal_distributions_for_cos_tilt,
    JointDistribution,
    ScaledMixture,
)


build_spin_distributions = create_truncated_normal_distributions
build_tilt_distributions = create_truncated_normal_distributions_for_cos_tilt
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
    r"""Build distributions for non-mass parameters.

    :param N: Number of components
    :param component_type: type of component, either "pl" or "g"
    :param mass_distributions: list of mass distributions
    :param use_spin: whether to include spin
    :param use_tilt: whether to include tilt
    :param use_eccentricity: whether to include eccentricity
    :param use_redshift: whether to include redshift
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :return: list of distributions
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
        tilt1_dists = build_tilt_distributions(
            N=N,
            parameter_name="cos_tilt1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        tilt2_dists = build_tilt_distributions(
            N=N,
            parameter_name="cos_tilt2",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        build_distributions = combine_distributions(build_distributions, tilt1_dists)
        build_distributions = combine_distributions(build_distributions, tilt2_dists)

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
    r"""Build distributions for power-law components.

    :param N: Number of components
    :param use_spin: whether to include spin
    :param use_tilt: whether to include tilt
    :param use_eccentricity: whether to include eccentricity
    :param use_redshift: whether to include redshift
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :return: list of JointDistribution
    """
    smoothed_powerlaws = create_smoothed_powerlaws(
        N=N, params=params, validate_args=validate_args
    )

    mass_distributions = jtr.map(
        lambda powerlaw: [powerlaw],
        smoothed_powerlaws,
        is_leaf=lambda x: isinstance(x, SmoothedPowerlawPrimaryMassRatio),
    )

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
    r"""Build distributions for Gaussian components.

    :param N: Number of components
    :param use_spin: whether to include spin
    :param use_tilt: whether to include tilt
    :param use_eccentricity: whether to include eccentricity
    :param use_redshift: whether to include redshift
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :return: list of JointDistribution
    """
    m1_q_dists = create_smoothed_gaussians(
        N=N,
        params=params,
        validate_args=validate_args,
    )

    mass_distributions = jtr.map(
        lambda m1q: [m1q],
        m1_q_dists,
        is_leaf=lambda x: isinstance(x, SmoothedGaussianPrimaryMassRatio),
    )

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

    :param N_pl: Number of power-law components
    :param N_g: Number of Gaussian components
    :param use_spin: whether to include spin, defaults to False
    :param use_tilt: whether to include tilt, defaults to False
    :param use_eccentricity: whether to include eccentricity, defaults to False
    :param use_redshift: whether to include redshift, defaults to False
    :param validate_args: whether to validate arguments, defaults to None
    :return: Mixture of power-law and Gaussian components
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
    log_rates = jnp.stack([params.get(f"log_rate_{i}", 0.0) for i in range(N)], axis=-1)

    return ScaledMixture(
        log_rates,
        component_dists,
        support=constraints.real_vector,
        validate_args=validate_args,
    )
