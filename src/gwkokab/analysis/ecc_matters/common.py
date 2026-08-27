# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared definitions for the eccentricity-matters analysis family.

The simplest of the four families: a single-component population whose masses follow the
:class:`~gwkokab.models.mass.Wysocki2019MassModel` and whose eccentricity is a truncated
normal. There are no component counts and no ``use_*`` flags -- the parameter list is
fixed -- which makes it the natural family to read first when following how the analysis
matrix fits together.

See :
file:`docs/source/examples/ecc_plus_spin/`
for a complete worked run.
"""

from typing import List, Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import TruncatedNormal

from gwkokab.models import Wysocki2019MassModel
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P


def EccentricityMattersModel(
    log_rate: Array,
    alpha_m: Array,
    mmin: Array,
    mmax: Array,
    loc: Array,
    scale: Array,
    low: Array,
    high: Array,
    *,
    validate_args: Optional[bool] = None,
) -> ScaledMixture:
    r"""Build the eccentricity-matters population model.

    A single-component :class:`~gwkokab.models.utils.ScaledMixture` whose one component
    is the product of a :class:`~gwkokab.models.mass.Wysocki2019MassModel` over the two
    masses -- a truncated power law on :math:`m_1`, with :math:`m_2` uniform between
    :math:`m_{\text{min}}` and :math:`m_1` -- and a truncated normal over eccentricity.

    Parameters
    ----------
    log_rate : Array
        Log merger rate of the population, in natural logarithm.
    alpha_m : Array
        Power law index of the primary mass.
    mmin : Array
        Minimum mass.
    mmax : Array
        Maximum mass.
    loc : Array
        Location of the eccentricity distribution.
    scale : Array
        Scale of the eccentricity distribution.
    low : Array
        Lower truncation of the eccentricity distribution.
    high : Array
        Upper truncation of the eccentricity distribution.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    ScaledMixture
        The population model, over :math:`(m_1, m_2, \varepsilon)`.
    """
    comp_dist = JointDistribution(
        Wysocki2019MassModel(
            alpha_m=alpha_m, mmin=mmin, mmax=mmax, validate_args=validate_args
        ),
        TruncatedNormal(
            loc=loc, scale=scale, low=low, high=high, validate_args=validate_args
        ),
    )
    return ScaledMixture(
        log_scales=jnp.array([log_rate]),
        component_distributions=[comp_dist],
        support=comp_dist.support,
        validate_args=validate_args,
    )


class EccentricityMattersCore:
    """Family half of the eccentricity-matters analysis.

    Mixed with a data-representation base -- :class:`~gwkokab.analysis.core.discrete_base.DiscreteBase`
    or :class:`~gwkokab.analysis.core.analytical_gwalk_base.AnalyticalGWalkBase` -- and a
    sampler mixin, to form a complete analysis.
    """

    model_fn = EccentricityMattersModel
    """The population model factory, :func:`EccentricityMattersModel`."""

    @property
    def parameters(self) -> Tuple[str, ...]:
        """The event coordinates this analysis reads from the data.

        Returns
        -------
        Tuple[str, ...]
            The two source-frame component masses and the eccentricity, in that order.
        """
        return (P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.ECCENTRICITY)

    @property
    def model_parameters(self) -> List[str]:
        """The flat list of population hyper-parameters to be inferred.

        Fixed, since this family has no optional physical parameters. These names are what
        the regex keys of ``prior_cfg.json`` are matched against.

        Returns
        -------
        List[str]
            Hyper-parameter names.
        """
        return ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"]
