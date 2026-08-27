# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Spin models for compact binary populations.

Covers both spin parameterisations in use: the effective/precessing pair
(:math:`\chi_{\text{eff}}, \chi_p`), modelled by
:func:`~gwkokab.models.spin.GaussianSpinModel` and
:class:`~gwkokab.models.spin.GWTC4EffectiveSpinSkewNormalModel`, and the component
magnitudes and tilts, modelled by :func:`~gwkokab.models.spin.BetaFromMeanVar` and
:func:`~gwkokab.models.spin.GenericTiltModel`.
"""

from ._models import (
    BetaFromMeanVar as BetaFromMeanVar,
    GaussianSpinModel as GaussianSpinModel,
    GenericTiltModel as GenericTiltModel,
    GWTC4EffectiveSpinSkewNormalModel as GWTC4EffectiveSpinSkewNormalModel,
)
