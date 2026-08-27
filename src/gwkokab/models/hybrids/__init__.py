# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Hybrid population models assembled from per-parameter building blocks.

Each family here is a :class:`~gwkokab.models.utils.ScaledMixture` whose components are
:class:`~gwkokab.models.utils.JointDistribution`\ s over the physical parameters that
were switched on. The per-parameter factories live in ``_ncombination``; the family
modules combine them:

- :func:`~gwkokab.models.hybrids.NPowerlawMGaussian` -- :math:`N` power-law and
  :math:`M` Gaussian mass components (component tags ``pl``/``g``).
- :func:`~gwkokab.models.hybrids.MultiSourceModel` -- smoothed, broken and Gaussian
  power-law components plus a Gaussian-Gaussian one (tags ``spl``/``bpl``/``gpl``/``gg``).
- :func:`~gwkokab.models.hybrids.SubPopulationModel` -- smoothed, broken and Gaussian
  power-law sub-populations (tags ``spl``/``bpl``/``gpl``).

Adding a physical parameter to a family means touching a ``create_*`` factory, the
family's ``_build_*_distributions`` wiring, the ``model_parameters``/``parameters``
properties in :mod:`gwkokab.analysis`, and the family's ``--add-*`` CLI flag.
"""

from ._multisource import MultiSourceModel as MultiSourceModel
from ._npowerlawmgaussian import NPowerlawMGaussian as NPowerlawMGaussian
from ._subpopulation import SubPopulationModel as SubPopulationModel
