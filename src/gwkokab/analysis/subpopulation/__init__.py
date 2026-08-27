# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""The sub-population analysis family.

A mixture over three mass families -- a truncated power law (``spl``), a broken power law
(``bpl``) and a truncated normal (``gpl``) -- mixed at the level of the *primary mass*
rather than of the whole component. The primary mass models are combined into a single
mixture with branching fractions ``lambda_<i>``, tapered by a shared smoothing window and
normalised, and one overall ``log_rate`` is then split across the components in
proportion to their weights.

So where :mod:`~gwkokab.analysis.multisource` has one rate per component, this family has
one rate for the population and a set of branching fractions -- the natural
parameterisation when the components are sub-populations of a single channel rather than
separate channels.

Console scripts
---------------

- ``discrete_subpopulation`` -- infer the population from per-event posterior samples.
- ``analytical_gwalk_subpopulation`` -- infer it from per-event Gaussian summaries.

The sampler backend is chosen at runtime by ``sampler_cfg.json``, not by the entry point.

See Also
--------
gwkokab.models.hybrids.SubPopulationModel :
    The population model this family fits.
gwkokab.analysis.multisource :
    The same component families, but mixed at the level of the whole component.
"""
