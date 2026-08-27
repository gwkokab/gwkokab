# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""The multi-source analysis family.

A mixture over four kinds of mass component, meant to describe a population drawn from
several astrophysical formation channels at once:

- ``spl`` -- smoothed power law primary mass with a conditional mass ratio power law;
- ``bpl`` -- smoothed *broken* power law, same conditional mass ratio;
- ``gpl`` -- Gaussian primary mass, same conditional mass ratio;
- ``gg`` -- independent truncated normals on :math:`m_1` and :math:`m_2`.

Each component carries its own log rate, ``log_rate_<index>``, indexed across the whole
mixture -- one rate per channel.

Console scripts
---------------

- ``discrete_multisource`` -- infer the population from per-event posterior samples.
- ``analytical_gwalk_multisource`` -- infer it from per-event Gaussian summaries.

The sampler backend is chosen at runtime by ``sampler_cfg.json``, not by the entry point.

See Also
--------
gwkokab.models.hybrids.MultiSourceModel :
    The population model this family fits.
gwkokab.analysis.subpopulation :
    The same component families, but mixed at the level of the primary mass.
"""
