# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""The eccentricity-matters analysis family.

A single-component population whose masses follow the
:class:`~gwkokab.models.mass.Wysocki2019MassModel` and whose eccentricity is a truncated
normal. There are no component counts and no ``use_*`` flags, which makes this the
simplest of the four families and the natural one to read first when following how the
analysis matrix fits together.

Console scripts
---------------

- ``synthetic_events_ecc_matter`` -- draw a population from the model.
- ``discrete_ecc_matters`` -- infer the population from per-event posterior samples.
- ``analytical_gwalk_ecc_matters`` -- infer it from per-event Gaussian summaries.

The sampler backend is chosen at runtime by ``sampler_cfg.json``, not by the entry point.

See :file:`docs/source/examples/ecc_plus_spin/` for a complete worked run.
"""
