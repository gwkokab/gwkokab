# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""The :math:`N` power-law plus :math:`M` Gaussian analysis family.

The canonical family of the package: a mixture whose first :math:`N` components have
power-law masses (component tag ``pl``) and whose next :math:`M` have Gaussian masses
(tag ``g``). Each component may additionally carry spin, tilt, eccentricity, redshift
and extrinsic marginals, switched on by the ``--add-*`` command line flags. Every
component has its own log rate, ``log_rate_<index>``, indexed across the whole mixture.

Console scripts
---------------

- ``synthetic_events_n_pls_m_gs`` -- draw a population from the model.
- ``discrete_n_pls_m_gs`` -- infer the population from per-event posterior samples.
- ``analytical_gwalk_n_pls_m_gs`` -- infer it from per-event Gaussian summaries.

The sampler backend is chosen at runtime by ``sampler_cfg.json``, not by the entry point.

See Also
--------
gwkokab.models.hybrids.NPowerlawMGaussian : The population model this family fits.
"""
