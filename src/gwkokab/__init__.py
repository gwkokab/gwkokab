# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import importlib.metadata


__version__ = importlib.metadata.version("gwkokab")


from gwkokab.internal import lazy_loader


constants = lazy_loader.LazyLoader("constants", globals(), "gwkokab.constants")
cosmology = lazy_loader.LazyLoader("cosmology", globals(), "gwkokab.cosmology")
errors = lazy_loader.LazyLoader("errors", globals(), "gwkokab.errors")
inference = lazy_loader.LazyLoader("inference", globals(), "gwkokab.inference")
logger = lazy_loader.LazyLoader("logger", globals(), "gwkokab.logger")
models = lazy_loader.LazyLoader("models", globals(), "gwkokab.models")
parameters = lazy_loader.LazyLoader("parameters", globals(), "gwkokab.parameters")
poisson_mean = lazy_loader.LazyLoader("poisson_mean", globals(), "gwkokab.poisson_mean")
population = lazy_loader.LazyLoader("population", globals(), "gwkokab.population")
utils = lazy_loader.LazyLoader("utils", globals(), "gwkokab.utils")
vts = lazy_loader.LazyLoader("vts", globals(), "gwkokab.vts")
