# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


class ModelRegistry(object):
    r"""Decorator class to register models and their parameters."""

    def __init__(self) -> None:
        self._registry = {}

    @property
    def registry(self) -> dict:
        r"""Hashmap of registered models and their parameters."""
        return self._registry

    def register(self, parameter, model=None):
        r"""Registers a model with the parameter(s) it yields."""
        if model is None:
            return lambda model: self.register(parameter, model)
        if isinstance(parameter, str):
            parameter = (parameter,)
        elif isinstance(parameter, tuple):
            assert all(isinstance(p, str) for p in parameter)
        else:
            raise ValueError("Parameter must be a string or tuple of strings")

        self._registry[parameter] = model
        return model

    def __call__(self, parameter):
        try:
            model = self._registry[parameter]
        except KeyError as e:
            raise NotImplementedError from e

        return model(parameter)
