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


__all__ = ["ModelRegistry"]


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
