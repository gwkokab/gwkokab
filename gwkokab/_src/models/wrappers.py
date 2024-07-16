#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from numpyro.distributions import Distribution, Unit

from .utils import JointDistribution


__all__ = ["gwk_magazine", "add_log_factor"]


def add_log_factor(name, model=None):
    r"""Adds log factor to logarithm of the probability of the model,
    equivalent to multiplying the model by the factor.

    :param name: Name of the parameter to be added as a log factor.
    :param model: Model to be added the log factor to.

    .. doctest::

        >>> from functools import partial
        >>> from jax import numpy as jnp, random as jrd
        >>> from numpyro.distributions import Normal

        >>> model_by_function = add_log_factor("log_rate", Normal)

        >>> @partial(add_log_factor, "log_rate")
        >>> def model_by_decorator(loc, scale):
        ...     return Normal(loc, scale, validate_args=True)

        >>> xx = jrd.uniform(jrd.PRNGKey(0), (100,))
        >>> model1 = model_by_function(log_rate=2.0, loc=10.0, scale=2.0, validate_args=True)
        >>> model2 = model_by_decorator(log_rate=2.0, loc=10.0, scale=2.0)
        >>> assert jnp.equal(model1.log_prob(xx), model2.log_prob(xx))

    """

    if model is None:
        raise ValueError("Model must be provided.")

    def rate_times_model(*args, **kwargs) -> Distribution:
        log_rate = kwargs.pop(name)
        log_rate_factor_dist = Unit(log_rate, validate_args=True)
        return JointDistribution(log_rate_factor_dist, model(*args, **kwargs))

    return rate_times_model


class ModelRegistry(object):
    r"""Decorator class to register models and their parameters"""

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


gwk_magazine = ModelRegistry()
