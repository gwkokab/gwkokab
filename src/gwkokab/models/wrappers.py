# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry mapping parameter names to the models that produce them.

:class:`ModelRegistry` is a small decorator-based lookup: a model registers itself
against the parameter name (or tuple of names) it is a distribution over, and calling
the registry with that name builds the model. This lets a family's ``--add-*`` CLI flags
select models by parameter name without a hard-coded dispatch table.
"""

__all__ = ["ModelRegistry"]


class ModelRegistry(object):
    """Decorator class to register models and their parameters.

    A model is registered against the parameter name, or tuple of names, it is a
    distribution over; calling the registry with that key constructs the model.

    Examples
    --------
    .. code-block:: python

        registry = ModelRegistry()


        @registry.register("redshift")
        def redshift_model(parameter): ...


        model = registry("redshift")
    """

    def __init__(self) -> None:
        self._registry = {}

    @property
    def registry(self) -> dict:
        """Hashmap of registered models and their parameters.

        Returns
        -------
        dict
            Maps each registered tuple of parameter names to its model factory.
        """
        return self._registry

    def register(self, parameter, model=None):
        """Register a model with the parameter(s) it yields.

        Usable directly or as a decorator: omitting ``model`` returns a decorator that
        registers the function it is applied to.

        Parameters
        ----------
        parameter : str | tuple[str, ...]
            The parameter name, or tuple of names, the model is a distribution over. A bare
            string is promoted to a one-element tuple.
        model : Callable, optional
            The model factory. Defaults to :data:`None`, which returns a decorator.

        Returns
        -------
        Callable
            The registered model, or a decorator when ``model`` is :data:`None`.

        Raises
        ------
        ValueError
            If ``parameter`` is neither a string nor a tuple of strings.
        """
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
        """Build the model registered for a parameter key.

        Parameters
        ----------
        parameter : tuple[str, ...]
            The key the model was registered under. It is also passed to the factory, so a
            model registered against several names can tell which it is building.

        Returns
        -------
        Any
            The constructed model.

        Raises
        ------
        NotImplementedError
            If no model is registered for ``parameter``.
        """
        try:
            model = self._registry[parameter]
        except KeyError as e:
            raise NotImplementedError from e

        return model(parameter)
