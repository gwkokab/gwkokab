# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Selection-function configuration: ``pmean_cfg.json``.

The ``estimator_type`` field is a Pydantic discriminator that picks one of four loaders,
each of which validates the knobs its estimator accepts and then calls into
:mod:`gwkokab.poisson_mean`. The ``custom`` loader is the escape hatch: it imports a
Python file by path and calls its ``custom_poisson_mean_estimator(key, parameters,
filename, **kwargs)``, so an estimator can be supplied without modifying the package.
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from jaxtyping import Array, PRNGKeyArray
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from gwkokab.analysis.utils.common import read_json
from gwkokab.poisson_mean import (
    poisson_mean_from_neural_pdet,
    poisson_mean_from_neural_vt,
    poisson_mean_from_sensitivity_injections,
)
from gwkokab.utils.exceptions import LoggedImportError, LoggedValueError


class BaseLoader(BaseModel):
    """Base logic shared across all loaders.

    Holds the three things every estimator needs, and rejects unrecognised fields so a
    typo in the configuration file is an error rather than a silent no-op.
    """

    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    model_config = ConfigDict(extra="forbid")

    filename: str
    """Path to the file the estimator reads -- a trained network, or an injection
    set.
    """

    key: Any
    """JAX random key, injected by :meth:`PoissonMeanEstimationLoader.read_from_json`
    rather than read from the configuration file.
    """

    parameters: Tuple[str, ...]
    """Names of the event coordinates, injected the same way."""

    def get_estimators(self):
        """Build the Poisson mean estimator this loader describes.

        Returns
        -------
        tuple
            The triple described in :mod:`gwkokab.poisson_mean`.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class NeuralVolumeTimeSensitivityPoissonMeanLoader(BaseLoader):
    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    """Configuration for ``estimator_type: "neural_vt"`` -- a neural sensitive volume-
    time regressor.

    Validates the knobs of :func:`~gwkokab.poisson_mean.poisson_mean_from_neural_vt` and calls it.
    """

    model_config = ConfigDict(extra="forbid")

    estimator_type: Literal["neural_vt"]
    """The discriminator selecting this loader."""

    batch_size: Optional[PositiveInt] = None
    """Chunk size for evaluating the network.

    Defaults to :data:`None`.
    """

    num_samples: PositiveInt = 1_000
    """Monte Carlo samples drawn per mixture component.

    Defaults to ``1000``.
    """

    time_scale: PositiveFloat = 1.0
    """Observing time.

    Defaults to ``1.0``.
    """

    def get_estimators(self):
        """Build the estimator by calling
        :func:`~gwkokab.poisson_mean.poisson_mean_from_neural_vt`.

        Returns
        -------
        tuple
            The triple described in :mod:`gwkokab.poisson_mean`.
        """
        return poisson_mean_from_neural_vt(
            key=self.key,
            parameters=self.parameters,
            filename=self.filename,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            time_scale=self.time_scale,
        )


class NeuralVolumeProbabilityOfDetectionPoissonMeanLoader(BaseLoader):
    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    """Configuration for ``estimator_type: "neural_pdet"`` -- a neural detection-
    probability regressor.

    Validates the knobs of :func:`~gwkokab.poisson_mean.poisson_mean_from_neural_pdet` and calls it.
    """

    model_config = ConfigDict(extra="forbid")

    estimator_type: Literal["neural_pdet"]
    """The discriminator selecting this loader."""

    batch_size: Optional[PositiveInt] = None
    """Chunk size for evaluating the network.

    Defaults to :data:`None`.
    """

    num_samples: PositiveInt = 1_000
    """Monte Carlo samples drawn per mixture component.

    Defaults to ``1000``.
    """

    time_scale: PositiveFloat = 1.0
    """Observing time.

    Defaults to ``1.0``.
    """

    def get_estimators(self):
        """Build the estimator by calling
        :func:`~gwkokab.poisson_mean.poisson_mean_from_neural_pdet`.

        Returns
        -------
        tuple
            The triple described in :mod:`gwkokab.poisson_mean`.
        """
        return poisson_mean_from_neural_pdet(
            key=self.key,
            parameters=self.parameters,
            filename=self.filename,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            time_scale=self.time_scale,
        )


class GWTCInjectionLoader(BaseLoader):
    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    """Configuration for ``estimator_type: "injection"`` -- reweighted sensitivity
    injections.

    Validates the knobs of :func:`~gwkokab.poisson_mean.poisson_mean_from_sensitivity_injections` and calls it.
    """

    model_config = ConfigDict(extra="forbid")

    estimator_type: Literal["injection"]
    """The discriminator selecting this loader."""

    batch_size: Optional[PositiveInt] = None
    """Chunk size for evaluating the population density over the injections.

    Defaults to :data:`None`.
    """

    far_cut: PositiveFloat = 1.0
    """False alarm rate threshold, in inverse years.

    Defaults to ``1.0``.
    """

    snr_cut: PositiveFloat = 10.0
    """SNR threshold.

    Defaults to ``10.0``.
    """

    def get_estimators(self):
        """Build the estimator by calling
        :func:`~gwkokab.poisson_mean.poisson_mean_from_sensitivity_injections`.

        Returns
        -------
        tuple
            The triple described in :mod:`gwkokab.poisson_mean`.
        """
        return poisson_mean_from_sensitivity_injections(
            key=self.key,
            parameters=self.parameters,
            filename=self.filename,
            batch_size=self.batch_size,
            far_cut=self.far_cut,
            snr_cut=self.snr_cut,
        )


class CustomPoissonMeanEstimationLoader(BaseLoader):
    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    """Configuration for ``estimator_type: "custom"`` -- an estimator supplied by path.

    The escape hatch for a selection function the package does not implement.
    """

    model_config = ConfigDict(extra="forbid")

    estimator_type: Literal["custom"]
    """The discriminator selecting this loader."""

    python_module_path: str
    """Path to a Python file defining ``custom_poisson_mean_estimator(key, parameters,
    filename, **kwargs)``.
    """

    kwargs: Dict[str, Any]
    """Extra keyword arguments forwarded to that function."""

    def get_estimators(self):
        """Import the module by path and call its estimator factory.

        Returns
        -------
        tuple
            The triple described in :mod:`gwkokab.poisson_mean`, as the custom function
            returns it.

        Raises
        ------
        LoggedImportError
            If the file at ``python_module_path`` cannot be loaded as a module.
        LoggedValueError
            If the module has no ``custom_poisson_mean_estimator`` attribute.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.python_module_path
        )
        if spec is None or spec.loader is None:
            raise LoggedImportError(
                f"Could not load spec for module at {self.python_module_path}"
            )

        custom_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(custom_module)  # type: ignore

        if not hasattr(custom_module, "custom_poisson_mean_estimator"):
            raise LoggedValueError(
                "The custom module must have a 'custom_poisson_mean_estimator' function."
            )
        return custom_module.custom_poisson_mean_estimator(
            self.key,
            self.parameters,
            self.filename,
            **self.kwargs,
        )


class PoissonMeanEstimationLoader(BaseModel):
    """The parsed ``pmean_cfg.json``.

    A thin wrapper whose single ``loader`` field is a discriminated union: the
    ``estimator_type`` in the JSON selects which of the four loaders validates the rest
    of the file.
    """

    loader: Union[
        NeuralVolumeTimeSensitivityPoissonMeanLoader,
        NeuralVolumeProbabilityOfDetectionPoissonMeanLoader,
        GWTCInjectionLoader,
        CustomPoissonMeanEstimationLoader,
    ] = Field(discriminator="estimator_type")
    """The selected loader, discriminated on ``estimator_type``."""

    @classmethod
    def read_from_json(
        cls, config_path: str, key: PRNGKeyArray, parameters: Tuple[str, ...]
    ):
        """Parse a ``pmean_cfg.json`` and inject the run-time values.

        The random key and the parameter names are not part of the configuration file --
        they come from the running analysis -- so they are merged in here before validation.

        Parameters
        ----------
        config_path : str
            Path to the JSON configuration file.
        key : PRNGKeyArray
            JAX random key for the estimator.
        parameters : Tuple[str, ...]
            Names of the event coordinates, in the order the population model produces them.

        Returns
        -------
        PoissonMeanEstimationLoader
            The validated configuration.

        Raises
        ------
        ValidationError
            If the file has an unrecognised ``estimator_type``, a missing required field, or
            an extra one.
        """
        raw_data = read_json(config_path)

        payload = {"loader": {**raw_data, "key": key, "parameters": parameters}}

        return cls(**payload)

    def get_estimators(
        self,
    ) -> Tuple[
        Optional[Callable[[Array], Array]],
        Callable[..., Array],
        dict[str, Any],
    ]:
        """Build the Poisson mean estimator the configuration describes.

        Returns
        -------
        Tuple[Optional[Callable[[Array], Array]], Callable[..., Array], dict[str, Any]]
            The log sensitivity function (or :data:`None`), the estimator, and the extra
            arguments to splat into it. See :mod:`gwkokab.poisson_mean`.
        """
        return self.loader.get_estimators()
