# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Dict, Optional, Union

import equinox as eqx
from jaxtyping import Array


class VolumeTimeSensitivityInterface(eqx.Module):
    """Interface for volume time sensitivity."""

    shuffle_indices: Optional[Sequence[int]] = eqx.field(static=True, default=None)
    """The indices to shuffle the input to the model."""
    batch_size: Optional[int] = eqx.field(static=True, default=None)
    """The batch size used by :func:`jax.lax.map` in mapped functions."""
    parameter_ranges: Dict[str, Union[int, float]] = eqx.field(default=None)

    @abstractmethod
    def get_logVT(self) -> Callable[[Array], Array]:
        """Gets the log volume-time sensitivity function.

        Returns
        -------
        Callable[[Array], Array]
            A function that takes an input array of shape (n_features,) and
            returns the log volume-time sensitivity as an array of shape ().
        """
        raise NotImplementedError

    @abstractmethod
    def get_mapped_logVT(self) -> Callable[[Array], Array]:
        """Gets a mapped log volume-time sensitivity function for batch processing.

        Returns
        -------
        Callable[[Array], Array]
            A function that takes a stack of inputs as an array of shape
            (n_example, n_features) and returns an array of log volume-time
            sensitivities with shape (n_example,).
        """
        raise NotImplementedError
