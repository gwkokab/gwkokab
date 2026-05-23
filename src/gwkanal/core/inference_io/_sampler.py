# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import ast
import warnings
from typing import Annotated, Literal

import jax
import numpy as np
from loguru import logger
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PositiveFloat,
    PositiveInt,
)

from gwkanal.utils.common import read_json
from gwkokab.utils.exceptions import LoggedUserWarning


def _nd_array_before_validator(arr: str | list | np.ndarray) -> np.ndarray:
    """Parses a string or list input into a NumPy array.

    Parameters
    ----------
    arr : str | list | np.ndarray
        The input to be parsed, which can be a JSON string representation of a list,
        a Python list, or already a NumPy array.

    Returns
    -------
    np.ndarray
        The parsed NumPy array.
    """
    arr_converted = arr
    if isinstance(arr, str):
        lst = ast.literal_eval(arr)
        arr_converted = np.array(lst)
    if isinstance(arr, list):
        arr_converted = np.array(arr)
    return arr_converted  # type: ignore


def _nd_array_serializer(arr: np.ndarray) -> list:
    """Serializes a NumPy array to a list for JSON compatibility.

    Parameters
    ----------
    arr : np.ndarray
        The NumPy array to serialize.

    Returns
    -------
    list
        The serialized list representation of the NumPy array.
    """
    return arr.tolist()


NumPyArrayTypeForPydantic = Annotated[
    np.ndarray,
    BeforeValidator(_nd_array_before_validator),
    PlainSerializer(_nd_array_serializer, return_type=list),
]
"""Type annotation for NumPy arrays that supports parsing from JSON strings or lists,
and serializing back to JSON-compatible lists.

A Solution to handle numpy arrays in Pydantic, proposed at:
https://www.flowphysics.com/2024/02/12/numpy-arrays-in-pydantic.html
"""


class NumpyroNUTSSamplerConfig(BaseModel):
    """Configuration for the Numpyro NUTS."""

    model_config = ConfigDict(
        # raise error whenever an extra field is passed
        # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    step_size: PositiveFloat = Field(default=1.0)
    """Determines the size of a single step taken by the verlet integrator while
    computing the trajectory using Hamiltonian dynamics.

    If not specified, it will be set to 1.
    """

    inverse_mass_matrix: None | NumPyArrayTypeForPydantic | dict = Field(default=None)
    r"""Initial value for inverse mass matrix.

    This may be adapted during warmup if `adapt_mass_matrix = True`. If no value is
    specified, then it is initialized to the identity matrix. For a `potential_fn`
    with general JAX pytree parameters, the order of entries of the mass matrix is
    the order of the flattened version of pytree parameters obtained with
    :func:`~jax.tree_flatten`, which is a bit ambiguous (see more at
    https://jax.readthedocs.io/en/latest/pytrees.html). If model is not None, here
    we can specify a structured block mass matrix as a dictionary, where keys are
    tuple of site names and values are the corresponding block of the mass matrix.
    For more information about structured mass matrix, see dense_mass argument.
    """

    adapt_step_size: bool = Field(default=True)
    """A flag to decide if we want to adapt step_size during warm-up phase using Dual
    Averaging scheme.
    """

    adapt_mass_matrix: bool = Field(default=True)
    """A flag to decide if we want to adapt mass matrix during warm-up phase using
    Welford scheme.
    """

    dense_mass: bool | list[tuple[str, ...]] = Field(default=False)
    """This flag controls whether mass matrix is dense (i.e. full-rank) or diagonal
    (defaults to dense_mass=False).

    To specify a structured mass matrix, users can provide a list of tuples of site
    names. Each tuple represents a block in the joint mass matrix. For example,
    assuming that the model has latent variables "x", "y", "z" (where each variable
    can be multi-dimensional), possible specifications and corresponding mass matrix
    structures are as follows:

    - `dense_mass=[("x", "y")]`: use a dense mass matrix for the joint (x, y) and a
        diagonal mass matrix for z
    - `dense_mass=[]` (equivalent to `dense_mass=False`): use a diagonal mass matrix
        for the joint (x, y, z)
    - `dense_mass=[("x", "y", "z")]` (equivalent to `full_mass=True`): use a dense
        mass matrix for the joint (x, y, z)
    - `dense_mass=[("x",), ("y",), ("z")]`: use dense mass matrices for each of x, y,
        and z (i.e. block-diagonal with 3 blocks)
    """

    target_accept_prob: Annotated[float, Field(gt=0.0, le=1.0)] = Field(default=0.8)
    """Target acceptance probability for step size adaptation using Dual Averaging.

    Increasing this value will lead to a smaller step size, hence the sampling will be
    slower but more robust. Defaults to 0.8.
    """

    max_tree_depth: PositiveInt | tuple[PositiveInt, PositiveInt] = Field(default=8)
    """Max depth of the binary tree created during the doubling scheme of NUTS sampler.

    Defaults to 8. This argument also accepts a tuple of integers (d1, d2), where d1 is
    the max tree depth during warmup phase and d2 is the max tree depth during post
    warmup phase.
    """

    find_heuristic_step_size: bool = Field(default=False)
    """Whether or not to use a heuristic function to adjust the step size at the
    beginning of each adaptation window.

    Defaults to False.
    """

    forward_mode_differentiation: bool = Field(default=False)
    """Whether to use forward-mode differentiation or reverse-mode differentiation.

    By default, we use reverse mode but the forward mode can be useful in some cases to
    improve the performance. In addition, some control flow utility on JAX such as
    jax.lax.while_loop or jax.lax.fori_loop only supports forward-mode differentiation.
    """


class NumpyroMCMCConfig(BaseModel):
    """Configuration for the Numpyro MCMC."""

    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    model_config = ConfigDict(extra="forbid")

    num_warmup: PositiveInt = Field(default=1000)
    """Number of warmup steps."""

    num_samples: PositiveInt = Field(default=2000)
    """Number of samples to generate from the Markov chain."""

    thinning: PositiveInt = Field(default=1)
    """Positive integer that controls the fraction of post-warmup samples that are
    retained.

    For example if thinning is 2 then every other sample is retained. Defaults to 1,
    i.e. no thinning.
    """

    num_chains: PositiveInt = Field(default=1)
    """Number of MCMC chains to run.

    By default, chains will be run in parallel using :func:`~jax.pmap()`. If there are
    not enough devices available, chains will be run in sequence.
    """

    chain_method: Literal["parallel", "sequential", "vectorized"] = Field(
        default="parallel"
    )
    """A callable jax transform like :func:`~jax.vmap` or one of `"parallel"` (default),
    `"sequential"` `"vectorized"`.

    The method `"parallel"` is used to execute the drawing process in parallel on XLA
    devices (CPUs/GPUs/TPUs), If there are not enough devices for `"parallel"`, we fall
    back to `"sequential"` method to draw chains sequentially. `"vectorized"` method is
    an experimental feature which vectorizes the drawing method, hence allowing us to
    collect samples in parallel on a single device.
    """

    progress_bar: bool = Field(default=True)
    """Whether to enable progress bar updates.

    Defaults to `True`.
    """

    progress_rate: PositiveInt | None = Field(default=None)
    """Number of iterations per progress bar update.

    Defaults to `None`, which is 5% of total iterations when there are more than 20
    iterations, otherwise every iteration.
    """

    jit_model_args: bool = Field(default=True)
    """If set to `True`, this will compile the potential energy computation as a
    function of model arguments.

    As such, calling :func:`~numpyro.infer.MCMC.run` again on a same sized but different
    dataset will not result in additional compilation cost. Note that currently, this
    does not take effect for the case `num_chains > 1` and `chain_method == 'parallel'`.
    """


class NumpyroGlobalConfig(BaseModel):
    """Configuration for the Numpyro sampler, including both kernel and MCMC
    settings.
    """

    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    model_config = ConfigDict(extra="forbid")

    sampler_name: Literal["numpyro"] = "numpyro"
    """The name of the sampler to use.

    Currently only 'numpyro' is supported.
    """

    kernel: NumpyroNUTSSamplerConfig = Field(default_factory=NumpyroNUTSSamplerConfig)
    """Configuration for the NUTS sampler kernel."""

    mcmc: NumpyroMCMCConfig = Field(default_factory=NumpyroMCMCConfig)
    """Configuration for the MCMC sampling procedure."""

    @classmethod
    def from_json(cls, config_path: str) -> "NumpyroGlobalConfig":
        """Initializes the loader from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON file containing loader settings.

        Returns
        -------
        NumpyroGlobalConfig
            An instance of NumpyroGlobalConfig.
        """
        sampler_cfg = read_json(config_path)
        instance = cls.model_validate(sampler_cfg)
        n_devices = jax.device_count()
        if instance.mcmc.chain_method != "parallel" and n_devices > 1:
            warnings.warn(
                f"Multiple devices detected ({n_devices}), but chain_method is set to "
                f"'{instance.mcmc.chain_method}'. Overriding to 'parallel'.",
                LoggedUserWarning,
            )
            instance.mcmc.chain_method = "parallel"
        else:
            logger.info(
                f"Using chain method: '{instance.mcmc.chain_method}' with {n_devices} device(s)."
            )
        return instance


class FlowMCGlobalConfig(BaseModel):
    """Configuration for the FlowMC sampler."""

    model_config = ConfigDict(
        # raise error whenever an extra field is passed
        # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    sampler_name: Literal["flowMC"] = "flowMC"
    """The name of the sampler to use.

    Currently only 'flowMC' is supported.
    """

    chain_batch_size: Annotated[int, Field(ge=0)] = Field(default=0)
    """Batch size for processing chains.

    If 0, processes all chains simultaneously.
    """

    n_chains: PositiveInt = Field(default=10)
    """Number of chains to sample."""

    batch_size: PositiveInt = Field(default=1000)
    """Number of samples per training batch for the Normalizing Flow."""

    history_window: PositiveInt = Field(default=500)
    """Size of the rolling history window used for training data or adaptation."""

    n_epochs: PositiveInt = Field(default=4)
    """Number of training epochs for the Normalizing Flow per training loop."""

    n_max_examples: PositiveInt = Field(default=100_000)
    """Maximum number of total samples/examples to store in the training history."""

    n_NFproposal_batch_size: PositiveInt = Field(default=10)
    """Batch size used when generating proposal steps from the Normalizing Flow."""

    global_thinning: PositiveInt = Field(default=1)
    """Thinning factor applied to global (Normalizing Flow) proposals."""

    local_thinning: PositiveInt = Field(default=1)
    """Thinning factor applied to local sampler steps."""

    n_global_steps: PositiveInt = Field(default=20)
    """Number of global production/exploration steps to take using the NF proposal."""

    n_local_steps: PositiveInt = Field(default=300)
    """Number of local sampler steps to take between global updates."""

    n_production_loops: PositiveInt = Field(default=40)
    """Number of production loops to run after the model is trained."""

    n_training_loops: PositiveInt = Field(default=15)
    """Number of initial loops dedicated to tuning and training the Normalizing Flow."""

    local_sampler_name: Literal["mala", "hmc"] = Field(default="hmc")
    """The underlying local MCMC sampler to use ('mala' for MALA or 'hmc' for HMC)."""

    step_size: PositiveFloat = Field(default=1e-2)
    """The initial step size (or integration step size) for the local sampler."""

    n_leapfrog: PositiveInt = Field(default=10)
    """Number of leapfrog steps per HMC trajectory (ignored if using MALA)."""

    mass_matrix: PositiveFloat | NumPyArrayTypeForPydantic = Field(default=1.0)
    """Mass matrix diagonal elements or scalar value for HMC trajectory dynamics."""

    learning_rate: PositiveFloat = Field(default=1e-3)
    """Learning rate for the Normalizing Flow optimizer."""

    rq_spline_hidden_units: list[PositiveInt] = Field(default=[128, 128])
    """Layer widths of the neural network conditioning the Rational-Quadratic
    Splines.
    """

    rq_spline_n_bins: PositiveInt = Field(default=8)
    """Number of bins used in each Rational-Quadratic Spline transformation layer."""

    rq_spline_n_layers: PositiveInt = Field(default=10)
    """Total number of flow layers (coupling blocks) in the Normalizing Flow."""

    rq_spline_range: tuple[float, float] = Field(default=(-10.0, 10.0))
    """The bounding box interval (min, max) where the spline transformations are
    active.
    """

    verbose: bool = Field(default=False)
    """If True, prints execution progress logs and loss metrics to the console."""

    @classmethod
    def from_json(cls, config_path: str) -> "FlowMCGlobalConfig":
        """Initializes the loader from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON file containing loader settings.

        Returns
        -------
        FlowMCGlobalConfig
            An instance of FlowMCGlobalConfig.
        """
        sampler_cfg = read_json(config_path)
        return cls(**sampler_cfg)


class SamplerConfig(BaseModel):
    """Configuration for the sampler, which can be either Numpyro or flowMC."""

    # raise error whenever an extra field is passed
    # https://pydantic.dev/docs/validation/latest/concepts/models/#extra-data
    model_config = ConfigDict(extra="forbid")

    loader: NumpyroGlobalConfig | FlowMCGlobalConfig = Field(
        discriminator="sampler_name"
    )

    @classmethod
    def from_json(cls, config_path: str) -> "SamplerConfig":
        """Initializes the loader from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON file containing loader settings.

        Returns
        -------
        SamplerConfig
            An instance of SamplerConfig.
        """
        sampler_cfg = read_json(config_path)
        return cls(loader=sampler_cfg)  # type: ignore


def _dump_numpyro_cfg() -> None:
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(
        description="Creates a template config for NumPyro NUTS sampler and MCMC",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="numpyro_cfg_template.json",
        help="Output JSON filename",
    )
    args = parser.parse_args()

    from pathlib import Path

    cfg = NumpyroGlobalConfig()

    Path(args.output).write_text(cfg.model_dump_json(indent=4))

    print(f"Successfully generated NumPyro configuration at: {args.output}")
    print("\nResources for these parameters:")
    print(" - MCMC: https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mcmc.MCMC")
    print(" - NUTS: https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS")


def _dump_flowMC_cfg() -> None:
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(
        description="Creates a template config for flowMC",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="flowMC_cfg_template.json",
        help="Output JSON filename",
    )
    args = parser.parse_args()

    cfg = FlowMCGlobalConfig()

    from pathlib import Path

    Path(args.output).write_text(cfg.model_dump_json(indent=4))

    print(f"Successfully generated flowMC configuration at: {args.output}")
