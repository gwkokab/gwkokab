# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The flowMC sampler backend.

:class:`FlowMCBase` supplies ``driver`` for analyses run with flowMC, which alternates a
gradient-based local sampler with global proposals drawn from a normalizing flow trained
on the chains so far. Chains, acceptance rates, training loss and the thinned posterior
samples are written to the output HDF5 after every loop, so a run can be inspected -- or
salvaged -- while it is still going.

The backend is chosen at runtime from ``sampler_cfg.json``'s ``sampler_name``, not by
the console script.

:class:`Local_Global_Sampler_Bundle` and :class:`Sampler` are vendored from `flowMC
<https://github.com/kazewong/flowMC>`_, with their own copyright notices, so that the
per-loop checkpointing can be woven into the sampling loop. They are marked not to be
modified.

See Also
--------
gwkokab.analysis.core.numpyro_base : The alternative sampler backend.
"""

from typing import Any, Callable, Dict, List, Literal, Optional

import equinox as eqx
import h5py
import jax
import numpy as np
import tqdm
from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.kernel.HMC import HMC
from flowMC.resource.kernel.MALA import MALA
from flowMC.resource.kernel.NF_proposal import NFProposal
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.resource.states import State
from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle
from flowMC.strategy.base import Strategy
from flowMC.strategy.lambda_function import Lambda
from flowMC.strategy.take_steps import TakeGroupSteps, TakeSerialSteps
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.update_state import UpdateState
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from loguru import logger

from gwkokab.analysis.core.analysis_base import analysis_base_arg_parser, AnalysisBase
from gwkokab.analysis.core.inference_io import FlowMCGlobalConfig
from gwkokab.analysis.core.utils import write_to_hdf5
from gwkokab.analysis.utils.literals import (
    CHAIN_GROUP_FORMAT,
    INFERENCE_OUTPUT_FILENAME,
    SAMPLES_GROUP_NAME,
)
from gwkokab.models.utils import JointDistribution


# WARNING: do not change anything in this class


# Copyright (c) 2022 Kaze Wong & contributor
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class Local_Global_Sampler_Bundle(ResourceStrategyBundle):
    """A bundle that uses a Rational Quadratic Spline as a normalizing flow model and
    the Metropolis Adjusted Langevin Algorithm as a local sampler.

    This is the base algorithm described in
    https://www.pnas.org/doi/full/10.1073/pnas.2109420119
    """

    def __init__(
        self,
        rng_key: PRNGKeyArray,
        n_chains: int,
        n_dims: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_local_steps: int,
        n_global_steps: int,
        n_training_loops: int,
        n_production_loops: int,
        n_epochs: int,
        local_sampler_name: Literal["mala", "hmc"] = "mala",
        step_size: float = 1e-1,
        condition_matrix: Array = 1.0,  # type: ignore
        n_leapfrog: int = 10,
        chain_batch_size: int = 0,
        rq_spline_hidden_units: list[int] = [32, 32],
        rq_spline_n_bins: int = 8,
        rq_spline_n_layers: int = 4,
        rq_spline_range: tuple[float, float] = (-10.0, 10.0),
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        history_window: int = 100,
        local_thinning: int = 1,
        global_thinning: int = 1,
        n_NFproposal_batch_size: int = 10000,
        verbose: bool = False,
    ):
        n_local_steps_per_loop = n_local_steps // local_thinning
        n_global_steps_per_loop = n_global_steps // global_thinning
        n_training_steps = (
            n_local_steps_per_loop + n_global_steps_per_loop
        ) * n_training_loops

        n_production_steps = (
            n_local_steps_per_loop + n_global_steps_per_loop
        ) * n_production_loops

        n_total_epochs = n_training_loops * n_epochs

        positions_training = Buffer(
            "positions_training", (n_chains, n_training_steps, n_dims), 1
        )
        log_prob_training = Buffer("log_prob_training", (n_chains, n_training_steps), 1)
        local_accs_training = Buffer(
            "local_accs_training", (n_chains, n_training_steps), 1
        )
        global_accs_training = Buffer(
            "global_accs_training", (n_chains, n_training_steps), 1
        )
        loss_buffer = Buffer("loss_buffer", (n_total_epochs,), 0)

        position_production = Buffer(
            "positions_production", (n_chains, n_production_steps, n_dims), 1
        )
        log_prob_production = Buffer(
            "log_prob_production", (n_chains, n_production_steps), 1
        )
        local_accs_production = Buffer(
            "local_accs_production", (n_chains, n_production_steps), 1
        )
        global_accs_production = Buffer(
            "global_accs_production", (n_chains, n_production_steps), 1
        )

        if local_sampler_name.strip().lower() == "mala":
            local_sampler = MALA(step_size=step_size)
        else:
            local_sampler = HMC(
                condition_matrix=condition_matrix,
                step_size=step_size,
                n_leapfrog=n_leapfrog,
            )
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            n_dims,
            rq_spline_n_layers,
            rq_spline_hidden_units,
            rq_spline_n_bins,
            subkey,
            rq_spline_range,
        )
        global_sampler = NFProposal(
            model, n_NFproposal_batch_size=n_NFproposal_batch_size
        )
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf = LogPDF(logpdf, n_dims=n_dims)

        sampler_state = State(
            {
                "target_positions": "positions_training",
                "target_log_prob": "log_prob_training",
                "target_local_accs": "local_accs_training",
                "target_global_accs": "global_accs_training",
                "training": True,
            },
            name="sampler_state",
        )

        self.resources = {
            "logpdf": logpdf,
            "positions_training": positions_training,
            "log_prob_training": log_prob_training,
            "local_accs_training": local_accs_training,
            "global_accs_training": global_accs_training,
            "loss_buffer": loss_buffer,
            "positions_production": position_production,
            "log_prob_production": log_prob_production,
            "local_accs_production": local_accs_production,
            "global_accs_production": global_accs_production,
            "local_sampler": local_sampler,
            "global_sampler": global_sampler,
            "model": model,
            "optimizer": optimizer,
            "sampler_state": sampler_state,
        }

        local_stepper = TakeSerialSteps(
            "logpdf",
            "local_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_local_accs"],
            n_local_steps,
            thinning=local_thinning,
            chain_batch_size=chain_batch_size,
            verbose=verbose,
        )

        global_stepper = TakeGroupSteps(
            "logpdf",
            "global_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_global_accs"],
            n_global_steps,
            thinning=global_thinning,
            chain_batch_size=chain_batch_size,
            verbose=verbose,
        )

        model_trainer = TrainModel(
            "model",
            "positions_training",
            "optimizer",
            loss_buffer_name="loss_buffer",
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            history_window=history_window,
            verbose=verbose,
        )

        update_state = UpdateState(
            "sampler_state",
            [
                "target_positions",
                "target_log_prob",
                "target_local_accs",
                "target_global_accs",
                "training",
            ],
            [
                "positions_production",
                "log_prob_production",
                "local_accs_production",
                "global_accs_production",
                False,
            ],
        )

        def reset_steppers(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Reset the steppers to the initial position."""
            local_stepper.set_current_position(0)
            global_stepper.set_current_position(0)
            return rng_key, resources, initial_position

        reset_steppers_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: reset_steppers(
                rng_key, resources, initial_position, data
            )
        )

        update_global_step = Lambda(
            lambda rng_key, resources, initial_position, data: (
                global_stepper.set_current_position(local_stepper.current_position)
            )
        )
        update_local_step = Lambda(
            lambda rng_key, resources, initial_position, data: (
                local_stepper.set_current_position(global_stepper.current_position)
            )
        )

        def update_model(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Update the model."""
            model = resources["model"]
            resources["global_sampler"] = eqx.tree_at(
                lambda x: x.model,
                resources["global_sampler"],
                model,
            )
            return rng_key, resources, initial_position

        update_model_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: update_model(
                rng_key, resources, initial_position, data
            )
        )

        self.strategies = {
            "local_stepper": local_stepper,
            "global_stepper": global_stepper,
            "model_trainer": model_trainer,
            "update_state": update_state,
            "update_global_step": update_global_step,
            "update_local_step": update_local_step,
            "reset_steppers": reset_steppers_lambda,
            "update_model": update_model_lambda,
        }

        training_phase = [
            "local_stepper",
            "update_global_step",
            "model_trainer",
            "update_model",
            "global_stepper",
            "update_local_step",
        ]
        production_phase = [
            "local_stepper",
            "update_global_step",
            "global_stepper",
            "update_local_step",
        ]
        strategy_order = []
        for _ in range(n_training_loops):
            strategy_order.extend(training_phase)

        strategy_order.append("reset_steppers")
        strategy_order.append("update_state")
        for _ in range(n_production_loops):
            strategy_order.extend(production_phase)

        self.strategy_order = strategy_order


# WARNING: do not change anything in this class


# Copyright (c) 2022 Kaze Wong & contributor
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class Sampler:
    """Top level API that the users primarily interact with.

    Args:
        n_dim (int): Dimension of the parameter space.
        n_chains (int): Number of chains to sample.
        rng_key (PRNGKeyArray): Jax PRNGKey.
        logpdf (Callable[[Float[Array, "n_dim"], dict], Float):
            Log probability function.
        resources (dict[str, Resource]): Resources to be used by the sampler.
        strategies (dict[str, Strategy]): Strategies to be used by the sampler.
        verbose (bool): Whether to print out progress. Defaults to False.
        logging (bool): Whether to log the progress. Defaults to True.
        outdir (str): Directory to save the logs. Defaults to "./outdir/".
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: PRNGKeyArray
    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: Optional[list[str]]

    # Logging hyperparameters
    verbose: bool = False
    logging: bool = True
    outdir: str = "./outdir/"

    def __init__(
        self,
        n_dim: int,
        n_chains: int,
        rng_key: PRNGKeyArray,
        resources: None | dict[str, Resource] = None,
        strategies: None | dict[str, Strategy] = None,
        strategy_order: None | list[str] = None,
        resource_strategy_bundles: None | ResourceStrategyBundle = None,
        **kwargs,
    ):
        # Copying input into the model

        self.n_dim = n_dim
        self.n_chains = n_chains
        self.rng_key = rng_key

        if resources is not None and strategies is not None:
            print(
                "Resources and strategies provided. Ignoring resource strategy bundles."
            )
            self.resources = resources
            self.strategies = strategies
            self.strategy_order = strategy_order

        else:
            print(
                "Resources or strategies not provided. Using resource strategy bundles."
            )
            if resource_strategy_bundles is None:
                raise ValueError(
                    "Resource strategy bundles not provided."
                    "Please provide either resources and strategies or resource strategy bundles."
                )
            self.resources = resource_strategy_bundles.resources
            self.strategies = resource_strategy_bundles.strategies
            self.strategy_order = resource_strategy_bundles.strategy_order

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def sample(
        self,
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
        n_local_steps_per_loop: int,
        n_global_steps_per_loop: int,
        labels: List[str],
    ):
        """Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.
            data (dict): Data to be used by the likelihood functions
        """
        initial_position = jnp.atleast_2d(initial_position)  # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        assert isinstance(self.strategy_order, list)
        for strategy in self.strategy_order:
            if strategy not in self.strategies:
                raise ValueError(
                    f"Invalid strategy name '{strategy}' provided. "
                    f"Available strategies are: {list(self.strategies.keys())}."
                )

        n_total_strategy = len(self.strategy_order)
        n_global_steps = self.strategy_order.count("model_trainer")
        n_local_steps = (n_total_strategy - 6 * n_global_steps - 2) // 4

        with tqdm.tqdm(range(n_global_steps), total=n_global_steps) as pbar:
            pbar.set_description("Global Tuning")
            for i in pbar:
                for strategy in self.strategy_order[6 * i : 6 * (i + 1)]:
                    (
                        rng_key,
                        self.resources,
                        last_step,
                    ) = self.strategies[strategy](
                        rng_key, self.resources, last_step, data
                    )
                    _save_acceptances(self.resources)
                    _save_chains(self.resources, labels, is_training=True)
                    _save_loss(self.resources)

        logger.info("Transitioning to production: Cleaning up global training samples.")
        for strategy in self.strategy_order[
            6 * n_global_steps : 6 * n_global_steps + 2
        ]:
            (
                rng_key,
                self.resources,
                last_step,
            ) = self.strategies[strategy](rng_key, self.resources, last_step, data)

        with tqdm.tqdm(range(n_local_steps), total=n_local_steps) as pbar:
            pbar.set_description("Global Sampling")
            offset = 6 * n_global_steps + 2
            for i in pbar:
                for strategy in self.strategy_order[
                    offset + 4 * i : offset + 4 * (i + 1)
                ]:
                    (
                        rng_key,
                        self.resources,
                        last_step,
                    ) = self.strategies[strategy](
                        rng_key, self.resources, last_step, data
                    )
                    _save_acceptances(self.resources)
                    _save_chains(self.resources, labels, is_training=False)
                    _save_samples(
                        self.resources,
                        labels,
                        n_local_steps_per_loop,
                        n_global_steps_per_loop,
                    )

    # TODO: Implement quick access and summary functions that operates on buffer

    def serialize(self):
        """Serialize the sampler object."""
        raise NotImplementedError

    def deserialize(self):
        """Deserialize the sampler object."""
        raise NotImplementedError


def _save_acceptances(resources: dict) -> None:
    """Overwrite the global and local acceptance rates in the HDF5 file.

    Parameters
    ----------
    resources : dict
        The sampler's resource buffers. Missing or empty acceptance buffers are skipped.
    """
    with h5py.File(INFERENCE_OUTPUT_FILENAME, "a") as f:
        for acc_type in ["global", "local"]:
            train_key = f"{acc_type}_accs_training"
            prod_key = f"{acc_type}_accs_production"

            if train_key in resources and len(resources[train_key].data) > 0:
                train_data = np.array(resources[train_key].data).mean(0)
                write_to_hdf5(f, f"acceptances/{acc_type}/train", train_data)

            if prod_key in resources and len(resources[prod_key].data) > 0:
                prod_data = np.array(resources[prod_key].data).mean(0)
                write_to_hdf5(f, f"acceptances/{acc_type}/prod", prod_data)


def _save_chains(resources: dict, labels: list[str], *, is_training: bool) -> None:
    """Overwrite the chains and log probabilities in the HDF5 file.

    Each chain is written as its own dataset under ``chains/<phase>/chain_<n>``, and the
    parameter labels are recorded as a file attribute so the columns can be named
    afterwards.

    Parameters
    ----------
    resources : dict
        The sampler's resource buffers.
    labels : list[str]
        Names of the sampled dimensions, in column order.
    is_training : bool
        Whether these are the training-phase chains or the production ones.
    """
    phase = "train" if is_training else "prod"
    pos_key = f"positions_{'training' if is_training else 'production'}"
    lp_key = f"log_prob_{'training' if is_training else 'production'}"

    if pos_key not in resources or lp_key not in resources:
        return

    positions = np.array(resources[pos_key].data)  # Shape: (n_chains, n_steps, n_dims)
    log_probs = np.array(resources[lp_key].data)  # Shape: (n_chains, n_steps)

    n_chains = positions.shape[0]

    # Store parameter labels as metadata attributes
    with h5py.File(INFERENCE_OUTPUT_FILENAME, "a") as f:
        f.attrs["labels"] = labels

        # Overwrite each chain dataset individually with the complete updated sequence
        for n in range(n_chains):
            dataset_suffix = (
                f"chains/{phase}/" + CHAIN_GROUP_FORMAT.format(chain_id=n) + "/"
            )
            write_to_hdf5(f, dataset_suffix + "positions", positions[n])
            write_to_hdf5(f, dataset_suffix + "log_probs", log_probs[n])


def _save_samples(
    resources: dict,
    labels: list[str],
    n_local_steps_per_loop: int,
    n_global_steps_per_loop: int,
) -> None:
    r"""Overwrite the posterior samples dataset in the HDF5 file.

    Only the local sampler's steps are kept: the global steps are normalizing-flow
    proposals whose acceptance is already reflected in the local chain, so including them
    would double-count. Rows containing :math:`-\infty` are dropped.

    Parameters
    ----------
    resources : dict
        The sampler's resource buffers.
    labels : list[str]
        Names of the sampled dimensions, in column order.
    n_local_steps_per_loop : int
        Local steps recorded per loop, after thinning.
    n_global_steps_per_loop : int
        Global steps recorded per loop, after thinning.
    """
    if "positions_production" not in resources:
        return

    positions = np.array(resources["positions_production"].data)
    _, n_production_steps, n_dims = positions.shape

    selected_indices = [
        idx
        for idx in range(n_production_steps)
        if (idx % (n_local_steps_per_loop + n_global_steps_per_loop))
        < n_local_steps_per_loop
    ]

    local_sampler_positions = positions[:, selected_indices, :].reshape(-1, n_dims)
    local_sampler_positions = local_sampler_positions[
        ~np.isneginf(local_sampler_positions).any(axis=1)
    ]

    write_to_hdf5(
        INFERENCE_OUTPUT_FILENAME, SAMPLES_GROUP_NAME, local_sampler_positions
    )


def _save_loss(resources: dict) -> None:
    """Overwrite the training loss dataset in the HDF5 file.

    Parameters
    ----------
    resources : dict
        The sampler's resource buffers. A missing loss buffer is skipped.
    """
    if "loss_buffer" not in resources:
        return

    train_loss_vals = np.array(resources["loss_buffer"].data).reshape(-1)
    write_to_hdf5(INFERENCE_OUTPUT_FILENAME, "loss", train_loss_vals)


class FlowMCBase(AnalysisBase):
    """Sampler mixin that runs the analysis with flowMC.

    Mixed with a data-representation base, which supplies ``run`` and ``read_data``, and
    with a model family's ``Core`` class. Selected at runtime by ``sampler_cfg.json``.
    """

    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Dict[str, Any],
        labels: List[str],
    ) -> None:
        """Run flowMC over the given log posterior and data.

        Chains are started from prior draws, so the initial positions are automatically
        within the support. The sampler configuration is written to the output HDF5 before
        sampling starts, so the run can be reconstructed from the output file alone. The
        debugging flags ``debug_nans``, ``profile_memory`` and ``check_leaks`` each wrap the
        run differently; they are mutually exclusive, the first set winning.

        Parameters
        ----------
        logpdf : Callable[[Array, Dict[str, Any]], Array]
            The log posterior built by :mod:`gwkokab.inference`.
        priors : JointDistribution
            Joint prior over the sampled variables, used to draw the initial chain positions.
        data : Dict[str, Any]
            The event data, passed through to ``logpdf`` on every evaluation.
        labels : List[str]
            Names of the sampled dimensions, in ``variables_index`` order.
        """
        sampler_cfg: FlowMCGlobalConfig = self.sampler_cfg

        n_chains = sampler_cfg.n_chains
        initial_position = priors.sample(self.rng_key, (n_chains,))
        n_dims = initial_position.shape[1]

        bundle = Local_Global_Sampler_Bundle(
            rng_key=self.rng_key,
            n_chains=n_chains,
            n_dims=n_dims,
            logpdf=logpdf,
            n_local_steps=sampler_cfg.n_local_steps,
            n_global_steps=sampler_cfg.n_global_steps,
            n_training_loops=sampler_cfg.n_training_loops,
            n_production_loops=sampler_cfg.n_production_loops,
            n_epochs=sampler_cfg.n_epochs,
            local_sampler_name=sampler_cfg.local_sampler_name,
            step_size=sampler_cfg.step_size,
            condition_matrix=sampler_cfg.condition_matrix,
            n_leapfrog=sampler_cfg.n_leapfrog,
            chain_batch_size=sampler_cfg.chain_batch_size,
            rq_spline_hidden_units=sampler_cfg.rq_spline_hidden_units,
            rq_spline_n_bins=sampler_cfg.rq_spline_n_bins,
            rq_spline_n_layers=sampler_cfg.rq_spline_n_layers,
            rq_spline_range=sampler_cfg.rq_spline_range,
            learning_rate=sampler_cfg.learning_rate,
            batch_size=sampler_cfg.batch_size,
            n_max_examples=sampler_cfg.n_max_examples,
            history_window=sampler_cfg.history_window,
            local_thinning=sampler_cfg.local_thinning,
            global_thinning=sampler_cfg.global_thinning,
            n_NFproposal_batch_size=sampler_cfg.n_NFproposal_batch_size,
            verbose=sampler_cfg.verbose,
        )
        logger.success("Local_Global_Sampler_Bundle created.")

        logger.info("Saving sampler configuration to HDF5.")
        sampler_cfg.write_to_hdf5(INFERENCE_OUTPUT_FILENAME, n_dims=n_dims)
        logger.success("Sampler configuration saved.")

        sampler = Sampler(
            n_dims,
            n_chains,
            self.rng_key,
            resource_strategy_bundles=bundle,
        )

        logger.debug("Sampler initialized, starting sampling.")

        n_local_steps_per_loop = sampler_cfg.n_local_steps // sampler_cfg.local_thinning
        n_global_steps_per_loop = (
            sampler_cfg.n_global_steps // sampler_cfg.global_thinning
        )

        if self.debug_nans:
            with jax.debug_nans(True):
                sampler.sample(
                    initial_position,
                    data,
                    n_local_steps_per_loop=n_local_steps_per_loop,
                    n_global_steps_per_loop=n_global_steps_per_loop,
                    labels=labels,
                )
        elif self.profile_memory:
            sampler.sample(
                initial_position,
                data,
                n_local_steps_per_loop=n_local_steps_per_loop,
                n_global_steps_per_loop=n_global_steps_per_loop,
                labels=labels,
            )
            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
            logger.debug("Memory profile saved as {filename}", filename=filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                sampler.sample(
                    initial_position,
                    data,
                    n_local_steps_per_loop=n_local_steps_per_loop,
                    n_global_steps_per_loop=n_global_steps_per_loop,
                    labels=labels,
                )
        else:
            sampler.sample(
                initial_position,
                data,
                n_local_steps_per_loop=n_local_steps_per_loop,
                n_global_steps_per_loop=n_global_steps_per_loop,
                labels=labels,
            )

        logger.info("Sampling and data saving complete.")


flowMC_arg_parser = analysis_base_arg_parser
