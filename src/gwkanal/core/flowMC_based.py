# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import equinox as eqx
import jax
import numpy as np
import tqdm
from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.resource.states import State
from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle
from flowMC.strategy.base import Strategy
from flowMC.strategy.lambda_function import Lambda
from flowMC.strategy.take_steps import TakeGroupSteps, TakeSerialSteps
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.update_state import UpdateState
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from loguru import logger

from gwkanal.core.guru import Guru, guru_arg_parser
from gwkanal.utils.common import read_json
from gwkanal.utils.literals import INFERENCE_DIRECTORY, POSTERIOR_SAMPLES_FILENAME
from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import error_if, warn_if


_INFERENCE_DIRECTORY = "flowMC_" + INFERENCE_DIRECTORY


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
class _HMC(ProposalBase):
    """Hamiltonian Monte Carlo sampler class building the hmc_sampler method from target
    logpdf.

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    mass_matrix: Float[Array, " n_dim n_dim"]
    step_size: float
    leapfrog_coefs: Float[Array, " n_leapfrog n_dim"]

    @property
    def n_leapfrog(self) -> int:
        return self.leapfrog_coefs.shape[0] - 2

    def __init__(
        self,
        mass_matrix: Float[Array, " n_dim n_dim"],
        step_size: float = 0.1,
        n_leapfrog: int = 10,
    ):
        self.mass_matrix = mass_matrix
        self.step_size = step_size

        coefs = jnp.ones((n_leapfrog + 2, 2))
        coefs = coefs.at[0].set(jnp.array([0, 0.5]))
        coefs = coefs.at[-1].set(jnp.array([1, 0.5]))
        self.leapfrog_coefs = coefs

    def get_initial_hamiltonian(
        self,
        potential: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        kinetic: Callable[
            [Float[Array, " n_dim"], Float[Array, " n_dim n_dim"]], Float[Array, "1"]
        ],
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        data: PyTree,
    ):
        L = jnp.linalg.cholesky(self.mass_matrix)
        momentum = L @ jax.random.normal(rng_key, shape=position.shape)

        return potential(position, data) + kinetic(momentum, self.mass_matrix)

    def leapfrog_kernel(self, kinetic, potential, carry, extras):
        position, momentum, data, metric, index = carry

        # Note: jax.grad(kinetic) with respect to momentum will return M^-1 @ p
        # which is the velocity, effectively handling the dense matrix logic.
        position = position + self.step_size * self.leapfrog_coefs[index][0] * jax.grad(
            kinetic
        )(momentum, metric)

        momentum = momentum - self.step_size * self.leapfrog_coefs[index][1] * jax.grad(
            potential
        )(position, data)

        index = index + 1
        return (position, momentum, data, metric, index), extras

    def leapfrog_step(
        self,
        leapfrog_kernel: Callable,
        position: Float[Array, " n_dim"],
        momentum: Float[Array, " n_dim"],
        data: PyTree,
        metric: Float[Array, " n_dim n_dim"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        (position, momentum, data, metric, _), _ = jax.lax.scan(
            leapfrog_kernel,
            (position, momentum, data, metric, 0),
            jnp.arange(self.n_leapfrog + 2),
        )
        return position, momentum

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        def potential(x: Float[Array, " n_dim"], data: PyTree) -> Float[Array, "1"]:
            return -logpdf(x, data)

        # CHANGED: Kinetic energy for dense mass matrix
        # K(p) = 0.5 * p^T * M^-1 * p
        def kinetic(
            p: Float[Array, " n_dim"], metric: Float[Array, " n_dim n_dim"]
        ) -> Float[Array, "1"]:
            # We solve Mx = p for x (which is M^-1 p), then dot with p
            velocity = jnp.linalg.solve(metric, p)
            return 0.5 * jnp.dot(p, velocity)

        leapfrog_kernel = jax.tree_util.Partial(
            self.leapfrog_kernel, kinetic, potential
        )
        leapfrog_step = jax.tree_util.Partial(self.leapfrog_step, leapfrog_kernel)

        key1, key2 = jax.random.split(rng_key)

        # CHANGED: Correct Sampling of Momentum ~ N(0, M)
        # We need the lower triangular matrix L such that L @ L.T = M
        L = jnp.linalg.cholesky(self.mass_matrix)
        momentum = L @ jax.random.normal(key1, shape=position.shape)

        H = -log_prob + kinetic(momentum, self.mass_matrix)

        proposed_position, proposed_momentum = leapfrog_step(
            position, momentum, data, self.mass_matrix
        )

        proposed_PE = potential(proposed_position, data)
        proposed_ham = proposed_PE + kinetic(proposed_momentum, self.mass_matrix)
        log_acc = H - proposed_ham
        log_uniform = jnp.log(jax.random.uniform(key2))

        do_accept = log_uniform < log_acc

        position = jnp.where(do_accept, proposed_position, position)  # type: ignore
        log_prob = jnp.where(do_accept, -proposed_PE, log_prob)  # type: ignore

        return position, log_prob, do_accept

    def print_parameters(self):
        print("HMC parameters:")
        print(f"step_size: {self.step_size}")
        print(f"n_leapfrog: {self.n_leapfrog}")
        print(f"condition_matrix shape: {self.condition_matrix.shape}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError


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
        mass_matrix: Array = 1.0,  # type: ignore
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
            local_sampler = _HMC(
                mass_matrix=mass_matrix,
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

        os.makedirs(_INFERENCE_DIRECTORY, exist_ok=True)

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


def _same_length_arrays(length: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """This function pads the arrays with None to make them the same length.

    Parameters
    ----------
    length : int
        The length of the arrays.
    arrays : np.ndarray
        The arrays to pad.

    Returns
    -------
    tuple[np.ndarray, ...]
        The padded arrays.
    """
    padded_arrays = []
    for array in arrays:
        padded_array = np.empty((length,))
        padded_array[..., : array.shape[0]] = array
        padded_array[..., array.shape[0] :] = None
        padded_arrays.append(padded_array)
    return tuple(padded_arrays)


def _save_acceptances(resources: dict) -> None:
    """Saves global and local acceptance rates to disk."""
    # Mean acceptances
    for acc_type in ["global", "local"]:
        train_key = f"{acc_type}_accs_training"
        prod_key = f"{acc_type}_accs_production"

        # Check if training data was cleared
        train_data = (
            np.array(resources[train_key].data).mean(0)
            if train_key in resources
            else np.array([])
        )
        prod_data = (
            np.array(resources[prod_key].data).mean(0)
            if prod_key in resources
            else np.array([])
        )

        max_len = max(len(train_data), len(prod_data))
        warn_if(
            max_len == 0,
            msg=f"No data found for {acc_type} acceptance rates in both phases.",
        )

        np.savetxt(
            f"{_INFERENCE_DIRECTORY}/{acc_type}_accs.dat",
            np.column_stack(_same_length_arrays(max_len, train_data, prod_data)),
            header="train prod",
            comments="#",
        )


def _save_chains(resources: Dict, labels: List[str], *, is_training: bool) -> None:
    """Saves the chains to disk.

    Parameters
    ----------
    resources : Dict
        dictionary of resources
    labels : List[str]
        list of parameter labels
    is_training : bool
        whether the phase is training or production
    """
    header = " ".join(labels)

    if is_training:
        phase = "train"
        pos_key = "positions_training"
        lp_key = "log_prob_training"
    else:
        phase = "prod"
        pos_key = "positions_production"
        lp_key = "log_prob_production"

    if pos_key not in resources:
        logger.warning(f"Key {pos_key} not found in resources. Skipping save.")
        return

    positions = np.array(resources[pos_key].data)
    log_probs = np.array(resources[lp_key].data)

    n_chains = positions.shape[0]
    width = len(str(n_chains - 1))

    for n in range(n_chains):
        tag = str(n).zfill(width)
        np.savetxt(
            f"{_INFERENCE_DIRECTORY}/{phase}_chains_{tag}.dat",
            positions[n],
            header=header,
        )
        np.savetxt(
            f"{_INFERENCE_DIRECTORY}/log_prob_{phase}_{tag}.dat",
            log_probs[n],
            header=phase,
        )


def _save_samples(
    resources: Dict,
    labels: List[str],
    n_local_steps_per_loop: int,
    n_global_steps_per_loop: int,
) -> None:
    """Saves the posterior samples from the local sampler to disk.

    Parameters
    ----------
    resources : Dict
        dictionary of resources
    labels : List[str]
        list of parameter labels
    n_local_steps_per_loop : int
        number of local steps per loop
    n_global_steps_per_loop : int
        number of global steps per loop
    """
    header = " ".join(labels)

    positions = np.array(resources["positions_production"].data)

    _, n_production_steps, n_dims = positions.shape

    selected_indices = list(
        filter(
            lambda idx: (
                (idx % (n_local_steps_per_loop + n_global_steps_per_loop))
                < n_local_steps_per_loop
            ),
            range(n_production_steps),
        )
    )

    local_sampler_positions = positions[:, selected_indices, :].reshape(-1, n_dims)
    local_sampler_positions = local_sampler_positions[
        ~np.isneginf(local_sampler_positions).any(axis=1)
    ]

    np.savetxt(
        rf"{_INFERENCE_DIRECTORY}/{POSTERIOR_SAMPLES_FILENAME}",
        local_sampler_positions,
        header=header,
    )


def _save_loss(resources: dict) -> None:
    """Saves the training loss to disk.

    Parameters
    ----------
    resources : Dict
        dictionary of resources
    """
    train_loss_vals = np.array(resources["loss_buffer"].data)
    np.savetxt(
        rf"{_INFERENCE_DIRECTORY}/loss.dat", train_loss_vals.reshape(-1), header="loss"
    )


class FlowMCBased(Guru):
    output_directory: str = _INFERENCE_DIRECTORY

    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Dict[str, Any],
        labels: List[str],
    ) -> None:
        sampler_cfg: dict = read_json(self.sampler_settings_filename)

        batch_size = sampler_cfg["batch_size"]
        chain_batch_size = sampler_cfg["chain_batch_size"]
        global_thinning = sampler_cfg["global_thinning"]
        learning_rate = sampler_cfg["learning_rate"]
        local_thinning = sampler_cfg["local_thinning"]
        local_sampler_name: str = sampler_cfg.get("local_sampler_name", "mala")
        step_size = sampler_cfg["step_size"]
        mass_matrix = sampler_cfg.get("mass_matrix", 1.0)
        n_leapfrog = sampler_cfg.get("n_leapfrog", 10)
        n_chains = sampler_cfg["n_chains"]
        n_epochs = sampler_cfg["n_epochs"]
        n_global_steps = sampler_cfg["n_global_steps"]
        n_local_steps = sampler_cfg["n_local_steps"]
        n_max_examples = sampler_cfg["n_max_examples"]
        history_window = sampler_cfg.get("history_window", 100)
        n_NFproposal_batch_size = sampler_cfg["n_NFproposal_batch_size"]
        n_production_loops = sampler_cfg["n_production_loops"]
        n_training_loops = sampler_cfg["n_training_loops"]
        rq_spline_hidden_units = sampler_cfg["rq_spline_hidden_units"]
        rq_spline_n_bins = sampler_cfg["rq_spline_n_bins"]
        rq_spline_n_layers = sampler_cfg["rq_spline_n_layers"]
        rq_spline_range = tuple(sampler_cfg.get("rq_spline_range", (-10.0, 10.0)))
        verbose = sampler_cfg["verbose"]

        logger.debug("Validation for Sampler parameters starting")

        for var, var_name, expected_type in (
            (batch_size, "batch_size", int),
            (chain_batch_size, "chain_batch_size", int),
            (global_thinning, "global_thinning", int),
            (learning_rate, "learning_rate", (int, float)),
            (local_thinning, "local_thinning", int),
            (local_sampler_name, "local_sampler_name", str),
            (step_size, "step_size", (int, float)),
            (mass_matrix, "mass_matrix", (int, float, Sequence)),
            (n_leapfrog, "n_leapfrog", int),
            (n_chains, "n_chains", int),
            (n_epochs, "n_epochs", int),
            (n_global_steps, "n_global_steps", int),
            (n_local_steps, "n_local_steps", int),
            (n_max_examples, "n_max_examples", int),
            (history_window, "history_window", int),
            (n_NFproposal_batch_size, "n_NFproposal_batch_size", int),
            (n_production_loops, "n_production_loops", int),
            (n_training_loops, "n_training_loops", int),
            (rq_spline_hidden_units, "rq_spline_hidden_units", list),
            (rq_spline_n_bins, "rq_spline_n_bins", int),
            (rq_spline_n_layers, "rq_spline_n_layers", int),
            (rq_spline_range, "rq_spline_range", tuple),
            (verbose, "verbose", bool),
        ):
            error_if(
                not isinstance(var, expected_type),
                err=TypeError,
                msg=f"expected a {expected_type}, got {type(var)} for {var_name}",
            )
            if expected_type is int:
                error_if(
                    var < 1 and var_name != "chain_batch_size",
                    err=ValueError,
                    msg=f"expected a positive integer, got {var} for {var_name}",
                )
            logger.debug(f"{var_name}: {var}")
        error_if(
            not all(isinstance(x, int) and x > 0 for x in rq_spline_hidden_units),
            msg=f"expected a list of positive integers, got {rq_spline_hidden_units} for rq_spline_hidden_units",
        )

        valid_local_samplers = ("mala", "hmc")
        error_if(
            local_sampler_name.strip().lower() not in valid_local_samplers,
            msg="local_sampler_name must be one of " + ", ".join(valid_local_samplers),
        )

        initial_position = priors.sample(self.rng_key, (n_chains,))

        n_dims = initial_position.shape[1]

        if isinstance(mass_matrix, float) or isinstance(mass_matrix, int):
            error_if(mass_matrix <= 0.0, msg="mass_matrix must be positive")
            mass_matrix = jnp.eye(n_dims) * float(mass_matrix)
        elif isinstance(mass_matrix, list):
            mass_matrix = jnp.array(mass_matrix)
            error_if(mass_matrix.ndim > 2, msg="mass_matrix must be 1D or 2D array")
            _shape = mass_matrix.shape
            error_if(
                _shape != (n_dims, n_dims) and _shape != (n_dims,),
                msg=f"mass_matrix must be of shape ({n_dims}, {n_dims}) or ({n_dims},), got {_shape}",
            )
            if _shape == (n_dims,):
                error_if(
                    jnp.any(mass_matrix <= 0),
                    msg="mass_matrix diagonal elements must be positive",
                )
                mass_matrix = jnp.diag(mass_matrix)
            else:
                eigvals = jnp.linalg.eigvalsh(mass_matrix)
                error_if(
                    jnp.any(eigvals <= 0),
                    msg="mass_matrix must be positive definite",
                )

        bundle = Local_Global_Sampler_Bundle(
            rng_key=self.rng_key,
            n_chains=n_chains,
            n_dims=n_dims,
            logpdf=logpdf,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_training_loops=n_training_loops,
            n_production_loops=n_production_loops,
            n_epochs=n_epochs,
            local_sampler_name=local_sampler_name,
            step_size=step_size,
            mass_matrix=mass_matrix,
            n_leapfrog=n_leapfrog,
            chain_batch_size=chain_batch_size,
            rq_spline_hidden_units=rq_spline_hidden_units,
            rq_spline_n_bins=rq_spline_n_bins,
            rq_spline_n_layers=rq_spline_n_layers,
            rq_spline_range=rq_spline_range,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            history_window=history_window,
            local_thinning=local_thinning,
            global_thinning=global_thinning,
            n_NFproposal_batch_size=n_NFproposal_batch_size,
            verbose=verbose,
        )
        logger.info("Local_Global_Sampler_Bundle created successfully.")

        sampler = Sampler(
            n_dims,
            n_chains,
            self.rng_key,
            resource_strategy_bundles=bundle,
        )

        logger.debug("Sampler initialized, starting sampling.")

        n_local_steps_per_loop = n_local_steps // local_thinning
        n_global_steps_per_loop = n_global_steps // global_thinning

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


flowMC_arg_parser = guru_arg_parser
