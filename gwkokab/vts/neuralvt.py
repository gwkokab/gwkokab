#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Sequence

import h5py
import jax
import jax.numpy as jnp
import optax
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax import numpy as jnp
from jaxtyping import Array


def read_vt_file(file_path: str = "./vt_1_200_1000.hdf5") -> Sequence[jnp.ndarray]:
    """Interpolates the VT values from an HDF5 file based on given m1 and m2 coordinates.

    :param m1: The m1 coordinate.
    :param m2: The m2 coordinate.
    :param file_path: The path to the HDF5 file, defaults to "./vt_1_200_1000.hdf5"
    :return: The interpolated VT value.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        m1_grid = hdf5_file["m1"][:]
        m2_grid = hdf5_file["m2"][:]
        VT_grid = hdf5_file["VT"][:]
        m1_coord = m1_grid[0]
        m2_coord = m2_grid[:, 0]

    return m1_coord, m2_coord, VT_grid


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


def create_train_state(
    neural_vt: nn.Module,
    rng: Array,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
):
    params = neural_vt.init(rng, jnp.ones((2,)))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=neural_vt.apply,
        params=params,
        tx=tx,
        metric=Metrics.create(accuracy=metrics.Accuracy(), loss=metrics.Average()),
    )


class NeuralVT(nn.Module):
    """A neural network that approximates the VT function.

    Dense(2)->ReLU->Dense(128)->ReLU->Dense(128)->ReLU->Dense(1)
    """

    @nn.compact
    def __call__(self, *args, **kwargs):
        x = args[0]
        x = nn.Dense(2)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def create_train_state(
    module: nn.Module,
    rng: Array,
    learning_rate: float,
    momentum: float,
):
    """Creates an initial `TrainState`."""
    params = module.init(
        rng,
        jnp.ones([2, 128, 128, 1]),
    )["params"]
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metric=Metrics.create(
            accuracy=metrics.Accuracy(),
            loss=metrics.Average(),
        ),
    )


def train_step(
    state: train_state.TrainState,
    batch: Array,
    rng: Array,
) -> train_state.TrainState:
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch["m1"],
            batch["m2"],
        )
        loss = jnp.mean((logits - batch["VT"]) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)
    return new_state.replace(metric=state.metric.merge(loss=loss))


def eval_step(
    state: train_state.TrainState,
    batch: Array,
) -> train_state.TrainState:
    """Evaluate for a single step."""
    logits = state.apply_fn(
        {"params": state.params},
        batch["m1"],
        batch["m2"],
    )
    loss = jnp.mean((logits - batch["VT"]) ** 2)
    return state.replace(metric=state.metric.merge(loss=loss))


def compute_metrics(
    state: train_state.TrainState,
    batch: Array,
) -> train_state.TrainState:
    """Compute metrics for a single step."""
    logits = state.apply_fn(
        {"params": state.params},
        batch["m1"],
        batch["m2"],
    )
    loss = jnp.mean((logits - batch["VT"]) ** 2)
    accuracy = jnp.mean(jnp.abs(logits - batch["VT"]))
    return state.replace(metric=state.metric.merge(loss=loss, accuracy=accuracy))


def train_epoch(
    state: train_state.TrainState,
    rng: Array,
    train_ds: Array,
    batch_size: int,
) -> train_state.TrainState:
    """Train for a single epoch."""
    perms = jax.random.permutation(rng, len(train_ds))
    perms = perms[: len(train_ds) - (len(train_ds) % batch_size)]
    perms = perms.reshape(-1, batch_size)
    for perm in perms:
        batch = {k: v[perm] for k, v in train_ds.items()}
        state = train_step(state, batch, rng)

    loss = state.metric.loss.compute()
    return state, loss


def eval_epoch(
    state: train_state.TrainState,
    test_ds: Array,
    batch_size: int,
) -> train_state.TrainState:
    """Evaluate for a single epoch."""
    perms = jnp.arange(len(test_ds))
    perms = perms[: len(test_ds) - (len(test_ds) % batch_size)]
    perms = perms.reshape(-1, batch_size)
    for perm in perms:
        batch = {k: v[perm] for k, v in test_ds.items()}
        state = eval_step(state, batch)

    loss = state.metric.loss.compute()
    return state, loss
