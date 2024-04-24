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


import h5py
import optax
from clu import metrics
from flax import linen as nn, struct
from flax.training import train_state
from jax import numpy as jnp


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metric: Metrics


def create_train_state(neural_vt, rng, learning_rate=1e-3, momentum=0.9):
    params = neural_vt.init(rng, jnp.ones((2,)))["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
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

    def vt_loss_batch(self, params, m1, m2, vt):
        """Computes the loss for the VT function.

        :param params: The parameters of the neural network.
        :param m1: The m1 coordinate.
        :param m2: The m2 coordinate.
        :param vt: The VT value.
        :return: The loss value.
        """
        vt_pred = self.apply({"params": params}, m1, m2)
        return jnp.mean((vt_pred - vt) ** 2, axis=0)


def read_vt_file(file_path: str = "./vt_1_200_1000.hdf5"):
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
