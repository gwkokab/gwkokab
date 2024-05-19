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


from __future__ import annotations

import json

import equinox as eqx
import h5py
import jax
import numpy as np
import optax
import polars as pl
from jax import numpy as jnp

from ..utils import get_key


@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    """Mean squared error loss function for the neural network.

    :param model: Model to approximate the log of the VT function
    :param x: input data
    :param y: output data
    :return: mean squared error
    """
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y - y_pred))  # mean squared error


class NeuralVT:
    """
    A class to approximate the log of the VT function using a neural network.

    >>> from gwkokab.vts.neuralvt import NeuralVT
    >>> neural_vt = NeuralVT(
    ...     input_keys=["m1", "m2"],
    ...     output_keys=["VT"],
    ...     hidden_layers=[64, 64, 512, 512, 64, 64],
    ...     epochs=2,
    ...     batch_size=128,
    ... )
    >>> neural_vt.train(plot_loss=True)
    """

    def __init__(
        self,
        input_keys: list[str] = ["m1", "m2"],
        output_keys: list[str] = ["VT"],
        hidden_layers: list[int] = [128],
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        model_name: str = "log(VT) approximation model",
        activation: str = "relu",
        data_path: str = "./vt_1_200_1000.hdf5",
        batch_size: int = 128,
        epochs: int = 15,
        validation_split: float = 0.2,
        model_path: str = "model.eqx",
    ) -> None:
        """Initialize the NeuralVT class.

        :param input_keys: keys of the input data, defaults to ["m1", "m2"]
        :param output_key: key of the output data, defaults to ["VT"]
        :param hidden_layers: hidden layers of the neural network, defaults to [128]
        :param optimizer: optimizer for the neural network, defaults to "adam"
        :param loss: loss function for the neural network, defaults to "mean_squared_error"
        :param model_name: name of the model, defaults to "log(VT) approximation model"
        :param activation: activation function for the hidden layers, defaults to "relu"
        :param data_path: path to the HDF5 file containing the data, defaults to "./vt_1_200_1000.hdf5"
        :param batch_size: batch size for training the neural network, defaults to 128
        :param epochs: number of epochs for training the neural network, defaults to 15
        :param validation_split: validation split for training the neural network, defaults to 0.2
        :param model_path: path to save the trained model, defaults to "model.keras"
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = model_name
        self.activation = activation
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.model_path = model_path
        self.optimizer = optax.adam(1e-3)

    def read_data(self) -> pl.DataFrame:
        """Read the data from the HDF5 file.

        :return: a polars DataFrame containing the data
        """
        with h5py.File(self.data_path, "r") as vt_file:
            keys = list(vt_file.keys())
            df = pl.DataFrame(
                data={key: pl.Series(key, np.array(vt_file[key][:]).flatten()) for key in keys},
            )
        return df

    def build_model(self):
        keys = jax.random.split(get_key(), 2 + len(self.hidden_layers))

        layers = [
            eqx.nn.Linear(
                in_features=len(self.input_keys),
                out_features=self.hidden_layers[0],
                key=keys[0],
            ),
            eqx.nn.Lambda(jax.nn.relu),
        ]
        for i in range(len(self.hidden_layers) - 1):
            layers.append(
                eqx.nn.Linear(
                    in_features=self.hidden_layers[i],
                    out_features=self.hidden_layers[i + 1],
                    key=keys[i],
                ),
            )
            layers.append(eqx.nn.Lambda(jax.nn.relu))
        layers.append(
            eqx.nn.Linear(
                in_features=self.hidden_layers[-1],
                out_features=len(self.output_keys),
                key=keys[i],
            )
        )

        model = eqx.nn.Sequential(layers)

        return model

    @eqx.filter_jit
    def make_step(self, model, x, y, opt_state):
        """Make a step in the optimization process.

        :param model: Model to approximate the log of the VT function
        :param x: input data
        :param y: output data
        :param opt_state: optimizer state
        :return: optimizer state
        """
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_epoch(self, model, x, y, opt_state):
        """Train the model for one epoch.

        :param model: Model to approximate the log of the VT function
        :param x: input data
        :param y: output data
        :param opt_state: optimizer state
        :return: optimizer state
        """
        model, opt_state, loss = self.make_step(model, x, y, opt_state)
        return model, opt_state, loss

    def train(self, *, plot_loss: bool = True):
        """Train the neural network to approximate the log of the VT function.

        :param plot_loss: plot the loss function of the model, defaults to True
        """
        df = self.read_data()

        data_X = df[self.input_keys].to_numpy()
        data_Y = df[self.output_keys].to_numpy()

        log_data_Y = np.log(data_Y)

        model = self.build_model()

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        for epoch in range(self.epochs):
            model, opt_state, loss = self.train_epoch(model, data_X, log_data_Y, opt_state)
            print(f"Epoch {epoch + 1}: loss {loss}")

        self.save(self.model_path, {"hidden_layers": self.hidden_layers}, model)

    def save(self, filename, hyperparams, model):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)
