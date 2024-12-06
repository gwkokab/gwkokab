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


import warnings
from collections.abc import Sequence
from typing import Any, List, Optional, Tuple

import equinox as eqx
import h5py
import jax
import numpy as np
import pandas as pd
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from numpyro.util import is_prng_key


__all__ = [
    "load_model",
    "make_model",
    "mse_loss_fn",
    "predict",
    "read_data",
    "save_model",
]


@eqx.filter_value_and_grad
def mse_loss_fn(model: PyTree, x: Float[Array, ""], y: Float[Array, ""]) -> Array:
    """Mean squared error loss function.

    :param model: Model to approximate the log of the VT function
    :param x: input data
    :param y: output data
    :return: mean squared error
    """
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y - y_pred))  # mean squared error


@eqx.filter_jit
def predict(model: PyTree, x: Float[Array, ""]) -> Array:
    """Predict the output of the model given the input data.

    :param model: Model to approximate the log of the VT function
    :param x: input data
    :return: predicted output
    """
    return jax.vmap(model)(x)


def read_data(data_path: str) -> pd.DataFrame:
    """Read the data from the given path.

    :param data_path: path to the data
    :return: data in a DataFrame
    """
    with h5py.File(data_path, "r") as vt_file:
        keys = list(vt_file.keys())
        df = pd.DataFrame(data={key: np.array(vt_file[key]).flatten() for key in keys})
    return df


def make_model(
    *,
    key: PRNGKeyArray,
    input_layer: int,
    output_layer: int,
    hidden_layers: Optional[list[int]] = None,
) -> PyTree:
    """Make a neural network model to approximate the log of the VT function.

    :param key: jax random key
    :param input_layer: input layer of the model
    :param output_layer: output layer of the model
    :param hidden_layers: hidden layers of the model
    :return: neural network model
    """
    assert is_prng_key(key)
    if hidden_layers is None:
        keys = jrd.split(key, 2)
        layers: List[eqx.nn.Linear | eqx.nn.Lambda] = [
            eqx.nn.Linear(
                in_features=input_layer,
                out_features=output_layer,
                key=keys[0],
            )
        ]
        model = eqx.nn.Sequential(layers)
        return model

    keys = jrd.split(key, 2 + len(hidden_layers))

    layers = [
        eqx.nn.Linear(
            in_features=input_layer,
            out_features=hidden_layers[0],
            key=keys[0],
        ),
        eqx.nn.Lambda(jnn.relu),
    ]
    for i in range(len(hidden_layers) - 1):
        layers.append(
            eqx.nn.Linear(
                in_features=hidden_layers[i],
                out_features=hidden_layers[i + 1],
                key=keys[i + 1],
            ),
        )
        layers.append(eqx.nn.Lambda(jnn.relu))
    layers.append(
        eqx.nn.Linear(
            in_features=hidden_layers[-1], out_features=output_layer, key=keys[-1]
        )
    )

    model = eqx.nn.Sequential(layers)

    return model


def save_model(
    *, filename: str, model: eqx.nn.Sequential, names: Optional[Sequence[str]] = None
) -> None:
    """Save the model to the given file.

    :param filename: Name of the file to save the model
    :param model: Model to approximate the log of the VT function
    """
    if not filename.endswith(".hdf5"):
        if "." in filename:
            old_filename = filename
            filename = filename.split(".")[0] + ".hdf5"
            warnings.warn(
                f"Neural VT path does not end with .hdf5: {old_filename}. Saving to {filename} instead."
            )
    num_layers = len(model.layers)
    with h5py.File(filename, "w") as f:
        if names is not None:
            f.create_dataset("names", data=np.array(names, dtype="S"))
        for i in range(0, num_layers, 2):
            layer_number = i >> 1
            layer_i = f.create_group(f"layer_{layer_number}")
            layer_i.create_dataset(
                f"weight_{layer_number}", data=model.layers[i].weight
            )
            layer_i.create_dataset(f"bias_{layer_number}", data=model.layers[i].bias)


def load_model(filename) -> Tuple[List[str], eqx.nn.Sequential]:
    """Load the model from the given file.

    :param filename: Name of the file to load the model
    :return: names of the parameters and the model
    """
    layers: List[Any] = []
    with h5py.File(filename, "r") as f:
        names = f["names"][:]
        names = names.astype(str).tolist()
        i = 0
        while f.get(f"layer_{i}"):
            layer_i = f[f"layer_{i}"]
            weight_i = layer_i[f"weight_{i}"][:]
            bias_i = layer_i[f"bias_{i}"][:]
            nn = eqx.nn.Linear(
                in_features=weight_i.shape[1],
                out_features=weight_i.shape[0],
                key=jrd.PRNGKey(0),
            )
            nn = eqx.tree_at(lambda l: l.weight, nn, weight_i)
            nn = eqx.tree_at(lambda l: l.bias, nn, bias_i)
            layers.append(nn)
            layers.append(eqx.nn.Lambda(jnn.relu))
            i += 1
    layers.pop(-1)  # remove the last relu layer
    new_model = eqx.nn.Sequential(layers)

    return names, new_model
