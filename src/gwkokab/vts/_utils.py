# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple

import equinox as eqx
import h5py
import jax
import numpy as np
import pandas as pd
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray, PyTree
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
@eqx.filter_jit
def mse_loss_fn(
    model: PyTree, x: Array, y: Array, batch_size: Optional[int] = 256
) -> Array:
    """Mean squared error loss function.

    Parameters
    ----------
    model : PyTree
        Model to approximate the log of the VT function
    x : Array
        input data
    y : Array
        output data
    batch_size : Optional[int], optional
        batch size for training, by default 256. This is used to avoid OOM errors when
        training large datasets. If None, the entire dataset will be trained
        sequentially. This is not recommended for large datasets.

    Returns
    -------
    Array
        mean squared error
    """
    y_pred = jax.lax.map(model, x, batch_size=batch_size)
    return jnp.mean(jnp.square(y - y_pred))  # mean squared error


@eqx.filter_jit
def predict(model: PyTree, x: Array, batch_size: Optional[int] = 256) -> Array:
    """Predict the output of the model given the input data.

    Parameters
    ----------
    model : PyTree
        Model to approximate the log of the VT function
    x : Array
        input data
    batch_size : Optional[int], optional
        batch size for prediction, by default 256. This is used to avoid OOM errors when
        predicting large datasets. If None, the entire dataset will be predicted
        sequentially. This is not recommended for large datasets.

    Returns
    -------
    Array
        predicted output
    """
    return jax.lax.map(model, x, batch_size=batch_size)


def read_data(data_path: str, keys: Sequence[str]) -> pd.DataFrame:
    """Read the data from the given path.

    Parameters
    ----------
    data_path : str
        path to the data
    keys : Sequence[str]
        keys to read from the data file

    Returns
    -------
    pd.DataFrame
        data in a DataFrame
    """
    with h5py.File(data_path, "r") as vt_file:
        df = pd.DataFrame(data={key: np.array(vt_file[key]).flatten() for key in keys})
    return df


def make_model(
    *,
    key: PRNGKeyArray,
    input_layer: int,
    output_layer: int,
    width_size: int,
    depth: int,
) -> eqx.nn.MLP:
    """Make a neural network model to approximate the log of the VT function.

    Parameters
    ----------
    key : PRNGKeyArray
        jax random key
    input_layer : int
        input layer of the model
    output_layer : int
        output layer of the model
    width_size : int
        width size of the model
    depth : int
        depth of the model

    Returns
    -------
    eqx.nn.MLP
        neural network model
    """
    assert is_prng_key(key)

    model = eqx.nn.MLP(
        in_size=input_layer,
        out_size=output_layer,
        width_size=width_size,
        depth=depth,
        activation=jnn.relu,
        key=key,
    )
    return model


def save_model(
    *,
    filepath: str,
    datafilepath: str,
    model: eqx.nn.MLP,
    names: Optional[Sequence[str]] = None,
    is_log: bool = False,
) -> None:
    """Save the model to the given file.

    Parameters
    ----------
    filepath : str
        Name of the file to save the model
    datafilepath : str
        Path to the data file, used to save the names of the parameters
    model : eqx.nn.MLP
        Model to approximate the log of the VT function
    names : Optional[Sequence[str]], optional
        names of the parameters, by default None
    is_log : bool, optional
        Whether the model was trained on log-transformed data, by default False
    """
    if not filepath.endswith(".hdf5"):
        if "." in filepath:
            old_filename = filepath
            filepath = filepath.split(".")[0] + ".hdf5"
            warnings.warn(
                f"Neural VT path does not end with .hdf5: {old_filename}. Saving to {filepath} instead."
            )

    with h5py.File(datafilepath, "r") as f:
        # read all attributes from the data file
        attr = dict(f.attrs)
    model: eqx.nn.MLP = model._fun  # type: ignore
    with h5py.File(filepath, "w") as f:
        for key, value in attr.items():
            f.attrs[key] = value  # copy attributes from the data file
        if names is not None:
            f.create_dataset("names", data=np.array(names, dtype="S"))
        f.create_dataset("in_size", data=model.in_size)  # type: ignore
        f.create_dataset("out_size", data=model.out_size)  # type: ignore
        f.create_dataset("width_size", data=model.width_size)  # type: ignore
        f.create_dataset("depth", data=model.depth)  # type: ignore
        f.attrs["is_log"] = is_log
        num_layers = len(model.layers)  # type: ignore
        for i in range(num_layers):
            layer_i = f.create_group(f"layer_{i}")
            layer_i.create_dataset(f"weight_{i}", data=model.layers[i].weight)  # type: ignore
            layer_i.create_dataset(f"bias_{i}", data=model.layers[i].bias)  # type: ignore


def load_model(filename: str) -> Tuple[List[str], eqx.nn.MLP]:
    """Load the model from the given file.

    Parameters
    ----------
    filename : str
        Name of the file to load the model

    Returns
    -------
    Tuple[List[str], eqx.nn.MLP]
        names of the parameters and the model
    """
    with h5py.File(filename, "r") as f:
        names = f["names"][:]
        names = names.astype(str).tolist()
        in_size = int(f["in_size"][()])
        out_size = int(f["out_size"][()])
        width_size = int(f["width_size"][()])
        depth = int(f["depth"][()])
        new_model = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.relu,
            key=jrd.PRNGKey(0),
        )
        i = 0
        while f.get(f"layer_{i}"):
            layer_i = f[f"layer_{i}"]
            weight_i = jax.device_put(np.asarray(layer_i[f"weight_{i}"][:]))
            bias_i = jax.device_put(np.asarray(layer_i[f"bias_{i}"][:]))
            new_model = eqx.tree_at(lambda m: m.layers[i].weight, new_model, weight_i)
            new_model = eqx.tree_at(lambda m: m.layers[i].bias, new_model, bias_i)
            i += 1

    return names, new_model
