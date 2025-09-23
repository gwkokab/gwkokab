# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import tqdm
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray, PyTree
from loguru import logger
from numpyro.util import is_prng_key


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
        raise ValueError("Model save path must end with .hdf5")

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


def _train_test_data_split(
    X: Array, Y: Array, batch_size: int, test_size: float = 0.2
) -> tuple[Array, Array, Array, Array]:
    """Split the data into training and testing sets.

    Parameters
    ----------
    X : Array
        Input data
    Y : Array
        Output data
    batch_size : int
        batch size for training
    test_size : float, optional
        fraction of the data to use for testing, by default 0.2

    Returns
    -------
    tuple[Array, Array, Array, Array]
        training and testing sets
    """
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    split = (split // batch_size) * batch_size
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_X = X[train_indices]
    train_Y = Y[train_indices]
    test_X = X[test_indices]
    test_Y = Y[test_indices]

    return train_X, test_X, train_Y, test_Y


def train_regressor(
    *,
    input_keys: list[str],
    output_keys: list[str],
    width_size: int,
    depth: int,
    batch_size: int,
    data_path: str,
    checkpoint_path: Optional[str] = None,
    epochs: int = 50,
    validation_split: float = 0.2,
    learning_rate: float = 1e-3,
    train_in_log: bool = False,
) -> None:
    """Train the model to approximate the log of the VT function.

    input_keys : list[str]
        list of input keys
    output_keys : list[str]
        list of output keys
    width_size : int
        width size of the model
    depth : int
        depth of the model
    batch_size : int
        batch size for training
    data_path : str
        path to the data
    checkpoint_path : Optional[str]
        path to save the model, by default None
    epochs : int
        number of epochs to train the model
    validation_split : float
        fraction of the data to use for validation, by default 0.2
    learning_rate : float
        learning rate for the optimizer, by default 1e-3
    train_in_log : bool
        whether to train the model in log space, by default False

    Raises
    ------
    ValueError
        if checkpoint path does not end with :code:`.hdf5`
    """
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided, model will not be saved.")
    if not checkpoint_path.endswith(".hdf5"):
        raise ValueError("Checkpoint path must end with .hdf5")

    optimizer = optax.adam(learning_rate=learning_rate)

    @eqx.filter_jit
    def make_step(
        model: eqx.nn.MLP,
        x: Array,
        y: Array,
        opt_state: optax.OptState,
    ) -> tuple[eqx.nn.MLP, optax.OptState, Array]:
        """Make a step in the optimization process.

        Parameters
        ----------
        model : eqx.nn.MLP
            Model to approximate the log of the VT function
        x : Array
            input data
        y : Array
            output data
        opt_state : optax.OptState
            optimizer state

        Returns
        -------
        tuple[eqx.nn.MLP, optax.OptState, Array]
            optimizer state
        """
        loss, grads = mse_loss_fn(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_step(
        model: eqx.nn.MLP,
        x: Array,
        y: Array,
        opt_state: optax.OptState,
    ) -> tuple[eqx.nn.MLP, optax.OptState, Array]:
        """Train the model for one epoch.

        Parameters
        ----------
        model : eqx.nn.MLP
            Model to approximate the log of the VT function
        x : Array
            input data
        y : Array
            output data
        opt_state : optax.OptState
            optimizer state

        Returns
        -------
        tuple[eqx.nn.MLP, optax.OptState, Array]
            optimizer state
        """
        model, opt_state, loss = make_step(model, x, y, opt_state)
        return model, opt_state, loss

    df = read_data(data_path, keys=input_keys + output_keys)

    data_X = jax.device_put(df[input_keys].to_numpy(), may_alias=True)
    data_Y = jax.device_put(df[output_keys].to_numpy(), may_alias=True)

    if train_in_log:
        data_Y = jnp.log(data_Y)
        data_Y = jnp.where(
            jnp.isneginf(data_Y), -jnp.finfo(jnp.result_type(float)).tiny, data_Y
        )

    train_X, test_X, train_Y, test_Y = _train_test_data_split(
        data_X,
        data_Y,
        batch_size,
        test_size=validation_split,
    )

    logger.info("Input Keys: " + ", ".join(input_keys))
    logger.info("Output Keys: " + ", ".join(output_keys))
    logger.info("Width Size: " + str(width_size))
    logger.info("Depth: " + str(depth))
    logger.info("Data Path: " + data_path)
    logger.info("Checkpoint Path: " + checkpoint_path)
    logger.info("Train Size: " + str(len(train_X)))
    logger.info("Test Size: " + str(len(test_X)))
    logger.info("Validation Split: " + str(validation_split))
    logger.info("Batch Size: " + str(batch_size))
    logger.info("Epochs: " + str(epochs))
    logger.info("Learning Rate: " + str(learning_rate))

    model = make_model(
        key=jrd.PRNGKey(np.random.randint(0, 2**32 - 1)),
        input_layer=len(input_keys),
        output_layer=len(output_keys),
        width_size=width_size,
        depth=depth,
    )

    model: Callable = eqx.filter_checkpoint(model)  # type: ignore

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_vals = []
    val_loss_vals = []

    with tqdm.tqdm(range(epochs), unit="epochs") as pbar:
        for epoch in pbar:
            epoch_loss = jnp.zeros(())
            for i in range(0, len(train_X), batch_size):
                x = train_X[i : i + batch_size]
                y = train_Y[i : i + batch_size]

                model, opt_state, loss = train_step(model, x, y, opt_state)
                epoch_loss += loss
                pbar.set_postfix({"epoch": epoch + 1, "loss": f"{loss:.5E}"})

            loss = epoch_loss / (len(train_X) // batch_size)
            loss_vals.append(loss)

            val_loss, _ = mse_loss_fn(model, test_X, test_Y)
            val_loss_vals.append(val_loss)
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.5E}, Val Loss: {val_loss:.5E}"
            )
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.5E}, Val Loss: {val_loss:.5E}"
                )

    if checkpoint_path is not None:
        save_model(
            filepath=checkpoint_path,
            datafilepath=data_path,
            model=model,
            names=input_keys,
            is_log=train_in_log,
        )  # type: ignore
        plt.plot(loss_vals, label="loss")
        plt.plot(val_loss_vals, label="val loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Average loss per epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(checkpoint_path + "_loss.png")
        plt.close("all")

        Y_hat = predict(model, data_X)
        total_loss = jnp.square(data_Y - Y_hat)
        total_loss = np.asarray(total_loss)  # type: ignore
        quantiles = np.quantile(total_loss, [0.05, 0.5, 0.95])
        total_loss = total_loss.tolist()  # type: ignore
        total_loss = sorted(total_loss)  # type: ignore

        plt.plot(total_loss, label="loss per data instance")
        plt.axhline(quantiles[0], color="red", linestyle="--", label="5% quantile")
        plt.axhline(quantiles[1], color="green", linestyle="--", label="50% quantile")
        plt.axhline(quantiles[2], color="blue", linestyle="--", label="95% quantile")
        plt.yscale("log")
        plt.xlabel("data instance")
        plt.ylabel("Loss per data instance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(checkpoint_path + "_total_loss.png")
        plt.close("all")
