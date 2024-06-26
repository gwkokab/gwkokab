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
import warnings
from typing_extensions import Any, Optional

import equinox as eqx
import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import polars as pl
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from numpyro.util import is_prng_key
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table


@eqx.filter_value_and_grad
def loss_fn(model: PyTree, x: Float[Array, ""], y: Float[Array, ""]) -> Array:
    """Mean squared error loss function for the neural network.

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


@eqx.filter_jit
def compute_accuracy(model: PyTree, x: Float[Array, ""], y: Float[Array, ""]) -> Array:
    """Compute the accuracy of the model given the input and output data.

    :param model: Model to approximate the log of the VT function
    :param x: input data
    :param y: output data
    :return: accuracy of the model
    """
    y_pred = predict(model, x)
    return jnp.mean(jnp.square(y - y_pred))


def read_data(data_path: str) -> pl.DataFrame:
    """Read the data from the given path.

    :param data_path: path to the data
    :return: data in a DataFrame
    """
    with h5py.File(data_path, "r") as vt_file:
        keys = list(vt_file.keys())
        df = pl.DataFrame(
            data={
                key: pl.Series(key, np.array(vt_file[key]).flatten()) for key in keys
            },
        )
    return df


def train_test_split(
    X: Float[Array, ""], Y: Float[Array, ""], batch_size: int, test_size: float = 0.2
) -> tuple[Array, Array, Array, Array]:
    """Split the data into training and testing sets.

    :param X: Input data
    :param Y: Output data
    :param batch_size: batch size for training
    :param test_size: fraction of the data to use for testing, defaults to 0.2
    :return: training and testing sets
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


def make(
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
        layers = [
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
        eqx.nn.Lambda(jax.nn.relu),
    ]
    for i in range(len(hidden_layers) - 1):
        layers.append(
            eqx.nn.Linear(
                in_features=hidden_layers[i],
                out_features=hidden_layers[i + 1],
                key=keys[i],
            ),
        )
        layers.append(eqx.nn.Lambda(jax.nn.relu))
    layers.append(
        eqx.nn.Linear(
            in_features=hidden_layers[-1],
            out_features=output_layer,
            key=keys[-1],
        )
    )

    model = eqx.nn.Sequential(layers)

    return model


def save_model(*, filename: str, hyperparams: dict[str, Any], model) -> None:
    """Save the model to the given file.

    :param filename: Name of the file to save the model
    :param hyperparams: Hyperparameters of the model
    :param model: Model to approximate the log of the VT function
    """
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename) -> tuple[dict[str, Any], PyTree]:
    """Load the model from the given file.

    :param filename: Name of the file to load the model
    :return: Hyperparameters and the model
    """
    with open(filename, "rb") as f:
        hyperparam_str = f.readline().decode()
        hyperparams = json.loads(hyperparam_str)
        model = make(key=jrd.PRNGKey(0), **hyperparams)
        model = eqx.tree_deserialise_leaves(f, model)
    return hyperparams, model


def train_regressor(
    *,
    input_keys: list[str],
    output_keys: list[str],
    hidden_layers: list[int],
    batch_size: int,
    data_path: str,
    checkpoint_path: Optional[str] = None,
    epochs: int = 50,
    validation_split: float = 0.2,
    learning_rate: float = 1e-3,
    plot_loss: bool = True,
) -> None:
    """Train the model to approximate the log of the VT function.

    :param input_keys: list of input keys
    :param output_keys: list of output keys
    :param hidden_layers: list of hidden layers
    :param batch_size: batch size for training
    :param data_path: path to the data
    :param checkpoint_path: path to save the model
    :param epochs: number of epochs to train the model
    :param validation_split: fraction of the data to use for validation,
        defaults to 0.2
    :param learning_rate: learning rate for the optimizer, defaults to 1e-3
    :param plot_loss: whether to plot the loss, defaults to True
    :raises ValueError: if checkpoint path does not end with .eqx
    """
    if checkpoint_path is not None:
        if not checkpoint_path.endswith(".eqx"):
            raise ValueError("Checkpoint path must end with .eqx")
    else:
        warnings.warn("No checkpoint path provided, model will not be saved.")

    optimizer = optax.adam(learning_rate=learning_rate)

    @eqx.filter_jit
    def make_step(
        model: PyTree,
        x: Array,
        y: Array,
        opt_state: optax.OptState,
    ) -> tuple[PyTree, optax.OptState, Array]:
        """Make a step in the optimization process.

        :param model: Model to approximate the log of the VT function
        :param x: input data
        :param y: output data
        :param opt_state: optimizer state
        :return: optimizer state
        """
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_step(
        model: PyTree,
        x: Array,
        y: Array,
        opt_state: optax.OptState,
    ) -> tuple[PyTree, optax.OptState, Array]:
        """Train the model for one epoch.

        :param model: Model to approximate the log of the VT function
        :param x: input data
        :param y: output data
        :param opt_state: optimizer state
        :return: optimizer state
        """
        model, opt_state, loss = make_step(model, x, y, opt_state)
        return model, opt_state, loss

    df = read_data(data_path)

    data_X = jnp.asarray(df[input_keys].to_numpy())
    data_Y = df[output_keys].to_numpy()

    log_data_Y = jnp.log(data_Y)

    train_X, test_X, train_Y, test_Y = train_test_split(
        data_X,
        log_data_Y,
        batch_size,
        test_size=validation_split,
    )

    table = Table(title="Summary of the Neural Network Model", highlight=True)
    table.add_column("Parameter", justify="left")
    table.add_column("Value", justify="left")
    table.add_row("Input Keys", ", ".join(input_keys))
    table.add_row("Output Keys", ", ".join(output_keys))
    table.add_row("Hidden Layers", ", ".join(map(str, hidden_layers)))
    table.add_row("Data Path", data_path)
    table.add_row("Checkpoint Path", checkpoint_path)
    table.add_row("Train Size", str(len(train_X)))
    table.add_row("Test Size", str(len(test_X)))
    table.add_row("Validation Split", str(validation_split))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Epochs", str(epochs))
    table.add_row("Learning Rate", str(learning_rate))

    console = Console()
    console.print(table)

    model = make(
        key=jrd.PRNGKey(np.random.randint(0, 2**32 - 1)),
        input_layer=len(input_keys),
        output_layer=len(output_keys),
        hidden_layers=hidden_layers,
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_vals = []
    val_loss_vals = []

    with Progress(
        SpinnerColumn(),
        TextColumn("Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
        TextColumn("[progress.description]{task.description}"),
        refresh_per_second=5,
    ) as progress:
        total = int(len(train_X) // batch_size)
        for epoch in range(epochs):
            task_id = progress.add_task(
                "",
                total=total,
                epoch=epoch + 1,
            )
            epoch_loss = 0
            for i in range(0, len(train_X), batch_size):
                x = train_X[i : i + batch_size]
                y = train_Y[i : i + batch_size]

                model, opt_state, loss = train_step(model, x, y, opt_state)
                epoch_loss += loss
                progress.update(
                    task_id,
                    advance=1,
                    description=f"loss: {loss:.4f}",
                )

            loss = epoch_loss / (len(train_X) // batch_size)
            loss_vals.append(loss)

            val_loss = compute_accuracy(model, test_X, test_Y)
            val_loss_vals.append(val_loss)
            progress.update(
                task_id,
                completed=total,
                description=f"loss: {loss:.4f} - val loss: {val_loss:.4f}",
            )

    if checkpoint_path is not None:
        save_model(
            filename=checkpoint_path,
            hyperparams={
                "input_layer": len(input_keys),
                "output_layer": len(output_keys),
                "hidden_layers": hidden_layers,
            },
            model=model,
        )

    if plot_loss:
        plt.plot(loss_vals, label="loss")
        plt.plot(val_loss_vals, label="val loss")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss.png")
        plt.show()
