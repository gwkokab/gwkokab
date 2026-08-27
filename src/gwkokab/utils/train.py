# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Training utilities for the neural selection-function regressors.

The ``neural_vt`` and ``neural_pdet`` Poisson-mean estimators consume an
:class:`equinox.nn.MLP` serialised to HDF5. This module provides the whole round
trip: reading a training set out of HDF5 (:func:`read_data`), building
(:func:`make_model`) and fitting (:func:`train_regressor`) the network, and
serialising it in the layout that :func:`load_model` -- and hence
:mod:`gwkokab.poisson_mean` -- expects.
"""

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


# ==============================
# Losses & Prediction
# ==============================


@eqx.filter_value_and_grad
@eqx.filter_jit
def mse_loss_fn(
    model: PyTree, x: Array, y: Array, batch_size: Optional[int] = 256
) -> Array:
    """Mean squared error loss.

    Wrapped in :func:`equinox.filter_value_and_grad`, so calling it returns both the
    loss and the gradients with respect to the inexact-array leaves of ``model``.

    Parameters
    ----------
    model : PyTree
        The network to evaluate; called once per row of ``x``.
    y : Array
        Target outputs, of the same shape as the model outputs.
    x : Array
        Input features, mapped over the leading axis.
    batch_size : Optional[int]
        Chunk size for :func:`jax.lax.map`. Defaults to ``256``.

    Returns
    -------
    Array
        Scalar mean squared error.
    """
    y_pred = jax.lax.map(model, x, batch_size=batch_size)
    return jnp.mean(jnp.square(y - y_pred))


@eqx.filter_value_and_grad
@eqx.filter_jit
def bce_logits_loss_fn(
    model: PyTree,
    x: Array,
    y: Array,
    batch_size: Optional[int] = 256,
    eps: float = 1e-6,
) -> Array:
    r"""Binary cross-entropy with logits (numerically stable).

    The model is treated as emitting logits; targets are expected in :math:`[0, 1]`
    and are clipped to :math:`[\varepsilon, 1 - \varepsilon]` to keep the loss finite.
    Wrapped in :func:`equinox.filter_value_and_grad`, so calling it returns both the
    loss and the gradients.

    Parameters
    ----------
    model : PyTree
        The network to evaluate; its outputs are interpreted as logits.
    x : Array
        Input features, mapped over the leading axis.
    y : Array
        Target probabilities in :math:`[0, 1]`.
    batch_size : Optional[int]
        Chunk size for :func:`jax.lax.map`. Defaults to ``256``.
    eps : float
        Clipping tolerance applied to ``y``. Defaults to ``1e-6``.

    Returns
    -------
    Array
        Scalar mean binary cross-entropy.
    """
    logits = jax.lax.map(model, x, batch_size=batch_size)
    y = jnp.clip(y, eps, 1.0 - eps)
    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=y)
    return jnp.mean(loss)


@eqx.filter_jit
def predict(model: PyTree, x: Array, batch_size: Optional[int] = 256) -> Array:
    """Predict outputs for inputs ``x``.

    Parameters
    ----------
    model : PyTree
        The network to evaluate.
    x : Array
        Input features, mapped over the leading axis.
    batch_size : Optional[int]
        Chunk size for :func:`jax.lax.map`. Defaults to ``256``.

    Returns
    -------
    Array
        Model outputs, stacked along the leading axis of ``x``.
    """
    return jax.lax.map(model, x, batch_size=batch_size)


# ==============================
# IO Helpers
# ==============================


def read_data(data_path: str, keys: Sequence[str]) -> pd.DataFrame:
    """Read an HDF5 dataset into a :class:`pandas.DataFrame`.

    Each requested key must name a dataset in the file; it is flattened and becomes one
    column of the frame.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file.
    keys : Sequence[str]
        Dataset names to read, which become the columns of the returned frame.

    Returns
    -------
    pd.DataFrame
        Frame whose columns are ``keys``, in the given order.
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
    """Build a multilayer perceptron with ReLU activations.

    Parameters
    ----------
    key : PRNGKeyArray
        PRNG key used to initialise the weights.
    input_layer : int
        Number of input features.
    output_layer : int
        Number of output features.
    width_size : int
        Number of units in each hidden layer.
    depth : int
        Number of hidden layers.

    Returns
    -------
    eqx.nn.MLP
        Freshly initialised network.
    """
    assert is_prng_key(key)
    return eqx.nn.MLP(
        in_size=input_layer,
        out_size=output_layer,
        width_size=width_size,
        depth=depth,
        activation=jnn.relu,
        key=key,
    )


def save_model(
    *,
    filepath: str,
    datafilepath: str,
    model: eqx.nn.MLP,
    names: Optional[Sequence[str]] = None,
    is_log: bool = False,
) -> None:
    """Persist model weights and metadata to HDF5.

    The layout is the one :func:`load_model` and the neural Poisson-mean estimators
    expect: scalar datasets for the architecture (``in_size``, ``out_size``,
    ``width_size``, ``depth``), one ``layer_<i>`` group per layer holding
    ``weight_<i>``/``bias_<i>``, the input parameter ``names``, and the ``is_log``
    attribute recording whether the network was trained on log-targets. Attributes of
    the training data file are copied across so the saved network carries its
    provenance.

    Parameters
    ----------
    filepath : str
        Destination path; must end in ``.hdf5``.
    datafilepath : str
        Path to the training-data HDF5 file whose attributes are copied over.
    model : eqx.nn.MLP
        The trained network, wrapped by :func:`equinox.filter_checkpoint`.
    names : Optional[Sequence[str]]
        Names of the input parameters, in input order. Defaults to :data:`None`, in
        which case no ``names`` dataset is written.
    is_log : bool
        Whether the network predicts the logarithm of the target. Defaults to
        :data:`False`.

    Raises
    ------
    ValueError
        If ``filepath`` does not end in ``.hdf5``.
    """
    if not filepath.endswith(".hdf5"):
        raise ValueError("Model save path must end with .hdf5")

    with h5py.File(datafilepath, "r") as f:
        attr = dict(f.attrs)

    model: eqx.nn.MLP = model._fun  # type: ignore

    compression_args = {"compression": "gzip", "compression_opts": 9}

    with h5py.File(filepath, "w") as f:
        for key, value in attr.items():
            f.attrs[key] = value
        if names is not None:
            f.create_dataset(
                "names", data=np.array(names, dtype="S"), **compression_args
            )
        f.create_dataset("in_size", data=model.in_size)
        f.create_dataset("out_size", data=model.out_size)
        f.create_dataset("width_size", data=model.width_size)
        f.create_dataset("depth", data=model.depth)
        f.attrs["is_log"] = is_log
        num_layers = len(model.layers)
        for i in range(num_layers):
            layer_i = f.create_group(f"layer_{i}")
            layer_i.create_dataset(
                f"weight_{i}", data=model.layers[i].weight, **compression_args
            )  # type: ignore
            layer_i.create_dataset(
                f"bias_{i}", data=model.layers[i].bias, **compression_args
            )  # type: ignore


def load_model(filename: str) -> Tuple[List[str], eqx.nn.MLP]:
    """Load a network and its input parameter names from HDF5.

    The architecture is rebuilt from the scalar datasets written by
    :func:`save_model` and the stored weights and biases are grafted onto it, so the
    returned network reproduces the trained one exactly.

    Parameters
    ----------
    filename : str
        Path to an HDF5 file written by :func:`save_model`.

    Returns
    -------
    Tuple[List[str], eqx.nn.MLP]
        The input parameter names and the reconstructed network.
    """
    with h5py.File(filename, "r") as f:
        names = f["names"][:].astype(str).tolist()
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
            key=jrd.key(0),
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


# ==============================
# Train/Validation Split
# ==============================


def _train_test_data_split(
    X: Array,
    Y: Array,
    batch_size: int,
    test_size: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[Array, Array, Array, Array]:
    """Split ``X``/``Y`` into train and validation sets, aligned to ``batch_size``.

    The split point is snapped to a multiple of ``batch_size`` so that every training
    epoch sees whole batches, which keeps the validation curve from jittering. Passing
    a ``seed`` makes the permutation -- and hence the validation set -- reproducible.

    Parameters
    ----------
    X : Array
        Input features, split along the leading axis.
    Y : Array
        Targets, split along the leading axis in the same order as ``X``.
    batch_size : int
        Batch size the split point is snapped to.
    test_size : float
        Fraction of the data held out for validation. Defaults to ``0.2``.
    seed : Optional[int]
        Seed for the shuffling RNG. Defaults to :data:`None`, which draws from the
        global NumPy RNG.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(train_X, test_X, train_Y, test_Y)``.
    """
    n = len(X)
    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)

    split = int(n * (1 - test_size))
    split = (
        max(split, batch_size) // batch_size
    ) * batch_size  # keep multiple of batch_size
    split = (
        min(split, n - batch_size) if n >= 2 * batch_size else max(batch_size, split)
    )

    train_indices = indices[:split]
    test_indices = indices[split:]
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]


# ==============================
# Trainer
# ==============================


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
    # New, optional stabilizers (keep old calls working):
    loss_type: str = "mse",  # "mse" (Mean Squared Error) or "bce_logits" (Binary Cross-Entropy with logits)
    grad_clip_norm: float = 1.0,
    weight_decay: float = 1e-4,
    use_cosine_decay: bool = True,
    min_lr: float = 1e-6,
    warmup_epochs: int = 3,
    seed: Optional[int] = 42,
) -> None:
    r"""Train an MLP regressor and save it to an HDF5 checkpoint.

    Optimisation uses AdamW with global-norm gradient clipping and, by default, a
    warmup-plus-cosine-decay learning-rate schedule. After training, the network is
    written with :func:`save_model` and the loss curves are saved alongside it.

    Parameters
    ----------
    input_keys : list[str]
        Datasets in ``data_path`` used as inputs; their order fixes the input order of
        the saved network.
    output_keys : list[str]
        Datasets in ``data_path`` used as targets.
    width_size : int
        Number of units in each hidden layer.
    depth : int
        Number of hidden layers.
    batch_size : int
        Mini-batch size.
    data_path : str
        Path to the training-data HDF5 file.
    checkpoint_path : Optional[str]
        Destination for the trained network; must end in ``.hdf5``. Required.
    epochs : int
        Number of training epochs. Defaults to ``50``.
    validation_split : float
        Fraction of the data held out for validation. Defaults to ``0.2``.
    learning_rate : float
        Peak learning rate. Defaults to ``1e-3``.
    train_in_log : bool
        Fit :math:`\log y` instead of :math:`y`. Defaults to :data:`False`.
    loss_type : str
        Either ``"mse"`` or ``"bce_logits"``. Defaults to ``"mse"``.
    grad_clip_norm : float
        Global gradient-norm clipping threshold. Defaults to ``1.0``.
    weight_decay : float
        AdamW weight decay. Defaults to ``1e-4``.
    use_cosine_decay : bool
        Use a warmup-plus-cosine-decay schedule rather than a constant learning rate.
        Defaults to :data:`True`.
    min_lr : float
        Final learning rate of the cosine decay. Defaults to ``1e-6``.
    warmup_epochs : int
        Number of warmup epochs. Defaults to ``3``.
    seed : Optional[int]
        Seed fixing the train/validation split. Defaults to ``42``.

    Raises
    ------
    ValueError
        If ``checkpoint_path`` is :data:`None` or does not end in ``.hdf5``, or if
        ``loss_type`` is not one of the recognised names.

    Notes
    -----
    - For detection probabilities in :math:`[0, 1]`, prefer ``loss_type="bce_logits"``
      and do *not* set ``train_in_log=True`` -- BCE expects probability targets, not
      log-values.
    - ``seed`` fixes the validation split, giving a less jittery validation loss.
    """
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided, model will not be saved.")
    if not checkpoint_path.endswith(".hdf5"):
        raise ValueError("Checkpoint path must end with .hdf5")

    # --------------------------
    # Read & prepare data
    # --------------------------
    df = read_data(data_path, keys=input_keys + output_keys)

    data_X = jax.device_put(df[input_keys].to_numpy())
    data_Y = jax.device_put(df[output_keys].to_numpy())

    if train_in_log:
        # Only for regression on log(y). Avoid for BCE loss.
        data_Y = jnp.log(data_Y)
        data_Y = jnp.where(
            jnp.isneginf(data_Y),
            -jnp.finfo(jnp.result_type(float)).tiny,
            data_Y,
        )

    train_X, test_X, train_Y, test_Y = _train_test_data_split(
        data_X, data_Y, batch_size, test_size=validation_split, seed=seed
    )

    logger.info("Input Keys: " + ", ".join(input_keys))
    logger.info("Output Keys: " + ", ".join(output_keys))
    logger.info(f"Width Size: {width_size}")
    logger.info(f"Depth: {depth}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Checkpoint Path: {checkpoint_path}")
    logger.info(f"Train Size: {len(train_X)}")
    logger.info(f"Test Size: {len(test_X)}")
    logger.info(f"Validation Split: {validation_split}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning Rate (peak): {learning_rate}")
    logger.info(f"Loss Type: {loss_type}")
    logger.info(f"Scheduler: {'cosine+warmup' if use_cosine_decay else 'constant'}")
    logger.info(
        f"Weight Decay: {weight_decay}, Grad Clip: {grad_clip_norm}, Seed: {seed}"
    )
    if train_in_log and loss_type.lower() == "bce_logits":
        logger.warning(
            "train_in_log=True with BCE is not recommended; BCE expects probabilities."
        )

    # --------------------------
    # Model & Optimizer
    # --------------------------
    model = make_model(
        key=jrd.key(np.random.randint(0, 2**32 - 1)),
        input_layer=len(input_keys),
        output_layer=len(output_keys),
        width_size=width_size,
        depth=depth,
    )
    model: Callable = eqx.filter_checkpoint(model)  # type: ignore

    steps_per_epoch = max(1, len(train_X) // batch_size)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(total_steps, max(0, warmup_epochs) * steps_per_epoch)

    if use_cosine_decay:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps),
            end_value=min_lr,
        )
    else:
        schedule = learning_rate

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Choose loss function
    loss_fn: Callable[..., PyTree]
    if loss_type.lower() == "mse":
        loss_fn = mse_loss_fn
    elif loss_type.lower() in ("bce", "bce_logits", "bcelogits"):
        loss_fn = bce_logits_loss_fn
    else:
        raise ValueError("loss_type must be 'mse' or 'bce_logits'")

    @eqx.filter_jit
    def make_step(
        model: eqx.nn.MLP, x: Array, y: Array, opt_state: optax.OptState
    ) -> tuple[eqx.nn.MLP, optax.OptState, Array]:
        """Run one optimiser step on a single mini-batch.

        Parameters
        ----------
        model : eqx.nn.MLP
            Current network.
        x : Array
            Mini-batch of inputs.
        y : Array
            Mini-batch of targets.
        opt_state : optax.OptState
            Current optimiser state.

        Returns
        -------
        tuple[eqx.nn.MLP, optax.OptState, Array]
            The updated network, the updated optimiser state, and the batch loss.
        """
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # --------------------------
    # Training Loop
    # --------------------------
    loss_vals: list[float] = []
    val_loss_vals: list[float] = []

    with tqdm.tqdm(range(epochs), unit="epochs") as pbar:
        for epoch in pbar:
            epoch_loss = jnp.zeros(())
            # mini-batch iterate
            for i in range(0, len(train_X), batch_size):
                x = train_X[i : i + batch_size]
                y = train_Y[i : i + batch_size]
                model, opt_state, loss = make_step(model, x, y, opt_state)
                epoch_loss = epoch_loss + loss
                pbar.set_postfix({"epoch": epoch + 1, "loss": f"{loss:.5E}"})

            # epoch metrics
            loss_epoch = epoch_loss / max(1, (len(train_X) // batch_size))
            loss_vals.append(float(loss_epoch))

            val_loss, _ = loss_fn(model, test_X, test_Y)
            val_loss_vals.append(float(val_loss))

            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss_epoch:.5E}, Val Loss: {val_loss:.5E}"
            )
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {loss_epoch:.5E}, Val Loss: {val_loss:.5E}"
                )

    # --------------------------
    # Save & Diagnostics
    # --------------------------
    save_model(
        filepath=checkpoint_path,
        datafilepath=data_path,
        model=model,
        names=input_keys,
        is_log=train_in_log,
    )  # type: ignore

    # Loss curves

    # Create subplots with shared x and y axes

    plt.rcParams.update({"font.size": 18})
    _, axes = plt.subplots(3, 1, figsize=(15, 15), dpi=300, sharex=True, sharey=True)

    import glasbey

    colors = glasbey.create_palette(palette_size=2)

    # Plot 1: Training loss
    axes[0].plot(loss_vals, label="loss", color=colors[0])
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_ylabel("Loss")

    # Plot 2: Validation loss
    axes[1].plot(val_loss_vals, label="val loss", color=colors[1])
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_ylabel("Validation Loss")

    # Plot 3: Combined
    axes[2].plot(loss_vals, label="loss", alpha=0.7, color=colors[0])
    axes[2].plot(val_loss_vals, label="val loss", alpha=0.7, color=colors[1])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Average loss per epoch")
    axes[2].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(checkpoint_path + "_loss.png")
    plt.close("all")

    # Per-instance loss distribution
    Y_hat = predict(model, data_X)
    per_instance = jnp.square(data_Y - Y_hat).squeeze(axis=-1)
    per_instance_np = np.asarray(per_instance)
    q05, q50, q95 = np.quantile(per_instance_np, [0.05, 0.5, 0.95])
    ordered = np.sort(per_instance_np)

    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 6), dpi=300)
    colors = glasbey.create_palette(palette_size=4)
    plt.plot(ordered, label="loss per data instance", color=colors[0])
    plt.axhline(q05, linestyle="--", label="5 percent quantile", color=colors[1])
    plt.axhline(q50, linestyle="--", label="50 percent quantile", color=colors[2])
    plt.axhline(q95, linestyle="--", label="95 percent quantile", color=colors[3])
    plt.yscale("log")
    plt.xlabel(r"$x_i$")
    plt.ylabel(r"$\left(y(x_i) - \hat{y}(x_i)\right)^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(checkpoint_path + "_total_loss.png")
    plt.close("all")
