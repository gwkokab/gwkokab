# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing_extensions import Optional

import numpy as np
from flowMC.Sampler import Sampler
from jax import random as jrd


def _same_length_arrays(length: int, *arrays: np.ndarray) -> tuple[np.ndarray]:
    """This function pads the arrays with None to make them the same length.

    Parameters
    ----------
    length : int
        The length of the arrays.
    arrays : np.ndarray
        The arrays to pad.

    Returns
    -------
    tuple[np.ndarray]
        The padded arrays.
    """
    padded_arrays = []
    for array in arrays:
        padded_array = np.empty((length,))
        padded_array[..., : array.shape[0]] = array
        padded_array[..., array.shape[0] :] = None
        padded_arrays.append(padded_array)
    return tuple(padded_arrays)


def save_data_from_sampler(
    sampler: Sampler,
    *,
    out_dir: str,
    labels: Optional[list[str]] = None,
    n_samples: int = 5000,
) -> None:
    """This functions saves the data from a sampler to disk. The data saved includes the
    samples from the flow, the chains from the training and production phases, the log
    probabilities, the global and local acceptance rates, and the loss values.

    Parameters
    ----------
    sampler : Sampler
        The sampler object.
    out_dir : str
        The output directory.
    labels : Optional[list[str]], optional
        list of labels for the samples, by default None
    n_samples : int, optional
        number of samples to draw from the flow, by default 5000
    """
    if labels is None:
        labels = [f"x{i}" for i in range(sampler.n_dim)]

    out_train = sampler.get_sampler_state(training=True)

    train_chains = np.array(out_train["chains"])
    train_global_accs = np.array(out_train["global_accs"])
    train_local_accs = np.array(out_train["local_accs"])
    train_loss_vals = np.array(out_train["loss_vals"])
    train_log_prob = np.array(out_train["log_prob"])

    out_prod = sampler.get_sampler_state(training=False)

    prod_chains = np.array(out_prod["chains"])
    prod_global_accs = np.array(out_prod["global_accs"])
    prod_local_accs = np.array(out_prod["local_accs"])
    prod_log_prob = np.array(out_prod["log_prob"])

    os.makedirs(out_dir, exist_ok=True)

    samples = np.array(
        sampler.sample_flow(
            n_samples=n_samples,
            rng_key=jrd.PRNGKey(np.random.randint(1, 2**32 - 1)),
        )
    )

    header = " ".join(labels)

    np.savetxt(rf"{out_dir}/nf_samples.dat", samples, header=header)

    n_chains = sampler.n_chains

    np.savetxt(
        rf"{out_dir}/global_accs.dat",
        np.column_stack(
            _same_length_arrays(
                max(train_global_accs.shape[1], prod_global_accs.shape[1]),
                train_global_accs.mean(0),
                prod_global_accs.mean(0),
            )
        ),
        header="train prod",
        comments="#",
    )
    np.savetxt(
        rf"{out_dir}/local_accs.dat",
        np.column_stack(
            _same_length_arrays(
                max(train_local_accs.shape[1], prod_local_accs.shape[1]),
                train_local_accs.mean(0),
                prod_local_accs.mean(0),
            )
        ),
        header="train prod",
        comments="#",
    )

    np.savetxt(rf"{out_dir}/loss.dat", train_loss_vals.reshape(-1), header="loss")

    for i in range(n_chains):
        np.savetxt(
            rf"{out_dir}/train_chains_{i}.dat",
            train_chains[i, :, :],
            header=header,
        )
        np.savetxt(
            rf"{out_dir}/prod_chains_{i}.dat",
            prod_chains[i, :, :],
            header=header,
        )
        np.savetxt(
            rf"{out_dir}/log_prob_{i}.dat",
            np.column_stack(
                _same_length_arrays(
                    max(train_log_prob[i].shape[0], prod_log_prob[i].shape[0]),
                    train_log_prob[i],
                    prod_log_prob[i],
                )
            ),
            header="train prod",
            comments="#",
        )
