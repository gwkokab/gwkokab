#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import os
from typing_extensions import Optional

import numpy as np
from flowMC.Sampler import Sampler
from jax import random as jrd
from jaxtyping import Int


def save_data_from_sampler(
    sampler: Sampler,
    *,
    out_dir: str,
    labels: Optional[list[str]] = None,
    n_samples: Int = 5000,
) -> None:
    """This functions saves the data from a sampler to disk. The data saved
    includes the samples from the flow, the chains from the training and
    production phases, the log probabilities, the global and local acceptance
    rates, and the loss values.

    :param sampler: Sampler object
    :param out_dir: path to the output directory
    :param labels: list of labels for the samples, defaults to None
    :param n_samples: number of samples to draw from the flow, defaults to 5000
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
        np.column_stack((train_global_accs.mean(0), prod_global_accs.mean(0))),
        header="train prod",
        comments="#",
    )
    np.savetxt(
        rf"{out_dir}/local_accs.dat",
        np.column_stack((train_local_accs.mean(0), prod_local_accs.mean(0))),
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
            np.column_stack((train_log_prob[i, :], prod_log_prob[i, :])),
            header="train prod",
            comments="#",
        )
