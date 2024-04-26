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

import os

import corner
import numpy as np
import variables as var
from matplotlib import pyplot as plt

from gwkokab.utils import get_key


def plot(nf_sampler, dir_path, labels, N_DIM):
    out_train = nf_sampler.get_sampler_state(training=True)
    print("Logged during tuning:", out_train.keys())

    train_chains = np.array(out_train["chains"])
    train_global_accs = np.array(out_train["global_accs"])
    train_local_accs = np.array(out_train["local_accs"])
    train_loss_vals = np.array(out_train["loss_vals"])
    train_log_prob = np.array(out_train["log_prob"])
    train_nf_samples = np.array(nf_sampler.sample_flow(n_samples=5000, rng_key=get_key()))

    os.makedirs(dir_path, exist_ok=True)

    # Plot Nf samples
    figure = corner.corner(
        train_nf_samples,
        labels=labels,
        bins=50,
        show_titles=True,
        smooth=True,
        quantiles=(0.25, 0.5, 0.75),
    )
    figure.set_size_inches(15, 15)
    figure.suptitle("Visualize NF samples (Training)")

    figure.savefig(rf"{dir_path}/nf_samples_train.png")
    plt.close()

    out_prod = nf_sampler.get_sampler_state(training=False)
    print("Logged during tuning:", out_prod.keys())

    prod_chains = np.array(out_prod["chains"])
    prod_global_accs = np.array(out_prod["global_accs"])
    prod_local_accs = np.array(out_prod["local_accs"])
    prod_log_prob = np.array(out_prod["log_prob"])

    # log_prob.shape
    fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
    for i in range(var.N_CHAINS):
        axes[0].plot(train_log_prob[i, :])
        axes[1].plot(prod_log_prob[i, :])
    fig.suptitle("Log of Probability [Training (Up) Production (Down)]")
    fig.savefig(rf"{dir_path}/log_prob.png")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(var.N_CHAINS):
        axes[0].plot(train_local_accs[i, :])
        axes[1].plot(prod_local_accs[i, :])
    fig.suptitle("Local Accs [Training (Up) Production (Down)]")
    fig.savefig(rf"{dir_path}/local_accs.png")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(var.N_CHAINS):
        axes[0].plot(train_global_accs[i, :])
        axes[1].plot(prod_global_accs[i, :])
    fig.suptitle("Global Accs [Training (Up) Production (Down)]")
    fig.savefig(rf"{dir_path}/global_accs.png")
    plt.close()

    for i in range(var.N_CHAINS):
        plt.plot(train_loss_vals[i, :], label=f"chain {i}")
    plt.xlabel("num_epoch")
    plt.ylabel("loss")
    plt.savefig(rf"{dir_path}/loss.png")
    plt.close()

    fig, axes = plt.subplots(N_DIM, 2, figsize=(20, 20), sharex=True)
    for j in range(N_DIM):
        for i in range(var.N_CHAINS):
            axes[j][0].plot(train_chains[i, :, j], alpha=0.5)
            axes[j][0].set_ylabel(labels[j])

            axes[j][1].plot(prod_chains[i, :, j], alpha=0.5)
            axes[j][1].set_ylabel(labels[j])
    axes[-1][0].set_xlabel("Iteration")
    axes[-1][1].set_xlabel("Iteration")
    plt.suptitle("Chains\n[Training (Left) Production (Right)]")
    fig.savefig(rf"{dir_path}/chains.png")
    plt.close()

    np.savetxt(rf"{dir_path}/nf_samples_train.dat", train_nf_samples, header=" ".join(labels))
    np.savetxt(rf"{dir_path}/chains_prod.dat", prod_chains, header=" ".join(labels))
    np.savetxt(rf"{dir_path}/chains_train.dat", train_chains, header=" ".join(labels))

    print("Plots saved in", dir_path)
