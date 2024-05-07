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

import glob
import os

import corner
import numpy as np
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.Sampler import Sampler
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ..utils import get_key
from .lippl import LogInhomogeneousPoissonProcessLikelihood


class NFMCMCHandler:
    def __init__(
        self,
        *,
        posterior_regex: str,
        likelihood_obj: LogInhomogeneousPoissonProcessLikelihood,
        #
        n_chains: int = 4,
        #
        step_size: float = 1e-1,
        #
        n_layers: int = 5,
        hidden_size: list[int] = [32, 32],
        num_bins: int = 8,
        #
        n_loop_training: int = 100,
        n_loop_production: int = 100,
        n_local_steps: int = 100,
        n_global_steps: int = 10,
        num_epochs: int = 5,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        batch_size: int = 5000,
        max_samples: int = 5000,
        #
        results_dir: str = "results",
    ) -> None:
        self._posterior_regex = posterior_regex
        self._likelihood_obj = likelihood_obj
        self._n_chains = n_chains
        self._step_size = step_size
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._num_bins = num_bins
        self._n_loop_training = n_loop_training
        self._n_loop_production = n_loop_production
        self._n_local_steps = n_local_steps
        self._n_global_steps = n_global_steps
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._batch_size = batch_size
        self._max_samples = max_samples
        self._use_global = True
        self._results_dir = results_dir

    def load_dataset(self) -> dict[int, np.ndarray]:
        posterior_files = glob.glob(self._posterior_regex)
        data_set = {
            "data": {i: np.loadtxt(file) for i, file in enumerate(posterior_files)},
            "N": len(posterior_files),
        }
        return data_set

    def run_sampler(self) -> Sampler:
        model = MaskedCouplingRQSpline(
            self._likelihood_obj.n_dim,
            self._n_layers,
            self._hidden_size,
            self._num_bins,
            get_key(),
        )
        MALA_Sampler = MALA(
            self._likelihood_obj.likelihood,
            True,
            self._step_size,
        )

        initial_positions = jnp.column_stack(
            [prior.sample(get_key(), (self._n_chains,)) for prior in self._likelihood_obj.priors]
        )

        nf_sampler = Sampler(
            self._likelihood_obj.n_dim,
            get_key(),
            None,
            MALA_Sampler,
            model,
            n_loop_training=self._n_loop_training,
            n_loop_production=self._n_loop_production,
            n_local_steps=self._n_local_steps,
            n_global_steps=self._n_global_steps,
            n_chains=self._n_chains,
            n_epochs=self._num_epochs,
            learning_rate=self._learning_rate,
            momentum=self._momentum,
            batch_size=self._batch_size,
            use_global=self._use_global,
        )

        nf_sampler.sample(
            initial_positions,
            data=self.load_dataset(),
        )

        return nf_sampler

    def generate_plots(self, nf_sampler: Sampler) -> None:
        out_train = nf_sampler.get_sampler_state(training=True)
        print("Logged during tuning:", out_train.keys())
        train_chains = np.array(out_train["chains"])
        train_global_accs = np.array(out_train["global_accs"])
        train_local_accs = np.array(out_train["local_accs"])
        train_loss_vals = np.array(out_train["loss_vals"])
        train_log_prob = np.array(out_train["log_prob"])
        train_nf_samples = np.array(nf_sampler.sample_flow(n_samples=5000, rng_key=get_key()))
        labels = [self._likelihood_obj.labels[i] for i in range(self._likelihood_obj.n_dim)]

        os.makedirs(self._results_dir, exist_ok=True)
        np.savetxt(
            rf"{self._results_dir}/nf_samples_train.dat",
            train_nf_samples,
            header=" ".join(labels),
        )

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

        figure.savefig(rf"{self._results_dir}/nf_samples_train.png")

        out_prod = nf_sampler.get_sampler_state(training=False)
        print("Logged during tuning:", out_prod.keys())

        prod_chains = np.array(out_prod["chains"])
        prod_global_accs = np.array(out_prod["global_accs"])
        prod_local_accs = np.array(out_prod["local_accs"])
        prod_log_prob = np.array(out_prod["log_prob"])

        # log_prob.shape
        fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
        for i in range(self._n_chains):
            axes[0].plot(train_log_prob[i, :])
            axes[1].plot(prod_log_prob[i, :])
        plt.suptitle("Log of Probability [Training (Up) Production (Down)]")
        fig.savefig(rf"{self._results_dir}/log_prob.png")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i in range(self._n_chains):
            axes[0].plot(train_local_accs[i, :])
            axes[1].plot(prod_local_accs[i, :])
        plt.suptitle("Local Accs [Training (Up) Production (Down)]")
        fig.savefig(rf"{self._results_dir}/local_accs.png")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i in range(self._n_chains):
            axes[0].plot(train_global_accs[i, :])
            axes[1].plot(prod_global_accs[i, :])
        plt.suptitle("Global Accs [Training (Up) Production (Down)]")
        fig.savefig(rf"{self._results_dir}/global_accs.png")
        plt.close("all")

        for i in range(self._n_chains):
            plt.plot(train_loss_vals[i, :], label=f"chain {i}")
        plt.xlabel("num_epoch")
        plt.ylabel("loss")
        plt.savefig(rf"{self._results_dir}/loss.png")

        fig, axes = plt.subplots(
            self._likelihood_obj.n_dim,
            2,
            figsize=(20, 20),
            sharex=True,
        )
        for j in range(self._likelihood_obj.n_dim):
            for i in range(self._n_chains):
                axes[j][0].plot(train_chains[i, :, j], alpha=0.5)
                axes[j][0].set_ylabel(labels[j])

                axes[j][1].plot(prod_chains[i, :, j], alpha=0.5)
                axes[j][1].set_ylabel(labels[j])
        axes[-1][0].set_xlabel("Iteration")
        axes[-1][1].set_xlabel("Iteration")
        plt.suptitle("Chains\n[Training (Left) Production (Right)]")
        fig.savefig(rf"{self._results_dir}/chains.png")

    def run(self) -> None:
        nf_sampler = self.run_sampler()
        self.generate_plots(nf_sampler)
