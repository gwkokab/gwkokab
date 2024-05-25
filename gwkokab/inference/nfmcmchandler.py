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
import time
from typing_extensions import Self

import numpy as np
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.Sampler import Sampler
from jax import numpy as jnp

from ..utils import get_key
from .lippl import LogInhomogeneousPoissonProcessLikelihood


class NFMCMCHandler:
    def __init__(
        self: Self,
        *,
        posterior_regex: str,
        headers: list[str],
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
        self._headers = headers
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
        self._results_dir = results_dir + f"_{time.strftime('%Y%m%d_%H%M%S')}"

    def get_column_from_keys(self: Self, input_keys: list[str], headers: list[str]) -> list[int]:
        return [headers.index(key) for key in input_keys]

    def load_dataset(self: Self) -> dict[int, np.ndarray]:
        columns = self.get_column_from_keys(self._likelihood_obj.input_keys, self._headers)
        posterior_files = glob.glob(self._posterior_regex)
        data_set = {
            "data": {i: np.loadtxt(file)[..., columns] for i, file in enumerate(posterior_files)},
            "N": len(posterior_files),
        }
        return data_set

    def run_sampler(self: Self) -> Sampler:
        model = MaskedCouplingRQSpline(
            self._likelihood_obj.n_dim,
            self._n_layers,
            self._hidden_size,
            self._num_bins,
            get_key(),
        )
        MALA_Sampler = MALA(
            self._likelihood_obj.log_posterior,
            True,
            self._step_size,
        )

        initial_positions = jnp.column_stack(
            [prior.sample(get_key(), (self._n_chains,)) for prior in self._likelihood_obj.priors]
        )

        nf_sampler = Sampler(
            self._likelihood_obj.n_dim,
            get_key(),
            self.load_dataset(),
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

    def save_data(self, nf_sampler: Sampler) -> None:
        labels = [self._likelihood_obj.labels[i] for i in range(self._likelihood_obj.n_dim)]

        out_train = nf_sampler.get_sampler_state(training=True)

        train_chains = np.array(out_train["chains"])
        train_global_accs = np.array(out_train["global_accs"])
        train_local_accs = np.array(out_train["local_accs"])
        train_loss_vals = np.array(out_train["loss_vals"])
        train_log_prob = np.array(out_train["log_prob"])

        out_prod = nf_sampler.get_sampler_state(training=False)

        prod_chains = np.array(out_prod["chains"])
        prod_global_accs = np.array(out_prod["global_accs"])
        prod_local_accs = np.array(out_prod["local_accs"])
        prod_log_prob = np.array(out_prod["log_prob"])

        os.makedirs(self._results_dir, exist_ok=True)

        nf_samples = np.array(nf_sampler.sample_flow(n_samples=5000, rng_key=get_key()))
        np.savetxt(
            rf"{self._results_dir}/nf_samples.dat",
            nf_samples,
            header=" ".join(labels),
        )

        for i in range(self._n_chains):
            np.savetxt(
                rf"{self._results_dir}/train_chains_{i}.dat",
                train_chains[i, :, :],
                header=" ".join(labels),
            )
            np.savetxt(
                rf"{self._results_dir}/prod_chains_{i}.dat",
                prod_chains[i, :, :],
                header=" ".join(labels),
            )
            np.savetxt(
                rf"{self._results_dir}/log_prob_{i}.dat",
                np.column_stack((train_log_prob[i, :], prod_log_prob[i, :])),
                header="train prod",
                comments="#",
            )
            np.savetxt(
                rf"{self._results_dir}/global_accs_{i}.dat",
                np.column_stack((train_global_accs[i, :], prod_global_accs[i, :])),
                header="train prod",
                comments="#",
            )
            np.savetxt(
                rf"{self._results_dir}/local_accs_{i}.dat",
                np.column_stack((train_local_accs[i, :], prod_local_accs[i, :])),
                header="train prod",
                comments="#",
            )
            np.savetxt(
                rf"{self._results_dir}/loss_{i}.dat",
                train_loss_vals[i, :],
                header="loss",
            )

    def run(self) -> None:
        nf_sampler = self.run_sampler()
        self.save_data(nf_sampler)
