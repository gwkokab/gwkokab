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

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Tuple
from typing_extensions import Optional

import numpy as np
import numpyro.distributions as dist
from flowMC.Sampler import Sampler
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, Int, PRNGKeyArray

from gwkokab.models.utils import JointDistribution
from gwkokab.parameters import Parameter
from gwkokab.vts import NeuralNetVolumeTimeSensitivity


def save_data_from_sampler(
    sampler: Sampler,
    *,
    out_dir: str,
    labels: Optional[list[str]] = None,
    n_samples: Int = 5000,
) -> None:
    """This functions saves the data from a sampler to disk. The data saved includes
    the samples from the flow, the chains from the training and production phases,
    the log probabilities, the global and local acceptance rates, and the loss
    values.

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


def log_weights_and_samples(
    key: PRNGKeyArray,
    parameters: Sequence[Parameter],
    vt_filename: str,
    num_samples: int,
    add_peak: bool = False,
) -> Tuple[Array, Array]:
    r"""Get the weights and samples from the VT.

    :param parameters: list of parameters
    :param vt_filename: VT filename
    :param num_samples: number of samples
    :param add_peak: whether to add a normal distribution peak, defaults to False
    :return: tuple of weights and samples
    """
    nvt = NeuralNetVolumeTimeSensitivity(
        [param.name for param in parameters], vt_filename
    )
    logVT_vmap = nvt.get_vmapped_logVT()
    hyper_uniform = JointDistribution(
        *[param.prior for param in parameters], validate_args=True
    )
    hyper_log_uniform = JointDistribution(
        *[
            dist.LogUniform(
                low=param.prior.low, high=param.prior.high, validate_args=True
            )
            for param in parameters
        ],
        validate_args=True,
    )

    uniform_key, proposal_key = jrd.split(key)
    component_distributions = [hyper_uniform, hyper_log_uniform]
    if add_peak:
        uniform_samples = hyper_uniform.sample(uniform_key, (num_samples,))

        logVT_val = logVT_vmap(uniform_samples)

        VT_max_at = jnp.argmax(logVT_val)
        loc_vector_at_highest_density = uniform_samples[VT_max_at]

        loc_vector_by_expectation = jnp.average(
            uniform_samples, axis=0, weights=jnp.exp(logVT_val)
        )
        covariance_matrix = jnp.cov(uniform_samples.T)
        component_distributions.append(
            JointDistribution(
                *[
                    dist.TruncatedNormal(
                        loc_vector_by_expectation[i],
                        jnp.sqrt(covariance_matrix[i, i]),
                        low=param.prior.low,
                        high=param.prior.high,
                        validate_args=True,
                    )
                    for i, param in enumerate(parameters)
                ],
                validate_args=True,
            )
        )
        component_distributions.append(
            JointDistribution(
                *[
                    dist.TruncatedNormal(
                        loc_vector_at_highest_density[i],
                        jnp.sqrt(covariance_matrix[i, i]),
                        low=param.prior.low,
                        high=param.prior.high,
                        validate_args=True,
                    )
                    for i, param in enumerate(parameters)
                ],
                validate_args=True,
            )
        )

    n = len(component_distributions)

    proposal_dist = dist.MixtureGeneral(
        dist.Categorical(probs=jnp.ones(n) / n, validate_args=True),
        component_distributions,
        support=hyper_uniform.support,
        validate_args=True,
    )

    proposal_samples = proposal_dist.sample(proposal_key, (num_samples,))

    mask = parameters[0].prior.support(proposal_samples[..., 0])
    for i in range(1, len(parameters)):
        mask &= parameters[i].prior.support(proposal_samples[..., i])

    proposal_samples = proposal_samples[mask]

    log_weights = logVT_vmap(proposal_samples) - proposal_dist.log_prob(
        proposal_samples
    )
    return log_weights, proposal_samples
