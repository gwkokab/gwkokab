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


import glob
import os
import sys
from datetime import datetime
from typing import Optional

import corner
import jax
import numpy as np
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.Sampler import Sampler
from jax import jit, numpy as jnp
from jax.scipy.stats import beta, multivariate_normal, truncnorm
from jaxtyping import Array
from matplotlib import pyplot as plt
from numpyro import distributions as dist


sys.path.append("gwkokab")

from gwkokab.models import *
from gwkokab.utils import get_key


CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

POSTERIOR_REGEX = "/home/muhammad.zeeshan/o4a-analysis/data/o3bns/*.dat"  # set the path to the posterior files regex
posteriors = glob.glob(POSTERIOR_REGEX)
data_set = {i: np.loadtxt(event) for i, event in enumerate(posteriors)}

N_CHAINS = 250


# MaskedCouplingRQSpline parameters

N_LAYERS = 5
HIDDEN_SIZE = [32, 32]
NUM_BINS = 5

# MALA Sampler parameters

STEP_SIZE = 1e-1

# Sampler parameters

N_LOOP_TRAINING = 500
N_LOOP_PRODUCTION = 500
N_LOCAL_STEPS = 150
N_GLOBAL_STEPS = 20
NUM_EPOCHS = 5

LEARNING_RATE = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 5000
MAX_SAMPLES = 5000


# Prior distributions

labels = [
    r"$\mu_{m_1}$",
    r"$\mu_{m_2}$",
    r"$\sigma_{m_1}$",
    r"$\sigma_{m_2}$",
    r"$E[\chi_1]$",
    r"$E[\chi_2]$",
    r"$\sigma_{\chi_1}$",
    r"$\sigma_{\chi_2}$",
    r"$\sigma_{\cos{(\theta_1)}}$",
    r"$\sigma_{\cos{(\theta_2)}}$",
]


mu_mass_1_prior_dist = dist.Uniform(low=1.0, high=3.0, validate_args=True)
mu_mass_2_prior_dist = dist.Uniform(low=1.0, high=3.0, validate_args=True)
sigma_mass_1_prior_dist = dist.Uniform(low=0.05, high=2.0, validate_args=True)
sigma_mass_2_prior_dist = dist.Uniform(low=0.05, high=2.0, validate_args=True)
E_chi_1_prior_dist = dist.Uniform(low=0, high=1, validate_args=True)
E_chi_2_prior_dist = dist.Uniform(low=0, high=1, validate_args=True)
sigma_chi_1_prior_dist = dist.Uniform(low=0, high=0.25, validate_args=True)
sigma_chi_2_prior_dist = dist.Uniform(low=0, high=0.25, validate_args=True)
sigma_cos_theta_1_prior_dist = dist.Uniform(low=0, high=4, validate_args=True)
sigma_cos_theta_2_prior_dist = dist.Uniform(low=0, high=4, validate_args=True)


priors = [
    mu_mass_1_prior_dist,
    mu_mass_2_prior_dist,
    sigma_mass_1_prior_dist,
    sigma_mass_2_prior_dist,
    E_chi_1_prior_dist,
    E_chi_2_prior_dist,
    sigma_chi_1_prior_dist,
    sigma_chi_2_prior_dist,
    sigma_cos_theta_1_prior_dist,
    sigma_cos_theta_2_prior_dist,
]


initial_position = np.column_stack(
    jax.tree.map(
        lambda x: x.sample(key=get_key(), sample_shape=(N_CHAINS,)),
        priors,
        is_leaf=lambda x: isinstance(x, dist.Distribution),
    )
)


N_DIM = initial_position.shape[1]


model = MaskedCouplingRQSpline(N_DIM, N_LAYERS, HIDDEN_SIZE, NUM_BINS, get_key())


@jit
def get_alpha_beta(mu: Array, sigma: Array) -> Array:
    r"""Transforms expectation and variance into alpha and
    beta parameters of a Beta distribution.

    .. math::

        \alpha = \mu \left( \frac{\mu(1-\mu)}{\sigma^2} - 1 \right)
        \beta = \alpha \left( \frac{1}{\mu} - 1 \right)

    :param mu: expectation
    :param sigma: variance
    :return: alpha, beta
    """
    alpha = mu * (mu * (1 - mu) * jnp.power(sigma, -2.0) - 1)
    beta = alpha * (jnp.reciprocal(mu) - 1)
    return alpha, beta


@jit
def likelihood_fn(x: Array, data: Optional[dict] = None):
    mu_mass_1 = x[..., 0]
    mu_mass_2 = x[..., 1]
    sigma_mass_1 = x[..., 2]
    sigma_mass_2 = x[..., 3]
    E_chi_1 = x[..., 4]
    E_chi_2 = x[..., 5]
    sigma_chi_1 = x[..., 6]
    sigma_chi_2 = x[..., 7]
    sigma_cos_theta_1 = x[..., 8]
    sigma_cos_theta_2 = x[..., 9]

    alpha1, beta1 = get_alpha_beta(E_chi_1, sigma_chi_1)
    alpha2, beta2 = get_alpha_beta(E_chi_2, sigma_chi_2)

    integral_individual = jax.tree.map(
        lambda y: jax.scipy.special.logsumexp(
            # log likelihood of mass model
            multivariate_normal.logpdf(
                y[..., [0, 1]],
                jnp.array([mu_mass_1, mu_mass_2]),
                jnp.array([[sigma_mass_1**2, 0], [0, sigma_mass_2**2]]),
            )
            # log likelihood of spin1 model
            + beta.logpdf(
                y[..., 2],
                a=alpha1,
                b=beta1,
            )
            # log likelihood of spin2 model
            + beta.logpdf(
                y[..., 3],
                a=alpha2,
                b=beta2,
            )
            # log likelihood of tilt1 model
            + truncnorm.logpdf(
                y[..., 4],
                a=-1,
                b=1,
                loc=0,
                scale=sigma_cos_theta_1,
            )
            # log likelihood of tilt2 model
            + truncnorm.logpdf(
                y[..., 5],
                a=-1,
                b=1,
                loc=0,
                scale=sigma_cos_theta_2,
            )
            - jnp.log(y.shape[0])
        ),
        data["data"],
    )

    log_likelihood = jnp.sum(jnp.asarray(jax.tree.leaves(integral_individual)))

    log_prior = (
        mu_mass_1_prior_dist.log_prob(mu_mass_1)
        + mu_mass_2_prior_dist.log_prob(mu_mass_2)
        + sigma_mass_1_prior_dist.log_prob(sigma_mass_1)
        + sigma_mass_2_prior_dist.log_prob(sigma_mass_2)
        + E_chi_1_prior_dist.log_prob(E_chi_1)
        + E_chi_2_prior_dist.log_prob(E_chi_2)
        + sigma_chi_1_prior_dist.log_prob(sigma_chi_1)
        + sigma_chi_2_prior_dist.log_prob(sigma_chi_2)
        + sigma_cos_theta_1_prior_dist.log_prob(sigma_cos_theta_1)
        + sigma_cos_theta_2_prior_dist.log_prob(sigma_cos_theta_2)
    )

    return log_likelihood + log_prior


MALA_Sampler = MALA(likelihood_fn, True, STEP_SIZE)

rng_key_set = get_key()


nf_sampler = Sampler(
    N_DIM,
    rng_key_set,
    None,
    MALA_Sampler,
    model,
    n_loop_training=N_LOOP_TRAINING,
    n_loop_production=N_LOOP_PRODUCTION,
    n_local_steps=N_LOCAL_STEPS,
    n_global_steps=N_GLOBAL_STEPS,
    n_chains=N_CHAINS,
    n_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM,
    batch_size=BATCH_SIZE,
    use_global=True,
)

nf_sampler.sample(initial_position, data={"data": data_set})


out_train = nf_sampler.get_sampler_state(training=True)
print("Logged during tuning:", out_train.keys())


train_chains = np.array(out_train["chains"])
train_global_accs = np.array(out_train["global_accs"])
train_local_accs = np.array(out_train["local_accs"])
train_loss_vals = np.array(out_train["loss_vals"])
train_log_prob = np.array(out_train["log_prob"])
train_nf_samples = np.array(nf_sampler.sample_flow(n_samples=5000, rng_key=get_key()))

os.makedirs(rf"results/{CURRENT_TIME}", exist_ok=True)

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

figure.savefig(rf"results/{CURRENT_TIME}/nf_samples_train.png")
plt.close()

out_prod = nf_sampler.get_sampler_state(training=False)
print("Logged during tuning:", out_prod.keys())


prod_chains = np.array(out_prod["chains"])
prod_global_accs = np.array(out_prod["global_accs"])
prod_local_accs = np.array(out_prod["local_accs"])
prod_log_prob = np.array(out_prod["log_prob"])
prod_nf_samples = np.array(nf_sampler.sample_flow(n_samples=10000, rng_key=get_key()))


# log_prob.shape
fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
for i in range(N_CHAINS):
    axes[0].plot(train_log_prob[i, :])
    axes[1].plot(prod_log_prob[i, :])
fig.suptitle("Log of Probability [Training (Up) Production (Down)]")
fig.savefig(rf"results/{CURRENT_TIME}/log_prob.png")
plt.close()


fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for i in range(N_CHAINS):
    axes[0].plot(train_local_accs[i, :])
    axes[1].plot(prod_local_accs[i, :])
fig.suptitle("Local Accs [Training (Up) Production (Down)]")
fig.savefig(rf"results/{CURRENT_TIME}/local_accs.png")
plt.close()


fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for i in range(N_CHAINS):
    axes[0].plot(train_global_accs[i, :])
    axes[1].plot(prod_global_accs[i, :])
fig.suptitle("Global Accs [Training (Up) Production (Down)]")
fig.savefig(rf"results/{CURRENT_TIME}/global_accs.png")
plt.close()

for i in range(N_CHAINS):
    plt.plot(train_loss_vals[i, :], label=f"chain {i}")
plt.xlabel("num_epoch")
plt.ylabel("loss")
plt.savefig(rf"results/{CURRENT_TIME}/loss.png")
plt.close()

fig, axes = plt.subplots(N_DIM, 2, figsize=(20, 20), sharex=True)
for j in range(N_DIM):
    for i in range(N_CHAINS):
        axes[j][0].plot(train_chains[i, :, j], alpha=0.5)
        axes[j][0].set_ylabel(labels[j])

        axes[j][1].plot(prod_chains[i, :, j], alpha=0.5)
        axes[j][1].set_ylabel(labels[j])
axes[-1][0].set_xlabel("Iteration")
axes[-1][1].set_xlabel("Iteration")
plt.suptitle("Chains\n[Training (Left) Production (Right)]")
fig.savefig(rf"results/{CURRENT_TIME}/chains.png")
plt.close()


np.savetxt(rf"results/{CURRENT_TIME}/nf_samples_train.dat", train_nf_samples, header=" ".join(labels))
np.savetxt(rf"results/{CURRENT_TIME}/initial_position.dat", initial_position, header=" ".join(labels))
np.savetxt(rf"results/{CURRENT_TIME}/chains_prod.dat", prod_chains, header=" ".join(labels))
np.savetxt(rf"results/{CURRENT_TIME}/chains_train.dat", train_chains, header=" ".join(labels))


print("\n\n\n\n\nEverything is saved at: ", f"results/{CURRENT_TIME}")
