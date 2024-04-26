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
from datetime import datetime
from typing import Optional

import jax
import numpy as np
import plot_utils as pu
import variables as var
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.Sampler import Sampler
from jax import jit, numpy as jnp
from jaxtyping import Array
from numpyro import distributions as dist

from gwkokab.models import PowerLawPrimaryMassRatio
from gwkokab.utils import get_key


CURRENT_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # Year-Month-Day-Hour-Minute-Second

posteriors = glob.glob(var.POSTERIOR_REGEX)
data_set = {i: np.loadtxt(event) for i, event in enumerate(posteriors)}


# Prior distributions

labels = [
    r"$\alpha$",
    r"$\beta$",
    r"$\m_{\text{min}}$",
    r"$\m_{\text{max}}$",
]


alpha_prior_dist = dist.Uniform(low=0.0, high=5.0, validate_args=True)
beta_prior_dist = dist.Uniform(low=0.0, high=5.0, validate_args=True)
mmin_prior_dist = dist.Uniform(low=1.0, high=100.0, validate_args=True)
mmax_prior_dist = dist.Uniform(low=1.0, high=100.0, validate_args=True)


priors = [
    alpha_prior_dist,
    beta_prior_dist,
    mmin_prior_dist,
    mmax_prior_dist,
]


@jit
def sum_log_prior(x: Array) -> Array:
    return jnp.sum(jnp.asarray([prior.log_prob(x[i]) for i, prior in enumerate(priors)]))


initial_position = np.column_stack(
    jax.tree.map(
        lambda x: x.sample(key=get_key(), sample_shape=(var.N_CHAINS,)),
        priors,
        is_leaf=lambda x: isinstance(x, dist.Distribution),
    )
)


N_DIM = initial_position.shape[1]


model = MaskedCouplingRQSpline(N_DIM, var.N_LAYERS, var.HIDDEN_SIZE, var.NUM_BINS, get_key())


@jit
def likelihood_fn(x: Array, data: Optional[dict] = None):
    alpha = x[..., 0]
    beta = x[..., 1]
    mmin = x[..., 2]
    mmax = x[..., 3]

    m1q_model = PowerLawPrimaryMassRatio(
        alpha=alpha,
        beta=beta,
        mmin=mmin,
        mmax=mmax,
    )

    integral_individual = jax.tree.map(
        lambda y: jax.scipy.special.logsumexp(
            m1q_model.log_prob(jnp.column_stack((y[..., 0], y[..., 1] / y[..., 0]))) - jnp.log(y.shape[0])
        ),
        data["data"],
    )

    log_likelihood = jnp.sum(jnp.asarray(jax.tree.leaves(integral_individual)))

    log_prior = sum_log_prior(x)

    return log_likelihood + log_prior


MALA_Sampler = MALA(likelihood_fn, True, var.STEP_SIZE)

rng_key = get_key()

nf_sampler = Sampler(
    N_DIM,
    rng_key,
    None,
    MALA_Sampler,
    model,
    n_loop_training=var.N_LOOP_TRAINING,
    n_loop_production=var.N_LOOP_PRODUCTION,
    n_local_steps=var.N_LOCAL_STEPS,
    n_global_steps=var.N_GLOBAL_STEPS,
    n_chains=var.N_CHAINS,
    n_epochs=var.NUM_EPOCHS,
    learning_rate=var.LEARNING_RATE,
    momentum=var.MOMENTUM,
    batch_size=var.BATCH_SIZE,
    use_global=True,
)

nf_sampler.sample(initial_position, data={"data": data_set})

pu.plot(nf_sampler, f"reults/{CURRENT_TIME}", labels, N_DIM)
