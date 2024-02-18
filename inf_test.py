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


# import numpy as np

# from gwkokab.inference.model_test import expval_mc
# from gwkokab.vts.utils import interpolate_hdf5, load_hdf5, mass_grid_coords

# total_events = 100
# posterior_size = 100
# posterior_regex = "realization_6/posteriors/event_{}.dat"
# injection_regex = "realization_6/injections/event_{}.dat"
# true_values = np.loadtxt("realization_6/configuration.dat")[:3]


# expval = expval_mc(*true_values, 1.0, interpolate_hdf5(load_hdf5("mass_vt.hdf5")))
# print(expval)


import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from numpyro import handlers
from numpyro.distributions.util import validate_sample
from numpyro.infer import MCMC, NUTS

# X = np.random.randn(128, 3)
# y = np.random.randn(128)


# def model(X, y):
#     beta = numpyro.sample("beta", dist.Normal(0, 1).expand([3]))
#     numpyro.sample("obs", dist.Normal(X @ beta, 1), obs=y)


# def model(x):
#     numpyro.sample("obs", dist.Uniform(0.0, 1.0), obs=x)


# mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
# # See https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
# sharding = PositionalSharding(mesh_utils.create_device_mesh((1,)))
# X_shard = jax.device_put(
#     X,
#     sharding.reshape(
#         1,
#     ),
# )
# # y_shard = jax.device_put(y, sharding.reshape(1))
# mcmc.run(jax.random.PRNGKey(0), x=X)

# print(mcmc.get_samples())

# print(mcmc.print_summary())


def importance_sampling(target_dist, proposal_dist, num_samples):
    # Draw samples from the proposal distribution
    samples = proposal_dist.sample(jax.random.PRNGKey(0), sample_shape=(num_samples,))

    # Compute the importance weights
    weights = target_dist.log_prob(samples) - proposal_dist.log_prob(samples)
    weights = jnp.exp(weights - jnp.max(weights))

    return samples, weights


# Number of samples
num_samples = 100000


target_distribution = dist.Normal()
proposal_distribution = dist.Uniform(-1.0, 1.0)

# Perform importance sampling
samples, weights = importance_sampling(
    target_distribution,
    proposal_distribution,
    num_samples,
)

# Estimate the integral using the weighted samples
integral_estimate = jnp.mean(jnp.exp(target_distribution.log_prob(samples)) / weights)
print("Importance sampling integral estimate:", integral_estimate)
