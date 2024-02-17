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

import corner
import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from numpyro.infer import MCMC, NUTS

from gwkokab.inference.model_test import model

total_events = 100
posterior_size = 100
posterior_regex = "realization_6/posteriors/event_{}.dat"
injection_regex = "realization_6/injections/event_{}.dat"
true_values = np.loadtxt("realization_6/configuration.dat")[:3]


X = np.zeros((total_events, posterior_size, 2))
y = np.zeros((total_events, 2))


for i in range(total_events):
    posterior = np.loadtxt(posterior_regex.format(i))
    injection = np.loadtxt(injection_regex.format(i))
    X[i] = posterior[:, :2]
    y[i] = injection[:2]


mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=10000)
# See https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
sharding = PositionalSharding(mesh_utils.create_device_mesh((1,)))
X_shard = jax.device_put(X, sharding.reshape(1, 1))
y_shard = jax.device_put(y, sharding.reshape(1))
mcmc.run(jax.random.PRNGKey(int(np.random.rand() * 100)), X_shard, y_shard)

samples = mcmc.get_samples()

try:
    fig = corner.corner(
        samples,
        labels=[r"$\alpha$", r"$m_{\text{min}}$", r"$m_{\text{max}}$"],
        show_titles=True,
        truths=[true_values[0], true_values[1], true_values[2]],
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        hist_kwargs={"density": True},
    )
except:
    fig = corner.corner(
        samples,
        labels=[r"$\alpha$", r"$m_{\text{min}}$", r"$m_{\text{max}}$"],
        show_titles=True,
        truths=[true_values[0], true_values[1], true_values[2]],
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        hist_kwargs={"density": True},
        range=[0.99] * 3,
    )
fig.savefig("test.png")
