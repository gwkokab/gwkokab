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


import arviz as az
import corner
import jax
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from numpyro.infer import MCMC, NUTS

from gwkokab.inference.model_test import model
from gwkokab.vts.utils import interpolate_hdf5, load_hdf5, mass_grid_coords

total_events = 100
posterior_size = 5000
posterior_regex = "syn_data/realization_0/posteriors/event_{}.dat"
injection_regex = "syn_data/realization_0/injections/event_{}.dat"
true_values = np.loadtxt("syn_data/realization_0/configuration.dat")


posteriors = jnp.empty(shape=(total_events, posterior_size, 2))


for event in range(total_events):
    posteriors = posteriors.at[event].set(np.loadtxt(posterior_regex.format(event)))


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=2500, num_chains=10)
# See https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html

mcmc.run(
    jax.random.PRNGKey(0), lambda_n=posteriors
)  # , raw_interpolatror=interpolate_hdf5(load_hdf5("vt_m1_m2.hdf5")))

samples = mcmc.get_samples(group_by_chain=True)

print(mcmc.print_summary())


# make corner plots
fig = corner.corner(
    np.array([samples["alpha"], samples["mmin"], samples["mmax"], samples["rate"]]).T,
    labels=[r"$\alpha$", r"$m_\text{min}$", r"$m_\text{max}$", r"$\mathcal{R}$"],
    # truths=true_values,
    show_titles=True,
    truth_color="r",
)

fig.savefig("corner.png")


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
az.plot_trace(
    samples,
    var_names=["alpha", "mmin", "mmax", "rate"],
    kind="trace",
    combined=False,
    compact=False,
)
fig.tight_layout()
plt.savefig("trace.png")
