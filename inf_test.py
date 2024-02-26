import glob
import sys

import arviz as az
import corner
import numpy as np
from matplotlib import pyplot as plt
from numpyro.infer import MCMC, NUTS

sys.path.append("gwkokab")

from gwkokab.inference.model_test import model
from gwkokab.utils import get_key

#total_events = 100
#posterior_size = 5000
posterior_regex = "events/event_*.dat"
injection_regex = "events/event_*.dat"
#true_values = np.loadtxt("syn_data/realization_0/configuration.dat")


posteriors = [np.loadtxt(event) for event in glob.glob(posterior_regex)]


mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=2500, num_chains=20)
# See https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html

mcmc.run(get_key(), lambda_n=posteriors)

samples = mcmc.get_samples(group_by_chain=True)

print(mcmc.print_summary())


# make corner plots
fig = corner.corner(
    np.array([samples["alpha"], samples["mmin"], samples["mmax"]]).T,
    labels=[r"alpha", r"$m_{min}$", r"$m_{max}$"],
    show_titles=True,
    truth_color="r",
)

fig.savefig("corner.png")


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
az.plot_trace(
    samples,
    var_names=["alpha", "mmin", "mmax"],
    kind="trace",
    combined=False,
    compact=False,
)
fig.tight_layout()
plt.savefig("trace.png")