# Hierarchical Bayesian Inference (Discrete Method)

[![Open in GitHub](https://img.shields.io/badge/Open-GitHub-black?logo=github)][REPRODUCIBILITY_LINK]

## Introduction

It is recommended to read [Introduction](./generating_mock_posterior_estimates.ipynb#introduction) and [Model Specification](./generating_mock_posterior_estimates.ipynb#model-specification) sections of
[Generating Mock Posterior Estimates](./generating_mock_posterior_estimates.ipynb) tutorial to get familiar
with the model and the data used in this tutorial.

## Sensitivity of Detectors

Sensitivity of detectors is provided in the same way as described in
[Generating Mock Posterior Estimates/Sensitivity of Detectors](./generating_mock_posterior_estimates.ipynb#sensitivity-of-detectors).

## Priors

Priors are also NumPyro distributions. They are provided through a json file. Each key
corresponds to a model parameter and its value is a dictionary, which specifies the
type of distribution and its parameters. For example, the prior for $x$ is given
by a normal distribution with mean 2 and standard deviation 3.

$$
x \sim \mathcal{N}(2, 3)
$$

then the json file will look like,

```json
{
  "x": {
    "dist": "Normal",
    "loc": 2,
    "scale": 3
  }
}
```

Prior distributions and constants for the model parameters used in this tutorial are,

$$
\begin{align*}
    \ln{\mathcal{R}_0}    &\sim \mathrm{Unif}(-11.5, 11.5) \\
    \alpha_{0}            &\sim \mathrm{Unif}(-5.0, 5.0) \\
    \beta_{0}             &=0                             \\
    m_{\mathrm{min,pl},0} &\sim \mathrm{Unif}(1.0, 10.0)  \\
    m_{\mathrm{max,pl},0} &\sim \mathrm{Unif}(30.0, 80.0) \\
    \sigma_\epsilon       &\sim \mathrm{Unif}(0.0, 4.0)
\end{align*}
$$

User can provide any NumPyro distribution which takes only scalar parameters. Their json
representation is saved in
[`priors.json`](https://github.com/gwkokab/hello-gwkokab/tree/main/hbi_discrete_method/priors.json),

```json
{
    "log_rate_0": {
        "dist": "Uniform",
        "low": -6.0,
        "high": 10.0
    },
    "alpha_pl_0": {
        "dist": "Uniform",
        "low": -4.0,
        "high": 12.0
    },
    "beta_pl_0": 0.0,
    "mmin_pl_0": {
        "dist": "Uniform",
        "low": 1.0,
        "high": 10.0
    },
    "mmax_pl_0": {
        "dist": "Uniform",
        "low": 30.0,
        "high": 80.0
    },
    "eccentricity_scale_pl_0": {
        "dist": "Uniform",
        "low": 0.0,
        "high": 4.0
    },
    "eccentricity_loc_pl_0": 0.0,
    "eccentricity_low_pl_0": 0.0
}
```

## MCMC Sampler Configurations

### NumPyro

We provide No U-Turn Sampler (NUTS) through NumPyro as one of the MCMC samplers.
The configuration for the sampler is also provided through a json file. Configuration
has two sub-configurations, one for the NUTS kernel and another for the MCMC sampler.
MCMC configuration are provided under the key `mcmc` while NUTS kernel configurations
are provided under the key `kernel`. For example, if we want to set the maximum tree
depth of the NUTS kernel to 7 and number of warmup steps to 2000, number of samples
to 1000, number of chains to 5, thinning to 1, chain method to "parallel", enable
progress bar and JIT model args, then the json file will look like below.

```json
{
    "kernel": {
        "max_tree_depth": 7
    },
    "mcmc": {
        "num_warmup": 2000,
        "num_samples": 1000,
        "num_chains": 5,
        "thinning": 1,
        "chain_method": "parallel",
        "progress_bar": true,
        "jit_model_args": true
    }
}
```

Please see the
[`numpyro.infer.MCMC`](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.mcmc.MCMC)
and
[`numpyro.infer.NUTS`](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS)
for more details on the available configurations.

These configurations are saved in
[`numpyro_config.json`](https://github.com/gwkokab/hello-gwkokab/tree/main/hbi_discrete_method/sampler_config.json).
Then you can run the following command to perform Hierarchical Bayesian Inference using
NumPyro NUTS sampler.

```bash
n_sage_n_pls_m_gs \
 --seed 37 \
 --n-pl 1 \
 --n-g 0 \
 --add-truncated-normal-eccentricity \
 --posterior-regex "../generating_mock_posterior_estimates/data/realization_0/posteriors/event_*.dat" \
 --posterior-columns mass_1_source mass_2_source eccentricity \
 --pmean-json pmean.json \
 --prior-json prior.json \
 --sampler-config numpyro_config.json \
 --n-buckets 10
```

- `seed` is the random seed for reproducibility.
- `posterior-regex` is the regex pattern to locate posterior samples of individual events.
- `posterior-columns` are the columns in the posterior samples corresponding to
  primary mass, secondary mass and eccentricity.
- `pmean-json` is the json file containing the detector sensitivity information.
- `prior-json` is the json file containing the prior distributions of the population
  parameters.
- `sampler-config` is the json file containing the MCMC sampler configurations.
- `n-buckets` is an optimization parameter to speed up the likelihood evaluations.

### FlowMC

Similarly, we can use Normalizing flows enhanced MALA (FlowMC) as the MCMC sampler.
The configuration for FlowMC is also provided through a json file and saved in
[`flowMC_config.json`](https://github.com/gwkokab/hello-gwkokab/tree/main/hbi_discrete_method/flowMC_config.json).
We will talk about the various configurations in detail in another tutorial.

```json
{
    "data_dump": {
        "n_samples": 5000
    },
    "RQSpline_MALA_Bundle": {
        "chain_batch_size": 0,
        "n_chains": 100,
        "batch_size": 10000,
        "n_epochs": 5,
        "n_max_examples": 200000,
        "history_window": 2000,
        "n_NFproposal_batch_size": 50,
        "n_global_steps": 100,
        "n_local_steps": 100,
        "n_production_loops": 20,
        "n_training_loops": 20,
        "global_thinning": 4,
        "local_thinning": 4,
        "mala_step_size": 0.01,
        "rq_spline_hidden_units": [
            64,
            64
        ],
        "rq_spline_n_bins": 10,
        "rq_spline_n_layers": 8,
        "rq_spline_range": [
            -10.0,
            10.0
        ],
        "learning_rate": 0.001,
        "verbose": false
    }
}
```

Then you can run the following command to perform Hierarchical Bayesian Inference using
FlowMC sampler.

```bash
f_sage_n_pls_m_gs \
 --seed 37 \
 --n-pl 1 \
 --n-g 0 \
 --add-truncated-normal-eccentricity \
 --posterior-regex "../generating_mock_posterior_estimates/data/realization_0/posteriors/event_*.dat" \
 --posterior-columns mass_1_source mass_2_source eccentricity \
 --pmean-json pmean.json \
 --prior-json prior.json \
 --sampler-config flowMC_config.json \
 --n-buckets 10
```

## Analysis of Results

```{toggle} GPU Status
```md
Sat Nov  1 16:23:27 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
|  0%   36C    P8             35W /  450W |      40MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1286      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            1474      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
```

---

All the code and files used in this tutorial can be found in
[hello-gwkokab/hbi_discrete_method][REPRODUCIBILITY_LINK].

[REPRODUCIBILITY_LINK]: https://github.com/gwkokab/hello-gwkokab/tree/main/hbi_discrete_method

## References

```{bibliography} refs.bib
```
