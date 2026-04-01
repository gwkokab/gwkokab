# Hierarchical Bayesian Inference (Continuous Method)

[![Open in GitHub](https://img.shields.io/badge/Open-GitHub-black?logo=github)][REPRODUCIBILITY_LINK]

## Introduction

It is recommended to read [Introduction](./generating_mock_posterior_estimates.ipynb#introduction) and [Model Specification](./generating_mock_posterior_estimates.ipynb#model-specification) sections of
[Generating Mock Posterior Estimates](./generating_mock_posterior_estimates.ipynb) tutorial to get familiar with the model and the data used in this tutorial, and [Hierarchical Bayesian Inference (Discrete Method)](./hbi_discrete_method.md) tutorial to understand the discrete method.

## MCMC Sampler Configurations

Hierarchical Bayesian Inference with continuous method can only be performed on flowMC
at the moment. The configuration for FlowMC is also provided through a json file and saved in
[`flowMC_config.json`](https://github.com/gwkokab/hello-gwkokab/blob/main/hbi_continuous_method/flowMC_config.json). We will talk about the various configurations in detail in another
tutorial.

```json
{
    "data_dump": {
        "n_samples": 10000
    },
    "bundle_config": {
        "chain_batch_size": 0,
        "n_chains": 100,
        "batch_size": 10000,
        "n_epochs": 4,
        "n_max_examples": 200000,
        "history_window": 2000,
        "n_NFproposal_batch_size": 50,
        "n_global_steps": 100,
        "n_local_steps": 100,
        "n_production_loops": 10,
        "n_training_loops": 10,
        "global_thinning": 4,
        "local_thinning": 4,
        "local_sampler_name": "hmc",
        "step_size": 0.01,
        "condition_matrix": 1.0,
        "n_leapfrog": 5,
        "rq_spline_hidden_units": [64, 64],
        "rq_spline_n_bins": 10,
        "rq_spline_n_layers": 8,
        "rq_spline_range": [-10.0, 10.0],
        "learning_rate": 0.001,
        "verbose": false
    }
}
```

Then you can run the following command to perform Hierarchical Bayesian Inference using
FlowMC sampler.

```bash
f_monk_n_pls_m_gs \
    --seed 37 \
    --n-pl 1 \
    --n-g 0 \
    --data-filename "../generating_mock_posterior_estimates/data/realization_0/means_covs.hdf5" \
    --n-samples 100 \
    --minimum-mc-error 0.01 \
    --n-checkpoints 10 \
    --n-max-steps 3 \
    --pmean-json pmean.json \
    --prior-json prior.json \
    --sampler-config flowMC_config.json
```

## Analysis of Results

```{toggle}
```md
Mon Nov  3 03:38:17 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
|  0%   39C    P8             34W /  450W |      40MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1326      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            1501      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
```

<img src="https://raw.githubusercontent.com/gwkokab/hello-gwkokab/refs/heads/main/hbi_continuous_method/figs_flowMC/nf_samples_unweighted.png"/>

---

All the code and files used in this tutorial can be found in
[hello-gwkokab/hbi_continuous_method][REPRODUCIBILITY_LINK].

[REPRODUCIBILITY_LINK]: https://github.com/gwkokab/hello-gwkokab/blob/main/hbi_continuous_method
