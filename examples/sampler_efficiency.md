# FlowMC Convergence Conditions Guide

This document outlines **all convergence/critical rules and conditions** for using the [FlowMC](https://github.com/kazewong/flowMC) sampler effectively. These ensure reliable, stable, and efficient sampling across various stages of training and inference.

For the inference using FlowMC sampler, you will have two main steps training using local sampler and production using normalizing flows trained on data produced by local sampler.

So, for the effective training using local sampler, the following condition must be fulfilled to avoid the model collapse.

`((n_chains * n_local_steps) / train_thinning) << n_max_examples`

or

Effective sample size (ESS) = `((max_samples * train_thinning) / n_chains) > n_local_steps`.


For the global sampler the following condition is required.

`n_flow_samples ≥ n_global_steps × n_chains`

## Overall convergence checks and associated parameters to tune

| Parameter     | Condition                                     | 
| ------------- | --------------------------------------------- |
| `m` | Number of parameters you want to recover |
| `n_chains` | should be ≥ $m^2$ |
| `n_flow_samples`| Number of samples NF draws per loop,  `n_flow_samples ≥ n_global_steps × n_chains` , start with `n_flow_samples = 100*m` and increase as needed |
| `n_global_steps` | How many times you attempt a global proposal per training loop `n_flow_samples/n_chains` |
| `n_local_steps` | Number of steps local sampler take to generate points to train NFs. |
| `number of samples in first training loop` | `(n_chains * n_local_steps) / train_thinning` = points generated in each loop for training and keep adding in next loop until reaches the `n_max_samples` |
| `n_max_samples`    | maximum number of samples allowed to use for training, when this number reaches, sampler starts losing previous information, atleast use the previous data for half of the given loops for training |
| Flow loss behavior     | Should converge (decreasing + stable)          |
| MCMC acceptance rate   | Between 50%--80%                               |
| Mode jumping           | Flow-based proposals must connect all modes    |
| Chain independence     | Fast ACF decay, visible across multiple chains |
| Effective Sample Size  | High ESS, low autocorrelation                  |
| Gelman-Rubin statistic | R̂ < 1.1 across chains for training and production |

## I. Normalizing Flow (NF) Model Convergence

### 1. Flow Expressivity

| Parameter     | Condition                                     | 
| ------------- | --------------------------------------------- |
| `hidden_size` | ≥ \[64, 64] for expressive flow               |
| `n_layers`    | ≥ 5 for multimodal or complex targets         |
| `num_bins`    | ≥ 8 (smooth), ≥ 10 (sharp features or ridges) |

* **Rule**: Flow loss (e.g., NLL) should decrease consistently and not saturate early. We can see this in plots.

### 2. Flow Training Stability

| Parameter        | Condition                                  |
| ---------------- | ------------------------------------------ |
| `learning_rate`  | [0.0003 to 0.001] for stable convergence |
| `batch_size`     | ≥ 256 for stable gradients                 |
| `n_epochs`       | ≥ 7 Sufficient to avoid under fitting      |
| `train_thinning` | Frequent training updates (≤ 2)            |

* Monitor flow loss variance to diagnose convergence.

---

## II. Local Sampler (MALA/HMC) Convergence

### 3. Step Size and Acceptance Rate

| Parameter        | Condition                                 |
| ---------------- | ----------------------------------------- |
| `step_size`      | Should give 50%-80% acceptance (MALA/HMC) |
| `local_autotune` | Must be `true` to enable adaptive tuning  |
| `n_local_steps`  | ≥ 50 for useful exploration               |

* **Rule**: Adjust `step_size` if acceptance is too low/high.

---

## III. Global Proposal Convergence

### 4. Global Move Effectiveness

| Parameter        | Condition                                     |
| ---------------- | --------------------------------------------- |
| `n_global_steps` | ≥ 50 for effective cross-mode proposals       |
| `use_global`     | Must be `true` for complex/multimodal targets |
| `n_flow_sample`  | ≥ `n_global_steps × n_chains`                 |

* **Rule**: Flow must be well-trained enough to produce viable global proposals. start with minimum required n_flow_sample = n_global_steps × n_chains for memory efficiency. This number cause the memory peaks and may cause to resource exhaustion.

---

## IV. MCMC Chains and Sampling Conditions

### 5. Mixing and Production

| Parameter           | Condition                              |
| ------------------- | -------------------------------------- |
| `output_thinning`   | Use if chains are auto-correlated      |
| `n_loop_production` | ≥ 10 for meaningful production samples |

* Use trace plots and ACF to verify mixing quality.

---

## V. Computational Stability

### 6. Efficiency and Safety Checks

| Parameter         | Condition                                            |
| ----------------- | ---------------------------------------------------- |
| `precompile`      | Use `true` to avoid JIT overhead                     |
| `batch_size`      | ≤ 5000 or GPU-capacity dependent |

* **Warning**: Check for memory leaks with `torch.cuda.memory_allocated()`.


