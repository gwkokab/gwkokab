# FlowMC Convergence Conditions Guide

This document outlines **all convergence/critical rules and conditions** for using the [FlowMC](https://github.com/kazewong/flowMC) sampler effectively. These ensure reliable, stable, and efficient sampling across various stages of training and inference.

For the inference using FlowMC sampler, you will have two main steps training using local sampler and production using normalizing flows trained on data produced by local sampler.

So, for the effective training using local sampler, the following condition must be fulfilled to avoid the model collapse.

`(((n_chains * n_local_steps) / train_thinning))*N_training_loop << n_max_examples`

or

Effective sample size (ESS) = `((max_samples * train_thinning) / n_chains) > n_local_steps`.

For the global sampler the following condition is required.

`n_flow_samples <= n_global_steps × n_chains`

## Overall convergence checks and associated parameters to tune

| Parameter     | Condition                                     |
| ------------- | --------------------------------------------- |
| `m` | Number of parameters you want to recover |
| `n_chains` | should be ≥ $m^2$ |
| `n_flow_samples`| Number of samples NF draws per loop,  `n_flow_samples <= n_global_steps × n_chains` , start with `n_flow_samples = 100*m` and increase as needed |
| `n_global_steps` | How many times you attempt a global proposal per training loop `n_flow_samples/n_chains` |
| `n_local_steps` | Number of steps local sampler take to generate points to train NFs. |
| `number of samples in first training loop` | `(n_chains * n_local_steps) / train_thinning` = points generated in each loop for training and keep adding in next loop until reaches the `n_max_samples` |
| `n_max_samples`    | maximum number of samples allowed to use for training, when this number reaches, sampler starts losing previous information, at least use the previous data for half of the given loops for training |
| Flow loss behavior     | Should converge (decreasing + stable), No sudden peaks       |
| Global acceptance rate   | Between 30%--60% (Ideally 40% - 60%)                       |
| Local acceptance rate    |  should be ≥ 60%   |
| Chain independence     | Fast ACF decay, visible across multiple chains |
| Effective Sample Size  | ≥ 0.2,  High ESS, low autocorrelation                  |
| Gelman-Rubin statistic | R̂ < 1.1 across chains for training and production |
| Better Mixed chains | Use HMC (Bit slower) instead of MALA (Fast) |

## I. Normalizing Flow (NF) Model Convergence

### 1. Flow Expressivity

| Parameter     | Condition                                     |
| ------------- | --------------------------------------------- |
| `hidden_size` | ≥ \[64, 64] for expressive flow               |
| `n_layers`    | ≥ 5 for multimodal or complex targets         |
| `num_bins`    | ≥ 7 (smooth), ≥ 10 (sharp features or ridges) |

* **Rule**: Flow loss (e.g., NLL) should decrease consistently and not saturate early. We can see this in plots.

### 2. Flow Training Stability

| Parameter        | Condition                                  |
| ---------------- | ------------------------------------------ |
| `learning_rate`  | [0.0003 to 0.001] for stable convergence |
| `batch_size`     | ≥ 256 for stable gradients                 |
| `n_epochs`       | ≥ 5 Sufficient to avoid under fitting      |
| `train_thinning` | ≥ 5            |

* Monitor flow loss variance to diagnose convergence.

---

## II. Local Sampler (MALA/HMC) Convergence

### 3. Step Size and Acceptance Rate

| Parameter        | Condition                                 |
| ---------------- | ----------------------------------------- |
| `local_acceptance_rate` | Should give >60% acceptance (MALA/HMC) |
| `local_autotune` | should be `true` to enable adaptive tuning  |
| `n_local_steps`  | ≥ 40 for useful exploration               |

* **Rule**: Adjust `step_size` or `n_local_steps` if acceptance is too low/high.

---

## III. Global Proposal Convergence

### 4. Global Move Effectiveness

| Parameter        | Condition                                     |
| ---------------- | --------------------------------------------- |
| `global_acceptance_rate` | Between 30%--60% (Ideally 40% - 60%)  for effective cross-mode proposals       |
| `use_global`     | Must be `true` for complex/multimodal targets |
| `n_flow_sample`  | <= `n_global_steps × n_chains`                 |

* **Rule**: Flow must be well-trained enough to produce viable global proposals. start with minimum required n_flow_sample = n_global_steps × n_chains for memory efficiency. This number cause the memory peaks and may cause to resource exhaustion.

---

## IV. MCMC Chains and Sampling Conditions

### 5. Mixing and Production

| Parameter           | Condition                              |
| ------------------- | -------------------------------------- |
| `output_thinning`   | Use if chains are auto-correlated      |
| `n_loop_production` | ≥ 8 for meaningful production samples |

* Use trace plots and ACF to verify mixing quality.

---

## V. Computational Stability

### 6. Efficiency and Safety Checks

| Parameter         | Condition                                            |
| ----------------- | ---------------------------------------------------- |
| `precompile`      | Use `true` to avoid JIT overhead                     |
| `batch_size`      | GPU-capacity dependent |

* **Warning**: Check for memory leaks with `torch.cuda.memory_allocated()`.

The following are the some of the plots to see the convergence of the sampler based on minimal run.

![local_accs](https://github.com/user-attachments/assets/12a606c2-d472-4238-944c-1c8ce7ee8d96)
![global_accs-2](https://github.com/user-attachments/assets/90b043b6-9261-43ab-8d55-e2c837fda58a)
![r_hat_prod.pdf](https://github.com/user-attachments/files/20819292/r_hat_prod.pdf)
![Chain.pdf](https://github.com/user-attachments/assets/2f414748-a9ae-42c0-b4bb-d88d395a2c2c)
![ess.pdf](https://github.com/user-attachments/files/20819315/ess.pdf)
