# Hyperparameters

This document explains the different hyperparameters used by GWKokab, flowMC, and NumPyro. It also explains the relationships between them, where applicable.

## GWKokab

.. note::

    This section is not ready yet.

## flowMC

### `n_chains`

The total number of chains.

### `chain_batch_size`

Controls parallelization by breaking the computation of `n_chains` into batches of size `chain_batch_size`.

### `batch_size`

The batch size used for training the normalizing flows.

### `n_epochs`

The number of epochs used for training the normalizing flows.

### `n_max_examples` and `history_window`

These two parameters control the data selection for training the normalizing flows.

We maintain a buffer containing all samples collected by the MCMC sampler and Normalizing Flows. Its shape is `(n_chains, n_training_steps, n_dims)`. A training step is defined as:

$$
    n_t := \left(\left\lfloor\frac{n_l}{n_{l,t}}\right\rfloor +\left\lfloor\frac{n_{g}}{n_{g,t}}\right\rfloor\right)n_{t,l}
$$

where $n_t$ is `n_training_steps`, $n_l$ is the number of local steps, $n_{l,t}$ is local thinning, $n_g$ is the number of global steps, $n_{g,t}$ is global thinning, and $n_{t,l}$ is the number of training loops.

During training, this buffer is transformed to prepare the data. The `n_max_examples` and `history_window` parameters are used during this transformation process:

1. First, all finite values are retained from the buffer (i.e., removing all `jax.numpy.nan` and `jax.numpy.inf` values). The new shape becomes `(n_chains, n_finite_value, n_dims)`.
2. This new buffer is transformed using `history_window`; only the last `history_window` samples are taken from the `n_finite_value` axis.
3. The buffer is then flattened to the shape `(-1, n_dims)`.
4. Finally, `n_max_examples` are randomly sampled from this buffer for training.

Note that if the buffer has fewer than `n_max_examples` entries, all entries are used for training.

### `n_NFproposal_batch_size`

A convenience parameter to control the evaluation of the log posterior.

If `n_NFproposal_batch_size` is greater than `n_global_steps`, all evaluations occur simultaneously. Otherwise, the data is split into batches of size `n_NFproposal_batch_size` and evaluated sequentially.

There are a total of `n_global_steps` entries in the leading axis of the buffer; therefore, to avoid JIT recompilation, `n_global_steps` should be divisible by `n_NFproposal_batch_size`.

## NumPyro

.. note::

    This section is not ready yet.
