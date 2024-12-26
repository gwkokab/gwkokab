# Frequently Asked Questions

We are collecting answers to frequently asked questions here. Contributions welcome!
Besides the questions below, you can also check the following resources:

- [flowMC](https://flowmc.readthedocs.io/en/main/FAQ/)
- [JAX](https://jax.readthedocs.io/en/latest/faq.html)
- [NumPyro](https://num.pyro.ai/en/stable/getting_started.html#frequently-asked-questions)

## How to write my own population model?

GWKokab is designed to be flexible and user-friendly. To create your own population
model, you can subclass the [`numpyro.distributions.distribution.Distribution`](numpyro.distributions.distribution.Distribution)
class. If you need to generate a population and run inference, you should implement
both the [`sample`](numpyro.distributions.distribution.Distribution.sample) and [`log_prob`](numpyro.distributions.distribution.Distribution.log_prob) methods. However, if you only need to run
inference, you can just implement the [`log_prob`](numpyro.distributions.distribution.Distribution.log_prob) method.

Here is an example of a simple population model:

```{code-block} python
>>> import numpyro
>>> from jax import random as jrd
>>> from numpyro.distributions import Distribution
>>>
>>> class MyPopulationModel(Distribution):
...     def __init__(self, loc, scale):
...         self.loc = loc
...         self.scale = scale
...
...     def sample(self, key, sample_shape=()):
...         return numpyro.distributions.Normal(self.loc, self.scale).sample(
...             key, sample_shape
...         )
...
...     def log_prob(self, value):
...         return numpyro.distributions.Normal(self.loc, self.scale).log_prob(value)
>>>
>>> samples = MyPopulationModel(0.0, 1.0).sample(jrd.PRNGKey(0), (1000,))
>>> samples.shape
(1000,)
```

## GWKokab is slow on GPU

There can be several reasons why GWKokab is slow on GPU. Here are some common issues:

- JAX is not installed correctly. Make sure you have the correct version of JAX installed along with the necessary dependencies. Check installation instructions in the [GWKokab's documentation](#installation) and the [JAX's documentation](https://jax.readthedocs.io/en/latest/installation.html)
- Use appropriate environment variables. Following are the common environment variables GWKokab dev team uses,

    ```{code-block} bash
    export NPROC=16
    export intra_op_parallelism_threads=4
    export OPENBLAS_NUM_THREADS=4
    export TF_CPP_MIN_LOG_LEVEL=1
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    export TF_FORCE_GPU_ALLOW_GROWTH=false
    export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
    ```

    Their values are adjusted based on the system configuration and hardware. You may need to adjust them based on your system configuration. Therefore it is recommended to experiment with different values to find the best configuration for your system and check their documentation for more information.
- See following articles,
  - [GPU performance tips](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
  - [Persistent compilation cache](https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html)
  - [List of XLA compiler flags](https://jax.readthedocs.io/en/latest/xla_flags.html)
