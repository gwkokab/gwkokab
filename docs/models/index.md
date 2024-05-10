---
hide:
    - toc
---

Models in GWKokab are inherited from the [`numpyro.distributions.Distribution`](https://num.pyro.ai/en/stable/distributions.html) class. This allows for easy sampling and evaluation of the models. The models are also compatible with the [`NumPyro`](https://num.pyro.ai/en/stable/index.html) library, which is a probabilistic programming library built on top of [`JAX`](https://jax.readthedocs.io/en/latest/). This means that every model present in [`NumPyro`](https://num.pyro.ai/en/stable/index.html) which is inherited from [`numpyro.distributions.Distribution`](https://num.pyro.ai/en/stable/distributions.html) can be used in GWKokab. We have implemented a few models that are not present in [`NumPyro`](https://num.pyro.ai/en/stable/index.html) and are specific to GWKokab. These models are listed below.
