Frequently Asked Questions
==========================

We are collecting answers to frequently asked questions here. Contributions welcome!
Besides the questions below, you can also check the following resources:

- `flowMC <https://flowmc.readthedocs.io/en/main/FAQ/>`_
- `JAX <https://jax.readthedocs.io/en/latest/faq.html>`_
- `NumPyro <https://num.pyro.ai/en/stable/getting_started.html#frequently-asked-questions>`_


How to write my own population model?
-------------------------------------

GWKokab is designed to be flexible and user-friendly. To create your own population
model, you can subclass the :class:`~numpyro.distributions.distribution.Distribution`
class. If you need to generate a population and run inference, you should implement
both the :meth:`sample` and :meth:`log_prob` methods. However, if you only need to run
inference, you can just implement the :meth:`log_prob` method.

Here is an example of a simple population model:

.. code-block::

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
