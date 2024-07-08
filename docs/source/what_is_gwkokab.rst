What is GWKokab?
================

GWKokab_ is a comprehensive package offering an extensive suite of probability density function (PDF) models for describing the distribution of parameters in compact binary coalescence (CBC) sources. These models are essential for understanding CBC events, such as neutron star or black hole mergers, which are key gravitational wave sources.

Researchers use these models to generate synthetic populations of CBC sources, aiding in population inference studies. This allows scientists to draw statistical conclusions about the underlying astrophysical population of CBC events based on observed data. The models explore various aspects of CBC sources, such as mass distribution, spin, and redshift.

Implemented using :class:`numpyro.distributions.distribution.Distribution` objects, these models benefit from NumPyro's integration with JAX_ for automatic differentiation and GPU acceleration, ensuring mathematical rigor and computational efficiency for large-scale simulations and analyses.

GWKokab_ includes a wide range of models frequently cited in the literature, providing well-established tools for research. The package is continuously evolving, with new models regularly added to keep pace with field advancements. Community contributions are encouraged to expand and enhance the repository.

GWKokab_ supports research in gravitational wave astronomy, offering tools for both detailed parameter estimation of single events and large-scale population studies. Its flexibility and extensibility make it invaluable for both novice and experienced researchers.


.. _GWKokab: www.github.com/gwkokab/gwkokab
.. _NumPyro: www.github.com/pyro-ppl/numpyro
.. _JAX: www.github.com/google/jax
.. _Python: https://www.python.org/