Models
======

GWKokab_ is a comprehensive package that provides an extensive suite of probability density function (PDF) models designed to describe the distribution of parameters associated with compact binary coalescence (CBC) sources. These models are crucial for understanding and simulating the complex nature of CBC events, such as the merging of neutron stars or black holes, which are key sources of gravitational waves.

By employing these models, researchers can generate synthetic populations of CBC sources, which are essential for population inference studies. This process allows scientists to draw meaningful statistical conclusions about the underlying astrophysical population of CBC events based on the observed data. The models help in exploring various aspects of CBC sources, such as their mass distribution, spin, and redshift.

The implementation of these models is facilitated through the use of :class:`numpyro.distributions.distribution.Distribution` objects. NumPyro_ is a lightweight library for probabilistic programming in Python_, which leverages JAX_ for automatic differentiation and GPU acceleration. This integration ensures that the models are not only mathematically rigorous but also computationally efficient, making them suitable for large-scale simulations and analyses.

GWKokab_ includes a wide range of models that are frequently cited in the literature, ensuring that users have access to well-established tools for their research. However, the package is continually evolving, with new models being regularly added to keep pace with advancements in the field. We encourage contributions from the community to expand and enhance the repository, fostering a collaborative environment where cutting-edge research can thrive.

Whether you are conducting detailed parameter estimation for a single event or performing large-scale population studies, GWKokab_ provides the necessary tools to support your research in gravitational wave astronomy. The package's flexibility and extensibility make it an invaluable resource for both novice users and experienced researchers in the field.

Broken Power Law Mass Model
---------------------------
.. autoclass:: gwkokab._src.models.models.BrokenPowerLawMassModel
    :no-index:

Gaussian Spin Model
-------------------
.. autoclass:: gwkokab._src.models.models.GaussianSpinModel
    :no-index:

Independent Spin Orientation Gaussian Isotropic
-----------------------------------------------
.. autoclass:: gwkokab._src.models.models.IndependentSpinOrientationGaussianIsotropic
    :no-index:

Multi Peak Mass Model
---------------------
.. autoclass:: gwkokab._src.models.models.MultiPeakMassModel
    :no-index:

Mixture of N-Distributions
--------------------------
.. autoclass:: gwkokab._src.models.models.NDistribution
    :no-index:

Power Law Peak Mass Model
-------------------------
.. autoclass:: gwkokab._src.models.models.PowerLawPeakMassModel
    :no-index:

Power Law Primary Mass Ratio
----------------------------
.. autoclass:: gwkokab._src.models.models.PowerLawPrimaryMassRatio
    :no-index:

Truncated Power Law
-------------------
.. autoclass:: gwkokab._src.models.models.TruncatedPowerLaw
    :no-index:

Wysocki2019MassModel
--------------------
.. autoclass:: gwkokab._src.models.models.Wysocki2019MassModel
    :no-index:



.. _GWKokab: www.github.com/gwkokab/gwkokab
.. _NumPyro: www.github.com/pyro-ppl/numpyro
.. _JAX: www.github.com/google/jax
.. _Python: https://www.python.org/