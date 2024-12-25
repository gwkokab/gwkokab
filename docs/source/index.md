<!-- GWKokab documentation master file, created by
sphinx-quickstart on Sun Jul  7 22:40:56 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

<!-- :github_url: <https://github.com/gwkokab/gwkokab> -->

# GWKokab documentation

[GWKokab][GWKokab] offers a robust suite of probability density function (PDF) models for
analyzing the distribution of parameters in compact binary coalescence (CBC) sources,
such as neutron star or black hole mergers.

Researchers use these models to create synthetic populations of CBC sources, aiding in
population inference studies. This helps scientists draw statistical conclusions about
the astrophysical population of CBC events based on observed data. The models cover
various aspects of CBC sources, including mass distribution, spin, and redshift.

Built with :py::class:`~numpyro.distributions.distribution.Distribution` objects, these
models leverage NumPyro's integration with [JAX][JAX] for automatic differentiation and GPU
acceleration, ensuring mathematical rigor and computational efficiency for large-scale
simulations and analyses.

[GWKokab][GWKokab] also incorporates [flowMC][flowMC], a normalizing flow-enhanced sampling package
for probabilistic inference, providing a powerful framework for Bayesian parameter
estimation from complex, high-dimensional distributions.

The package includes a wide range of well-established
[models] <https://gwkokab.readthedocs.io/en/latest/gwkokab.models.html#models>`_
frequently cited in the literature. It is continuously updated with new models to keep
pace with advancements in the field, and community contributions are encouraged.

[GWKokab][GWKokab] supports research in gravitational wave astronomy, offering tools for
detailed parameter estimation of single events and large-scale population studies. Its
flexibility and extensibility make it invaluable for researchers at all levels.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting started

installation
FAQs
cite
examples
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Further resources

dev_docs
```

```{toctree}
:caption: Links
:hidden:

GitHub Repository <https://github.com/gwkokab/gwkokab>
Changelog <https://github.com/gwkokab/gwkokab/releases>
```

[GWKokab]: www.github.com/gwkokab/gwkokab
[JAX]: www.github.com/google/jax
[flowMC]: www.github.com/kazewong/flowMC
