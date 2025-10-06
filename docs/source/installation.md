# Installation

[GWKokab][GWKokab] is available on [PyPI][PyPI] and can be easily installed using pip. For optimal setup, it is recommended to install [GWKokab][GWKokab] in a virtual environment. You can install [GWKokab][GWKokab] with the following command:

::::{tab-set}

:::{tab-item} Stable Release 📦

```{code-block} bash
pip install --upgrade gwkokab
```

:::

:::{tab-item} Nightly 🍺

```{code-block} bash
pip install --upgrade git+https://github.com/gwkokab/gwkokab
```

:::

::::

GWKokab assumes default accelerator is GPU. If you are not a Linux user, you may check the support for you platform in the
[JAX supported platforms](https://docs.jax.dev/en/latest/installation.html#supported-platforms).

[GWKokab]: https://github.com/gwkokab/gwkokab
[PyPI]: https://pypi.org/project/gwkokab/
