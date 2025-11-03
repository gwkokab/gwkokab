# Installation

## Virtual Environment Setup

For optimal setup, we recommend using [UV][UV] package manager for creating and managing virtual environments. See details at [uv installation guidelines](https://docs.astral.sh/uv/getting-started/installation/).

Make a new virtual environment using [UV][UV],

```bash
uv venv -p 3.12
source .venv/bin/activate
```

Replace `3.12` with your desired Python version. After activating the virtual environment, you can proceed with the installation of [GWKokab][GWKokab].

## Stable Release

[GWKokab][GWKokab] is available on [PyPI][PyPI] and can be easily installed using pip. Setup virtual environment by following details in the [Virtual Environment Setup](#virtual-environment-setup) section. Then, depending on your hardware configuration, run one of the following commands to install the stable release:

::::{tab-set}
:sync-group: category

:::{tab-item} CPU 🐢
:sync: cpu

It is against the philosophy of [GWKokab][GWKokab] to use it with CPU support only, but if you really want to:

```{code-block} bash
uv pip install -U "gwkokab[cpu]"
```

This option is mainly intended for development purposes. See [JAX's official documentation](https://docs.jax.dev/en/latest/installation.html#cpu) for the extend of supported CPU platforms and features.

:::

:::{tab-item} Nvidia-GPU 🚀
:sync: gpu

We highly recommend using [GWKokab][GWKokab] with GPU support for better performance. Before installation, appropriate Nvidia drivers and CUDA-toolkit must be ensured on your system.

For CUDA-toolkit 12:

```{code-block} bash
uv pip install -U "gwkokab[cuda12]"
```

or for CUDA-toolkit 13:

```{code-block} bash
uv pip install -U "gwkokab[cuda13]"
```

See [JAX's official documentation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) for more details on supported Nvidia GPUs and drivers.

:::

:::{tab-item} TPU ☄️
:sync: tpu

TPU support is available for [GWKokab][GWKokab] users.

```{code-block} bash
uv pip install -U "gwkokab[tpu]"
```

See [JAX's official documentation](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu) for more details on using JAX with TPUs.

:::

::::

## Nightly Build

For bleeding-edge features and the latest updates, you can install the nightly build of [GWKokab][GWKokab] directly from the source. This is recommended for advanced users and developers who want to stay up-to-date with the latest changes.

You must have [GNU Make](https://www.gnu.org/software/make/) and [UV][UV] installed on your system.
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/gwkokab/gwkokab.git
cd gwkokab
```

Setup virtual environment by following details in the [Virtual Environment Setup](#virtual-environment-setup) section. Then, depending on your hardware configuration, run one of the following commands to install the nightly build:

::::{tab-set}
:sync-group: category

:::{tab-item} CPU 🐢
:sync: cpu

Linux with x86_64 is the sweet spot for CPU support with JAX.

```{code-block} bash
make install PIP_FLAGS=--upgrade EXTRA=cpu
```

See [JAX's official documentation](https://docs.jax.dev/en/latest/installation.html#cpu) for the extend of supported CPU platforms and features.

:::

:::{tab-item} Nvidia-GPU 🚀
:sync: gpu

Linux with x86_64 is the sweet spot for Nvidia-GPU support with JAX.

For CUDA-toolkit 12:

```{code-block} bash
make install PIP_FLAGS=--upgrade EXTRA=cuda12
```

or for CUDA-toolkit 13:

```{code-block} bash
make install PIP_FLAGS=--upgrade EXTRA=cuda13
```

See [JAX's official documentation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) for more details on supported Nvidia GPUs and drivers.

:::

:::{tab-item} TPU ☄️
:sync: tpu

TPU is the least tested hardware configuration for [GWKokab][GWKokab]. If you face any issues, please report them on our [GitHub Issues](https://github.com/gwkokab/gwkokab/issues).

```{code-block} bash
make install PIP_FLAGS=--upgrade EXTRA=tpu
```

See [JAX's official documentation](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu) for more details on using JAX with TPUs.

:::

::::

[GWKokab]: https://github.com/gwkokab/gwkokab
[PyPI]: https://pypi.org/project/gwkokab/
[UV]: https://docs.astral.sh/uv/
