# <p align="center">GWKokab</p>

## <p align="center">A JAX-based gravitational-wave population inference toolkit</p>

[![Python package](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml/badge.svg)](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/gwkokab/gwkokab/actions/workflows/python-publish.yml/badge.svg)](https://github.com/gwkokab/gwkokab/actions/workflows/python-publish.yml)
[![Versions](https://img.shields.io/pypi/pyversions/gwkokab.svg)](https://pypi.org/project/gwkokab/)

GWKokab is a JAX-based gravitational-wave population inference toolkit. It is designed to be a high-performance, flexible and easy-to-use library for sampling from a wide range of gravitational-wave population models. It is built on top of JAX, a high-performance numerical computing library, and is designed to be easily integrated into existing JAX workflows.

It is currently under active development and is not ready for production use. If you would like to contribute, please
see the [contributing guidelines](CONTRIBUTING.md).

## Installation

First, set the virtual environment,

```bash
pip install --upgrade venv
python -m venv gwkenv
source gwkenv/bin/activate
```

Now, you may install the latest released version of GWKokab through pip by doing

```bash
pip install --upgrade gwkokab
```

You may install the bleeding edge version by cloning this repo or doing

```bash
pip install --upgrade git+https://github.com/gwkokab/gwkokab
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Citing GWKokab

If you use GWKokab in your research, please cite the following paper:

```bibtex
@software{gwkokab2024github,
    author  = {Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy},
    title   = {{GWKokab}: A JAX-based gravitational-wave population inference},
    url     = {http://github.com/gwkokab/gwkokab},
    version = {0.0.1},
    year    = {2024}
}
```
