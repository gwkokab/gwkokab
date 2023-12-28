# Jaxtro 🔭 - A JAX-based gravitational-wave population inference

[![Python package](https://github.com/Qazalbash/jaxtro/actions/workflows/python-package.yml/badge.svg)](https://github.com/Qazalbash/jaxtro/actions/workflows/python-package.yml)
[![Versions](https://img.shields.io/pypi/pyversions/jaxtro.svg)](https://pypi.org/project/jaxtro/)

Jaxtro is a JAX-based gravitational-wave population inference package. It is built on top of [JAXampler](https://github.com/Qazalbash/jaxampler) and provides a high-level interface for sampling from a wide range of gravitational-wave population models.

It is currently under active development and is not ready for production use. If you would like to contribute, please see the [contributing guidelines](CONTRIBUTING.md).

<!-- ## Features

- [x] 🚀 High-Performance Sampling: Leverage the power of JAX for high-speed, accurate sampling.
- [x] 🧩 Versatile Algorithms: A wide range of sampling methods to suit various applications.
- [x] 🔗 Easy Integration: Seamlessly integrates with existing JAX workflows. -->

## Installation

You may install the latest released version of Jaxtro through pip by doing

```bash
pip3 install --upgrade jaxtro
```

You may install the bleeding edge version by cloning this repo, or doing

```bash
pip3 install --upgrade git+https://github.com/Qazalbash/jaxtro
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Requirements

Jaxtro requires Python 3.10 or higher. It also requires the following packages:

```bash
jaxampler
numpy
tqdm
```

The test suite is based on pytest. To run the tests, one needs to install pytest and run `pytest` at the root directory of this repo.

## Citing Jaxtro

If you use Jaxtro in your research, please cite the following paper:

```bibtex
@software{jaxtro2023github,
    author  = {Meesum Qazalbash, Muhammad Zeeshan},
    title   = {{jaxtro}: A JAX-based gravitational-wave population inference},
    url     = {http://github.com/Qazalbash/jaxtro},
    version = {0.0.2},
    year    = {2023}
}
```
