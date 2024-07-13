<div align="center">
<img src="https://raw.githubusercontent.com/gwkokab/gwkokab/main/docs/source/_static/logo.png" alt="logo" width="400px" height="90px"></img>
</div>

<h2 align="center">
A JAX-based gravitational-wave population inference toolkit
</h2>

![GitHub License](https://img.shields.io/github/license/gwkokab/gwkokab)
[![Python package](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml)
![PyPI - Version](https://img.shields.io/pypi/v/gwkokab)

GWKokab is a JAX-based gravitational-wave population inference toolkit. It is designed to be a high-performance, flexible and easy-to-use library for sampling from a wide range of gravitational-wave population models. It is built on top of JAX, a high-performance numerical computing library, and is designed to be easily integrated into existing JAX workflows.

If you like to contribute, please see the [contributing guidelines](docs/contributing/contributing.md).

## Installation

You can install the latest released version of GWKokab through pip by doing

```bash
pip install --upgrade gwkokab
```

You can install the bleeding edge version by cloning this repo or doing

```bash
pip install --upgrade git+https://github.com/gwkokab/gwkokab
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install -U "jax[cuda12]"
```

## Citing GWKokab

If you use GWKokab in your research, please cite the following paper:

```bibtex
@software{gwkokab2024github,
    author = {Meesum Qazalbash, Muhammad Zeeshan, Richard O'Shaughnessy},
    title = {{GWKokab}: A JAX-based gravitational-wave population inference toolkit},
    url = {https://github.com/gwkokab/gwkokab},
    version = {0.0.1},
    year = {2024}
}
```
