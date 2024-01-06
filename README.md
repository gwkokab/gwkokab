# Jaxtro 🔭 - A JAX-based gravitational-wave population inference

[![Python package](https://github.com/Qazalbash/jaxtro/actions/workflows/python-package.yml/badge.svg)](https://github.com/Qazalbash/jaxtro/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/Qazalbash/jaxtro/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Qazalbash/jaxtro/actions/workflows/python-publish.yml)
[![Versions](https://img.shields.io/pypi/pyversions/jaxtro.svg)](https://pypi.org/project/jaxtro/)

Jaxtro is a JAX-based gravitational-wave population inference package. It is built on top of [Jaxampler](https://github.com/Qazalbash/jaxampler) and provides a high-level interface for sampling from a wide range of gravitational-wave population models.

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
configargparse
jaxampler
numpy
tqdm
```

The test suite is based on pytest. To run the tests, one needs to install pytest and run `pytest` at the root directory of this repo.

## Usage

Jaxtro is designed to be used as a library. It provides a high-level interface for sampling from a wide range of gravitational-wave population models. Following example shows how to generate mock population data. It is a two step process:

1. **Generate a configuration file** such as the one shown below or one in the [repository](example_config.ini). This file specifies the population model to be used, the parameters to be sampled, and the names of the columns in the output file.

    ```ini
    [general]
    size=100
    error_scale=1.0
    error_size=4000
    root_container=data
    event_filename=event_{}.dat
    config_filename=configuration.csv

    [mass_model]
    model=Wysocki2019MassModel
    config_vars=['alpha','mmin','mmax']
    col_names=['m1_source','m2_source']
    params={'alpha_m':0.8,'k':0,'mmin':5.0,'mmax':40.0,'Mmax':80.0,'name':'Wysocki2019MassModel_test'}

    [spin_model]
    model=Wysocki2019SpinModel
    config_vars=['alpha_1','beta_1','alpha_2','beta_2']
    col_names=['chi1_source','chi2_source']
    params={'alpha_1':0.8,'beta_1':1.9,'alpha_2':2.2,'beta_2':3.1,'chimax':1.0,'name':'Wysocki2019SpinModel_test'}
    ```

2. **Generate mock population data** by running the following command,

    ```bash
    jaxtro_genie -c <path_to_config_file>
    ```

For this example the output directory will look like this,

```bash
data
├── configuration.csv
├── event_0.dat
├── event_1.dat
...
└── event_99.dat
```

**Note** this will only work for one model. Multiple models are not supported yet.

## Citing Jaxtro

If you use Jaxtro in your research, please cite the following paper:

```bibtex
@software{jaxtro2023github,
    author  = {Meesum Qazalbash, Muhammad Zeeshan},
    title   = {{Jaxtro}: A JAX-based gravitational-wave population inference},
    url     = {http://github.com/Qazalbash/jaxtro},
    version = {0.0.3},
    year    = {2023}
}
```
