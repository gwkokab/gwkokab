# <p align="center">GWKokab</p>

## <p align="center">A JAX-based gravitational-wave population inference</p>

[![Python package](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml/badge.svg)](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/gwkokab/gwkokab/actions/workflows/python-publish.yml/badge.svg)](https://github.com/gwkokab/gwkokab/actions/workflows/python-publish.yml)
[![Versions](https://img.shields.io/pypi/pyversions/gwkokab.svg)](https://pypi.org/project/gwkokab/)

GWKokab is a JAX-based gravitational-wave population inference package. It is currently under active development and is not ready for production use. If you would like to contribute, please
see the [contributing guidelines](CONTRIBUTING.md).

<!-- ## Features

- [x] 🚀 High-Performance Sampling: Leverage the power of JAX for high-speed, accurate sampling.
- [x] 🧩 Versatile Algorithms: A wide range of sampling methods to suit various applications.
- [x] 🔗 Easy Integration: Seamlessly integrates with existing JAX workflows. -->

## Installation

First, setup the virtual environment,

```bash
pip install --upgrade venv
python -m venv jvenv
source jvenv/bin/activate
```

Now, you may install the latest released version of GWKokab through pip by doing

```bash
pip install --upgrade gwkokab
```

You may install the bleeding edge version by cloning this repo, or doing

```bash
pip install --upgrade git+https://github.com/gwkokab/gwkokab
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Warning

We are in the development phase, so when you git pull new updates, please reinstall it as follows.

```bash
pip install build
python -m build
pip install --force-reinstall dist/gwkokab-0.0.10-py3-none-any.whl
```

## Requirements

GWKokab requires Python 3.10 or higher. It also requires the following packages:

```bash
configargparse
h5py
jax>=0.4.0
jaxlib>=0.4.0
matplotlib>=3.8.0
mplcursors
numpy
rift
setuptools
tfp-nightly
tqdm
```

The test suite is based on pytest. To run the tests, one needs to install pytest and run `pytest` at the root directory
of this repo.

## Usage

GWKokab is designed to be used as a library. It provides a high-level interface for sampling from a wide range of
gravitational-wave population models. Following example shows how to generate mock population data. It is a two step
process:

1. **Generate a configuration file** such as the one shown below or one in the [repository](example_config.ini). This
   file specifies the population model to be used, the parameters to be sampled, and the names of the columns in the
   output file.

    ```ini
    [general]
    size = 400
    error_size = 2000
    root_container = syn_data
    event_filename = event_{}.dat
    config_filename = configuration.csv
    num_realizations = 5

    ; optional params

    extra_size = 1500
    extra_error_size = 1000

    [selection_effect]
    vt_filename = mass_vt.hdf5

    [mass_model]
    model = Wysocki2019MassModel
    config_vars = ['alpha_m', 'mmin', 'mmax']
    col_names = ['m1_source', 'm2_source']
    params = {'alpha_m': 0.8, 'k': 0, 'mmin': 10.0, 'mmax': 50.0, 'Mmax': 100.0, 'error_scale': 1.0,}

    [spin1_model]
    model = Wysocki2019SpinModel
    config_vars = ['alpha', 'beta']
    col_names = ['a1']
    params = {'alpha': 1.8, 'beta': 0.9, 'chimax': 1.0, 'error_scale': 0.5,}

    [spin2_model]
    model = Wysocki2019SpinModel
    config_vars = ['alpha', 'beta']
    col_names = ['a2']
    params = {'alpha': 0.8, 'beta': 1.9, 'chimax': 1.0, 'error_scale': 0.5,}

    [ecc_model]
    model = EccentricityModel
    config_vars = ['sigma_ecc']
    col_names = ['ecc']
    params = {'sigma_ecc': 0.05, 'error_scale': 0.01,}
    ```

2. **Generate mock population data** by running the following command,

    ```bash
    gwk_genie -c <path_to_config_file>
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
