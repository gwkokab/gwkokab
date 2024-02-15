# <p align="center">GWKokab</p>

## <p align="center">A JAX-based gravitational-wave population inference toolkit</p>

[![Python package](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml/badge.svg)](https://github.com/gwkokab/gwkokab/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/gwkokab/gwkokab/actions/workflows/python-publish.yml/badge.svg)](https://github.com/gwkokab/gwkokab/actions/workflows/python-publish.yml)
[![Versions](https://img.shields.io/pypi/pyversions/gwkokab.svg)](https://pypi.org/project/gwkokab/)

GWKokab is a JAX-based gravitational-wave population inference toolkit. It is designed to be a high-performance, flexible and easy-to-use library for sampling from a wide range of gravitational-wave population models. It is built on top of JAX, a high-performance numerical computing library, and is designed to be easily integrated into existing JAX workflows.

It is currently under active development and is not ready for production use. If you would like to contribute, please
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
    size = 100
    error_size = 5000
    root_container = syn_data
    event_filename = event_{}.dat
    config_filename = configuration.dat
    num_realizations = 5

    ; optional params

    extra_size = 15000
    extra_error_size = 10000

    [selection_effect]
    vt_filename = mass_vt.hdf5

    [mass_model]
    model = Wysocki2019MassModel
    params = {'alpha_m': 0.8, 'k': 0, 'mmin': 10.0, 'mmax': 50.0, 'Mmax': 100.0,}
    config_vars = [('alpha_m', 'alpha'), ('mmin', 'mass_min'), ('mmax', 'mass_max')]
    col_names = ['m1_source', 'm2_source']
    error_type = banana

    [spin1_model]
    model = Beta
    config_vars = [('concentration1', 'alpha_1'), ('concentration0', 'beta_1')]
    col_names = ['a1']
    params = {'concentration1': 1.8, 'concentration0': 0.9}
    error_type = truncated_normal
    error_params = {'scale': 0.5, 'lower': 0.0, 'upper': 1.0, }

    [spin2_model]
    model = Beta
    config_vars = [('concentration1', 'alpha_2'), ('concentration0', 'beta_2')]
    col_names = ['a2']
    params = {'concentration1': 1.8, 'concentration0': 0.9}
    error_type = truncated_normal
    error_params = {'scale': 0.5, 'lower': 0.0, 'upper': 1.0, }

    [ecc_model]
    model = TruncatedNormal
    config_vars = [('scale', 'sigma_ecc')]
    col_names = ['ecc']
    params = {'loc': 0.0, 'scale': 0.05, 'low': 0.0, 'high': 1.0, }
    error_type = truncated_normal
    error_params = {'scale': 0.1, 'lower': 0.0, 'upper': 1.0,}
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
