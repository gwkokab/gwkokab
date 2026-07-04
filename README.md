<div align="center">
  <picture>
    <!-- Dark mode image -->
    <source srcset="https://raw.githubusercontent.com/kokabsc/gwkokab/main/docs/source/_static/noBgWhite.png" media="(prefers-color-scheme: dark)">
    <!-- Light mode image -->
    <source srcset="https://raw.githubusercontent.com/gwkokab/gwkokab/main/docs/source/_static/noBgBlack.png" media="(prefers-color-scheme: light)">
    <!-- Fallback image -->
    <img src="https://raw.githubusercontent.com/gwkokab/gwkokab/main/docs/source/_static/noBgColor.png" alt="GWKokab logo">
  </picture>
</div>

<h2 align="center">
A JAX-based gravitational-wave population inference toolkit for parametric models
</h2>

<p align="center">
  <a href="https://gwkokab.readthedocs.io/en/latest/installation.html"><b>Installation</b></a> |
  <a href="https://gwkokab.readthedocs.io/"><b>Documentation</b></a> |
  <a href="https://gwkokab.readthedocs.io/en/latest/examples.html"><b>Tutorials</b></a> |
  <a href="https://huggingface.co/datasets/kokabsc/GWKokab_example"><b>Analysis on 🤗</b></a> |
  <a href="https://gwkokab.readthedocs.io/en/latest/FAQs.html"><b>FAQs</b></a> |
  <a href="https://gwkokab.readthedocs.io/en/latest/cite.html"><b>Citing GWKokab</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/kokabsc/gwkokab?logo=open-source-initiative&logoColor=white&color=blue" alt="License">
  <img src="https://img.shields.io/pypi/v/gwkokab" alt="PyPI Version">
  <a href="https://gwkokab.readthedocs.io/en/latest/?badge=latest">
    <img src="https://img.shields.io/readthedocs/gwkokab?logo=Read-the-Docs" alt="Documentation Status">
  </a>
  <a href="https://github.com/kokabsc/gwkokab/actions/workflows/ci.yml">
    <img src="https://github.com/kokabsc/gwkokab/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

## Overview

GWKokab is a high-performance, flexible, and easy-to-use toolkit for **gravitational-wave population inference**. Built on top of **JAX**, it enables efficient Bayesian inference for a wide range of parametric population models while remaining fully compatible with modern GPU/TPU-accelerated workflows.

The framework is designed to support scalable hierarchical inference and rapid experimentation with astrophysical population models, including mass, spin, redshift, and eccentricity distributions of compact binary mergers.

## Contributing

We welcome contributions from the community. If you would like to contribute to GWKokab, please see the [contributing guidelines](https://gwkokab.readthedocs.io/en/latest/dev_docs/contributing.html).

## Citing GWKokab

If you use GWKokab in your research, please cite the following works:

```bibtex
@ARTICLE{2026PhRvD.113j3003Q,
  author          = {{Qazalbash}, M. and {Zeeshan}, M. and {O'Shaughnessy}, R.},
  title           = "{Implementation to identify the properties of multiple
                  populations of gravitational wave sources}",
  journal         = {Phys. Rev. D},
  keywords        = {Astrophysics and astroparticle physics, General Relativity
                  and Quantum Cosmology, High Energy Astrophysical Phenomena,
                  Instrumentation and Methods for Astrophysics},
  year            = 2026,
  month           = may,
  volume          = 113,
  number          = 10,
  eid             = 103003,
  pages           = 103003,
  doi             = {10.1103/krnm-3vrf},
  archivePrefix   = {arXiv},
  eprint          = {2509.13638},
  primaryClass    = {gr-qc},
  adsurl          = {https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3003Q},
  adsnote         = {Provided by the SAO/NASA Astrophysics Data System}
}

@Misc{gwkokab2024github,
  author          = {{Qazalbash}, Meesum and {Zeeshan}, Muhammad and
                  {O'Shaughnessy}, Richard},
  title           = {{GWKokab}: A JAX-based gravitational-wave population
                  inference toolkit for parametric models},
  url             = {https://github.com/kokabsc/gwkokab},
  year            = 2024
}
```
