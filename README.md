<div align="center">
  <picture>
    <!-- Dark mode image -->
    <source srcset="https://raw.githubusercontent.com/gwkokab/gwkokab/main/docs/source/_static/noBgWhite.png" media="(prefers-color-scheme: dark)">
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
  <a href="https://gwkokab.readthedocs.io/en/latest/examples.html"><b>Examples/Tutorials</b></a> |
  <a href="https://gwkokab.readthedocs.io/en/latest/FAQs.html"><b>FAQs</b></a> |
  <a href="#citing-gwkokab"><b>Citing GWKokab</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/gwkokab/gwkokab?logo=open-source-initiative&logoColor=white&color=blue" alt="License">
  <img src="https://img.shields.io/github/issues/gwkokab/gwkokab" alt="Issues">
  <img src="https://img.shields.io/pypi/v/gwkokab" alt="PyPI Version">
</p>

<p align="center">
  <a href="https://gwkokab.readthedocs.io/en/latest/?badge=latest">
    <img src="https://img.shields.io/readthedocs/gwkokab?logo=Read-the-Docs" alt="Documentation Status">
  </a>
  <a href="https://github.com/gwkokab/gwkokab/actions/workflows/ci.yml">
    <img src="https://github.com/gwkokab/gwkokab/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

---

## Overview

GWKokab is a high-performance, flexible, and easy-to-use toolkit for **gravitational-wave population inference**. Built on top of **JAX**, it enables efficient Bayesian inference for a wide range of parametric population models while remaining fully compatible with modern GPU/TPU-accelerated workflows.

The framework is designed to support scalable hierarchical inference and rapid experimentation with astrophysical population models, including mass, spin, redshift, and eccentricity distributions of compact binary mergers.

---

> [!IMPORTANT]
> ## Development Branch Notice
>
> The latest **tested features**, updates, and bug fixes are currently available on the `dev` branch.
>
> Until the ongoing documentation updates are finalized, we recommend users install and work from the `dev` branch instead of `main`.
>
> Clone directly using:
>
> ```bash
> git clone -b dev https://github.com/kokabsc/gwkokab.git
> ```
>
> or switch an existing clone:
>
> ```bash
> git checkout dev
> git pull origin dev
> ```
>
> The `main` branch will be updated after the current development cycle and documentation for the `dev` branch are completed.
>
> [!NOTE]
> For new users, we recommend starting with the **NumPyro** sampler before using **FlowMC**.  
> NumPyro is generally easier to configure, debug, and tune, making it a more accessible starting point for developing and validating population inference workflows.

---

## Contributing

We welcome contributions from the community.  
If you would like to contribute to GWKokab, please see the
[contributing guidelines](https://gwkokab.readthedocs.io/en/latest/dev_docs/contributing.html).

---

## Citing GWKokab

If you use GWKokab in your research, please cite the following works:

### GWKokab Paper

```bibtex
@article{arxiv:2509.13638,
    author  = {{Qazalbash}, Meesum and {Zeeshan}, Muhammad and {O'Shaughnessy}, Richard},
    title   = {GWKokab: An Implementation to Identify the Properties of Multiple Population of Gravitational Wave Sources},
    journal = {arXiv preprint arXiv:2509.13638},
    year    = {2025},
    url     = {https://arxiv.org/pdf/2509.13638v1}
}
@Misc{gwkokab2024github,
    author  = {{Qazalbash}, Meesum and {Zeeshan}, Muhammad and {O'Shaughnessy}, Richard},
    title   = {{GWKokab}: A JAX-based gravitational-wave population inference toolkit for parametric models},
    url     = {https://github.com/gwkokab/gwkokab},
    year    = {2024}
}
