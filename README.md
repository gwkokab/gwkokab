<div align="center">
  <picture>
    <!-- Dark mode image -->
    <source srcset="https://raw.githubusercontent.com/kokabsc/gwkokab/main/docs/source/_static/noBgWhite.png" media="(prefers-color-scheme: dark)">
    <!-- Light mode image -->
    <source srcset="https://raw.githubusercontent.com/kokabsc/gwkokab/main/docs/source/_static/noBgBlack.png" media="(prefers-color-scheme: light)">
    <!-- Fallback image (if no preference detected) -->
    <img src="https://raw.githubusercontent.com/kokabsc/gwkokab/main/docs/source/_static/noBgColor.png" alt="logo">
  </picture>
</div>

<h2 align="center">
A JAX-based gravitational-wave population inference toolkit for parametric models
</h2>

[**Installation**](https://gwkokab.readthedocs.io/en/latest/installation.html) |
[**Documentation**](https://gwkokab.readthedocs.io/) |
[**Examples/Tutorials**](https://gwkokab.readthedocs.io/en/latest/examples.html) |
[**FAQs**](https://gwkokab.readthedocs.io/en/latest/FAQs.html) |
[**Citing GWKokab**](#citing-gwkokab)

![GitHub License](https://img.shields.io/github/license/kokabsc/gwkokab?logo=open-source-initiative&logoColor=white&color=blue)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/kokabsc/gwkokab)
![PyPI - Version](https://img.shields.io/pypi/v/gwkokab)

[![Documentation Status](https://img.shields.io/readthedocs/gwkokab?logo=Read-the-Docs)](https://gwkokab.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/kokabsc/gwkokab/actions/workflows/ci.yml/badge.svg)](https://github.com/kokabsc/gwkokab/actions/workflows/ci.yml)

GWKokab is a JAX-based gravitational-wave population inference toolkit. It is designed to be a high-performance, flexible, easy-to-use library for sampling from a wide range of gravitational-wave population models. It is built on top of JAX, a high-performance numerical computing library, and is designed to be easily integrated into existing JAX workflows.

If you would like to contribute, please see the [contributing guidelines](https://gwkokab.readthedocs.io/en/latest/dev_docs/contributing.html).

## Citing GWKokab

If you use GWKokab in your research, please cite the following:

```bibtex
@ARTICLE{2025arXiv250913638Q,
       author = {{Qazalbash}, Meesum and {Zeeshan}, Muhammad and {O'Shaughnessy}, Richard},
        title = "{An Implementation to Identify the Properties of Multiple Population of Gravitational Wave Sources}",
      journal = {arXiv e-prints},
     keywords = {General Relativity and Quantum Cosmology, High Energy Astrophysical Phenomena, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = sep,
          eid = {arXiv:2509.13638},
        pages = {arXiv:2509.13638},
          doi = {10.48550/arXiv.2509.13638},
archivePrefix = {arXiv},
       eprint = {2509.13638},
 primaryClass = {gr-qc},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250913638Q},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

```bibtex
@software{gwkokab2024github,
    author  = {{Qazalbash}, Meesum and {Zeeshan}, Muhammad and {O'Shaughnessy}, Richard},
    title   = {{GWKokab}: A JAX-based gravitational-wave population inference toolkit for parametric models},
    url     = {https://github.com/kokabsc/gwkokab},
    year    = {2024}
}
```
