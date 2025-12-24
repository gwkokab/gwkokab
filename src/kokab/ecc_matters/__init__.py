# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0
r"""This module provides convenience command-line interfaces for the models and
analyses presented in: `Eccentricity matters: Impact of eccentricity on inferred binary
black hole populations <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.063009>`_
:cite:`PhysRevD.110.063009`.

Model Overview
--------------

The model defines a joint population distribution for component masses and eccentricity
of binary black holes (BBHs). It assumes:

+ A Powerlaw distribution for primary masses (:math:`m_1`), truncated at both ends.
+ A Uniform distribution for secondary masses (:math:`m_2`), conditioned on the primary mass.
+ A Half-Normal distribution for eccentricity (:math:`\epsilon`).

Mathematical Definition
-----------------------

The individual probability density functions are defined as:

.. math::

    \begin{align*}
    p(m_1 \mid \alpha, m_\mathrm{min}, m_\mathrm{max}) &\propto m_1^{-\alpha} \cdot \mathbb{I}_{[m_\mathrm{min}, m_\mathrm{max}]}(m_1) \\
    p(m_2 \mid m_1, m_\mathrm{min}) &= \frac{1}{m_1 - m_\mathrm{min}} \cdot \mathbb{I}_{[m_\mathrm{min}, m_1]}(m_2) \\
    p(\epsilon \mid \sigma) &= \sqrt{\frac{2}{\pi\sigma^2}} \exp\left(-\frac{\epsilon^2}{2\sigma^2}\right), \quad \epsilon \geq 0
    \end{align*}

where, :math:`\mathbb{I}` is the indicator function ensuring the variables lie within
specified bounds. The full phenomenological population model, including the merger rate density :math:`\mathcal{R}`, is expressed as:

.. math::

    \rho(m_1, m_2, \epsilon \mid \Lambda) = \mathcal{R} \cdot p(m_1 \mid \alpha, m_\mathrm{min}, m_\mathrm{max}) \cdot p(m_2 \mid m_1, m_\mathrm{min}) \cdot p(\epsilon \mid \sigma)

Hyperparameters Summary
-----------------------

.. list-table::
    :header-rows: 1

    *   - Parameter
        - Symbol
        - Domain
        - Description
    *   - :code:`log_rate`
        - :math:`\ln\mathcal{R}`
        - :math:`\mathbb{R}`
        - Natural logarithm of the merger rate density.
    *   - :code:`alpha_m`
        - :math:`\alpha`
        - :math:`\mathbb{R}`
        - Spectral index for the primary mass power-law.
    *   - :code:`mmin`
        - :math:`m_\mathrm{min}`
        - :math:`(0, m_\mathrm{max})`
        - Minimum mass cutoff for both components.
    *   - :code:`mmax`
        - :math:`m_\mathrm{max}`
        - :math:`(m_\mathrm{min}, \infty)`
        - Maximum mass cutoff for the primary distribution.
    *   - :code:`scale`
        - :math:`\sigma`
        - :math:`\mathbb{R}^+`
        - Scale (standard deviation) of the eccentricity distribution.
    *   - :code:`loc`
        - :math:`\mu`
        - :math:`\{0\}`
        - Location parameter; fixed to 0 for a Half-Normal distribution.
    *   - :code:`low`
        - ---
        - :math:`\{0\}`
        - Lower truncation bound for eccentricity.
    *   - :code:`high`
        - ---
        - :math:`\{\infty\}`
        - Upper truncation bound for eccentricity.
"""
