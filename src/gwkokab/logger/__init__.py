# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""A lightweight logging module built on top of Loguru_. Designed for seamless
integration with JAX_, it enables efficient logging within JIT-compiled functions,
making it a powerful tool for tracking and debugging JAX workflows. Fully compatible
with Loguru_, it retains a similar interface, allowing users to refer to the
`official Loguru documentation <https://loguru.readthedocs.io/en/stable/>`_ for usage
instructions.

.. _Loguru: https://github.com/Delgan/loguru
.. _JAX: https://github.com/jax-ml/jax
"""

from ._logger import enable_logging as enable_logging, logger as logger
