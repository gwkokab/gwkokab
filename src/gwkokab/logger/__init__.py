# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
