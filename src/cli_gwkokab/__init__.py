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
r"""During analysis, we often need to create various types of plots to visualize data
or results. The GWKokab core team has identified several common plot types based on
recurring data patterns. These plots may require input from a single file or
multiple files, depending on the specific requirements.

To streamline the process of generating these plots, we have developed a set of
command-line interfaces (CLIs) built on top of the following libraries:

- `matplotlib <https://matplotlib.org/>`_
- `seaborn <https://seaborn.pydata.org/>`_
- `corner <https://corner.readthedocs.io/>`_
- `arviz <https://python.arviz.org/en/stable/>`_

The available CLIs are as follows:

- :code:`gwk_batch_scatter2d`
- :code:`gwk_batch_scatter3d`
- :code:`gwk_chain_plot`
- :code:`gwk_corner_plot`
- :code:`gwk_ess_plot`
- :code:`gwk_joint_plot`
- :code:`gwk_ppd_plot`
- :code:`gwk_r_hat_plot`
- :code:`gwk_samples_from_vt`
- :code:`gwk_scatter2d`
- :code:`gwk_scatter3d`

For detailed usage instructions, use the help option :code:`-h` or :code:`--help` with
the respective CLI name.

We welcome feedback and community contributions to improve these CLIs. If you have
suggestions or wish to contribute, please refer to the :doc:`/contributing` guide for
more details.

.. note::

    The plots generated by these CLIs are primarily intended for testing purposes, so
    we have not prioritized aesthetic enhancements.
"""
