#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

import h5py
import numpy as np
from jax import numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator


def load_hdf5(hdf5_file):
    """
    Load the contents of an HDF5 file into a dictionary.

    :param hdf5_file: The path to the HDF5 file.

    :return: A dictionary of the HDF5 file's contents.
    """
    with h5py.File(hdf5_file, "r") as f:
        return {k: f[k][...] for k in f}


def mass_grid_coords(m1, m2, m_min, eps=1e-8):
    m1, m2 = np.asarray(m1), np.asarray(m2)

    M = m1 + m2

    logM = np.log(M)

    i_good = M > 2 * m_min + eps

    m1, m2, M = m1[i_good], m2[i_good], M[i_good]

    qtilde = np.ones_like(logM)
    qtilde[i_good] = M * (m2 - m_min) / (m1 * (M - 2 * m_min))

    return logM, qtilde


def interpolate_hdf5(hdf5_file):
    """
    A convenience function which wraps :py:func:`interpolate`, but given an HDF5
    file. The HDF5 file should contain (at least) the following three datasets:
    (``m1``, ``m2``, ``VT``), which should be arrays appropriate to pass as the
    (``m1_grid``, ``m2_grid``, ``VT_grid``) arguments to :py:func:`interpolate`.
    """
    # logM = jnp.asarray(hdf5_file["logM"])
    # qtilde = jnp.asarray(hdf5_file["qtilde"])
    # VT_grid = jnp.asarray(hdf5_file["VT"][:])

    # return interpolate((logM, qtilde), VT_grid)

    m1 = jnp.asarray(hdf5_file["m1"])
    m2 = jnp.asarray(hdf5_file["m2"])
    VT_grid = jnp.asarray(hdf5_file["VT"][:])

    return interpolate((m1[0], m2[:, 0]), VT_grid)


def interpolate(points, VT_grid):
    """
    Return a function, ``VT(m_1, m_2)``, given its value computed on a grid.
    Uses linear interpolation via ``scipy.interpolate.interp2d`` with
    ``kind="linear"`` option set.

    :param m1_grid: Source-frame mass 1.

    :param m2_grid: Source-frame mass 2.

    :param VT_grid: Sensitive volume-time products corresponding to each m1,m2.

    :return: A function ``VT(m_1, m_2)``.
    """

    interpolator = RegularGridInterpolator(  # scipy.interpolate.interp2d(
        points,
        VT_grid,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    return interpolator
