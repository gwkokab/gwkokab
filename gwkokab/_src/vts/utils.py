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
from jax.scipy.interpolate import RegularGridInterpolator


def load_hdf5(hdf5_file):
    """
    Load the contents of an HDF5 file into a dictionary.

    :param hdf5_file: The path to the HDF5 file.

    :return: A dictionary of the HDF5 file's contents.
    """
    with h5py.File(hdf5_file, "r") as f:
        return {k: f[k][...] for k in f}


def interpolate_hdf5(hdf5_file):
    """
    A convenience function which wraps :py:func:`interpolate`, but given an HDF5
    file. The HDF5 file should contain (at least) the following three datasets:
    (``m1``, ``m2``, ``VT``), which should be arrays appropriate to pass as the
    (``m1_grid``, ``m2_grid``, ``VT_grid``) arguments to :py:func:`interpolate`.
    """
    m1_grid = hdf5_file["m1"][:]
    m2_grid = hdf5_file["m2"][:]
    VT_grid = hdf5_file["VT"][:]

    return interpolate(m1_grid, m2_grid, VT_grid)


def interpolate(m1_grid, m2_grid, VT_grid):
    """
    Return a function, ``VT(m_1, m_2)``, given its value computed on a grid.
    Uses linear interpolation via ``scipy.interpolate.interp2d`` with
    ``kind="linear"`` option set.

    :param m1_grid: Source-frame mass 1.

    :param m2_grid: Source-frame mass 2.

    :param VT_grid: Sensitive volume-time products corresponding to each m1,m2.

    :return: A function ``VT(m_1, m_2)``.
    """

    #    print(m1_grid,m2_grid)
    points = (m1_grid[0], m2_grid[:, 0])
    #    values = VT_grid.flatten()
    #    print(points)
    interpolator = RegularGridInterpolator(  # scipy.interpolate.interp2d(
        points,
        VT_grid,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    return interpolator
