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
from jax import jit
from jax import numpy as jnp
from jaxtyping import Array

from ..typing import Numeric


@jit
def interpolate_hdf5(m1: float, m2: float, file_path: str = "./vt_1_200_1000.hdf5") -> Array:
    """Interpolates the VT values from an HDF5 file based on given m1 and m2 coordinates.

    :param m1: The m1 coordinate.
    :param m2: The m2 coordinate.
    :param file_path: The path to the HDF5 file, defaults to "./vt_1_200_1000.hdf5"
    :return: The interpolated VT value.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        m1_grid = hdf5_file["m1"][:]
        m2_grid = hdf5_file["m2"][:]
        VT_grid = hdf5_file["VT"][:]
        m1_coord = m1_grid[0]
        m2_coord = m2_grid[:, 0]
        interpolator = bispline_interp(m1, m2, m1_coord, m2_coord, VT_grid)
    return interpolator


@jit
def bispline_interp(
    xnew: Numeric,
    ynew: Numeric,
    xp: Numeric,
    yp: Numeric,
    zp: Numeric,
) -> Array:
    """Perform bivariate spline interpolation.
    Check `JAX discussion <https://github.com/google/jax/discussions/10689>`__.

    :param xnew: 1D vector of x-coordinates where to perform predictions.
    :param ynew: 1D vector of y-coordinates where to perform predictions.
    :param xp: 1D vector of original grid points along the x-axis.
    :param yp: 1D vector of original grid points along the y-axis.
    :param zp: 2D array of original values of functions, where zp[i,j] is the value at xp[i], yp[j].
    :return: Interpolated values at the specified coordinates (xnew, ynew).
    """
    M = 0.0625 * jnp.array(
        [
            [0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 16, -40, 32, -8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -8, 24, -24, 8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
            [4, 0, -4, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0],
            [-8, 20, -16, 4, 0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0],
            [4, -12, 12, -4, 0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0],
            [0, 16, 0, 0, 0, -40, 0, 0, 0, 32, 0, 0, 0, -8, 0, 0],
            [-8, 0, 8, 0, 20, 0, -20, 0, -16, 0, 16, 0, 4, 0, -4, 0],
            [16, -40, 32, -8, -40, 100, -80, 20, 32, -80, 64, -16, -8, 20, -16, 4],
            [-8, 24, -24, 8, 20, -60, 60, -20, -16, 48, -48, 16, 4, -12, 12, -4],
            [0, -8, 0, 0, 0, 24, 0, 0, 0, -24, 0, 0, 0, 8, 0, 0],
            [4, 0, -4, 0, -12, 0, 12, 0, 12, 0, -12, 0, -4, 0, 4, 0],
            [-8, 20, -16, 4, 24, -60, 48, -12, -24, 60, -48, 12, 8, -20, 16, -4],
            [4, -12, 12, -4, -12, 36, -36, 12, 12, -36, 36, -12, -4, 12, -12, 4],
        ],
        dtype=jnp.float32,
    )

    M1 = jnp.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0], [1, -1, -1, 1]], dtype=jnp.float32)

    def built_Ivec(zp: Array, ix: int, iy: int) -> Array:
        return jnp.array([zp[ix + i, iy + j] for j in range(-1, 3) for i in range(-1, 3)])

    def built_Ivec1(zp: Numeric, ix: int, iy: int) -> Array:
        return jnp.array([zp[ix + i, iy + j] for j in range(0, 2) for i in range(0, 2)])

    def compute_basis(x: int, order: int = 3) -> Array:
        """
        x in [0,1]
        """
        return jnp.array([x**i for i in jnp.arange(0, order + 1)])

    def tval(xnew: int, ix: Numeric, xp: Numeric) -> Numeric:
        return (xnew - xp[ix - 1]) / (xp[ix] - xp[ix - 1])

    ix = jnp.clip(jnp.searchsorted(xp, xnew, side="right"), 0, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, ynew, side="right"), 0, len(yp) - 1)

    def bilinear_interp(ix: Array, iy: Array) -> Array:
        Iv = built_Ivec1(zp, ix - 1, iy - 1)
        av = M1 @ Iv
        amtx = av.reshape(2, 2, -1)
        tx = tval(xnew, ix, xp)
        ty = tval(ynew, iy, yp)
        basis_x = compute_basis(tx, order=1)
        basis_y = compute_basis(ty, order=1)
        res = jnp.einsum("i...,ij...,j...", basis_y, amtx, basis_x)
        return res

    def bispline_interp(ix: Array, iy: Array) -> Array:
        Iv = built_Ivec(zp, ix - 1, iy - 1)
        av = M @ Iv
        amtx = av.reshape(4, 4, -1)
        tx = tval(xnew, ix, xp)
        ty = tval(ynew, iy, yp)
        basis_x = compute_basis(tx)
        basis_y = compute_basis(ty)
        res = jnp.einsum("i...,ij...,j...", basis_y, amtx, basis_x)
        return res

    condx = jnp.logical_and(ix >= 2, ix <= len(xp) - 2)
    condy = jnp.logical_and(iy >= 2, iy <= len(yp) - 2)

    cond = jnp.logical_and(condx, condy)
    return jnp.where(cond, bispline_interp(ix, iy), bilinear_interp(ix, iy))


# print(interpolate_hdf5(jnp.array([56.89,]),jnp.array([20.12,])))
