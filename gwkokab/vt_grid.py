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

import argparse

import h5py
from jax import numpy as jnp
from tqdm import tqdm

from ._src.vts import vt_from_mass


def vt_mass_grid():
    parser = argparse.ArgumentParser(description="Generate grid for VT from mass")
    parser.add_argument(
        "--min-mass",
        required=True,
        help="minimum mass",
        type=float,
    )
    parser.add_argument(
        "--max-mass",
        required=True,
        help="maximum mass",
        type=float,
    )
    parser.add_argument(
        "-n",
        required=True,
        help="number of points",
        type=int,
    )
    parser.add_argument(
        "--filename",
        required=True,
        help="path to output file",
        type=str,
    )
    parser.add_argument(
        "--threshold-snr",
        required=False,
        help="threshold SNR",
        default=8.0,
        type=float,
    )
    parser.add_argument(
        "--analysis-time",
        required=False,
        help="analysis time",
        default=1.0,
        type=float,
    )

    args = parser.parse_args()

    mmin = float(args.min_mass)
    mmax = float(args.max_mass)
    n = int(args.n)
    filename = args.filename
    threshold_snr = float(args.threshold_snr)
    analysis_time = float(args.analysis_time)

    m1 = jnp.linspace(mmin, mmax, n)
    m2 = jnp.linspace(mmin, mmax, n)

    m1_grid = jnp.repeat(m1, n)
    m2_grid = jnp.tile(m2, n)

    def vt(m1_, m2_):
        return vt_from_mass(
            m1_,
            m2_,
            threshold_snr,
            analysis_time,
        )

    bar = tqdm(desc="VT", total=n * n)
    vts = [[None for _ in range(n)] for _ in range(n)]
    norm = 0.0
    for i in range(n):
        for j in range(n):
            # if vts[j][i] != None:
            #     vts[i][j] = vts[j][i]
            # elif vts[i][j] == None:
            #     vts[i][j] = vt(float(m1[i]), float(m2[j]))
            vts[i][j] = vt(float(m1[i]), float(m2[j]))
            norm += vts[i][j]
            bar.update(1)
    bar.close()
    nbar = tqdm(desc="Normalizing", total=n * n)
    for i in range(n):
        for j in range(n):
            vts[i][j] /= norm
            nbar.update(1)
    nbar.close()

    with h5py.File(filename, "w") as f:
        f.create_dataset("m1", data=m1_grid, shape=(n, n))
        f.create_dataset("m2", data=m2_grid, shape=(n, n))
        f.create_dataset("VT", data=vts, shape=(n, n))
