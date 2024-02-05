#! /usr/bin/env python
import argparse
import os

import h5py
import numpy as np
from ._src.vts import interpolate_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=str, default=None)
    parser.add_argument("--vt", type=str, default=None)
    parser.add_argument("--n-out", type=int, default=100)
    parser.add_argument("--odat", type=str, default=None)
    opts = parser.parse_args()

    col_names = ["m1_source", "m2_source", "ecc"]

    # load in VT file
    with h5py.File(opts.vt, "r") as VTs:
        raw_interpolator = interpolate_hdf5(VTs)

    # load in data
    dat = np.genfromtxt(opts.dat, names=True)  # should work
    dat_mass = np.c_[dat["m1_source"], dat["m2_source"], dat["ecc"]]

    # compute VTs, which are weights for choices
    weights = raw_interpolator(dat_mass[:, :2])
    weights *= 1.0 / np.sum(weights)  # normalizes

    indexes_all = np.arange(len(dat_mass))
    downselected = np.random.choice(indexes_all, p=weights, size=opts.n_out)

    dat_mass = dat_mass[downselected]

    np.savetxt(opts.odat, dat_mass, delimiter="\t", header="\t".join(col_names))

    output_dir = "./weighted_events/"
    for i in range(dat_mass.shape[0]):
        row = dat_mass[i, :]
        data = np.repeat([row], 20000, axis=0)
        output_file_path = os.path.join(output_dir, f"event_{i + 1}.dat")
        np.savetxt(output_file_path, data, delimiter="\t", header="\t".join(col_names))
