#  Copyright 2023 The Jaxtro Authors
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
from configparser import ConfigParser

import h5py
from jax import numpy as jnp
from tqdm import tqdm
import lalsimulation as ls

from ._src.vts import vt_from_mass_spin, vt_from_mass
from .models import *


def vt_mass_spin():
    parser = argparse.ArgumentParser(description="VT from mass and spin")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="path to config",
    )

    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)

    inj_size = config["general"].getint("inj_size")
    filename = config["general"].get("filename")
    analysis_time = config["general"].getfloat("analysis_time")
    psd_fn = eval(config["general"].get("psd_fn", "None"))
    approximant = eval(config["general"].get("approximant", "None"))

    mass_model_obj = eval(config["mass_model"]["model"])
    mass_params = eval(config["mass_model"]["params"])

    spin1_model_obj = eval(config["spin1_model"]["model"])
    spin1_params = eval(config["spin1_model"]["params"])

    spin2_model_obj = eval(config["spin2_model"]["model"])
    spin2_params = eval(config["spin2_model"]["params"])

    mass_model = mass_model_obj(**mass_params)
    spin1_model = spin1_model_obj(**spin1_params)
    spin2_model = spin2_model_obj(**spin2_params)

    m1m2 = mass_model.samples(inj_size)
    a1 = spin1_model.samples(inj_size)
    a2 = spin2_model.samples(inj_size)

    m1 = m1m2[:, 0]
    m2 = m1m2[:, 1]

    def vt(m1, m2, a1, a2):
        return vt_from_mass_spin(
            m1,
            m2,
            a1,
            a2,
            8.0,
            analysis_time,
            psd_fn=psd_fn,
            approximant=approximant,
        )

    vts = []

    for i in tqdm(range(inj_size)):
        vts.append(vt(float(m1[i]), float(m2[i]), float(a1[i]), float(a2[i])))

    vts = jnp.asarray(vts)

    with h5py.File(filename, "w") as f:
        f.create_dataset("m1", data=m1)
        f.create_dataset("m2", data=m2)
        f.create_dataset("a1", data=a1)
        f.create_dataset("a2", data=a2)
        f.create_dataset("VT", data=vts)


def vt_mass():
    parser = argparse.ArgumentParser(description="VT from mass and spin")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="path to config",
    )

    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)

    inj_size = config["general"].getint("inj_size")
    filename = config["general"].get("filename")
    analysis_time = config["general"].getfloat("analysis_time")
    psd_fn = eval(config["general"].get("psd_fn", "None"))
    approximant = eval(config["general"].get("approximant", "None"))

    mass_model_obj = eval(config["mass_model"]["model"])
    mass_params = eval(config["mass_model"]["params"])

    mass_model = mass_model_obj(**mass_params)

    m1m2 = mass_model.samples(inj_size)

    m1 = m1m2[:, 0]
    m2 = m1m2[:, 1]

    def vt(m1, m2):
        return vt_from_mass(
            m1,
            m2,
            8.0,
            analysis_time,
            psd_fn=psd_fn,
            approximant=approximant,
        )

    vts = []

    for i in tqdm(range(inj_size)):
        vts.append(vt(float(m1[i]), float(m2[i])))

    vts = jnp.asarray(vts)

    with h5py.File(filename, "w") as f:
        f.create_dataset("m1", data=m1)
        f.create_dataset("m2", data=m2)
        f.create_dataset("VT", data=vts)
