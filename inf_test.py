#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np

from gwkokab.inference.model_test import expval_mc
from gwkokab.vts.utils import interpolate_hdf5, load_hdf5, mass_grid_coords

total_events = 100
posterior_size = 100
posterior_regex = "realization_6/posteriors/event_{}.dat"
injection_regex = "realization_6/injections/event_{}.dat"
true_values = np.loadtxt("realization_6/configuration.dat")[:3]


def m1m2_raw_interpolator(m1m2):
    m1 = m1m2[:, 0]
    m2 = m1m2[:, 1]
    logM, qtilde = mass_grid_coords(m1, m2, true_values[1])
    raw = interpolate_hdf5(load_hdf5("mass_vt.hdf5"))
    return raw(logM, qtilde)


expval = expval_mc(*true_values, 1.0, m1m2_raw_interpolator)
print(expval)
