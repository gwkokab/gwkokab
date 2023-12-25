# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from jaxtro.models import Wysocki2019MassModel
from jaxtro.utils import PopulationGenerator


def test():

    CONFIGS = {
        "model":
            Wysocki2019MassModel,
        "size":
            100,
        "error_scale":
            1,
        "error_size":
            4000,
        "root_container":
            "data",
        "container":
            "data_{}",
        "point_filename":
            "event_{}.dat",
        "config_filename":
            "configuration.csv",
        "config_vars": ["alpha", "mmin", "mmax"],
        "col_names": ["m1_source", "m2_source"],
        "params": [{
            "alpha": 0.8,
            "k": 0,
            "mmin": 5.0,
            "mmax": 40.0,
            "Mmax": 80.0,
            "name": "Wysocki2019MassModel0",
        }, {
            "alpha": -2.0,
            "k": 1,
            "mmin": 5.0,
            "mmax": 70.0,
            "Mmax": 140.0,
            "name": "Wysocki2019MassModel1",
        }, {
            "alpha": 0.8,
            "k": 5,
            "mmin": 30.0,
            "mmax": 45.0,
            "Mmax": 90.0,
            "name": "Wysocki2019MassModel2",
        }]
    }
    # check if the data directory exists
    assert not os.path.exists(CONFIGS["root_container"])
    pg = PopulationGenerator(CONFIGS)
    pg.generate()
    assert len(os.listdir(CONFIGS["root_container"])) == len(CONFIGS["params"])
    for i in range(len(CONFIGS["params"])):
        assert len(os.listdir(f"{CONFIGS['root_container']}/{CONFIGS['container'].format(i)}")) == CONFIGS["size"] + 1
    os.system("rm -rf data")
    assert not os.path.exists(CONFIGS["root_container"])
