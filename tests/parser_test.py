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

import sys

sys.path.append("../jaxtro")

from jaxtro.utils.parser import parse_config

dic = parse_config("tests/test_config.ini")


def test_general_arguments() -> None:
    assert dic["general"]["size"] == 100
    assert dic["general"]["error_scale"] == 1.0
    assert dic["general"]["error_size"] == 4000
    assert dic["general"]["root_container"] == "data"
    assert dic["general"]["event_filename"] == "event_{}.dat"
    assert dic["general"]["config_filename"] == "configuration.csv"


def test_mass_model() -> None:
    assert dic["mass_model"]["model"] == "Wysocki2019MassModel"
    assert dic["mass_model"]["params"]["alpha"] == 0.8
    assert dic["mass_model"]["params"]["k"] == 0
    assert dic["mass_model"]["params"]["mmin"] == 5.0
    assert dic["mass_model"]["params"]["mmax"] == 40.0
    assert dic["mass_model"]["params"]["Mmax"] == 80.0
    assert dic["mass_model"]["params"]["name"] == "Wysocki2019MassModel_test"
    assert dic["mass_model"]["config_vars"] == ["alpha", "mmin", "mmax"]
    assert dic["mass_model"]["col_names"] == ['m1_source', 'm2_source']


def test_spin_model() -> None:
    assert dic["spin_model"]["model"] == "Wysocki2019SpinModel"
    assert dic["spin_model"]["params"]["alpha_1"] == 0.8
    assert dic["spin_model"]["params"]["beta_1"] == 0.8
    assert dic["spin_model"]["params"]["alpha_2"] == 0.8
    assert dic["spin_model"]["params"]["beta_2"] == 0.8
    assert dic["spin_model"]["params"]["chimax"] == 1.0
    assert dic["spin_model"]["params"]["name"] == "Wysocki2019SpinModel_test"
    assert dic["spin_model"]["config_vars"] == ["alpha_1", "beta_1", "alpha_2", "beta_2"]
    assert dic["spin_model"]["col_names"] == ['chi1_source', 'chi2_source']
