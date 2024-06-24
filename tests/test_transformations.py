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

import pytest
from jax import numpy as jnp

from gwkokab.utils.transformations import (
    chirp_mass,
    m1_times_m2,
    mass_ratio,
    reduced_mass,
    symmetric_mass_ratio,
    total_mass,
)


# These masses are in such a way that we check both integers
# and floats along with the working of the functions
_primary_masses = [1, 2, 3.0]
_secondary_masses = [4.0, 5, 6.0]


@pytest.mark.parametrize("m1", _primary_masses)
@pytest.mark.parametrize("m2", _secondary_masses)
def test_different_mass_representations(m1, m2):
    m1m2 = m1 * m2
    assert jnp.equal(m1m2, m1_times_m2(m1=m1, m2=m2))
    M = m1 + m2
    assert jnp.equal(M, total_mass(m1=m1, m2=m2))
    q = m2 / m1
    assert jnp.equal(q, mass_ratio(m1=m1, m2=m2))
    Mc = m1m2**0.6 * M**-0.2
    assert jnp.equal(Mc, chirp_mass(m1=m1, m2=m2))
    eta = m1m2 * M**-2.0
    assert jnp.equal(eta, symmetric_mass_ratio(m1=m1, m2=m2))
    Mr = m1m2 / M
    assert jnp.equal(Mr, reduced_mass(m1=m1, m2=m2))
