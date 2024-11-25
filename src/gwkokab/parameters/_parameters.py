# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import equinox as eqx
from numpyro.distributions import Distribution, ImproperUniform, Uniform
from numpyro.distributions.constraints import real


class Parameter(eqx.Module):
    name: str = eqx.field(static=True, metadata={"help": "Name of the parameter."})
    prior: Distribution = eqx.field(
        default=ImproperUniform(
            support=real, batch_shape=(), event_shape=(), validate_args=True
        ),
    )


# TODO: Add more parameters as needed.


standard_uniform = Uniform(0.0, 1.0, validate_args=True)
two_sided_uniform = Uniform(-1.0, 1.0, validate_args=True)
uniform_for_masses = Uniform(0.5, 300.0, validate_args=True)

# Masses

CHIRP_MASS = Parameter(name="chirp_mass", prior=uniform_for_masses)
MASS_RATIO = Parameter(name="mass_ratio", prior=standard_uniform)
PRIMARY_MASS_DETECTED = Parameter(name="mass_1", prior=uniform_for_masses)
PRIMARY_MASS_SOURCE = Parameter(name="mass_1_source", prior=uniform_for_masses)
REDUCED_MASS = Parameter(name="reduced_mass", prior=uniform_for_masses)
SECONDARY_MASS_DETECTED = Parameter(name="mass_2", prior=uniform_for_masses)
SECONDARY_MASS_SOURCE = Parameter(name="mass_2_source", prior=uniform_for_masses)
SYMMETRIC_MASS_RATIO = Parameter(
    name="symmetric_mass_ratio", prior=Uniform(0.0, 0.25, validate_args=True)
)

# Spins

EFFECTIVE_SPIN_MAGNITUDE = Parameter(name="chi_eff", prior=two_sided_uniform)
PRIMARY_SPIN_MAGNITUDE = Parameter(name="a_1", prior=standard_uniform)
PRIMARY_SPIN_X = Parameter(name="spin_1x", prior=two_sided_uniform)
PRIMARY_SPIN_Y = Parameter(name="spin_1y", prior=two_sided_uniform)
PRIMARY_SPIN_Z = Parameter(name="spin_1z", prior=two_sided_uniform)
SECONDARY_SPIN_MAGNITUDE = Parameter(name="a_2", prior=standard_uniform)
SECONDARY_SPIN_X = Parameter(name="spin_2x", prior=two_sided_uniform)
SECONDARY_SPIN_Y = Parameter(name="spin_2y", prior=two_sided_uniform)
SECONDARY_SPIN_Z = Parameter(name="spin_2z", prior=two_sided_uniform)

# Tilt

COS_TILT_1 = Parameter(name="cos_tilt_1", prior=two_sided_uniform)
COS_TILT_2 = Parameter(name="cos_tilt_2", prior=two_sided_uniform)

# Eccentricity

ECCENTRICITY = Parameter(name="ecc", prior=standard_uniform)

# Redshift

REDSHIFT = Parameter(name="redshift", prior=Uniform(0.0, 10.0, validate_args=True))
