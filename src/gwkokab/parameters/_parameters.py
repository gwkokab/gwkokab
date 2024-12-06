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


import sys
from collections.abc import Mapping, Sequence

import equinox as eqx
from numpyro.distributions import Distribution, ImproperUniform, Uniform
from numpyro.distributions.constraints import real


class Parameter(eqx.Module):
    """Dataclass for a parameter.

    .. code::

        >>> from gwkokab.parameters import Parameter
        >>> from numpyro.distributions import Uniform
        >>> Mc = Parameter(name="chirp_mass", prior=Uniform(0.5, 300.0, validate_args=True))
        >>> Mc.name
        'chirp_mass'
        >>> Mc.prior.low
        0.5
        >>> Mc.prior.high
        300.0
    """

    name: str = eqx.field(static=True, metadata={"help": "Name of the parameter."})
    prior: Distribution = eqx.field(
        default=ImproperUniform(
            support=real, batch_shape=(), event_shape=(), validate_args=True
        ),
    )

    def __repr__(self):
        args = ", ".join(
            [f"{k}={getattr(self.prior, k)}" for k in self.prior.arg_constraints.keys()]
        )
        return f"Parameter(name={self.name}, prior={self.prior.__class__.__name__}({args}))"


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


class _Available:
    """Available parameters.

    .. code::

        >>> from gwkokab.parameters import available
        >>> Mc_by_var = available["CHIRP_MASS"]
        >>> Mc_by_var.name
        'chirp_mass'
        >>> Mc_by_name = available["chirp_mass"]
        >>> Mc_by_name.name
        'chirp_mass'

    Original implementation is in
    `wcosmo.astropy.available <https://github.com/ColmTalbot/wcosmo/blob/d15ee7d158b83226dcf0e1f319d96883472a05a5/wcosmo/astropy.py#L452-L464>`_
    """

    names_to_keys: Mapping[str, Parameter] = {
        CHIRP_MASS.name: CHIRP_MASS,
        COS_TILT_1.name: COS_TILT_1,
        COS_TILT_2.name: COS_TILT_2,
        ECCENTRICITY.name: ECCENTRICITY,
        EFFECTIVE_SPIN_MAGNITUDE.name: EFFECTIVE_SPIN_MAGNITUDE,
        MASS_RATIO.name: MASS_RATIO,
        PRIMARY_MASS_DETECTED.name: PRIMARY_MASS_DETECTED,
        PRIMARY_MASS_SOURCE.name: PRIMARY_MASS_SOURCE,
        PRIMARY_SPIN_MAGNITUDE.name: PRIMARY_SPIN_MAGNITUDE,
        PRIMARY_SPIN_X.name: PRIMARY_SPIN_X,
        PRIMARY_SPIN_Y.name: PRIMARY_SPIN_Y,
        PRIMARY_SPIN_Z.name: PRIMARY_SPIN_Z,
        REDSHIFT.name: REDSHIFT,
        REDUCED_MASS.name: REDUCED_MASS,
        SECONDARY_MASS_DETECTED.name: SECONDARY_MASS_DETECTED,
        SECONDARY_MASS_SOURCE.name: SECONDARY_MASS_SOURCE,
        SECONDARY_SPIN_MAGNITUDE.name: SECONDARY_SPIN_MAGNITUDE,
        SECONDARY_SPIN_X.name: SECONDARY_SPIN_X,
        SECONDARY_SPIN_Y.name: SECONDARY_SPIN_Y,
        SECONDARY_SPIN_Z.name: SECONDARY_SPIN_Z,
        SYMMETRIC_MASS_RATIO.name: SYMMETRIC_MASS_RATIO,
    }

    params: Sequence[str | Parameter] = [
        "CHIRP_MASS",
        "COS_TILT_1",
        "COS_TILT_2",
        "ECCENTRICITY",
        "EFFECTIVE_SPIN_MAGNITUDE",
        "MASS_RATIO",
        "PRIMARY_MASS_DETECTED",
        "PRIMARY_MASS_SOURCE",
        "PRIMARY_SPIN_MAGNITUDE",
        "PRIMARY_SPIN_X",
        "PRIMARY_SPIN_Y",
        "PRIMARY_SPIN_Z",
        "REDSHIFT",
        "REDUCED_MASS",
        "SECONDARY_MASS_DETECTED",
        "SECONDARY_MASS_SOURCE",
        "SECONDARY_SPIN_MAGNITUDE",
        "SECONDARY_SPIN_X",
        "SECONDARY_SPIN_Y",
        "SECONDARY_SPIN_Z",
        "SYMMETRIC_MASS_RATIO",
    ]

    def keys(self):
        return self.params + list(self.names_to_keys.keys())

    def __getitem__(self, key: str) -> Parameter:
        if key in self.names_to_keys:
            return self.names_to_keys[key]
        return getattr(sys.modules[__name__], key)

    def __repr__(self) -> str:
        return repr(self.keys())


available = _Available()
