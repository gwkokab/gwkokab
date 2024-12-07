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

from collections.abc import Callable, Sequence
from typing import Literal, overload, Tuple

import equinox as eqx
import h5py
import jax
from jax import numpy as jnp
from jax.numpy import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Array

from gwkokab.parameters import (
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)

from ._vt_abc import VolumeTimeSensitivityInterface


def _check_and_get(name: str, f: h5py.File) -> Array:
    r"""Check if the name is in the file and return the value.

    :param name: name of the dataset
    :param f: h5py file object
    :raises ValueError: if the name is not in the file
    :return: the value of the dataset
    """
    if name in f:
        return f[name][()]
    else:
        raise ValueError(f"{name} not found in file.")


class PopModelsVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    logVT_interpolator: RegularGridInterpolator
    m_min: float = eqx.field(converter=float, init=False, static=True)

    def __init__(
        self,
        parameters: Sequence[str],
        filename: str,
        zero_spin: bool,
        scale_factor: int = 1,
        m_min: float = 0.5,
    ) -> None:
        self.m_min = m_min
        if (
            PRIMARY_MASS_SOURCE.name not in parameters
            or SECONDARY_MASS_SOURCE.name not in parameters
        ):
            raise ValueError(
                f"{self.__class__.__name__} requires {PRIMARY_MASS_SOURCE.name} and {SECONDARY_MASS_SOURCE.name}"
            )
        spin_required = (
            PRIMARY_SPIN_MAGNITUDE.name in parameters
            or SECONDARY_SPIN_MAGNITUDE.name in parameters
        )

        if not spin_required and not zero_spin:
            VT, logM, qtilde, s1z, s2z = self._load_file(filename, zero_spin)
            VT = trapezoid(VT, s2z, axis=-1)
            VT = trapezoid(VT, s1z, axis=-1)
            del s1z, s2z
            points = (logM, qtilde)
            self.shuffle_indices = [
                parameters.index(PRIMARY_MASS_SOURCE.name),
                parameters.index(SECONDARY_MASS_SOURCE.name),
            ]
        elif spin_required and not zero_spin:
            VT, logM, qtilde, s1z, s2z = self._load_file(filename, zero_spin)
            points = (logM, qtilde, s1z, s2z)
            self.shuffle_indices = [
                parameters.index(PRIMARY_MASS_SOURCE.name),
                parameters.index(SECONDARY_MASS_SOURCE.name),
                parameters.index(PRIMARY_SPIN_MAGNITUDE.name),
                parameters.index(SECONDARY_SPIN_MAGNITUDE.name),
            ]
        elif not spin_required and zero_spin:
            VT, logM, qtilde = self._load_file(filename, zero_spin)
            points = (logM, qtilde)
            self.shuffle_indices = [
                parameters.index(PRIMARY_MASS_SOURCE.name),
                parameters.index(SECONDARY_MASS_SOURCE.name),
            ]
        else:
            raise ValueError(
                "Spin is required by the model, but zero_spin is set to True."
            )

        logVT = jnp.log(VT * scale_factor)

        self.logVT_interpolator = RegularGridInterpolator(
            points, logVT, bounds_error=False, fill_value=-jnp.inf
        )

    @overload
    @staticmethod
    def _load_file(
        self, filename: str, zero_spin: Literal[False]
    ) -> Tuple[Array, Array, Array, Array, Array]: ...

    @overload
    @staticmethod
    def _load_file(
        self, filename: str, zero_spin: Literal[True]
    ) -> Tuple[Array, Array, Array]: ...

    @staticmethod
    def _load_file(filename: str, zero_spin: bool):
        with h5py.File(filename, "r") as f:
            VT = _check_and_get("VT", f)
            logM = _check_and_get("logM", f)
            qtilde = _check_and_get("qtilde", f)

            if not zero_spin:
                s1z = _check_and_get("s1z", f)
                s2z = _check_and_get("s2z", f)

        if zero_spin:
            return VT, logM, qtilde
        else:
            return VT, logM, qtilde, s1z, s2z

    @eqx.filter_jit
    def mass_grid_coords_inverse(
        self, logM: Array, q_tilde: Array
    ) -> Tuple[Array, Array]:
        r"""Converts the :math:`\log{(M)}`, :math:`\tilde{q}` coordinates to m1, m2
        masses.

        .. math::
            m_1 = \frac{M(M-m_{\text{min}})}{M(1+\tilde{q}) - 2m_{\text{min}}\tilde{q}}

        .. math::
            m_2 = \frac{M(M\tilde{q} + m_{\text{min}}(1-2\tilde{q}))}{M(1+\tilde{q}) - 2m_{\text{min}}\tilde{q}}

        where :math:`M = \exp{(\log{(M)})}`.

        source: https://gitlab.com/dwysocki/bayesian-parametric-population-models/-/blob/master/src/pop_models/astro_models/gw_ifo_vt.py?ref_type=heads#L1025-1037

        :param logM: Logarithm of the total mass
        :param q_tilde: Reduced mass ratio
        :param m_min: Minimum mass
        :return: Primary and secondary masses
        """
        M = jnp.exp(logM)
        denominator = M * (1 + q_tilde) - 2 * self.m_min * q_tilde
        factor = (M - self.m_min) / denominator
        m1 = M * factor
        m2 = M - m1
        return m1, m2

    @eqx.filter_jit
    def mass_grid_coords(
        self, m1: Array, m2: Array, eps: float = 1e-8
    ) -> Tuple[Array, Array]:
        r"""Converts the :math:`m_1`, :math:`m_2` masses to :math:`\log{(M)}`,
        :math:`\tilde{q}` coordinates.

        .. note::

            When computing :math:`\tilde{q}`, there is a coordinate singularity at
            :math:`m_1 = m_2 = m_{\text{min}}`. While normally, equal mass corresponds to
            :math:`\tilde{q} = 1`, here the most extreme mass ratio is also equal mass, so
            we should have :math:`\tilde{q}` = 0. Here we break the singularity by just
            forcing :math:`\tilde{q} = 1` in these cases. We also allow for some numerical
            wiggle room beyond that point, by also including any points within some small
            epsilon of the singularity. VT does not change on a fast enough scale for this
            to make any measurable difference.

        source: https://gitlab.com/dwysocki/bayesian-parametric-population-models/-/blob/master/src/pop_models/astro_models/gw_ifo_vt.py?ref_type=heads#L998-1022

        :param m1: Primary mass
        :param m2: Secondary mass
        :param m_min: Minimum mass
        :param eps: tolerance for numerical stability, defaults to 1e-8
        :return: coordinates in the form of :math:`\log{(M)}`, :math:`\tilde{q}`
        """
        M = m1 + m2

        logM = jnp.log(M)

        i_good = M > 2 * self.m_min + eps

        # ~i_good is filled with random values
        m1 = jnp.where(i_good, m1, 1.0)
        m2 = jnp.where(i_good, m2, 2.0)
        M = jnp.where(i_good, M, 1.0)

        qtilde = jnp.where(
            i_good, (M * (m2 - self.m_min)) / (m1 * (M - 2 * self.m_min)), 1.0
        )

        return logM, qtilde

    def get_logVT(self) -> Callable[[Array], Array]:
        @eqx.filter_jit
        def _logVT(x: Array) -> Array:
            xs = x[..., self.shuffle_indices]
            m1, m2 = xs[..., 0], xs[..., 1]
            logM, qtilde = self.mass_grid_coords(m1, m2)
            query = (logM, qtilde, *xs[..., 2:])
            return self.logVT_interpolator(query)

        return _logVT

    def get_vmapped_logVT(self) -> Callable[[Array], Array]:
        return jax.vmap(self.get_logVT(), in_axes=0, out_axes=0)
