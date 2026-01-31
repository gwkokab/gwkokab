# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import enum
from collections import defaultdict
from typing import Any, Callable, Dict, Set, Tuple

import numpy as np

from .utils.transformations import (
    chi_costilt_to_chiz,
    chi_p_from_components,
    chieff,
    chirp_mass,
    delta_m,
    delta_m_to_symmetric_mass_ratio,
    eta_from_q,
    m1_m2_chi1_chi2_costilt1_costilt2_to_chieff,
    m1_m2_chi1z_chi2z_to_chiminus,
    m1_q_to_m2,
    m2_q_to_m1,
    m_det_z_to_m_source,
    m_source_z_to_m_det,
    mass_ratio,
    reduced_mass,
    spin_costilt_from_components,
    spin_magnitude_from_components,
    symmetric_mass_ratio,
    symmetric_mass_ratio_to_delta_m,
    total_mass,
)


class Parameters(str, enum.Enum):
    """Enumeration of common parameter names used in GWKokab."""

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    CHI_1 = "chi_1"
    CHI_2 = "chi_2"
    CHI_MINUS = "chiminus"
    CHIRP_MASS = "chirp_mass"
    CHIRP_MASS_DETECTOR = "chirp_mass_detector"
    CHIRP_MASS_SOURCE = "chirp_mass_source"
    COS_IOTA = "cos_iota"
    COS_TILT_1 = "cos_tilt_1"
    COS_TILT_2 = "cos_tilt_2"
    DELTA_M = "delta_m"
    DETECTION_TIME = "detection_time"
    ECCENTRICITY = "eccentricity"
    EFFECTIVE_SPIN = "chi_eff"
    LUMINOSITY_DISTANCE = "luminosity_distance"
    MASS_RATIO = "mass_ratio"
    MEAN_ANOMALY = "mean_anomaly"
    PHI_1 = "phi_1"
    PHI_12 = "phi_12"
    PHI_2 = "phi_2"
    PHI_ORB = "phi_orb"
    POLARIZATION_ANGLE = "psi"
    PRECESSING_SPIN = "chi_p"
    PRIMARY_MASS_DETECTED = "mass_1"
    PRIMARY_MASS_SOURCE = "mass_1_source"
    PRIMARY_SPIN_MAGNITUDE = "a_1"
    PRIMARY_SPIN_X = "spin_1x"
    PRIMARY_SPIN_Y = "spin_1y"
    PRIMARY_SPIN_Z = "spin_1z"
    REDSHIFT = "redshift"
    REDUCED_MASS = "reduced_mass"
    RIGHT_ASCENSION = "ra"
    SECONDARY_MASS_DETECTED = "mass_2"
    SECONDARY_MASS_SOURCE = "mass_2_source"
    SECONDARY_SPIN_MAGNITUDE = "a_2"
    SECONDARY_SPIN_X = "spin_2x"
    SECONDARY_SPIN_Y = "spin_2y"
    SECONDARY_SPIN_Z = "spin_2z"
    SIN_DECLINATION = "dec"
    SYMMETRIC_MASS_RATIO = "symmetric_mass_ratio"
    TOTAL_MASS = "total_mass"


class RelationMesh:
    def __init__(self):
        self.rules = []
        self._out_edges = defaultdict(list)
        self._all_params = set()

    def add_rule(self, inputs: Tuple[Any, ...], output: Any, func: Callable):
        """Adds a rule.

        Multiple rules can target the same output.
        """
        self.rules.append((inputs, output, func))
        self._all_params.update(inputs)
        if isinstance(output, tuple):
            self._all_params.update(output)
        else:
            self._all_params.add(output)

        for inp in inputs:
            self._out_edges[inp].append(len(self.rules) - 1)

    def resolve(self, initial_state: Dict[Any, Any]) -> Dict[Any, Any]:
        """Iteratively fills in missing parameters based on available rules.

        Does not overwrite values once they are set in the state.
        """
        state = dict(initial_state)
        # Track which rules have been successfully applied
        applied = [False] * len(self.rules)

        while True:
            progress = False
            for i, (inputs, output, func) in enumerate(self.rules):
                if applied[i]:
                    continue

                # Check if we have the necessary ingredients
                if all(inp in state for inp in inputs):
                    # Check if the target parameters are still missing
                    targets = output if isinstance(output, tuple) else (output,)
                    if any(t not in state for t in targets):
                        result = func(*[state[inp] for inp in inputs])

                        if isinstance(output, tuple):
                            # Handle multi-output functions (like coordinate transforms)
                            for name, val in zip(output, result):
                                state.setdefault(name, val)
                        else:
                            state.setdefault(output, result)

                        applied[i] = True
                        progress = True

            # If a full pass resulted in no new parameters, we are done
            if not progress:
                break

        return state

    def derive_only(
        self, initial_state: Dict[Any, Any], targets: Set[Any]
    ) -> Dict[Any, Any]:
        """Derives specified targets, allowing intermediate parameters to be computed as
        needed.
        """
        # Run a standard resolution to fill the state fully
        full_state = self.resolve(initial_state)

        # Return only what the user asked for
        return {t: full_state[t] for t in targets if t in full_state}

    def resolve_from_arrays(
        self, initial_state: np.ndarray, param_order: Tuple[Any, ...]
    ) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        state_dict = {param: initial_state[:, i] for i, param in enumerate(param_order)}
        resolved_dict = self.resolve(state_dict)
        # Sort keys to ensure a deterministic column order in the output
        resolved_order = tuple(sorted(resolved_dict.keys(), key=str))
        resolved_array = np.column_stack(
            [resolved_dict[param] for param in resolved_order]
        )
        return resolved_array, resolved_order


def default_relation_mesh() -> RelationMesh:
    """Constructs the default relation mesh with common gravitational wave parameter
    relations.
    """
    relation_mesh = RelationMesh()

    P = Parameters

    # --- Mass Relations ---
    # fmt: off
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.TOTAL_MASS, total_mass)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.MASS_RATIO, mass_ratio)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.CHIRP_MASS, chirp_mass)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.SYMMETRIC_MASS_RATIO, symmetric_mass_ratio)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.REDUCED_MASS, reduced_mass)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.DELTA_M, delta_m)

    # --- Component Mass Reconstructions ---
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.MASS_RATIO), P.SECONDARY_MASS_SOURCE, m1_q_to_m2)
    relation_mesh.add_rule((P.SECONDARY_MASS_SOURCE, P.MASS_RATIO), P.PRIMARY_MASS_SOURCE, m2_q_to_m1)
    relation_mesh.add_rule((P.PRIMARY_MASS_DETECTED, P.MASS_RATIO), P.SECONDARY_MASS_DETECTED, m1_q_to_m2)
    relation_mesh.add_rule((P.SECONDARY_MASS_DETECTED, P.MASS_RATIO), P.PRIMARY_MASS_DETECTED, m2_q_to_m1)

    # --- Symmetry/Ratio Conversions ---
    relation_mesh.add_rule((P.MASS_RATIO,), P.SYMMETRIC_MASS_RATIO, eta_from_q)
    relation_mesh.add_rule((P.DELTA_M, ), P.SYMMETRIC_MASS_RATIO, delta_m_to_symmetric_mass_ratio)
    relation_mesh.add_rule((P.SYMMETRIC_MASS_RATIO,), P.DELTA_M, symmetric_mass_ratio_to_delta_m)

    # --- Redshift / Source Frame ---
    relation_mesh.add_rule((P.PRIMARY_MASS_DETECTED, P.REDSHIFT), P.PRIMARY_MASS_SOURCE, m_det_z_to_m_source)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.REDSHIFT), P.PRIMARY_MASS_DETECTED, m_source_z_to_m_det)
    relation_mesh.add_rule((P.SECONDARY_MASS_DETECTED, P.REDSHIFT), P.SECONDARY_MASS_SOURCE, m_det_z_to_m_source)
    relation_mesh.add_rule((P.SECONDARY_MASS_SOURCE, P.REDSHIFT), P.SECONDARY_MASS_DETECTED, m_source_z_to_m_det)

    # --- Spin Relations ---
    relation_mesh.add_rule((P.PRIMARY_SPIN_X, P.PRIMARY_SPIN_Y, P.PRIMARY_SPIN_Z), P.PRIMARY_SPIN_MAGNITUDE, spin_magnitude_from_components)
    relation_mesh.add_rule((P.SECONDARY_SPIN_X, P.SECONDARY_SPIN_Y, P.SECONDARY_SPIN_Z), P.SECONDARY_SPIN_MAGNITUDE, spin_magnitude_from_components)
    relation_mesh.add_rule((P.PRIMARY_SPIN_X, P.PRIMARY_SPIN_Y, P.PRIMARY_SPIN_Z), P.CHI_1,spin_costilt_from_components)
    relation_mesh.add_rule((P.SECONDARY_SPIN_X, P.SECONDARY_SPIN_Y, P.SECONDARY_SPIN_Z), P.CHI_2, spin_costilt_from_components)
    relation_mesh.add_rule((P.CHI_1, P.COS_TILT_1), P.PRIMARY_SPIN_Z, chi_costilt_to_chiz)
    relation_mesh.add_rule((P.CHI_2, P.COS_TILT_2), P.SECONDARY_SPIN_Z, chi_costilt_to_chiz)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z), P.EFFECTIVE_SPIN, chieff)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z), P.CHI_MINUS,  m1_m2_chi1z_chi2z_to_chiminus)

    # Combined Spin (Effective)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.CHI_1, P.CHI_2, P.COS_TILT_1, P.COS_TILT_2), P.EFFECTIVE_SPIN, m1_m2_chi1_chi2_costilt1_costilt2_to_chieff)

    # Precessing Spin
    relation_mesh.add_rule((P.CHI_1, P.COS_TILT_1, P.CHI_2, P.COS_TILT_2, P.MASS_RATIO), P.PRECESSING_SPIN, chi_p_from_components)
    # fmt: on

    return relation_mesh
