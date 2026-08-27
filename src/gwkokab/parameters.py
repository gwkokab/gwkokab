# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical parameter names and the relation graph that connects them.

:class:`Parameters` is the single source of truth for how a gravitational-wave source
parameter is spelled on disk and in configuration files; it is imported almost
everywhere as ``from gwkokab.parameters import Parameters as P``.

:class:`RelationMesh` turns the closed-form maps in :mod:`gwkokab.utils.transformations`
into a rule graph over those names, so that given whichever parameters a dataset happens
to carry, every reachable parameter can be derived by forward chaining.
:func:`default_relation_mesh` assembles the standard rule set -- mass relations,
source/detector frame conversions, spin projections and the redshift/luminosity-distance
pair -- and is what backs the ``--derive-parameters`` flag of the synthetic-data CLIs.
"""

import enum
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Set, Tuple

import numpy as np

from .utils.transformations import (
    chi_costilt_to_chiz,
    chi_p_from_components,
    chieff,
    chirp_mass,
    chirp_mass_from_m1_q,
    delta_m,
    delta_m_to_symmetric_mass_ratio,
    eta_from_q,
    m1_m2_chi1_chi2_costilt1_costilt2_to_chieff,
    m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus,
    m1_m2_chi1z_chi2z_to_chiminus,
    m1_m2_chieff_chiminus_to_chi1z_chi2z,
    m1_q_to_m2,
    m2_q_to_m1,
    m_det_z_to_m_source,
    m_source_z_to_m_det,
    mass_ratio,
    q_from_eta,
    reduced_mass,
    spin_costilt_from_components,
    spin_magnitude_from_components,
    symmetric_mass_ratio,
    symmetric_mass_ratio_to_delta_m,
    total_mass,
)


class Parameters(str, enum.Enum):
    """Enumeration of common parameter names used in GWKokab.

    Each member's value is the string used for the parameter in data files, JSON
    configuration and column headers. The enum subclasses :class:`str`, and
    :meth:`__eq__`/:meth:`__hash__` are specialised so a member compares and hashes
    equal to its own value -- ``Parameters.REDSHIFT == "redshift"`` is :data:`True`, and
    a member and its string spelling index the same dictionary entry.
    """

    def __str__(self):
        """Return the parameter's string value.

        Returns
        -------
        str
            The enum member's value, e.g. ``"mass_1_source"``.
        """
        return str(self.value)

    def __hash__(self):
        """Hash the parameter by its string value.

        Returns
        -------
        int
            ``hash`` of the member's value, so a member and its string spelling collide in
            the same dictionary bucket.
        """
        return hash(self.value)

    def __eq__(self, other):
        """Compare equal to plain strings as well as to other members.

        Parameters
        ----------
        other : Any
            The object to compare against.

        Returns
        -------
        bool
            :data:`True` if ``other`` is this member's string value, or the member itself.
        """
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
    r"""A rule graph over gravitational-wave parameters.

    A rule is a triple ``(inputs, output, func)``: whenever every name in ``inputs`` is
    present, ``func`` is applied to their values and its result is stored under
    ``output``. ``output`` may be a tuple, for functions returning several parameters at
    once (for instance :func:`~gwkokab.utils.transformations.m1_m2_chieff_chiminus_to_chi1z_chi2z`).

    :meth:`resolve` forward-chains the rules to a fixed point. Rules are kept in a
    work queue keyed on how many of their inputs are still missing, so each rule fires
    at most once and only after its inputs exist. Several rules may target the same
    output; the first one to fire wins and the rest become no-ops, which means the graph
    tolerates redundant paths (:math:`q` from :math:`\eta` or from :math:`m_1, m_2`)
    without recomputing or contradicting itself.

    Registered rules are held in ``rules``, in insertion order.

    See Also
    --------
    default_relation_mesh : Builds the mesh with the standard GW parameter relations.
    """

    def __init__(self):
        self.rules = []
        self._out_edges = defaultdict(list)
        self._all_params = set()

    def add_rule(self, inputs: Tuple[Any, ...], output: Any, func: Callable):
        """Add a rule to the mesh.

        Multiple rules can target the same output; during :meth:`resolve` the first one
        whose inputs become available is the one that fires.

        Parameters
        ----------
        inputs : Tuple[Any, ...]
            Parameter names ``func`` consumes, in call order.
        output : Any
            Parameter name ``func`` produces, or a tuple of names when ``func`` returns
            several values.
        func : Callable
            The map from the input values to the output value(s).
        """
        self.rules.append((inputs, output, func))
        self._all_params.update(inputs)

        if isinstance(output, tuple):
            self._all_params.update(output)
        else:
            self._all_params.add(output)

        rule_idx = len(self.rules) - 1
        for inp in inputs:
            self._out_edges[inp].append(rule_idx)

    def resolve(self, initial_state: Dict[Any, Any]) -> Dict[Any, Any]:
        """Derive every parameter reachable from the given state.

        Rules are fired by forward chaining until no further rule can be applied: a rule
        becomes eligible once all of its inputs are present, and is skipped if all of its
        outputs already are. Values already in ``initial_state`` are never overwritten.

        Parameters
        ----------
        initial_state : Dict[Any, Any]
            The parameters that are already known, keyed by :class:`Parameters` member (or
            the equivalent string).

        Returns
        -------
        Dict[Any, Any]
            A new dictionary containing the initial state plus every derivable parameter.
        """
        state = dict(initial_state)

        n_rules = len(self.rules)
        missing = [0 for _ in range(n_rules)]
        applied = [False for _ in range(n_rules)]

        # Count missing inputs for each rule
        for i, (inputs, _, _) in enumerate(self.rules):
            missing[i] = sum(inp not in state for inp in inputs)

        # Initialize queue with rules that are already satisfied
        queue = deque(i for i in range(n_rules) if missing[i] == 0)

        while queue:
            i = queue.popleft()
            if applied[i]:
                continue

            inputs, output, func = self.rules[i]
            targets = output if isinstance(output, tuple) else (output,)

            # Skip if all targets already exist
            if all(t in state for t in targets):
                applied[i] = True
                continue

            result = func(*(state[inp] for inp in inputs))

            newly_added = []
            if isinstance(output, tuple):
                for name, val in zip(output, result):
                    if name not in state:
                        state[name] = val
                        newly_added.append(name)
            else:
                if output not in state:
                    state[output] = result
                    newly_added.append(output)

            applied[i] = True

            # Update dependent rules
            for param in newly_added:
                for j in self._out_edges.get(param, []):
                    missing[j] -= 1
                    if missing[j] == 0:
                        queue.append(j)

        return state

    def derive_only(
        self, initial_state: Dict[Any, Any], targets: Set[Any]
    ) -> Dict[Any, Any]:
        """Derive a chosen subset of parameters.

        The full resolution is run first, so intermediate parameters needed to reach the
        targets are computed even though they are not returned.

        Parameters
        ----------
        initial_state : Dict[Any, Any]
            The parameters that are already known.
        targets : Set[Any]
            The parameter names to return.

        Returns
        -------
        Dict[Any, Any]
            The subset of the resolved state restricted to ``targets``. Targets that could
            not be derived are silently absent.
        """
        # Run a standard resolution to fill the state fully
        full_state = self.resolve(initial_state)

        # Return only what the user asked for
        return {t: full_state[t] for t in targets if t in full_state}

    def resolve_from_arrays(
        self, initial_state: np.ndarray, param_order: Tuple[Any, ...]
    ) -> Tuple[np.ndarray, Tuple[Any, ...]]:
        """Resolve a 2-D array of samples whose columns are named by ``param_order``.

        Parameters
        ----------
        initial_state : np.ndarray
            Array of shape ``(n_samples, len(param_order))``; column ``i`` holds the values
            of ``param_order[i]``.
        param_order : Tuple[Any, ...]
            Names of the columns of ``initial_state``.

        Returns
        -------
        Tuple[np.ndarray, Tuple[Any, ...]]
            The resolved samples and the names of their columns. Columns are sorted by name
            so the output order is deterministic and independent of ``param_order``.
        """
        state_dict = {param: initial_state[:, i] for i, param in enumerate(param_order)}
        resolved_dict = self.resolve(state_dict)
        # Sort keys to ensure a deterministic column order in the output
        resolved_order = tuple(sorted(resolved_dict.keys(), key=str))
        resolved_array = np.column_stack([
            resolved_dict[param] for param in resolved_order
        ])
        return resolved_array, resolved_order


def default_relation_mesh() -> RelationMesh:
    r"""Build the default mesh of common gravitational-wave parameter relations.

    The registered rules cover mass combinations (total mass, mass ratio, chirp mass,
    symmetric mass ratio, reduced mass, :math:`\delta_m`) and their inversions,
    source/detector frame conversions through redshift, spin magnitude and tilt from
    Cartesian components, the effective and precessing spins, and the
    redshift/luminosity-distance pair through the default cosmology.

    Returns
    -------
    RelationMesh
        A freshly built mesh. The cosmology is captured at call time, so the mesh
        reflects the value of ``GWKOKAB_DEFAULT_COSMOLOGY`` as of that moment.
    """
    from gwkokab.cosmology import default_cosmology

    cosmo = default_cosmology()

    relation_mesh = RelationMesh()

    P = Parameters

    # --- Mass Relations ---
    # fmt: off
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.TOTAL_MASS, total_mass)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.MASS_RATIO, mass_ratio)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.CHIRP_MASS, chirp_mass)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.MASS_RATIO), P.CHIRP_MASS, chirp_mass_from_m1_q)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.SYMMETRIC_MASS_RATIO, symmetric_mass_ratio)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.REDUCED_MASS, reduced_mass)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE), P.DELTA_M, delta_m)

    # --- Component Mass Reconstructions ---
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.MASS_RATIO), P.SECONDARY_MASS_SOURCE, m1_q_to_m2)
    relation_mesh.add_rule((P.SECONDARY_MASS_SOURCE, P.MASS_RATIO), P.PRIMARY_MASS_SOURCE, m2_q_to_m1)
    relation_mesh.add_rule((P.PRIMARY_MASS_DETECTED, P.MASS_RATIO), P.SECONDARY_MASS_DETECTED, m1_q_to_m2)
    relation_mesh.add_rule((P.SECONDARY_MASS_DETECTED, P.MASS_RATIO), P.PRIMARY_MASS_DETECTED, m2_q_to_m1)
    relation_mesh.add_rule((P.CHIRP_MASS, P.MASS_RATIO), P.PRIMARY_MASS_SOURCE, lambda Mc, q: Mc * (1 + q) ** 0.2 * q ** (-0.6))

    # --- Symmetry/Ratio Conversions ---
    relation_mesh.add_rule((P.MASS_RATIO,), P.SYMMETRIC_MASS_RATIO, eta_from_q)
    relation_mesh.add_rule((P.SYMMETRIC_MASS_RATIO,), P.MASS_RATIO, q_from_eta)
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
    relation_mesh.add_rule((P.PRIMARY_SPIN_X, P.PRIMARY_SPIN_Y, P.PRIMARY_SPIN_Z), P.CHI_1, spin_costilt_from_components)
    relation_mesh.add_rule((P.SECONDARY_SPIN_X, P.SECONDARY_SPIN_Y, P.SECONDARY_SPIN_Z), P.CHI_2, spin_costilt_from_components)
    relation_mesh.add_rule((P.CHI_1, P.COS_TILT_1), P.PRIMARY_SPIN_Z, chi_costilt_to_chiz)
    relation_mesh.add_rule((P.CHI_2, P.COS_TILT_2), P.SECONDARY_SPIN_Z, chi_costilt_to_chiz)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z), P.EFFECTIVE_SPIN, chieff)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z), P.CHI_MINUS, m1_m2_chi1z_chi2z_to_chiminus)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.EFFECTIVE_SPIN, P.CHI_MINUS), (P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z), m1_m2_chieff_chiminus_to_chi1z_chi2z)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.CHI_1, P.CHI_2, P.COS_TILT_1, P.COS_TILT_2), P.CHI_MINUS, m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus)

    # Combined Spin (Effective)
    relation_mesh.add_rule((P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.CHI_1, P.CHI_2, P.COS_TILT_1, P.COS_TILT_2), P.EFFECTIVE_SPIN, m1_m2_chi1_chi2_costilt1_costilt2_to_chieff)

    # Precessing Spin
    relation_mesh.add_rule((P.CHI_1, P.COS_TILT_1, P.CHI_2, P.COS_TILT_2, P.MASS_RATIO), P.PRECESSING_SPIN, chi_p_from_components)

    # Redshift-Luminosity Distance Relation
    relation_mesh.add_rule((P.REDSHIFT,), P.LUMINOSITY_DISTANCE, lambda z: cosmo.z_to_DL(z))
    relation_mesh.add_rule((P.LUMINOSITY_DISTANCE,), P.REDSHIFT, lambda dL: cosmo.DL_to_z(dL))
    # fmt: on

    return relation_mesh
