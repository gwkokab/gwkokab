# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import enum


class Parameters(enum.Enum):
    CHIRP_MASS = "chirp_mass"
    CHIRP_MASS_SOURCE = "chirp_mass_source"
    COS_IOTA = "cos_iota"
    COS_TILT_1 = "cos_tilt_1"
    COS_TILT_2 = "cos_tilt_2"
    DETECTION_TIME = "detection_time"
    ECCENTRICITY = "eccentricity"
    EFFECTIVE_SPIN_MAGNITUDE = "chi_eff"
    MASS_RATIO = "mass_ratio"
    MEAN_ANOMALY = "mean_anomaly"
    PHI_1 = "phi_1"
    PHI_12 = "phi_12"
    PHI_2 = "phi_2"
    PHI_ORB = "phi_orb"
    POLARIZATION_ANGLE = "psi"
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
