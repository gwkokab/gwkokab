# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""Literals used in GWKokab Analysis."""

from typing import Final, Literal


INFERENCE_OUTPUT_FILENAME: Final[Literal["inference_data.hdf5"]] = "inference_data.hdf5"
"""Name of the file to store inference data in HDF5 format."""

SAMPLES_GROUP_NAME: Final[Literal["samples"]] = "samples"
"""Name of the group in the HDF5 file to store samples."""

CHAIN_GROUP_FORMAT: Final[Literal["chain_{chain_id}"]] = "chain_{chain_id}"
"""Format string for the group names in the HDF5 file to store chains."""
