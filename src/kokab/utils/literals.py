# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""Literals used in Kokab."""

from typing import Final, Literal


INFERENCE_DIRECTORY: Final[Literal["inference"]] = "inference"
"""Name of the directory to store inference results."""


POSTERIOR_SAMPLES_FILENAME: Final[Literal["samples.dat"]] = "samples.dat"
"""Name of the file to store posterior samples."""
