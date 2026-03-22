# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path


def normalize_path(raw_path: str) -> Path:
    """Expands environment variables and tildes, returning a resolved Path object.

    Args:
        raw_path (str): The string path (e.g., "$HOME/data" or "~/docs").

    Returns:
        Path: A fully resolved pathlib.Path object.

    Example:
        >>> normalize_path("$HOME/documents/data.csv")
        PosixPath('/home/user/documents/data.csv')

        >>> normalize_path("~/Desktop/test.txt")
        PosixPath('/home/user/Desktop/test.txt')
    """
    # Expand shell variables ($VAR)
    expanded = os.path.expandvars(raw_path)

    # Expand tilde (~) and resolve to an absolute path
    return Path(expanded).expanduser().resolve()
