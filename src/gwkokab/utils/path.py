# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path


def normalize_path(raw_path: str | os.PathLike) -> Path:
    """Expands environment variables and tildes, returning a resolved Path object.

    Args:
        raw_path (str | os.PathLike):
            The string path (e.g., "$HOME/data" or "~/docs").

    Returns:
        Path:
            A fully resolved pathlib.Path object.

    Example:
        >>> normalize_path("$HOME/documents/data.csv")
        PosixPath('/home/user/documents/data.csv')

        >>> normalize_path(Path("~/Desktop/test.txt"))
        PosixPath('/home/user/Desktop/test.txt')
    """
    path_str = os.fspath(raw_path)

    # Expand shell variables ($VAR)
    expanded = os.path.expandvars(path_str)

    # Expand tilde (~) and resolve to an absolute path
    return Path(expanded).expanduser().resolve()
