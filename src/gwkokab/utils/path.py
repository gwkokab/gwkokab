# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Filesystem path helpers."""

import os
from pathlib import Path


def normalize_path(raw_path: str | os.PathLike) -> Path:
    """Expand environment variables and ``~``, then resolve to an absolute path.

    Parameters
    ----------
    raw_path : str | os.PathLike
        The path to normalize, possibly containing shell variables (``$HOME/data``)
        or a leading tilde (``~/docs``).

    Returns
    -------
    Path
        A fully resolved, absolute :class:`pathlib.Path`.

    Examples
    --------
    .. code-block:: python

        >>> normalize_path("$HOME/documents/data.csv")  # doctest: +SKIP
        PosixPath('/home/user/documents/data.csv')
        >>> normalize_path(Path("~/Desktop/test.txt"))  # doctest: +SKIP
        PosixPath('/home/user/Desktop/test.txt')
    """
    path_str = os.fspath(raw_path)

    # Expand shell variables ($VAR)
    expanded = os.path.expandvars(path_str)

    # Expand tilde (~) and resolve to an absolute path
    return Path(expanded).expanduser().resolve()
