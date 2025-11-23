# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


__all__ = ["__version__"]


import datetime
import os
import subprocess
from typing import Final, Optional


MAJOR_VERSION: Final[int] = 0
MINOR_VERSION: Final[int] = 2
PATCH_VERSION: Final[int] = 1


def get_git_commit_hash() -> Optional[str]:
    """Attempts to get the commit hash from the git repository. Uses :code:`git rev-
    parse HEAD` to retrieve the current commit hash if possible.

    Returns
    -------
    Optional[str]
        The git commit hash or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,  # Raise exception for non-zero exit codes
            # Run command in script directory to ensure path context is correct
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_version() -> str:
    """Determines the final project version string.

    If the environment variable `GWKOKAB_NIGHTLY_BUILD` is set, appends the
    latest git commit hash (or a date fallback) to the base version.

    Returns
    -------
    str
        The version string, potentially including git commit hash or date for
        nightly builds.
    """
    NIGHTLY_ENV_VAR = "GWKOKAB_NIGHTLY_BUILD"
    version = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"

    if os.environ.get(NIGHTLY_ENV_VAR) is None:
        # Return the clean base version for standard release builds.
        return version

    commit_hash = get_git_commit_hash()

    if commit_hash:
        version += f"+g{commit_hash}"
    else:
        # not considering time because it changes too frequently
        utc_now = datetime.datetime.now().strftime("%Y%m%d")
        version += f"+d{utc_now}"

    return version


__version__ = get_version()
