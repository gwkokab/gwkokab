# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import datetime
import os
import subprocess
from typing import Final, Optional


MAJOR_VERSION: Final[int] = 0
MINOR_VERSION: Final[int] = 2
PATCH_VERSION: Final[int] = 1
BASE_VERSION: Final[str] = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"


def get_git_commit_hash() -> Optional[str]:
    """Attempts to get the commit hash from the git repository.

    Returns
    -------
    Optional[str]
        The git commit hash or None if unavailable.
    """
    try:
        # Use 'git rev-parse HEAD' to get the commit hash
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

    if os.environ.get(NIGHTLY_ENV_VAR) is None:
        # Return the clean base version for standard release builds.
        return BASE_VERSION

    commit_hash = get_git_commit_hash()

    if commit_hash:
        version = f"{BASE_VERSION}+g{commit_hash}"
    else:
        karachi_tz = datetime.timezone(datetime.timedelta(hours=5))
        # not considering time because it changes too frequently
        now = datetime.datetime.now(tz=karachi_tz).strftime("%Y%m%d")
        version = f"{BASE_VERSION}+d{now}"

    return version


__version__ = get_version()
