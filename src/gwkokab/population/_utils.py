# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def ensure_dat_extension(filename: str) -> str:
    """Transform a filename to end with .dat if it does not have an extension.

    Parameters
    ----------
    filename : str
        Name of the file

    Returns
    -------
    str
        Filename ending with .dat

    Raises
    ------
    ValueError
        If filename has an extension other than .dat
    """
    if filename.endswith(".dat"):
        return filename
    elif "." not in filename:
        return filename + ".dat"
    else:
        ext = filename.split(".")[-1]
        raise ValueError(
            f"Invalid filename {filename!r}: found extension '.{ext}' but must end with '.dat' or have no extension"
        )
