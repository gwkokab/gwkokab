# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


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


def get_progress_bar(
    name: str,
    verbose: bool = True,
    text_width: int = 25,
    bar_width: int = 40,
) -> Progress:
    """Create a progress bar with customizable columns.

    Parameters
    ----------
    name : str
        Name to display in the progress bar
    verbose : bool, optional
        Whether to show the progress bar, by default True
    text_width : int, optional
        Width of the name column, by default 25
    bar_width : int, optional
        Width of the progress bar, by default 40

    Returns
    -------
    Progress
        Configured Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(
            ("[bold blue]{name}".format(name=name)).ljust(text_width),
            justify="left",
        ),
        BarColumn(bar_width=bar_width),
        "[progress.percentage]{task.percentage:>3.2f}%",
        "•",
        TimeRemainingColumn(elapsed_when_finished=True),
        "•",
        MofNCompleteColumn(separator=" realizations out of "),
        disable=not verbose,
    )
