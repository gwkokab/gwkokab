# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


def name_endswith_dat_or_no_extension(filename: str) -> str:
    """Transform a filename to end with .dat if it does not have an extension.

    :param filename: name of the file
    :raises ValueError: if filename has an extension other than .dat
    :return: filename ending with .dat
    """
    if filename.endswith(".dat"):
        return filename
    elif "." not in filename:
        return filename + ".dat"
    else:
        raise ValueError(
            f"Invalid filename {filename!r}: must end with .dat or have no extension"
        )


def get_progress_bar(name: str, verbose: bool = True) -> Progress:
    progress_bar_text_witdh = 25
    return Progress(
        SpinnerColumn(),
        TextColumn(
            ("[bold blue]{name}".format(name=name)).ljust(progress_bar_text_witdh),
            justify="left",
        ),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.2f}%",
        "•",
        TimeRemainingColumn(elapsed_when_finished=True),
        "•",
        MofNCompleteColumn(separator=" realizations out of "),
        disable=not verbose,
    )
