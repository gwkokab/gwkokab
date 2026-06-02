# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from typing import Dict, List


def read_json(json_file: str) -> Dict:
    """Read json file and return.

    Parameters
    ----------
    json_file : str
        path of the json file

    Returns
    -------
    dict
        json file content as dict

    Raises
    ------
    ValueError
        If the file is not found or if the file is not a valid json file
    """
    try:
        with open(json_file, "r") as f:
            content = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading configuration: {e}")
    return content


def write_json(json_file: str, content: Dict) -> None:
    """Write a dictionary to a json file.

    Parameters
    ----------
    json_file : str
        path of the json file
    content : dict
        content to write to the json file

    Raises
    ------
    ValueError
        If the file is not writable or if the content is not a valid json serializable object
    """
    try:
        with open(json_file, "w") as f:
            json.dump(content, f, indent=4)
    except (FileNotFoundError, TypeError) as e:
        raise ValueError(f"Error writing configuration: {e}")


def expand_arguments(arg: str, n: int) -> List[str]:
    """Extend the argument with a number of strings.

    .. code:: python

        >>> expand_arguments("physics", 3)
        ["physics_0", "physics_1", "physics_2"]

    Parameters
    ----------
    arg : str
        argument to extend
    n : int
        number of strings to extend

    Returns
    -------
    List[str]
        list of extended arguments
    """
    return [f"{arg}_{i}" for i in range(n)]
