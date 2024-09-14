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

import json
from typing_extensions import List


def expand_arguments(arg: str, n: int) -> List[str]:
    r"""Extend the argument with a number of strings.

    .. code:: python

        >>> expand_arguments("physics", 3)
        ["physics_0", "physics_1", "physics_2"]

    :param arg: argument to extend
    :param n: number of strings to extend
    :return: list of extended arguments
    """
    return list(map(lambda i: arg + f"_{i}", range(n)))


def flowMC_json_read_and_process(json_file: str) -> dict:
    """
    Convert a json file to a dictionary
    """
    with open(json_file, "r") as f:
        flowMC_json = json.load(f)

    flowMC_json["data_dump_kwargs"]["out_dir"] = "sampler_data"

    flowMC_json["local_sampler_kwargs"]["jit"] = True
    flowMC_json["local_sampler_kwargs"]["sampler"] = "MALA"

    flowMC_json["nf_model_kwargs"]["model"] = "MaskedCouplingRQSpline"

    flowMC_json["sampler_kwargs"]["data"] = None
    flowMC_json["sampler_kwargs"]["logging"] = True
    flowMC_json["sampler_kwargs"]["outdir"] = "inf-plot"
    flowMC_json["sampler_kwargs"]["precompile"] = False
    flowMC_json["sampler_kwargs"]["use_global"] = True
    flowMC_json["sampler_kwargs"]["verbose"] = False

    return flowMC_json
