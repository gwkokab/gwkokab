#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

import glob
import os
import warnings
from typing_extensions import Callable

import numpy as np

from .aliases import NoisePopInfo


__all__ = ["run_noise_factory"]


def run_noise_factory(npopinfo: NoisePopInfo) -> None:
    filenames = glob.glob(npopinfo.FILENAME_REGEX)
    heads: list[list[int]] = []
    error_fns: list[Callable] = []
    for head, err_fn in npopinfo.ERROR_FUNCS:
        _head = []
        for h in head:
            i = npopinfo.HEADER.index(h)
            _head.append(i)
        heads.append(_head)
        error_fns.append(err_fn)

    index = 0

    for filename in filenames:
        noisey_data = np.empty((npopinfo.SIZE, len(npopinfo.HEADER)))
        data = np.loadtxt(filename)
        for head, err_fn in zip(heads, error_fns):
            noisey_data[:, head] = err_fn(data[head], npopinfo.SIZE)
        nan_mask = np.isnan(noisey_data).any(axis=1)
        masked_noisey_data = noisey_data[~nan_mask]
        count = np.count_nonzero(masked_noisey_data)
        if count == 0:
            warnings.warn(f"Skipping file {index} due to all NaN values", category=UserWarning)
            index += 1
            continue
        if masked_noisey_data.shape[0] == 1:
            masked_noisey_data = masked_noisey_data.reshape(1, -1)
        os.makedirs(os.path.dirname(npopinfo.OUTPUT_DIR.format(index)), exist_ok=True)
        np.savetxt(
            npopinfo.OUTPUT_DIR.format(index),
            masked_noisey_data,
            header=" ".join([h.value for h in npopinfo.HEADER]),
            comments="#",
        )
        index += 1
