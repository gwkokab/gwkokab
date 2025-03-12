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


from itertools import product
from multiprocessing import pool
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
from jaxtyping import ArrayLike
from numpyro.distributions.distribution import DistributionLike

from gwkokab.utils.tools import error_if


def wipe_log_rate(
    nf_samples: np.ndarray,
    nf_samples_mapping: Dict[str, int],
    constants: Dict[str, Union[int, float]],
) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
    """Set the log rate parameters to zero and remove them from the samples.

    Parameters
    ----------
    nf_samples : Array
        Normalizing flow samples.
    nf_samples_mapping : Dict[str, int]
        Mapping of the normalizing flow samples.
    constants : Dict[str, Union[int, float]]
        Constants.

    Returns
    -------
    Tuple[Array, Dict[str, Union[int, float]]]
        The normalizing flow samples and the updated constants.
    """
    for key in list(nf_samples_mapping.keys()).copy():
        if key.startswith("log_rate"):
            index = nf_samples_mapping.pop(key)
            constants[key] = 0.0
            nf_samples = np.delete(nf_samples, index, axis=-1)
            for key, value in nf_samples_mapping.items():
                if value > index:
                    nf_samples_mapping[key] -= 1
    return nf_samples, constants


def _compute_marginal_probs(
    probs_array: np.ndarray,
    axis: int,
    domain: List[Tuple[float, float, int]],
) -> np.ndarray:
    """Compute the marginal probabilities of a model.

    The function computes the marginal probabilities of a model by summing over the
    specified axis.

    Parameters
    ----------
    probs_array : Array
        The probabilities of the model.
    axis : int
        The axis along which to compute the marginal probabilities.
    domain : List[Tuple[float, float, int]]
        The domain of the axis.

    Returns
    -------
    Array
        The marginal probabilities of the model.
    """
    assert axis < probs_array.ndim, "Axis must be less than the number of dimensions."
    j = 0
    marginal_density = probs_array
    for i, (start, end, num_points) in enumerate(domain):
        if i == axis:
            continue
        num_points = int(num_points)
        marginal_density = np.trapezoid(
            y=marginal_density,
            x=np.linspace(start, end, num_points),
            axis=i - j,
        )
        j += 1

    return marginal_density


def get_all_marginals(
    probs: np.ndarray,
    domains: List[Tuple[float, float, int]],
) -> List[np.ndarray]:
    """Compute marginal probabilities for all axes.

    Parameters
    ----------
    probs : Array
        The probability array.
    domains : List[Tuple[float, float, int]]
        List of domains for each axis.

    Returns
    -------
    List[Array]
        List of marginal probability arrays, one for each axis.
    """
    return [_compute_marginal_probs(probs, axis, domains) for axis in range(probs.ndim)]


def save_probs(
    ppd_array: np.ndarray,
    marginal_probs: List[np.ndarray],
    filename: str,
    domains: List[Tuple[float, float, int]],
    headers: List[str],
) -> None:
    """Save the PPD and marginal probabilities to a file.

    Parameters
    ----------
    ppd_array : Array
        The ppd array.
    marginal_probs : List[Array]
        List of marginal probabilities.
    filename : str
        The name of the file to save the PPD and marginal probabilities.
    domains : List[Tuple[float, float, int]]
        List of domains for each axis
    headers : List[str]
        List of headers for the PPD and marginal probabilities
    """
    error_if(
        ppd_array.ndim != len(domains),
        AssertionError,
        "Number of ranges must match the number of dimensions of the PPD array.",
    )
    error_if(
        ppd_array.ndim != len(headers),
        AssertionError,
        "Number of headers must match the number of dimensions of the PPD array.",
    )

    with h5py.File(filename, "w") as f:
        f.create_dataset("domains", data=np.array(domains))
        f.create_dataset("headers", data=np.array(headers, dtype="S"))
        f.create_dataset("ppd", data=ppd_array)
        marginal_probs_group = f.create_group("marginals")
        for marginal_prob, head in zip(marginal_probs, headers):
            marginal_probs_group.create_dataset(head, data=marginal_prob)


def compute_and_save_ppd(
    model: DistributionLike,
    nf_samples: np.ndarray,
    domains: List[Tuple[float, float, int]],
    output_file: str,
    parameters: List[str],
    constants: Dict[str, ArrayLike],
    nf_samples_mapping: Dict[str, int],
    n_threads: int = 1,
) -> None:
    xx_mesh = [np.linspace(a, b, int(n)) for a, b, n in domains]
    shape = tuple(d[2] for d in domains)

    def _compute_probs(params: np.ndarray) -> np.ndarray:
        _model = model(
            **constants,
            **{k: params[v] for k, v in nf_samples_mapping.items()},
            validate_args=True,
        )
        prob_value = np.asarray(
            [np.exp(_model.log_prob(np.asarray(k))) for k in product(*xx_mesh)]
        )
        del params
        del _model
        return prob_value

    with pool.ThreadPool(n_threads) as p:
        prob_values = np.asarray(
            [
                p.apply_async(_compute_probs, args=(nf_samples[i],)).get()
                for i in range(nf_samples.shape[0])
            ]
        )

        p.close()
        p.join()

    del nf_samples
    prob_values = prob_values.reshape(-1, *shape[:-1])
    prob_values = np.moveaxis(prob_values, 0, -1)
    ppd_values = np.mean(prob_values, axis=-1)
    marginals = get_all_marginals(prob_values, domains)
    save_probs(ppd_values, marginals, output_file, domains, parameters)
