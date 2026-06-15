# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

import h5py
import jax
import numpy as np
import popsummary as ps
from jax import jit, numpy as jnp
from jaxtyping import Array
from matplotlib import pyplot as plt

from gwkokab.analysis.core.utils import (
    read_attrs_from_hdf5,
    read_from_hdf5,
    write_to_hdf5,
)
from gwkokab.cosmology import default_cosmology
from gwkokab.parameters import Parameters as P


@ft.partial(jit, static_argnums=(1, 3))
def _derive_marginal_densities(
    probs_array: Array,
    axis: int,
    domains: list[Array],
    normalize: bool = True,
) -> Array:
    """Derive marginal densities from a joint density array by integrating out all
    dimensions except the specified axis.

    Parameters
    ----------
    probs_array : Array
        An array representing the joint density over multiple dimensions.
    axis : int
        The index of the dimension to retain (i.e., the axis along which to compute the marginal density).
    domains : list[Array]
        A list of arrays representing the domain values for each dimension of the joint density. The length of this list should match the number of dimensions in :code:`probs_array`.
    normalize : bool, optional
        Whether to normalize the marginal densities, by default True

    Returns
    -------
    Array
        The computed marginal densities along the specified axis.
    """
    marginal_density = probs_array
    j = 0
    for i, domain in enumerate(domains):
        if i == axis:
            continue
        reduction_axis = i - j
        marginal_density = jnp.trapezoid(
            y=marginal_density, x=domain, axis=reduction_axis
        )
        j += 1

    if normalize:
        Z = jnp.trapezoid(marginal_density, domains[axis], axis=-1)

        # Avoid Division-by-Zero via safe masking
        Z_safe = jnp.where(Z > 0, Z, 1.0)
        marginal_density = jnp.where(Z > 0, marginal_density / Z_safe, 0.0)

    return marginal_density


def calculate_marginals_over_axes(
    probs: Array, domains: list[Array], normalize: list[bool] | None = None
) -> list[Array]:
    """Calculate marginal densities for each axis of a joint density array.

    This function iteratively integrates out all dimensions except the specified axis
    to derive the marginal densities for each axis.

    Parameters
    ----------
    probs : Array
        An array representing the joint density over multiple dimensions.
    domains : list[Array]
        A list of arrays representing the domain values for each dimension of the joint
        density. The length of this list should match the number of dimensions in
        :code:`probs`.
    normalize : list[bool] | None, optional
        A list of booleans indicating whether to normalize the marginal densities for
        each dimension. If None, all dimensions will be normalized, by default None

    Returns
    -------
    list[Array]
        A list of arrays representing the marginal densities for each dimension.
    """
    if normalize is None:
        normalize = [True] * len(domains)
    ndim = len(domains)
    return [
        _derive_marginal_densities(probs, axis, domains, normalize=normalize[axis])
        for axis in range(ndim)
    ]


def calculate_dist_layouts(
    shaped_values: list[tuple[int, int] | int],
) -> list[tuple[int, ...]]:
    """Calculate the layout of the distribution's support based on its shaped values.

    Parameters
    ----------
    shaped_values : list[tuple[int, int]  |  int]
        A list of shaped values for the distribution. Each element can be either a tuple
        representing a multi-dimensional support (with the second element indicating
        the number of dimensions), or an integer representing a one-dimensional support.

    Returns
    -------
    list[tuple[int, ...]]
        A list of tuples representing the layout of the distribution's support. Each
        tuple contains the indices of the dimensions that correspond to a particular
        component of the distribution.
    """
    dist_layouts: list[tuple[int, ...]] = []
    for shaped_value in shaped_values:
        if isinstance(shaped_value, tuple):
            a, b = shaped_value
            dist_layouts.append((a, b - 1))
            continue
        dist_layouts.append((shaped_value,))
    return dist_layouts


def compute_batched_marginals(
    model_meta_cls: type,
    samples_batch: Array,
    constants: dict,
    variables_index: dict,
    domains: list[Array],
    normalize: list[bool],
    batch_size: int | None = None,
):
    r"""Compute marginal densities.

    Parameters
    ----------
    model_meta_cls : type
        A class representing the meta-information of the model, which includes a method
        for constructing the model given specific parameters.
    samples_batch : Array
        A batch of samples, where each row corresponds to a single sample and each
        column corresponds to a specific parameter of the model.
    constants : dict
        A dictionary of constant values required for constructing the model. The keys
        should match the parameter names expected by the model's constructor.
    variables_index : dict
        A dictionary mapping parameter names to their corresponding column indices in
        the :code:`samples_batch` array. This mapping is used to extract the relevant
        parameter values from the samples when constructing the model.
    domains : list[Array]
        A list of arrays representing the domain values for each parameter of the model.
        The length of this list should match the number of parameters in the model.
    normalize : list[bool], optional
        A list of booleans indicating whether to normalize the marginal densities for
        each parameter.
    batch_size : int | None, optional
        The size of the batch to process at a time, by default None
    """

    def _compute_component_marginals_single_sample(
        sample: Array, domains: list[Array]
    ) -> list[list[Array]]:
        model = model_meta_cls.model_fn(  # type: ignore
            **constants,
            **{p: sample[m] for p, m in variables_index.items()},
            validate_args=True,
        )

        event_shaped_arr = jnp.zeros(model.event_shape)
        component_probs = []

        for comp_dist in model.component_distributions:
            dist_layouts = calculate_dist_layouts(comp_dist.shaped_values)

            marginal_probs_list = []

            for n_layout, dist_layout in enumerate(dist_layouts):
                _domain = [domains[i] for i in dist_layout]
                _normalize = [normalize[i] for i in dist_layout]

                grid = jnp.stack(
                    jnp.meshgrid(*_domain, indexing="ij"), axis=-1
                ).reshape(-1, len(dist_layout))

                feasible_points = comp_dist.support.feasible_like(event_shaped_arr)
                value = jnp.broadcast_to(
                    feasible_points, grid.shape[:-1] + model.event_shape
                )
                value = value.at[..., dist_layout].set(grid)

                _log_prob = comp_dist.marginal_log_probs(value)

                target_shape = tuple([ls.size for ls in _domain])
                probs = jnp.exp(_log_prob[..., n_layout]).reshape(target_shape)

                marginal_probs = calculate_marginals_over_axes(
                    probs, domains=_domain, normalize=_normalize
                )
                marginal_probs_list.extend(marginal_probs)

            component_probs.append(marginal_probs_list)

        return component_probs

    single_sample_fn = lambda s: _compute_component_marginals_single_sample(s, domains)
    return jax.lax.map(single_sample_fn, samples_batch, batch_size=batch_size)


def save_results_to_hdf5(
    constants: dict,
    variables_index: dict[str, int],
    samples: Array,
    batched_results: list[list[list[Array]]],
    parameters: list[str],
    domain_cfg: dict[str, tuple[float, float, int]],
    filepath: str | Path,
):
    """Save the computed marginal densities to an HDF5 file.

    Parameters
    ----------
    constants : dict
        A dictionary of constant values required for constructing the model. The keys
        should match the parameter names expected by the model's constructor.
    variables_index : dict[str, int]
        A dictionary mapping parameter names to their corresponding column indices in
        the samples array. This mapping is used to extract the relevant parameter values
        from the samples when constructing the model.
    samples : Array
        An array of samples used for computing the marginal densities. Each row
        corresponds to a single sample, and each column corresponds to a specific
        parameter of the model.
    batched_results : list[list[Array]]
        A nested list containing the computed marginal densities for each component of
        the model. The outer list corresponds to the components of the model, the
        inner list corresponds to the parameters of the model, and each leaf is a 2D
        array of shape (num_samples, domain_size).
    parameters : list[str]
        A list of parameter names for the model.
    domain_cfg : dict[str, tuple[float, float, int]]
        A dictionary mapping parameter names to their corresponding domain specifications.
    filepath : str | Path
        The path to the HDF5 file where the results will be saved.
    """
    # TODO(Qazalbash): save labels in numpyro sampler case and
    # use them instead of following logic
    inverted_variables_index = defaultdict(list)
    for param, idx in variables_index.items():
        inverted_variables_index[idx].append(param)

    hyperparameters = [0] * len(inverted_variables_index)
    for idx, params in inverted_variables_index.items():
        canonical_param = sorted(params)[0]
        hyperparameters[idx] = canonical_param

    result = ps.PopulationResult(
        fname=filepath,
        hyperparameters=hyperparameters,
        default_h5py_kwargs={"compression": "gzip", "compression_opts": 9},
    )
    N_components = len(batched_results)

    result.set_hyperparameter_samples(samples, overwrite=True)

    domains = {p: np.linspace(*info).reshape(1, -1) for p, info in domain_cfg.items()}

    for i in range(N_components):
        for idx, param in enumerate(parameters):
            param = str(param)
            rate_scaled_pdf = np.array(batched_results[i][idx])
            result.set_rates_on_grids(
                f"component_{i}_{param}",
                grid_params=param,
                positions=domains[param],
                rates=rate_scaled_pdf,
                overwrite=True,
            )

    write_to_hdf5(
        filepath,
        dataset_path="/posterior/hyperparameter_samples",
        attrs={"constants": constants, "variables_index": variables_index},
    )


def remove_comoving_volume_factor(
    marginal_density: Array, redshift_domain: Array
) -> Array:
    """Remove the comoving volume factor from a marginal density over redshift.

    This function takes a marginal density that includes the comoving volume factor and
    time dilation factor and divides it by the comoving volume element to obtain the
    underlying density without the volume factor.

    Parameters
    ----------
    marginal_density : Array
        The marginal density over redshift that includes the comoving volume factor.
    redshift_domain : Array
        The array of redshift values corresponding to the marginal density. This is used
        to compute the comoving volume element.

    Returns
    -------
    Array
        The marginal density with the comoving volume and time dilation factors removed,
        representing the underlying density over redshift.
    """
    cosmo = default_cosmology()
    dVc_dz = cosmo.dVcdz(redshift_domain)
    time_dilation = 1 / (1 + redshift_domain)
    factor = dVc_dz * time_dilation
    corrected_density = jnp.where(factor > 0, marginal_density / factor, 0.0)
    return corrected_density


def generate_marginal_probs(
    model_meta_cls: type,
    input_file_path: str | Path,
    output_file_path: str | Path,
    domain_cfg: dict[str, tuple[float, float, int]],
    max_samples: int | None = None,
    batch_size: int | None = None,
):
    """Generate marginal probability densities.

    Parameters
    ----------
    model_meta_cls : type
        A class representing the meta-information of the model, which includes a method
        for constructing the model given specific parameters.
    input_file_path : str | Path
        Path to hdf5 file containing the input data.
    output_file_path : str | Path
        The path to the HDF5 file where the computed marginal densities will be saved.
    domain_cfg : dict[str, tuple[float, float, int]]
        A dictionary mapping parameter names to their corresponding domain specifications.
        Each value in the dictionary should be a tuple containing the start, stop, and
        number of points for the domain of the parameter.
    filename : str
        The path to the HDF5 file where the results will be saved.
    max_samples : int | None, optional
        The maximum number of samples to use for computing marginal densities, by default None
    batch_size : int | None, optional
        The batch size for computing marginal densities, by default None

    Raises
    ------
    FileNotFoundError
        If the required samples file cannot be found in the specified base directory.
    """
    param_names = list(inspect.signature(model_meta_cls.__init__).parameters.keys())  # type: ignore
    param_names.remove("self")

    with h5py.File(input_file_path, "r") as f:
        constants = read_attrs_from_hdf5(f, "constants")
        variables_index = {
            p: int(idx) for p, idx in read_attrs_from_hdf5(f, "variables_index").items()
        }
        samples_arr = read_from_hdf5(f, "samples")

    if max_samples is not None:
        idx = np.random.choice(samples_arr.shape[0], size=max_samples, replace=False)
        samples_arr = samples_arr[idx]

    model_meta = model_meta_cls(**{p: constants[p] for p in param_names})

    domains = [jnp.linspace(*domain_cfg[p]) for p in model_meta.parameters]
    normalize = [p != P.REDSHIFT for p in model_meta.parameters]

    all_samples = jnp.array(samples_arr)

    batched_results = compute_batched_marginals(
        model_meta_cls,
        all_samples,
        constants,
        variables_index,
        domains,
        batch_size=batch_size,
        normalize=normalize,
    )

    if P.REDSHIFT in model_meta.parameters:
        redshift_idx = model_meta.parameters.index(P.REDSHIFT)
        redshift_domain = domains[redshift_idx]
        for i in range(len(batched_results)):
            batched_results[i][redshift_idx] = remove_comoving_volume_factor(
                batched_results[i][redshift_idx], redshift_domain
            )

    save_results_to_hdf5(
        constants=constants,
        variables_index=variables_index,
        samples=all_samples,
        batched_results=batched_results,
        parameters=model_meta.parameters,
        filepath=output_file_path,
        domain_cfg=domain_cfg,
    )


class PlotStyle(NamedTuple):
    """A named tuple representing the style for plotting marginal densities, including
    the color, label, and additional keyword arguments for line plots and fill-between
    plots.
    """

    color: str
    label: str
    line_plot_kwargs: dict = {
        "linewidth": 2.0,
    }
    fill_between_kwargs: dict = {
        "alpha": 0.3,
    }


def plot_marginal_with_intervals(
    ax: plt.Axes,
    filename: str,
    parameter: str,
    style: PlotStyle,
    component_idxs: list[int],
    scale: float | Callable = 1.0,
    weights: list[float | Callable] | None = None,
    normalize: bool = False,
):
    """Plot marginal densities with confidence intervals for a specified parameter.

    Parameters
    ----------
    ax : plt.Axes
        The Matplotlib Axes object on which to plot the marginal densities and confidence
        intervals.
    filename : str
        The path to the HDF5 file containing the marginal density data.
    parameter : str
        The name of the parameter for which to plot the marginal densities. This should
        correspond to a dataset in the HDF5 file under the "probs/component_{i}" groups.
    style : PlotStyle
        A list of PlotStyle objects specifying the plotting style for each component's
        marginal density. If an element is None, the corresponding component will be
        skipped in the plot.
    component_idxs : list[int]
        A list of indices specifying which components to plot.
    scale : float | Callable, optional
        A scaling factor for the marginal densities. If a callable is provided, it will
        be evaluated with the parameters from the HDF5 file, by default 1.0
    weights : list[float  |  Callable] | None, optional
        The weights for each component's marginal density. If None, equal weights are assumed, by default None
    normalize : bool, optional
        Whether to normalize the marginal densities, by default False
    """
    result = ps.PopulationResult(filename)

    datasets = [f"component_{i}_{parameter}" for i in component_idxs]

    samples = result.get_hyperparameter_samples()

    cv_dict = read_attrs_from_hdf5(filename, "/posterior/hyperparameter_samples")

    constants = cv_dict["constants"]
    variables_index = cv_dict["variables_index"]

    params = {p: samples[:, m][:, np.newaxis] for p, m in variables_index.items()}
    params.update(constants)

    weight_values = []
    if weights is None:
        weight_values = [1.0] * len(component_idxs)
    else:
        for i in range(len(component_idxs)):
            if callable(weights[i]):
                w = weights[i](params)
            else:
                w = weights[i]
            weight_values.append(w)

    pos_and_rates: tuple[np.ndarray, np.ndarray] = [
        result.get_rates_on_grids(dataset) for dataset in datasets
    ]  # type: ignore
    data = [rate for _, rate in pos_and_rates]

    # assume all components share the same domain for the parameter of interest
    domain = np.squeeze(pos_and_rates[0][0], axis=0)

    weighted_data = np.sum([w * d for w, d in zip(weight_values, data)], axis=0)

    if normalize:
        Z = np.trapezoid(weighted_data, domain)[:, np.newaxis]
        weighted_data /= Z

    weighted_data *= scale(params) if callable(scale) else scale

    lower = np.quantile(weighted_data, 0.05, axis=0)
    median = np.quantile(weighted_data, 0.5, axis=0)
    upper = np.quantile(weighted_data, 0.95, axis=0)

    ax.fill_between(
        domain, lower, upper, color=style.color, **style.fill_between_kwargs
    )
    ax.plot(
        domain,
        median,
        label=style.label,
        color=style.color,
        **style.line_plot_kwargs,
    )
    ax.set_xlim(domain[0], domain[-1])
