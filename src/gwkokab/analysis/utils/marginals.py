# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
import inspect
from pathlib import Path

import h5py
import jax
import numpy as np
from jax import jit, numpy as jnp
from jaxtyping import Array

from gwkokab.analysis.utils.common import read_json
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
        reduction_axis = (i - j) - probs_array.ndim + (i >= axis)
        marginal_density = jnp.trapezoid(
            y=marginal_density, x=domain, axis=reduction_axis
        )
        j += 1

    if normalize:
        Z = jnp.trapezoid(marginal_density, domains[axis], axis=-1)

        # Avoid Division-by-Zero via safe masking
        marginal_density = jnp.where(Z > 0, marginal_density / Z, 0.0)

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
    nf_samples_mapping: dict,
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
    nf_samples_mapping : dict
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
            **{p: sample[nf_samples_mapping[p]] for p in nf_samples_mapping.keys()},
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
    samples: Array,
    constants: dict,
    nf_samples_mapping: dict,
    batched_results: list[list[list[Array]]],
    parameters: list[str],
    domain_cfg: dict[str, tuple[float, float, int]],
    filepath: str | Path,
):
    """Save the computed marginal densities to an HDF5 file.

    Parameters
    ----------
    samples : Array
        An array of samples used for computing the marginal densities. Each row
        corresponds to a single sample, and each column corresponds to a specific
        parameter of the model.
    batched_results : list[list[list[Array]]]
        A nested list containing the computed marginal densities for each component of
        the model, for each sample in the batch. The outer list corresponds to the
        samples, the middle list corresponds to the components of the model, and the
        innermost list corresponds to the marginal densities for each parameter of the
        model.
    parameters : list[str]
        A list of parameter names for the model.
    domain_cfg : dict[str, tuple[float, float, int]]
        A dictionary mapping parameter names to their corresponding domain specifications.
    filepath : str | Path
        The path to the HDF5 file where the results will be saved.
    """
    compression_opts = {"compression": "gzip", "compression_opts": 9}
    N_components = len(batched_results)

    string_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(filepath, "w") as f:
        probs_group = f.create_group("probs")

        probs_group.attrs["domains"] = np.asarray(
            [(str(param), *info) for param, info in domain_cfg.items()],
            dtype=np.dtype([
                ("param", string_dt),
                ("start", np.float32),
                ("stop", np.float32),
                ("num_points", np.uint32),
            ]),
        )

        flat_constants = [(str(k), v) for k, v in constants.items()]
        probs_group.attrs["constants"] = np.asarray(
            flat_constants,
            dtype=np.dtype([("constant", string_dt), ("value", np.float32)]),
        )

        flat_nf_samples_mapping = [(str(k), v) for k, v in nf_samples_mapping.items()]
        probs_group.attrs["nf_samples_mapping"] = np.asarray(
            flat_nf_samples_mapping,
            dtype=np.dtype([("parameter", string_dt), ("column_index", np.uint32)]),
        )
        probs_group.create_dataset("samples", data=samples, **compression_opts)

        for i in range(N_components):
            comp_i_group = probs_group.create_group(f"component_{i}")
            for idx, param in enumerate(parameters):
                comp_i_group.create_dataset(
                    param,
                    data=np.array(batched_results[i][idx]),
                    **compression_opts,
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
    base_dir: str,
    domain_cfg: dict[str, tuple[float, float, int]],
    filename: str,
    max_samples: int | None = None,
    batch_size: int | None = None,
):
    """Generate marginal probability densities.

    Parameters
    ----------
    model_meta_cls : type
        A class representing the meta-information of the model, which includes a method
        for constructing the model given specific parameters.
    base_dir : str
        The base directory where the constants, samples, and other necessary files are
        located.
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
    base_path = Path(base_dir)
    constants = read_json(base_path / "constants.json")  # type: ignore
    nf_samples_mapping = read_json(base_path / "nf_samples_mapping.json")  # type: ignore

    param_names = list(inspect.signature(model_meta_cls.__init__).parameters.keys())  # type: ignore
    param_names.remove("self")

    model_meta = model_meta_cls(**{p: constants[p] for p in param_names})

    domains = [jnp.linspace(*domain_cfg[p]) for p in model_meta.parameters]
    normalize = [p != P.REDSHIFT for p in model_meta.parameters]

    inference_dirs = ("numpyro_inference", "flowMC_inference")
    samples_arr = None
    for inference_dir in inference_dirs:
        samples_path = base_path / inference_dir / "samples.dat"
        if samples_path.exists():
            samples_arr = np.loadtxt(samples_path, skiprows=1)
            break

    if samples_arr is None:
        raise FileNotFoundError(
            f"Could not locate samples.dat under search paths within: {base_dir}"
        )

    if max_samples is not None:
        idx = np.random.choice(samples_arr.shape[0], size=max_samples, replace=False)
        samples_arr = samples_arr[idx]

    all_samples = jnp.array(samples_arr)

    batched_results = compute_batched_marginals(
        model_meta_cls,
        all_samples,
        constants,
        nf_samples_mapping,
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
        all_samples,
        constants,
        nf_samples_mapping,
        batched_results,
        parameters=model_meta.parameters,
        filepath=filename,
        domain_cfg=domain_cfg,
    )
