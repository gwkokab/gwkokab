# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from abc import abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jax
from jax import lax, random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro._typing import DistributionT
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key

from gwkokab.models.utils import JointDistribution, LazyJointDistribution
from gwkokab.utils.tools import error_if
from kokab.utils.common import read_json, write_json
from kokab.utils.priors import get_processed_priors


def _topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    visited = set()
    result = []

    def dfs(node: str) -> None:
        visited.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)
    result.reverse()
    return result


def _check_cycles(graph: Dict[str, Set[str]]) -> bool:
    """Checks if a directed graph has cycles using Depth-First Search (DFS).

    Parameters
    ----------
    graph : Dict[str, Set[str]]
        A directed graph represented as an adjacency list.

    Returns
    -------
    bool
        True if the graph has cycles, False otherwise.
    """
    visited = set()
    rec_stack = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False


def _bake_model(
    dist_factory: Distribution | Callable[..., Distribution], **params: Any
) -> Tuple[
    Dict[str, int | float | None],
    Dict[str, Distribution],
    Dict[str, str],
    Callable[..., Distribution],
    Dict[str, Dict[str, str]],
    List[str],
]:
    """Separate constants, random variables, and lazy variables from model parameters.

    Parameters
    ----------
    dist_factory : Distribution | Callable[..., Distribution]
        A distribution class or a callable that constructs a distribution.

    Returns
    -------
    Tuple[
        Dict[str, int | float | None],
        Dict[str, Distribution],
        Dict[str, str],
        Callable[..., Distribution],
        Dict[str, Dict[str, str]],
        List[str],
    ]
        A tuple containing:
        - constants: fixed numeric or None values
        - variables: distribution instances or lazy callables
        - aliases: mapping of parameter aliases to their original names
        - baked_factory: callable with constants pre-applied
        - lazy_dependencies: mapping of lazy variable definitions
        - lazy_order: topological order of dependent lazy variables

    Raises
    ------
    ValueError
        If a parameter has an invalid type.
    """

    constants: Dict[str, int | float | None] = {}
    variables: Dict[str, Distribution] = {}
    aliases: Dict[str, str] = {}
    lazy_dependencies: Dict[str, Dict[str, str]] = {}

    dependency_graph: Dict[str, Set[str]] = defaultdict(set)
    variable_roots: List[str] = []

    # Pass 1: classify parameters
    for name, value in params.items():
        if value is None:
            constants[name] = None

        elif isinstance(value, Distribution):
            variables[name] = value
            variable_roots.append(name)

        elif isinstance(value, tuple):
            lazy_fn, lazy_args = value
            error_if(
                not isinstance(lazy_fn, jax.tree_util.Partial),
                msg=f"Lazy distribution '{name}' must be a `jax.tree_util.Partial`.",
            )
            error_if(
                not isinstance(lazy_args, dict),
                msg=f"Lazy distribution '{name}' must have a dictionary of dependencies.",
            )

            variables[name] = lazy_fn
            lazy_dependencies[name] = lazy_args

            for _, dep_name in lazy_args.items():
                dependency_graph[dep_name].add(name)

        elif isinstance(value, (int, float)):
            constants[name] = lax.stop_gradient(value)

        elif isinstance(value, str):
            # string aliases handled later
            continue

        else:
            error_if(
                True,
                msg=f"Invalid parameter '{name}' with type {type(value)} and value {value}",
            )

    # Pass 2: resolve string aliases
    for name, value in params.items():
        if isinstance(value, str):
            if value in constants:
                constants[name] = constants[value]
            elif value in variables:
                aliases[name] = value

    # Compute lazy evaluation order
    if lazy_dependencies:
        import pprint

        error_if(
            _check_cycles(dependency_graph),
            msg="Cyclic dependencies detected among lazy variables. Dependency graph:\n"
            + pprint.pformat(dependency_graph),
        )

        lazy_order = [
            var
            for var in _topological_sort(dependency_graph)
            if var not in variable_roots
        ]
    else:
        lazy_order = []

    baked_factory = jax.tree_util.Partial(dist_factory, **constants)

    return (
        constants,
        variables,
        aliases,
        baked_factory,
        lazy_dependencies,
        lazy_order,
    )


class Guru:
    """Guru is a class which contains all the common functionality among Genie, Sage and
    Guru classes.
    """

    _rng_key: PRNGKeyArray
    output_directory: str

    def __init__(
        self,
        *,
        analysis_name: str,
        check_leaks: bool,
        debug_nans: bool,
        model: Union[DistributionT, Callable[..., DistributionT]],
        poisson_mean_filename: str,
        prior_filename: str,
        profile_memory: bool,
        sampler_settings_filename: str,
        variance_cut_threshold: Optional[float] = None,
    ) -> None:
        self.analysis_name = analysis_name
        self.prior_filename = prior_filename
        self.model = model
        self.sampler_settings_filename = sampler_settings_filename
        self.debug_nans = debug_nans
        self.profile_memory = profile_memory
        self.check_leaks = check_leaks
        self.poisson_mean_filename = poisson_mean_filename
        self.variance_cut_threshold = variance_cut_threshold

    @property
    def rng_key(self) -> PRNGKeyArray:
        self._rng_key, subkey = jrd.split(self._rng_key)
        return subkey

    def set_rng_key(
        self, *, key: Optional[PRNGKeyArray] = None, seed: Optional[int] = None
    ) -> None:
        error_if(
            key is None and seed is None,
            msg="Either 'key' or 'seed' must be provided to set the random number generator key.",
        )
        if key is not None:
            error_if(
                not is_prng_key(key),
                msg=f"Expected a PRNGKeyArray, got {type(key)}.",
            )
            logger.info(f"Setting the random number generator key to {key}.")
            self._rng_key = key
        elif seed is not None:
            error_if(
                not isinstance(seed, int),
                msg=f"Expected an integer seed, got {type(seed)}.",
            )
            error_if(
                seed < 0,
                msg=f"Seed must be a non-negative integer, got {seed}.",
            )
            logger.info(f"Setting the random number generator key with seed {seed}.")
            key = jrd.PRNGKey(seed)
            self._rng_key = key

    def bake_model(
        self,
    ) -> Tuple[
        Dict[str, Union[int, float, bool, None]],
        Callable[..., DistributionT],
        JointDistribution,
        Dict[str, int],
        Dict[str, int],
    ]:
        """Returns a Bake object for the model.

        Returns
        -------
        Tuple[ Dict[str, Union[int, float, bool, None]], Callable[..., DistributionT], JointDistribution, Dict[str, int], ]
            A tuple containing the constants, the distribution function, the prior
            distribution, and the variables index.
        """
        prior_dict = read_json(self.prior_filename)
        model_prior_param = get_processed_priors(self.model_parameters, prior_dict)

        logger.debug("Baking the model")
        constants, variables, duplicates, dist_fn, lazy_deps, lazy_order = _bake_model(
            self.model, **self.constants, **model_prior_param
        )  # type: ignore

        variables_index: dict[str, int] = {
            key: i for i, key in enumerate(sorted(variables.keys()))
        }
        for key, value in duplicates.items():
            variables_index[key] = variables_index[value]

        group_variables: dict[int, list[str]] = {}
        for key, value in variables_index.items():  # type: ignore
            group_variables[value] = group_variables.get(value, []) + [key]  # type: ignore

        logger.debug(
            "Number of recovering variables: {num_vars}", num_vars=len(group_variables)
        )

        for key, value in constants.items():  # type: ignore
            logger.debug(
                "Constant variable: {name} = {variable}", name=key, variable=value
            )

        for value in group_variables.values():  # type: ignore
            logger.debug("Recovering variable: {variable}", variable=", ".join(value))

        write_json("constants.json", constants)
        write_json("nf_samples_mapping.json", variables_index)

        sorted_variables = sorted(variables.keys())
        if len(lazy_order) == 0:
            priors = JointDistribution(
                *[variables[key] for key in sorted_variables],
                validate_args=True,
            )
        else:
            lazy_deps_new = {
                sorted_variables.index(k): {
                    k_: sorted_variables.index(v) for k_, v in deps.items()
                }
                for k, deps in lazy_deps.items()
            }
            priors = LazyJointDistribution(
                *[variables[key] for key in sorted_variables],
                dependencies=lazy_deps_new,
                partial_order=[sorted_variables.index(k) for k in lazy_order],
                validate_args=True,
            )

        return constants, dist_fn, priors, variables, variables_index

    @property
    def parameters(self) -> List[str]:
        """Returns the parameters (intrinsic + extrinsic).

        Returns
        -------
        List[str]
            List of parameters.

        Raises
        ------
        NotImplementedError
            If the Guru class is used directly, this method raises a NotImplementedError.
            It is expected that subclasses of Guru will implement this method.
        """
        msg = (
            "The Guru class should not be used directly. Please use a subclass that "
            "implements the parameters property."
        )
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def model_parameters(self) -> List[str]:
        """Returns the model parameters.

        Returns
        -------
        List[str]
            List of model parameters.

        Raises
        ------
        NotImplementedError
            If the Guru class is used directly, this method raises a NotImplementedError.
            It is expected that subclasses of Guru will implement this method.
        """
        msg = (
            "The Guru class should not be used directly. Please use a subclass that "
            "implements the model_parameters property."
        )
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        """Returns the constants used in the model.

        Returns
        -------
        Dict[str, Union[int, float, bool]]
            A dictionary containing the constants used in the model.
        """
        return {}

    @abstractmethod
    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Any,
        labels: List[str],
    ) -> None:
        raise NotImplementedError()


def guru_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Guru script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    parser.add_argument(
        "--seed",
        help="Seed for the random number generator.",
        default=37,
        type=int,
    )

    pmean_group = parser.add_argument_group("Poisson Mean Options")
    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        default="pmean.json",
    )

    sampler_group = parser.add_argument_group("Sampler Options")
    sampler_group.add_argument(
        "--sampler-config",
        help="Path to the JSON file containing the sampler configuration.",
        type=str,
        required=True,
    )
    sampler_group.add_argument(
        "--variance-cut-threshold",
        help="Threshold for variance cut in the sampler.",
        type=float,
        default=None,
    )

    prior_group = parser.add_argument_group("Prior Options")
    prior_group.add_argument(
        "--prior-json",
        type=str,
        help="Path to a JSON file containing the prior distributions.",
        default="prior.json",
    )

    dev_group = parser.add_argument_group(
        "Developer Options",
        description="These options are intended for developers. They will not work in "
        "production mode.",
    )
    dev_group.add_argument(
        "--debug-nans",
        help="Checks for NaNs in each computation. See details in the documentation: "
        "https://jax.readthedocs.io/en/latest/_autosummary/jax.debug_nans.html#jax.debug_nans.",
        action="store_true",
    )
    dev_group.add_argument(
        "--profile-memory",
        help="Enable memory profiling.",
        action="store_true",
    )
    dev_group.add_argument(
        "--check-leaks",
        help="Check for JAX Tracer leaks. See details in the documentation: "
        "https://jax.readthedocs.io/en/latest/_autosummary/jax.checking_leaks.html#jax.checking_leaks.",
        action="store_true",
    )

    return parser
