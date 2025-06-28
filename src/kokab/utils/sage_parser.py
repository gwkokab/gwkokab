# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser

from numpyro.distributions.distribution import enable_validation


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Sage script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    # Global enable validation for all distributions
    enable_validation()

    sage_group = parser.add_argument_group("Sage Options")

    sage_group.add_argument(
        "--posterior-regex",
        help="Regex for the posterior samples.",
        type=str,
        required=True,
    )
    sage_group.add_argument(
        "--posterior-columns",
        help="Columns of the posterior samples.",
        nargs="+",
        type=str,
        required=True,
    )
    sage_group.add_argument(
        "--seed",
        help="Seed for the random number generator.",
        default=37,
        type=int,
    )

    vt_group = parser.add_argument_group("VT Options")

    vt_group.add_argument(
        "--vt-json",
        help="Path to the JSON file containing the VT options.",
        type=str,
        default="vt.json",
    )

    pmean_group = parser.add_argument_group("Poisson Mean Options")

    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        default="pmean.json",
    )

    flowMC_group = parser.add_argument_group("flowMC Options")

    flowMC_group.add_argument(
        "--flowMC-json",
        help="Path to a JSON file containing the flowMC options. It should contains"
        "keys: local_sampler_kwargs, nf_model_kwargs, sampler_kwargs, data_dump_kwargs,"
        " and their respective values.",
        default="flowMC.json",
        type=str,
    )

    optm_group = parser.add_argument_group("Optimization Options")
    optm_group.add_argument(
        "--adam-optimizer",
        help="Use Adam optimizer before running flowMC.",
        action="store_true",
    )
    optm_group.add_argument(
        "--adam-json",
        help="Path to a JSON file containing the Adam optimizer options.",
        type=str,
    )
    optm_group.add_argument(
        "--n-buckets",
        help="Number of buckets for the data arrays to be split into. "
        "This is useful for large datasets to avoid memory issues. "
        "See https://github.com/gwkokab/gwkokab/issues/568 for more details.",
        type=int,
        default=None,
    )
    optm_group.add_argument(
        "--threshold",
        help="Threshold to determine best number of buckets, if the number of buckets "
        "is not specified. It should be between 0 and 100.",
        type=float,
        default=3.0,
    )

    prior_group = parser.add_argument_group("Prior Options")
    prior_group.add_argument(
        "--prior-json",
        type=str,
        help="Path to a JSON file containing the prior distributions.",
        default="prior.json",
    )

    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--debug-nans",
        help="Checks for NaNs in each computation. See details in the documentation: "
        "https://jax.readthedocs.io/en/latest/_autosummary/jax.debug_nans.html#jax.debug_nans.",
        action="store_true",
    )
    debug_group.add_argument(
        "--profile-memory",
        help="Enable memory profiling.",
        action="store_true",
    )
    debug_group.add_argument(
        "--check-leaks",
        help="Check for JAX Tracer leaks. See details in the documentation: "
        "https://jax.readthedocs.io/en/latest/_autosummary/jax.checking_leaks.html#jax.checking_leaks.",
        action="store_true",
    )

    return parser
