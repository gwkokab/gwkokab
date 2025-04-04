# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser

from numpyro.distributions.distribution import enable_validation


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Sage script.

    :param parser: Parser to add the arguments to
    :return: the command line argument parser
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
    sage_group.add_argument(
        "--verbose",
        help="Verbose output.",
        action="store_true",
    )

    vt_group = parser.add_argument_group("VT Options")

    vt_group.add_argument(
        "--vt-json",
        help="Path to the JSON file containing the VT options.",
        type=str,
        required=True,
    )

    pmean_group = parser.add_argument_group("Poisson Mean Options")

    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        required=True,
    )

    flowMC_group = parser.add_argument_group("flowMC Options")

    flowMC_group.add_argument(
        "--flowMC-json",
        help="Path to a JSON file containing the flowMC options. It should contains"
        "keys: local_sampler_kwargs, nf_model_kwargs, sampler_kwargs, data_dump_kwargs,"
        " and their respective values.",
    )
    flowMC_group.add_argument(
        "--gradient-checkpoint-policy",
        choices=[
            "everything_saveable",
            "nothing_saveable",
            "dots_saveable",
            "checkpoint_dots",
            "dots_with_no_batch_dims_saveable",
            "checkpoint_dots_with_no_batch_dims",
            "offload_dot_with_no_batch_dims",
            "save_anything_except_these_names",
            "save_any_names_but_these",
            "save_only_these_names",
            "save_from_both_policies",
            "save_and_offload_only_these_names",
        ],
        default="everything_saveable",
        type=str,
        help="Gradient checkpoint policy for the log likelihood function",
    )

    adam_group = parser.add_argument_group("Adam Options")
    adam_group.add_argument(
        "--adam-optimizer",
        help="Use Adam optimizer before running flowMC.",
        action="store_true",
    )
    adam_group.add_argument(
        "--adam-json",
        help="Path to a JSON file containing the Adam optimizer options.",
        type=str,
    )

    prior_group = parser.add_argument_group("Prior Options")
    prior_group.add_argument(
        "--prior-json",
        type=str,
        help="Path to a JSON file containing the prior distributions.",
        required=True,
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
