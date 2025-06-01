# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser

from numpyro.distributions.distribution import enable_validation


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Genie
    script.

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

    genie_group = parser.add_argument_group("Genie Options")

    genie_group.add_argument(
        "--error-size",
        help="Size of the error.",
        default=2000,
        type=int,
    )
    genie_group.add_argument(
        "--num-realizations",
        help="Number of realizations.",
        default=5,
        type=int,
    )
    genie_group.add_argument(
        "--raw-injections",
        action="store_true",
        help="Save raw injections to file.",
    )
    genie_group.add_argument(
        "--raw-PEs",
        action="store_true",
        help="Save raw posterior estimates to file.",
    )
    genie_group.add_argument(
        "--weighted-injections",
        action="store_true",
        help="Save weighted injections to file.",
    )
    genie_group.add_argument(
        "--weighted-PEs",
        action="store_true",
        help="Save weighted posterior estimates to file.",
    )
    genie_group.add_argument(
        "--ref-probs",
        action="store_true",
        help="Save reference probabilities to file.",
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

    return parser
