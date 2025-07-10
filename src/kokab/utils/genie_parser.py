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
        "--seed",
        help="Random seed for reproducibility.",
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

    return parser
