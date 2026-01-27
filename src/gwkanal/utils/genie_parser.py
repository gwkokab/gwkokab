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

    pmean_group = parser.add_argument_group("Poisson Mean Options")
    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        default="pmean.json",
    )

    gaussian_group = parser.add_argument_group("Gaussian Error Options")
    gaussian_group.add_argument(
        "--tile-covariance",
        help="Tile the covariance of the parameters specified. Each tile will have its own "
        "covariance matrix estimated from the data. This is useful for large parameter "
        "spaces where the full covariance matrix is too large to estimate accurately. "
        "Each tile should be a list of parameters. Example: --tile-covariance mass1 "
        "mass2 --tile-covariance spin1z spin2z",
        nargs="+",
        action="append",
        default=None,
    )

    return parser
