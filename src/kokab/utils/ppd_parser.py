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

    ppd_group = parser.add_argument_group("PPD Options")

    ppd_group.add_argument(
        "--sample-filename",
        help="Path of the file to save the samples.",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--filename",
        help="Path of the file to save the PPD.",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--constants",
        help="Path to the JSON file containing the constant parameters.",
        type=str,
        default="constants.json",
    )
    ppd_group.add_argument(
        "--nf-samples-mapping",
        help="Path to the JSON file containing the mapping of the number of samples.",
        type=str,
        default="nf_samples_mapping.json",
    )
    ppd_group.add_argument(
        "--range",
        help="Range of the PPD for each parameter. The format is 'name min max step'. "
        "Repeat for each parameter.",
        nargs=4,
        action="append",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--batch-size",
        help="Batch size for the computation of log prob per sample.",
        type=int,
        default=1000,
    )

    return parser
