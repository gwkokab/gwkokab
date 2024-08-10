from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from gwkokab.errors import banana_error_m1_m2
from gwkokab.models import Wysocki2019MassModel
from gwkokab.parameters import (
    ECCENTRICITY,
    PRIMARY_MASS_SOURCE,
    SECONDARY_MASS_SOURCE,
)
from gwkokab.population import error_magazine, popfactory, popmodel_magazine
from gwkokab.vts.neuralvt import load_model
from jax import numpy as jnp, vmap
from jaxtyping import Array, Bool
from numpyro import distributions as dist


m1_source = PRIMARY_MASS_SOURCE.name
m2_source = SECONDARY_MASS_SOURCE.name
ecc = ECCENTRICITY.name


def constraint(x: Array) -> Bool:
    m1 = x[..., 0]
    m2 = x[..., 1]
    ecc = x[..., 2]
    mask = m2 <= m1
    mask &= m2 > 0.0
    mask &= m1 > 0.0
    mask &= ecc >= 0.0
    mask &= ecc <= 1.0
    return mask


def make_parser() -> ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = ArgumentParser(
        description="Generate a population of compact binary coalescences.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="This script generates a population of compact binary coalescences.",
    )
    parser.add_argument(
        "--alpha_m",
        help="Power-law index of the mass distribution.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--mmin",
        help="Minimum mass of the mass distribution.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--mmax",
        help="Maximum mass of the mass distribution.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--scale",
        help="Scale of the eccentricity distribution.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--loc",
        help="Location of the eccentricity distribution.",
        default=0.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--low",
        help="Lower bound of the eccentricity distribution.",
        default=0.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--high",
        help="Upper bound of the eccentricity distribution.",
        default=1.0,
        type=float,
        required=True,
    )

    parser.add_argument(
        "--err_scale_Mc",
        help="Scale of the error in chirp mass.",
        default=1.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--err_scale_eta",
        help="Scale of the error in symmetric mass ratio.",
        default=1.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--err_loc",
        help="Location of the error in eccentricity.",
        default=0.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--err_scale",
        help="Scale of the error in eccentricity.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--err_low",
        help="Lower bound of the error in eccentricity.",
        default=0.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--err_high",
        help="Upper bound of the error in eccentricity.",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--vt_path",
        help="Path to the neural VT",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--analysis_time",
        help="Analysis time of the VT",
        default=0.0,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--error_size",
        help="Size of the error.",
        default=2000,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_realizations",
        help="Number of realizations.",
        default=5,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--rate",
        help="Rate of binary mergers.",
        type=float,
        required=True,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    popmodel_magazine.register(
        (m1_source, m2_source),
        Wysocki2019MassModel(alpha_m=args.alpha_m, mmin=args.mmin, mmax=args.mmax),
    )

    popmodel_magazine.register(
        ecc,
        dist.TruncatedNormal(
            scale=args.scale,
            loc=args.loc,
            low=args.low,
            high=args.high,
            validate_args=True,
        ),
    )

    error_magazine.register(
        (m1_source, m2_source),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=args.err_scale_Mc,
            scale_eta=args.err_scale_eta,
        ),
    )

    @error_magazine.register(ecc)
    def ecc_error_fn(x, size, key):
        err_x = x + dist.TruncatedNormal(
            loc=args.err_loc,
            scale=args.err_scale,
            low=args.err_low,
            high=args.err_high,
        ).sample(key=key, sample_shape=(size,))
        mask = err_x < 0.0
        mask |= err_x > 1.0
        err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
        return err_x

    _, logVT = load_model(args.vt_path)
    logVT = vmap(logVT)

    popfactory.analysis_time = args.analysis_time
    popfactory.constraint = constraint
    popfactory.error_size = args.error_size
    popfactory.log_VT_fn = logVT
    popfactory.num_realizations = args.num_realizations
    popfactory.rate = args.rate
    popfactory.VT_params = [m1_source, m2_source]

    popfactory.produce()
