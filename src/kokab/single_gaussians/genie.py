from argparse import ArgumentParser

from gwkokab.errors import banana_error_m1_m2
from gwkokab.parameters import PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from gwkokab.population import error_magazine, popfactory, popmodel_magazine
from gwkokab.vts.neuralvt import load_model
from jax import numpy as jnp, vmap
from jaxtyping import Array, Bool
from numpyro.distributions import MultivariateNormal

from ..utils import genie_parser


m1_source = PRIMARY_MASS_SOURCE.name
m2_source = SECONDARY_MASS_SOURCE.name


def constraint(x: Array) -> Bool:
    m1 = x[..., 0]
    m2 = x[..., 1]
    mask = m2 <= m1
    mask &= m2 > 0.0
    mask &= m1 > 0.0
    return mask


def single_gaussian_model(
    mu1: Array, mu2: Array, sigma1: Array, sigma2: Array
) -> Array:
    return MultivariateNormal(
        jnp.stack([mu1, mu2], axis=-1),
        jnp.diag(jnp.array([sigma1**2, sigma2**2])),
        validate_args=True,
    )


def make_parser() -> ArgumentParser:
    parser = genie_parser.get_parser()

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--mu1",
        help="Mean of the primary mass source Gaussian.",
        type=float,
        required=True,
    )
    model_group.add_argument(
        "--mu2",
        help="Mean of the secondary mass source Gaussian.",
        type=float,
        required=True,
    )
    model_group.add_argument(
        "--sigma1",
        help="Mean of the primary mass source Gaussian.",
        type=float,
        required=True,
    )
    model_group.add_argument(
        "--sigma2",
        help="Mean of the secondary mass source Gaussian.",
        type=float,
        required=True,
    )

    err_group = parser.add_argument_group("Error Options")
    err_group.add_argument(
        "--err_scale_Mc",
        help="Scale of the error in chirp mass.",
        default=1.0,
        type=float,
        required=True,
    )
    err_group.add_argument(
        "--err_scale_eta",
        help="Scale of the error in symmetric mass ratio.",
        default=1.0,
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
        model=single_gaussian_model(
            mu1=args.mu1,
            mu2=args.mu2,
            sigma1=args.sigma1,
            sigma2=args.sigma2,
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
