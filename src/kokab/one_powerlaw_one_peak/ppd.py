# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Optional

import pandas as pd
from jax import numpy as jnp
from jax.lax import broadcast_shapes
from jaxtyping import Array
from numpyro._typing import DistributionLike
from numpyro.distributions import constraints, Distribution, Independent
from numpyro.distributions.util import promote_shapes, validate_sample

from gwkokab.models import PowerlawPeak, SmoothedTwoComponentPrimaryMassRatio
from gwkokab.models.spin import (
    BetaFromMeanVar,
    IndependentSpinOrientationGaussianIsotropic,
)
from gwkokab.models.utils import (
    JointDistribution,
    ScaledMixture,
)
from gwkokab.parameters import Parameters
from gwkokab.utils.tools import error_if
from kokab.utils import ppd, ppd_parser
from kokab.utils.common import ppd_ranges, read_json


class SimpleRedshiftPowerlaw(Distribution):
    arg_constraints = {
        "z_max": constraints.positive,
        "kappa": constraints.real,
    }
    reparametrized_params = ["z_max", "kappa"]
    pytree_data_fields = ("_support", "kappa", "z_max")

    def __init__(
        self, z_max: Array, kappa: Array, *, validate_args: Optional[bool] = None
    ):
        self.z_max, self.kappa = promote_shapes(z_max, kappa)
        batch_shape = broadcast_shapes(jnp.shape(z_max), jnp.shape(kappa))
        self._support = constraints.interval(0.0, z_max)
        super(SimpleRedshiftPowerlaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """The support of the distribution, which is the interval [0, z_max]."""
        return self._support

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        r"""Evaluate the psi function at a given redshift.

        .. math::

            \ln\psi(z) = \kappa \log(1 + z)

        Parameters
        ----------
        z : ArrayLike
            Redshift(s) to evaluate.

        Returns
        -------
        ArrayLike
            Values of the psi function.
        """
        return self.kappa * jnp.log1p(value)


def PowerlawPeak_raw(
    use_spin: bool = False,
    use_redshift: bool = False,
    use_tilt: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    smoothing_model = SmoothedTwoComponentPrimaryMassRatio(
        alpha=params["alpha"],
        beta=params["beta"],
        delta=params["delta"],
        lambda_peak=params["lambda_peak"],
        loc=params["loc"],
        mmax=params["mmax"],
        mmin=params["mmin"],
        scale=params["scale"],
        validate_args=validate_args,
    )

    component_distributions = [smoothing_model]

    if use_spin:
        chi_dist = Independent(
            BetaFromMeanVar(
                mean=jnp.stack([params["chi_mean"], params["chi_mean"]], axis=-1),
                variance=jnp.stack(
                    [params["chi_variance"], params["chi_variance"]], axis=-1
                ),
                validate_args=validate_args,
            ),
            reinterpreted_batch_ndims=1,
            validate_args=validate_args,
        )
        component_distributions.append(chi_dist)

    if use_tilt:
        tilt_dist = IndependentSpinOrientationGaussianIsotropic(
            zeta=params["cos_tilt_zeta"],
            scale1=params["cos_tilt_scale"],
            scale2=params["cos_tilt_scale"],
            validate_args=validate_args,
        )

        component_distributions.append(tilt_dist)

    if use_redshift:
        z_max = params["z_max"]
        kappa = params["kappa"]
        powerlaw_z = SimpleRedshiftPowerlaw(
            z_max=z_max, kappa=kappa, validate_args=validate_args
        )

        component_distributions.append(powerlaw_z)

    if len(component_distributions) > 1:
        component_distributions = [
            JointDistribution(*component_distributions, validate_args=validate_args)
        ]

    return ScaledMixture(
        log_scales=jnp.asarray([params["log_rate"]]),
        component_distributions=component_distributions,
        support=component_distributions[0].support,  # type: ignore
        validate_args=validate_args,
    )


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = ppd_parser.get_parser(parser)

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--raw",
        action="store_true",
        help="The raw parameters for this model are primary mass and mass ratio. To"
        "align with the rest of the codebase, we transform primary mass and mass ratio"
        "to primary and secondary mass. This flag will use the raw parameters i.e."
        "primary mass and mass ratio.",
    )
    return parser


def model(raw: bool, **params) -> DistributionLike:
    validate_args = params.pop("validate_args", True)
    _model = PowerlawPeak_raw if raw else PowerlawPeak
    return _model(**params, validate_args=validate_args)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    error_if(
        not str(args.filename).endswith(".hdf5"),
        msg="Output file must be an HDF5 file.",
    )

    constants = read_json(args.constants)
    nf_samples_mapping = read_json(args.nf_samples_mapping)

    use_spin = constants.get("use_spin", False)
    use_tilt = constants.get("use_tilt", False)
    use_redshift = constants.get("use_redshift", False)

    parameters = [Parameters.PRIMARY_MASS_SOURCE.value]
    if args.raw:
        parameters.append(Parameters.MASS_RATIO.value)
    else:
        parameters.append(Parameters.SECONDARY_MASS_SOURCE.value)

    if use_spin:
        parameters.extend(
            [
                Parameters.PRIMARY_SPIN_MAGNITUDE.value,
                Parameters.SECONDARY_SPIN_MAGNITUDE.value,
            ]
        )
    if use_tilt:
        parameters.extend([Parameters.COS_TILT_1.value, Parameters.COS_TILT_2.value])
    if use_redshift:
        parameters.append(Parameters.REDSHIFT.value)

    ranges = ppd_ranges(parameters, args.range)

    nf_samples = pd.read_csv(
        args.sample_filename, delimiter=" ", comment="#", header=None
    ).to_numpy()

    ppd.compute_and_save_ppd(
        ft.partial(model, raw=args.raw),
        nf_samples,
        ranges,
        "rate_scaled_" + args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )

    nf_samples, constants = ppd.wipe_log_rate(nf_samples, nf_samples_mapping, constants)

    ppd.compute_and_save_ppd(
        ft.partial(model, raw=args.raw),
        nf_samples,
        ranges,
        args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )
