# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable

from gwkanal.core.synthetic_events import (
    injection_generator_parser,
    SyntheticEventsBase,
)
from gwkanal.n_pls_m_gs.common import model_arg_parser, NPowerlawMGaussianCore
from gwkanal.utils.logger import log_info
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.utils import ScaledMixture


def main() -> None:
    """Main function of the script."""
    parser = injection_generator_parser()
    parser = model_arg_parser(parser)
    args = parser.parse_args()

    log_info(start=True)

    class NPowerlawMGaussianInjectionGenerator(
        NPowerlawMGaussianCore, SyntheticEventsBase
    ):
        def __init__(
            self,
            N_pl: int,
            N_g: int,
            use_beta_spin_magnitude: bool,
            use_spin_magnitude_mixture: bool,
            use_truncated_normal_spin_x: bool,
            use_truncated_normal_spin_y: bool,
            use_truncated_normal_spin_z: bool,
            use_chi_eff_mixture: bool,
            use_skew_normal_chi_eff: bool,
            use_truncated_normal_chi_p: bool,
            use_tilt: bool,
            use_eccentricity_mixture: bool,
            use_redshift: bool,
            use_cos_iota: bool,
            use_phi_12: bool,
            use_polarization_angle: bool,
            use_right_ascension: bool,
            use_sin_declination: bool,
            use_detection_time: bool,
            filename: str,
            model_fn: Callable[..., ScaledMixture],
            model_params_filename: str,
            poisson_mean_filename: str,
            derive_parameters: bool = False,
        ):
            NPowerlawMGaussianCore.__init__(
                self,
                N_pl=N_pl,
                N_g=N_g,
                use_beta_spin_magnitude=use_beta_spin_magnitude,
                use_spin_magnitude_mixture=use_spin_magnitude_mixture,
                use_truncated_normal_spin_x=use_truncated_normal_spin_x,
                use_truncated_normal_spin_y=use_truncated_normal_spin_y,
                use_truncated_normal_spin_z=use_truncated_normal_spin_z,
                use_chi_eff_mixture=use_chi_eff_mixture,
                use_skew_normal_chi_eff=use_skew_normal_chi_eff,
                use_truncated_normal_chi_p=use_truncated_normal_chi_p,
                use_tilt=use_tilt,
                use_eccentricity_mixture=use_eccentricity_mixture,
                use_redshift=use_redshift,
                use_cos_iota=use_cos_iota,
                use_phi_12=use_phi_12,
                use_polarization_angle=use_polarization_angle,
                use_right_ascension=use_right_ascension,
                use_sin_declination=use_sin_declination,
                use_detection_time=use_detection_time,
            )
            SyntheticEventsBase.__init__(
                self,
                filename=filename,
                model_fn=model_fn,
                model_params_filename=model_params_filename,
                poisson_mean_filename=poisson_mean_filename,
                derive_parameters=derive_parameters,
            )

        def modify_model_params(self, params: dict) -> dict:
            params.update(
                {
                    "N_pl": self.N_pl,
                    "N_g": self.N_g,
                    "use_beta_spin_magnitude": self.use_beta_spin_magnitude,
                    "use_spin_magnitude_mixture": self.use_spin_magnitude_mixture,
                    "use_truncated_normal_spin_x": self.use_truncated_normal_spin_x,
                    "use_truncated_normal_spin_y": self.use_truncated_normal_spin_y,
                    "use_truncated_normal_spin_z": self.use_truncated_normal_spin_z,
                    "use_chi_eff_mixture": self.use_chi_eff_mixture,
                    "use_skew_normal_chi_eff": self.use_skew_normal_chi_eff,
                    "use_truncated_normal_chi_p": self.use_truncated_normal_chi_p,
                    "use_tilt": self.use_tilt,
                    "use_eccentricity_mixture": self.use_eccentricity_mixture,
                    "use_redshift": self.use_redshift,
                    "use_cos_iota": self.use_cos_iota,
                    "use_phi_12": self.use_phi_12,
                    "use_polarization_angle": self.use_polarization_angle,
                    "use_right_ascension": self.use_right_ascension,
                    "use_sin_declination": self.use_sin_declination,
                    "use_detection_time": self.use_detection_time,
                }
            )
            return params

    NPowerlawMGaussianInjectionGenerator.init_rng_seed(seed=args.seed)

    generator = NPowerlawMGaussianInjectionGenerator(
        N_pl=args.n_pl,
        N_g=args.n_g,
        use_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        use_truncated_normal_spin_x=args.add_truncated_normal_spin_x,
        use_truncated_normal_spin_y=args.add_truncated_normal_spin_y,
        use_truncated_normal_spin_z=args.add_truncated_normal_spin_z,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_skew_normal_chi_eff=args.add_skew_normal_chi_eff,
        use_truncated_normal_chi_p=args.add_truncated_normal_chi_p,
        use_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
        use_redshift=args.add_redshift,
        use_cos_iota=args.add_cos_iota,
        use_phi_12=args.add_phi_12,
        use_polarization_angle=args.add_polarization_angle,
        use_right_ascension=args.add_right_ascension,
        use_sin_declination=args.add_sin_declination,
        use_detection_time=args.add_detection_time,
        filename=args.output_filename,
        model_fn=NPowerlawMGaussian,
        model_params_filename=args.model_params,
        poisson_mean_filename=args.pmean_cfg,
        derive_parameters=args.derive_parameters,
    )

    generator.from_inverse_transform_sampling()
