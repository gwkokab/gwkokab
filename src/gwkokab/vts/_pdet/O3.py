import os
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import wcosmo
from astropy import units
from jaxtyping import Array, PRNGKeyArray
from wcosmo.wcosmo import z_at_value

from ...parameters import (
    COS_INCLINATION,
    COS_TILT_1,
    COS_TILT_2,
    PHI_12,
    POLARIZATION_ANGLE,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    RIGHT_ASCENSION,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
    SIN_DECLINATION,
)
from ...utils.tools import error_if
from ...utils.transformations import eta_from_q, mass_ratio
from ._emulator import Emulator


# Disable units for wcosmo until the resolution of
# https://github.com/GalacticDynamics/quaxed/issues/133
wcosmo.disable_units()


Planck15: wcosmo.astropy.FlatLambdaCDM = getattr(wcosmo.astropy, "Planck15")


class pdet_O3(Emulator):
    """Class implementing the LIGO-Hanford, LIGO-Livingston, and Virgo network's
    selection function during their O3 observing run.

    Used to evaluate the detection probability of compact binaries, assuming a false
    alarm threshold of below 1 per year. The computed detection probabilities include
    all variation in the detectors' sensitivities over the course of the O3 run and
    accounts for time in which the instruments were not in observing mode. They should
    therefore be interpreted as the probability of a CBC detection if that CBC occurred
    during a random time between the startdate and enddate of O3.
    """

    all_parameters: Sequence[str] = eqx.field(static=True)
    proposal_dist: Dict[str, Tuple[float, float]] = eqx.field(static=True)
    parameters: Sequence[str] = eqx.field(static=True)
    extra_shape: Tuple[int, ...] = eqx.field(static=True)
    interp_DL: Array = eqx.field(static=False)
    interp_z: Array = eqx.field(static=False)
    is_full: bool = eqx.field(static=True)

    def __init__(
        self,
        model_weights=None,
        scaler=None,
        parameters: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ):
        """Instantiates a `p_det_O3` object, subclassed from the `emulator` class.

        Parameters
        ----------
        model_weights : `None` or `str`
            Filepath to .hdf5 file containing trained network weights, as saved
            by a `tensorflow.keras.Model.save_weights`, command, if one wishes
            to override the provided default weights (which are loaded when
            `model_weights==None`).
        scaler : `str`
            Filepath to saved `sklearn.preprocessing.StandardScaler` object, if
            one wishes to override the provided default (loaded when
            `scaler==None`).
        parameters : `list`
            List of parameters to be used by the emulator.
        batch_size : `int`
            Batch size to be used by `jax.vmap`
        """

        error_if(parameters is None, msg="Must provide list of parameters")

        self.all_parameters = (
            PRIMARY_MASS_SOURCE.name,
            SECONDARY_MASS_SOURCE.name,
            PRIMARY_SPIN_MAGNITUDE.name,
            SECONDARY_SPIN_MAGNITUDE.name,
            COS_TILT_1.name,
            COS_TILT_2.name,
            PHI_12.name,
            REDSHIFT.name,
            COS_INCLINATION.name,
            POLARIZATION_ANGLE.name,
            RIGHT_ASCENSION.name,
            SIN_DECLINATION.name,
        )

        error_if(
            PRIMARY_MASS_SOURCE.name not in parameters,
            msg="Must include {0} parameter".format(PRIMARY_MASS_SOURCE.name),
        )
        error_if(
            SECONDARY_MASS_SOURCE.name not in parameters,
            msg="Must include {0} parameter".format(SECONDARY_MASS_SOURCE.name),
        )

        self.is_full = set(self.all_parameters) == set(parameters)
        self.proposal_dist = {}
        if not self.is_full:
            # if not every parameter is provided, we are only storing the bounds of the
            # missing parameters
            missing_params = set(self.all_parameters) - set(parameters)

            proposal_dist = {
                REDSHIFT.name: (0.0, 10.0),
                COS_INCLINATION.name: (-1.0, 1.0),
                POLARIZATION_ANGLE.name: (0.0, jnp.pi),
                RIGHT_ASCENSION.name: (0.0, 2.0 * jnp.pi),
                SIN_DECLINATION.name: (-1.0, 1.0),
            }
            for param in missing_params:
                if param in proposal_dist:
                    self.proposal_dist[param] = proposal_dist[param]

        self.parameters = parameters

        self.extra_shape = (1000,)

        if model_weights is None:
            model_weights = os.path.join(
                os.path.dirname(__file__), "./weights_HLV_O3.hdf5"
            )
        else:
            warnings.warn("Overriding default weights", UserWarning)

        if scaler is None:
            scaler = os.path.join(os.path.dirname(__file__), "./scaler_HLV_O3.json")
        else:
            warnings.warn("Overriding default weights", UserWarning)

        input_dimension = 15
        hidden_width = 192
        hidden_depth = 4
        activation = lambda x: jax.nn.leaky_relu(x, 1e-3)
        final_activation = lambda x: (1.0 - 0.0589) * jax.nn.sigmoid(x)

        self.interp_DL = jnp.logspace(-4, jnp.log10(15.0), 500)
        self.interp_z = z_at_value(
            lambda z: Planck15.luminosity_distance(z) * units.Gpc,
            self.interp_DL,
        )

        super().__init__(
            model_weights,
            scaler,
            input_dimension,
            hidden_width,
            hidden_depth,
            activation,
            final_activation,
            batch_size,
        )

    def _transform_parameters(
        self,
        m1_trials: Array,
        m2_trials: Array,
        a1_trials: Array,
        a2_trials: Array,
        cost1_trials: Array,
        cost2_trials: Array,
        phi12_trials: Array,
        z_trials: Array,
        cos_inclination_trials: Array,
        pol_trials: Array,
        ra_trials: Array,
        sin_dec_trials: Array,
    ) -> Array:
        q = mass_ratio(m1=m1_trials, m2=m2_trials)
        eta = eta_from_q(q=q)
        Mtot_det = (m1_trials + m2_trials) * (1.0 + z_trials)
        Mc_det = (eta**0.6) * Mtot_det

        DL = jnp.interp(z_trials, self.interp_z, self.interp_DL)
        log_Mc_DL_ratio = (5.0 / 6.0) * jnp.log(Mc_det) - jnp.log(DL)
        amp_factor_plus = 2.0 * (
            log_Mc_DL_ratio + jnp.log1p(cos_inclination_trials**2) + jnp.log(0.5)
        )
        amp_factor_cross = 2.0 * (log_Mc_DL_ratio + jnp.log(cos_inclination_trials))

        # Effective spins
        chi_effective = (a1_trials * cost1_trials + q * a2_trials * cost2_trials) / (
            1.0 + q
        )
        chi_diff = (a1_trials * cost1_trials - a2_trials * cost2_trials) * 0.5

        # Generalized precessing spin
        Omg = q * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
        chi_1p = a1_trials * jnp.sqrt(1.0 - jnp.square(cost1_trials))
        chi_2p = a2_trials * jnp.sqrt(1.0 - jnp.square(cost2_trials))
        chi_p_gen = jnp.sqrt(
            jnp.square(chi_1p)
            + jnp.square(Omg * chi_2p)
            + 2.0 * Omg * chi_1p * chi_2p * jnp.cos(phi12_trials)
        )

        return jnp.stack(
            [
                amp_factor_plus,
                amp_factor_cross,
                Mc_det,
                Mtot_det,
                eta,
                q,
                DL,
                ra_trials,
                sin_dec_trials,
                jnp.abs(cos_inclination_trials),
                jnp.sin(pol_trials % jnp.pi),
                jnp.cos(pol_trials % jnp.pi),
                chi_effective,
                chi_diff,
                chi_p_gen,
            ],
            axis=-1,
        )

    def check_input(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[str, Array],
    ) -> Tuple[PRNGKeyArray, dict[str, Array]]:
        """Method to check provided set of compact binary parameters for any missing
        information, and/or to augment provided parameters with any additional derived
        information expected by the neural network. If extrinsic parameters (e.g. sky
        location, polarization angle, etc.) have not been provided, they will be
        randomly generated and appended to the given CBC parameters.

        Parameters
        ----------
        parameter_dict : `dict` or `pd.DataFrame`
            Set of compact binary parameters for which we want to evaluate pdet

        Returns
        -------
        parameter_dict : `dict`
            Dictionary of CBC parameters, augmented with necessary derived
            parameters
        """

        missing_params = {}

        if PRIMARY_SPIN_MAGNITUDE.name not in parameter_dict:
            missing_params[PRIMARY_SPIN_MAGNITUDE.name] = jnp.zeros(shape)

        if SECONDARY_SPIN_MAGNITUDE.name not in parameter_dict:
            missing_params[SECONDARY_SPIN_MAGNITUDE.name] = jnp.zeros(shape)

        if COS_TILT_1.name not in parameter_dict:
            missing_params[COS_TILT_1.name] = jnp.ones(shape)

        if COS_TILT_2.name not in parameter_dict:
            missing_params[COS_TILT_2.name] = jnp.ones(shape)

        if PHI_12.name not in parameter_dict:
            missing_params[PHI_12.name] = jnp.zeros(shape)

        parameter_dict.update(missing_params)

        return key, parameter_dict

    def predict(self, key: PRNGKeyArray, params: Array) -> Array:
        shape = jnp.shape(params)[:-1]
        if self.is_full:
            parameter_dict = {
                parameter: jax.lax.dynamic_index_in_dim(
                    params, i, axis=-1, keepdims=False
                )
                for i, parameter in enumerate(self.parameters)
            }

            @jax.jit
            def _predict(_features: Array) -> Array:
                prediction = self.__call__(_features)
                prediction = jnp.nan_to_num(
                    prediction, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf
                )
                return prediction

            features = jnp.stack(
                [parameter_dict[param] for param in self.all_parameters],
                axis=-1,
            )
        else:
            parameter_dict = {
                parameter: jnp.broadcast_to(
                    jax.lax.dynamic_index_in_dim(params, i, axis=-1, keepdims=False),
                    self.extra_shape + shape,
                )
                for i, parameter in enumerate(self.parameters)
            }
            key, parameter_dict = self.check_input(
                key, self.extra_shape + shape, parameter_dict
            )
            for param in self.proposal_dist.keys():
                parameter_dict[param] = jrd.uniform(
                    key,
                    self.extra_shape + shape,
                    minval=self.proposal_dist[param][0],
                    maxval=self.proposal_dist[param][1],
                )

            @jax.jit
            def _predict(_features: Array) -> Array:
                prediction = self.__call__(_features)
                prediction = jnp.nan_to_num(
                    prediction, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf
                )
                return jnp.mean(prediction, axis=0)

            features = jnp.stack(
                [parameter_dict[param] for param in self.all_parameters],
                axis=-1,
            )
            features = jnp.moveaxis(features, 1, 0)

        return jnp.squeeze(
            jax.lax.map(_predict, features, batch_size=self.batch_size), axis=-1
        )

    def get_logVT(self) -> Callable[[Array], Array]:
        raise NotImplementedError("logVT is not implemented for pdet_O3")

    def get_mapped_logVT(self) -> Callable[[Array], Array]:
        return lambda x: jnp.log(self.predict(jrd.PRNGKey(111), x))
