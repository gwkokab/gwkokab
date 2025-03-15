# Copyright (c) 2025 Tom Callister
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Tuple

import equinox as eqx
import h5py
import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ...utils.tools import warn_if
from .._abc import VolumeTimeSensitivityInterface


class Emulator(VolumeTimeSensitivityInterface):
    """Base class implementing a generic detection probability emulator.

    Intended to be subclassed when constructing emulators for particular
    networks/observing runs.
    """

    nn_vt: eqx.nn.MLP = eqx.field()
    scaler: dict[str, Array] = eqx.field()
    scale: float = eqx.field()

    def __init__(
        self,
        trained_weights: str,
        scaler: str,
        input_size: int,
        hidden_layer_width: int,
        hidden_layer_depth: int,
        activation: Callable,
        final_activation: Callable,
        batch_size: Optional[int] = None,
        scale: float = 1.0,
    ):
        """Instantiate an `emulator` object.

        Parameters
        ----------
        trained_weights : `str`
            Filepath to .hdf5 file containing trained network weights, as
            saved by a `tensorflow.keras.Model.save_weights` command
        scaler : `str`
            Filepath to saved `sklearn.preprocessing.StandardScaler` object,
            fitted during network training
        input_size : `int`
            Dimensionality of input feature vector
        hidden_layer_width : `int`
            Width of hidden layers
        hidden_layer_depth : `int`
            Number of hidden layers
        activation : `func`
            Activation function to be applied to hidden layers
        batch_size: `int`
            Batch size to be used by `jax.vmap`

        Returns
        -------
        None
        """

        warn_if(
            not jax.config.read("jax_enable_x64"),
            msg="jax_enable_x64 is not enabled; this may cause numerical instability.",
        )

        # Instantiate neural network
        nn = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            depth=hidden_layer_depth,
            width_size=hidden_layer_width,
            activation=activation,
            final_activation=final_activation,
            key=jax.random.PRNGKey(111),
        )

        # Load trained weights and biases
        weight_data = h5py.File(trained_weights, "r")

        # Load scaling parameters
        with open(scaler, "r") as f:
            _scaler = json.load(f)
            self.scaler = {
                "mean": jax.device_put(jnp.asarray(_scaler["mean"]), may_alias=True),
                "scale": jax.device_put(jnp.asarray(_scaler["scale"]), may_alias=True),
            }

        # Define helper functions with which to access MLP weights and biases
        # Needed by `eqx.tree_at`
        def get_weights(i: int) -> Callable[[eqx.nn.MLP], Array]:
            return lambda t: t.layers[i].weight

        def get_biases(i: int) -> Callable[[eqx.nn.MLP], Optional[Array]]:
            return lambda t: t.layers[i].bias

        # Loop across layers, load pre-trained weights and biases
        for i in range(hidden_layer_depth + 1):
            if i == 0:
                key = "dense"
            else:
                key = "dense_{0}".format(i)

            layer_weights = jax.device_put(
                weight_data["{0}/{0}/kernel:0".format(key)][()].T, may_alias=True
            )
            nn = eqx.tree_at(get_weights(i), nn, layer_weights)

            layer_biases = jax.device_put(
                weight_data["{0}/{0}/bias:0".format(key)][()].T, may_alias=True
            )
            nn = eqx.tree_at(get_biases(i), nn, layer_biases)

        weight_data.close()

        self.nn_vt = nn
        self.scale = scale
        self.batch_size = batch_size

    def _transform_parameters(self, *args, **kwargs) -> Array:
        """OVERWRITE UPON SUBCLASSING.

        Function to convert from a predetermined set of user-provided physical
        CBC parameters to the input space expected by the trained neural
        network. Used by `emulator.__call__` below.

        NOTE: This function should be JIT-able and differentiable, and so
        consistency/completeness checks should be performed upstream; we
        should be able to assume that `physical_params` is provided as
        expected.

        Parameters
        ----------
        *args : `jax.numpy.array`
            physical parameters characterizing CBC signals
        **kwargs : `jax.numpy.array`
            physical parameters characterizing CBC signals

        Returns
        -------
        transformed_parameters : `jax.numpy.array`
            Transformed parameter space expected by trained neural network
        """
        raise NotImplementedError

    def __call__(self, x):
        """Function to evaluate the trained neural network on a set of user- provided
        physical CBC parameters.

        NOTE: This function should be JIT-able and differentiable, and so any
        consistency or completeness checks should be performed upstream, such
        that we can assume the provided parameter vector `x` is already in the
        correct format expected by the `emulator._transform_parameters` method.
        """

        # Transform physical parameters to space expected by the neural network
        transformed_x = self._transform_parameters(
            *[x[..., i] for i in range(x.shape[-1])]
        )

        # Apply scaling, evaluate the network, and return
        scaled_x = (transformed_x - self.scaler["mean"]) / self.scaler["scale"]
        return jax.vmap(self.nn_vt)(scaled_x)

    @abstractmethod
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
        key: `jax.random.PRNGKey`
            Random key to be used for generating extrinsic parameters
        shape: `tuple`
            Shape of the input array
        parameter_dict : `dict`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        parameter_dict : `dict`
            Dictionary of CBC parameters, augmented with necessary derived parameters
        """
        raise NotImplementedError
