#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

import os

import h5py
import numpy as np
import polars as pl
from matplotlib import pyplot as plt


os.environ["KERAS_BACKEND"] = "jax"

import keras


class NeuralVT:
    """
    A class to approximate the log of the VT function using a neural network.

    >>> from gwkokab.vts.neuralvt import NeuralVT
    >>> neural_vt = NeuralVT(
    ...     input_keys=["m1", "m2"],
    ...     output_keys=["VT"],
    ...     hidden_layers=[64, 64, 512, 512, 64, 64],
    ...     epochs=2,
    ...     batch_size=128,
    ... )
    >>> neural_vt.train(plot_loss=True)
    """

    def __init__(
        self,
        input_keys: list[str] = ["m1", "m2"],
        output_keys: list[str] = ["VT"],
        hidden_layers: list[int] = [128],
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        model_name: str = "log(VT) approximation model",
        activation: str = "relu",
        data_path: str = "./vt_1_200_1000.hdf5",
        batch_size: int = 128,
        epochs: int = 15,
        validation_split: float = 0.2,
        model_path: str = "model.keras",
    ) -> None:
        """Initialize the NeuralVT class.

        :param input_keys: keys of the input data, defaults to ["m1", "m2"]
        :param output_key: key of the output data, defaults to ["VT"]
        :param hidden_layers: hidden layers of the neural network, defaults to [128]
        :param optimizer: optimizer for the neural network, defaults to "adam"
        :param loss: loss function for the neural network, defaults to "mean_squared_error"
        :param model_name: name of the model, defaults to "log(VT) approximation model"
        :param activation: activation function for the hidden layers, defaults to "relu"
        :param data_path: path to the HDF5 file containing the data, defaults to "./vt_1_200_1000.hdf5"
        :param batch_size: batch size for training the neural network, defaults to 128
        :param epochs: number of epochs for training the neural network, defaults to 15
        :param validation_split: validation split for training the neural network, defaults to 0.2
        :param model_path: path to save the trained model, defaults to "model.keras"
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = model_name
        self.activation = activation
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.model_path = model_path

    def read_data(self) -> pl.DataFrame:
        """Read the data from the HDF5 file.

        :return: a polars DataFrame containing the data
        """
        with h5py.File(self.data_path, "r") as vt_file:
            keys = list(vt_file.keys())
            df = pl.DataFrame(
                data={key: pl.Series(key, np.array(vt_file[key][:]).flatten()) for key in keys},
            )
        return df

    def build_model(self):
        model = keras.models.Sequential(name=self.model_name)

        for hidden_layer in self.hidden_layers:
            model.add(keras.layers.Dense(hidden_layer, activation=self.activation))
        model.add(keras.layers.Dense(len(self.output_keys), name="log(VT) Output Layer"))

        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def train(self, *, plot_loss: bool = True):
        """Train the neural network to approximate the log of the VT function.

        :param plot_loss: plot the loss function of the model, defaults to True
        """
        df = self.read_data()

        data_X = df[self.input_keys].to_numpy()
        data_Y = df[self.output_keys].to_numpy()

        log_data_Y = np.log(data_Y)

        model = self.build_model()

        history = model.fit(
            x=data_X,
            y=log_data_Y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_split=self.validation_split,
        )

        model.summary()

        keras.saving.save_model(model, self.model_path)

        if plot_loss:
            self.plot_loss(history)

    def plot_loss(self, history, save: bool = False, file_path: str = "loss.png"):
        """Plot the loss function of the model.

        :param history: history object from the model.fit method
        :param save: save the plot to a file, defaults to False
        :param file_path: path to save the plot, defaults to "loss.png"
        """
        plt.yscale("log")
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(file_path)
        plt.show()
