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
from matplotlib import pyplot as plt


os.environ["KERAS_BACKEND"] = "jax"

import keras


class NeuralVT:
    """
    A class to approximate the log of the VT function using a neural network.

    >>> from gwkokab.vts.neuralvt import NeuralVT
    >>> neural_vt = NeuralVT(hidden_layers=[64, 64, 512, 512, 64, 64])
    >>> neural_vt.train(plot_loss=True)
    """

    def __init__(
        self,
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

    def read_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the data from the HDF5 file.

        :return: m1, m2, VT
        """
        with h5py.File(self.data_path, "r") as hdf5_file:
            m1_grid = hdf5_file["m1"][:]
            m2_grid = hdf5_file["m2"][:]
            VT_grid = np.array(hdf5_file["VT"][:]).flatten()
            m1_coord = np.array(m1_grid).flatten()
            m2_coord = np.array(m2_grid).flatten()
        return m1_coord, m2_coord, VT_grid

    def build_model(self):
        model = keras.models.Sequential(name=self.model_name)

        model.add(keras.layers.InputLayer(shape=(2,), name="input"))
        for hidden_layer in self.hidden_layers:
            model.add(keras.layers.Dense(hidden_layer, activation=self.activation))
        model.add(keras.layers.Dense(1, name="log(VT) Output Layer"))

        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def train(self, *, plot_loss: bool = True):
        """Train the neural network to approximate the log of the VT function.

        :param plot_loss: plot the loss function of the model, defaults to True
        """
        m1_true, m2_true, VT_true = self.read_data()

        log_VT_true = np.log(VT_true)

        model = self.build_model()

        model.summary()

        history = model.fit(
            x=np.column_stack([m1_true, m2_true]),
            y=log_VT_true,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_split=self.validation_split,
        )

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
