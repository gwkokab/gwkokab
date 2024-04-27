#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import glob
from typing_extensions import Optional

import matplotlib.pyplot as plt
import mplcursors
import numpy as np


def scatter2d_batch_plot(
    file_pattern: str,
    output_filename: str,
    x_index: int,
    y_index: int,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    plt_title: Optional[str] = None,
) -> None:
    """Plot a batch of 2D scatter plots.

    This function plots a batch of 2D scatter plots from the given files
    and saves the plot to the given output file.

    :param file_pattern: regex pattern for the files
    :param output_filename: name of the output file
    :param x_index: index of the x-axis data
    :param y_index: index of the y-axis data
    :param x_label: label for the x-axis, defaults to `None`
    :param y_label: label for the y-axis, defaults to `None`
    :param plt_title: title for the plot, defaults to `None`
    """
    file_list = glob.glob(file_pattern)

    # Iterate over each file to make the scatter plots of each event in a figure.
    for file_path in file_list:
        # Load data from the file
        data = np.loadtxt(file_path)
        x = data[..., x_index]
        y = data[..., y_index]

        # Scatter plot with different colors for each file
        plt.scatter(x, y, s=5, alpha=0.3, marker=".")

    # Set plot title and labels
    if plt_title is not None:
        plt.title(plt_title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close("all")


def scatter3d_batch_plot(
    file_pattern: str,
    output_filename: str,
    x_index: int,
    y_index: int,
    z_index: int,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    plt_title: Optional[str] = None,
) -> None:
    """Create a 3D scatter plot from a batch of files.

    This function creates a 3D scatter plot from a batch of files and
    saves the plot to the given output file.

    :param file_pattern: regex pattern for the files
    :param output_filename: name of the output file
    :param x_index: index of the x-axis data
    :param y_index: index of the y-axis data
    :param z_index: index of the z-axis data
    :param x_label: label for the x-axis, defaults to `None`
    :param y_label: label for the y-axis, defaults to `None`
    :param z_label: label for the z-axis, defaults to `None`
    :param plt_title: title for the plot, defaults to `None`
    """
    file_list = glob.glob(file_pattern)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for file_path in file_list:
        # Load data from the file
        data = np.loadtxt(file_path)
        x = data[..., x_index]
        y = data[..., y_index]
        z = data[..., z_index]

        # Scatter plot with different colors for each file
        ax.scatter(x, y, z, c=z, cmap="plasma", marker="o", alpha=0.3)

    # Set labels for title and axes
    if plt_title is not None:
        ax.set_title(plt_title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    cursors = mplcursors.cursor(ax, hover=True)
    cursors.connect(
        "add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f}, {sel.target[2]:.2f})")
    )

    plt.savefig(output_filename, bbox_inches="tight")
    # Display the plot
    plt.close("all")


def scatter2d_plot(
    input_filename: str,
    output_filename: str,
    x_index: int,
    y_index: int,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    plt_title: Optional[str] = None,
) -> None:
    """Create a 2D scatter plot from a file.

    This function creates a 2D scatter plot from the given file and
    saves the plot to the given output file.

    :param input_filename: name of the input file
    :param output_filename: name of the output file
    :param x_index: index of the x-axis data
    :param y_index: index of the y-axis data
    :param x_label: label for the x-axis, defaults to `None`
    :param y_label: label for the y-axis, defaults to `None`
    :param plt_title: title for the plot, defaults to `None`
    """
    # Load data from the file
    data = np.loadtxt(input_filename)
    x = data[..., x_index]
    y = data[..., y_index]

    # Scatter plot with different colors for each file
    plt.scatter(x, y, alpha=0.3, marker=".")

    # Set plot title and labels
    if plt_title is not None:
        plt.title(plt_title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close("all")


def scatter3d_plot(
    input_filename: str,
    output_filename: str,
    x_index: int,
    y_index: int,
    z_index: int,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    plt_title: Optional[str] = None,
) -> None:
    """Create a 3D scatter plot from a file.

    This function creates a 3D scatter plot from the given file and
    saves the plot to the given output file.

    :param input_filename: name of the input file
    :param output_filename: name of the output file
    :param x_index: index of the x-axis data
    :param y_index: index of the y-axis data
    :param z_index: index of the z-axis data
    :param x_label: label for the x-axis, defaults to `None`
    :param y_label: label for the y-axis, defaults to `None`
    :param z_label: label for the z-axis, defaults to `None`
    :param plt_title: title for the plot, defaults to `None`
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    data = np.loadtxt(input_filename)
    x = data[..., x_index]
    y = data[..., y_index]
    z = data[..., z_index]

    # Plot the data points as dots with colors
    sc = ax.scatter(x, y, z, c=z, cmap="plasma", marker="o", alpha=0.3)
    cbar = fig.colorbar(sc)

    # Set a title for the plot
    if plt_title is not None:
        ax.set_title(plt_title)
    # Set labels for the axes
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        # ax.set_zlabel(z_label)
        cbar.set_label(z_label)

    # Add interactivity with mplcursors
    cursors = mplcursors.cursor(sc, hover=True)
    cursors.connect(
        "add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f}, {sel.target[2]:.2f})")
    )

    # Add a colorbar
    plt.savefig(output_filename, bbox_inches="tight")
    # Display the plot
    plt.close("all")
