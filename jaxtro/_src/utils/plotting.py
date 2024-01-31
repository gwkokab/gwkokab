#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:33:41 2023

@author: mzeeshan
"""
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
) -> None:
    file_list = glob.glob(file_pattern)

    # Iterate over each file to make the scatter plots of each event in a figure.
    for i, file_path in enumerate(file_list):
        # Load data from the file
        data = np.loadtxt(file_path)
        x = data[:, x_index]
        y = data[:, y_index]

        # Scatter plot with different colors for each file
        plt.scatter(x, y)

    # Set plot title and labels
    plt.title("Scatter Plot")
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
) -> None:
    file_list = glob.glob(file_pattern)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i, file_path in enumerate(file_list):
        # Load data from the file
        data = np.loadtxt(file_path)
        x = data[:, x_index]
        y = data[:, y_index]
        z = data[:, z_index]

        # Scatter plot with different colors for each file
        ax.scatter(x, y, z, c=z, cmap="plasma", marker="o")

    # Set labels for the axes
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    # if z_label is not None:
    #     ax.set_zlabel(z_label)

    # Set a title for the plot
    ax.set_title("3D Scatter Plot")

    # Add a colorbar
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
) -> None:
    # Load data from the file
    data = np.loadtxt(input_filename)
    x = data[:, x_index]
    y = data[:, y_index]

    # Scatter plot with different colors for each file
    plt.scatter(x, y)

    # Set plot title and labels
    plt.title("Scatter Plot")
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
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    data = np.loadtxt(input_filename)
    x = data[:, x_index]
    y = data[:, y_index]
    z = data[:, z_index]

    # Plot the data points as dots with colors
    sc = ax.scatter(x, y, z, c=z, cmap="plasma", marker="o")
    cbar = fig.colorbar(sc)

    # Set labels for the axes
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        # ax.set_zlabel(z_label)
        cbar.set_label(z_label)

    # Set a title for the plot
    ax.set_title("3D Scatter Plot")

    # Add interactivity with mplcursors
    cursors = mplcursors.cursor(sc, hover=True)
    cursors.connect(
        "add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f}, {sel.target[2]:.2f})")
    )

    # Add a colorbar
    plt.savefig(output_filename, bbox_inches="tight")
    # Display the plot
    plt.close("all")
