import shutil
from typing import NamedTuple

import corner
import glasbey
import numpy as np
from matplotlib import pyplot as plt

from gwkokab.analysis.core.utils import read_from_hdf5


use_tex = shutil.which("pdflatex") is not None


plt.rcParams.update({
    "axes.grid": True,
    "axes.labelpad": 6,
    "axes.linewidth": 1.0,
    "axes.titlepad": 8,
    "axes.labelsize": 15,
    "figure.autolayout": False,
    "figure.dpi": 200,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "grid.color": "#d3d3d3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "legend.borderaxespad": 0.3,
    "legend.fontsize": 15,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "lines.dash_capstyle": "round",
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "text.usetex": use_tex,
    "xtick.direction": "out",
    "xtick.major.size": 6,
    "xtick.major.width": 1.0,
    "xtick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.direction": "out",
    "ytick.major.size": 6,
    "ytick.major.width": 1.0,
    "ytick.minor.size": 3,
    "ytick.minor.visible": True,
})


class Info(NamedTuple):
    data: np.ndarray | list
    color: str
    legend: str


colors = glasbey.create_palette(5)

DISCRETE_FLOWMC = Info(
    read_from_hdf5("discrete_flowMC/inference_data.hdf5", "samples"),
    colors[0],
    "Discrete Method (flowMC)",
)
ANALYTICAL_GWALK_FLOWMC = Info(
    read_from_hdf5("analytical_gwalk_flowMC/inference_data.hdf5", "samples"),
    colors[1],
    "Analytical GWalk Method (flowMC)",
)
DISCRETE_NUMPYRO = Info(
    read_from_hdf5("discrete_numpyro/inference_data.hdf5", "samples"),
    colors[2],
    "Discrete Method (NumPyro)",
)
ANALYTICAL_GWALK_NUMPYRO = Info(
    read_from_hdf5("analytical_gwalk_numpyro/inference_data.hdf5", "samples"),
    colors[3],
    "Analytical GWalk Method (NumPyro)",
)
TRUE_VALUES = Info(
    [1.0, 0.0, 0.15, 4.0, 50.0, 5.0, 0.0, 0.4],
    colors[-1],
    "True Values",
)
LABELS = [
    r"$\alpha$",
    r"$\beta$",
    r"$\sigma_\epsilon$",
    r"$\ln(\mathcal{R})$",
    r"$m_{\mathrm{max}}$",
    r"$m_{\mathrm{min}}$",
    r"$\mu_{\chi}$",
    r"$\sigma_{\chi}$",
]


fig, ax = plt.subplots(len(LABELS), len(LABELS), figsize=(12, 12))

info_list = [
    DISCRETE_FLOWMC,
    ANALYTICAL_GWALK_FLOWMC,
    DISCRETE_NUMPYRO,
    ANALYTICAL_GWALK_NUMPYRO,
]


for info in info_list:
    fig = corner.corner(
        info.data,
        fig=fig,
        color=info.color,
        bins=30,
        smooth=True,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        hist_kwargs={"density": True},
        labels=LABELS,
    )

corner.overplot_lines(
    fig, TRUE_VALUES.data, color=TRUE_VALUES.color, linestyle="--", alpha=0.7
)
corner.overplot_points(fig, [TRUE_VALUES.data], marker="s", color=TRUE_VALUES.color)

handles = [
    plt.Line2D([], [], color=info.color, label=info.legend, lw=5) for info in info_list
]
handles.append(
    plt.Line2D(
        [],
        [],
        color=TRUE_VALUES.color,
        linestyle="--",
        marker="s",
        label=TRUE_VALUES.legend,
    )
)

fig.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.93, 0.93),
)

plt.savefig("figs/overplot_flowMC_numpyro.png", bbox_inches="tight", dpi=200)
