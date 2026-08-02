import shutil
from typing import NamedTuple

import glasbey
import numpy as np
from matplotlib import pyplot as plt
from numpyro.distributions import TruncatedNormal

from gwkokab.analysis.utils.marginals import plot_marginal_with_intervals, PlotStyle
from gwkokab.models import PowerlawPrimaryMassRatio
from gwkokab.models.transformations import (
    PrimaryMassAndMassRatioToComponentMassesTransform,
)
from gwkokab.models.utils import ExtendedSupportTransformedDistribution
from gwkokab.parameters import Parameters as P


use_tex = shutil.which("pdflatex") is not None

plt.rcParams.update({
    "axes.grid": True,
    "axes.labelpad": 4,
    "axes.linewidth": 1.0,
    "axes.titlepad": 8,
    "axes.labelsize": 12,
    "figure.autolayout": False,
    "figure.dpi": 200,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "grid.color": "#d3d3d3",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "legend.borderaxespad": 0.3,
    "legend.fontsize": 10,
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
    path: str
    color: str
    legend: str


colors = glasbey.create_palette(4)

analyses = [
    Info(
        path="analytical_flowMC",
        color=colors[0],
        legend="Analytical Method (flowMC)",
    ),
    Info(
        path="discrete_flowMC",
        color=colors[1],
        legend="Discrete Method (flowMC)",
    ),
    Info(
        path="analytical_numpyro",
        color=colors[2],
        legend="Analytical Method (numpyro)",
    ),
    Info(path="discrete_numpyro", color=colors[3], legend="Discrete Method (numpyro)"),
]


fig, ((ax_m1, ax_m2), (ax_spin_z, ax_ecc)) = plt.subplots(
    2, 2, figsize=(14, 7), constrained_layout=True
)


for i, analysis in enumerate(analyses):
    kwargs: dict = {
        "filename": f"{analysis.path}/marginal_probs.hdf5",
        "component_idxs": [0],
        "style": PlotStyle(
            color=colors[i],
            label=analysis.legend,
            line_plot_kwargs={"linewidth": 1.0, "linestyle": "-"},
            fill_between_kwargs={"alpha": 0.1, "linewidth": 0.0, "zorder": -1},
        ),
    }
    plot_marginal_with_intervals(
        ax=ax_m1,
        parameter=P.PRIMARY_MASS_SOURCE,
        scale=lambda p: np.exp(p["log_rate_0"]),
        **kwargs,
    )

    plot_marginal_with_intervals(
        ax=ax_m2,
        parameter=P.SECONDARY_MASS_SOURCE,
        scale=lambda p: np.exp(p["log_rate_0"]),
        **kwargs,
    )

    plot_marginal_with_intervals(ax=ax_spin_z, parameter=P.PRIMARY_SPIN_Z, **kwargs)

    plot_marginal_with_intervals(ax=ax_ecc, parameter=P.ECCENTRICITY, **kwargs)


mass_model = ExtendedSupportTransformedDistribution(
    base_distribution=PowerlawPrimaryMassRatio(
        alpha=1.0,
        beta=0.0,
        mmin=5.0,
        mmax=50.0,
        validate_args=True,
    ),
    transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
    validate_args=True,
)

m = np.linspace(5.0, 50.0, 500)


mass_grid = np.stack(np.meshgrid(m, m, indexing="ij"), axis=-1)

probs = np.exp(mass_model.log_prob(mass_grid))

dRdm1 = np.exp(4.0) * np.trapezoid(probs, x=m, axis=1)
dRdm2 = np.exp(4.0) * np.trapezoid(probs, x=m, axis=0)


ax_m1.plot(
    m,
    dRdm1,
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="True distribution",
)

ax_m2.plot(
    m,
    dRdm2,
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="True distribution",
)


ax_m1.set_xlabel(r"$m_1^\mathrm{source}$ [$M_\odot$]")
ax_m1.set_ylabel(
    r"$\displaystyle\frac{d\mathcal{R}}{d m_1^\mathrm{source}} \left[\mathrm{Gpc}^{-1} M_\odot^{-1}\right]$"
)
ax_m1.set_xlim(4.0, 50.0)
ax_m1.set_ylim(1e-1, None)
ax_m1.set_yscale("log")


ax_m2.set_xlabel(r"$m_2^\mathrm{source}$ [$M_\odot$]")
ax_m2.set_ylabel(
    r"$\displaystyle\frac{d\mathcal{R}}{d m_2^\mathrm{source}} \left[\mathrm{Gpc}^{-1} M_\odot^{-1}\right]$"
)
ax_m2.set_xlim(4.0, 50.0)
ax_m2.set_ylim(1e-2, None)
ax_m2.set_yscale("log")


spin_z = np.linspace(-1.0, 1.0, 200)

spin_z_model = TruncatedNormal(
    loc=0.0,
    scale=0.4,
    low=-1.0,
    high=1.0,
    validate_args=True,
)


ax_spin_z.plot(
    spin_z,
    np.exp(spin_z_model.log_prob(spin_z)),
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="True distribution",
)


ax_spin_z.set_xlabel(r"$\chi_z$")
ax_spin_z.set_ylabel(r"$p\left(\chi_z\right)$")
ax_spin_z.set_xlim(-1.0, 1.0)
ax_spin_z.set_ylim(0.0, None)


ecc = np.linspace(0.0, 1.0, 200)

ecc_model = TruncatedNormal(
    loc=0.0,
    scale=0.15,
    low=0.0,
    high=1.0,
    validate_args=True,
)


ax_ecc.plot(
    ecc,
    np.exp(ecc_model.log_prob(ecc)),
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="True distribution",
)


ax_ecc.set_xlabel(r"$e$")
ax_ecc.set_ylabel(r"$p\left(e\right)$")
ax_ecc.set_xlim(0.0, 0.5)
ax_ecc.set_ylim(0.0, None)


for ax in (ax_m1, ax_m2, ax_spin_z, ax_ecc):
    ax.legend()


fig.savefig("figs/marginals.png", dpi=100)
