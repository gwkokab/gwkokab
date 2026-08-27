# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-hoc marginalisation helpers.

The heavy entry points here (``compute_batched_marginals``, ``generate_marginal_probs``)
need a full inference output to run, so what is covered below are the pieces they are
assembled from: the quadrature that turns a joint density into per-axis marginals, the
layout bookkeeping that maps a joint's marginals onto grid axes, the redshift
correction, and the popsummary writer the report reads back.
"""

import matplotlib
import numpy as np
import popsummary as ps
import pytest
from jax import numpy as jnp
from numpyro import distributions as dist

from gwkokab.analysis.core.utils import read_attrs_from_hdf5
from gwkokab.analysis.utils.marginals import (
    calculate_dist_layouts,
    calculate_marginals_over_axes,
    plot_with_intervals,
    PlotStyle,
    remove_comoving_volume_factor,
    save_results_to_hdf5,
)
from gwkokab.cosmology import default_cosmology
from gwkokab.models.utils import JointDistribution


# No plotting test may try to open a window, on a developer machine or in CI.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _isolated_pyplot():
    from matplotlib import pyplot as plt

    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def separable():
    """A product density on a 2D grid, whose marginals are known in closed form."""
    x = jnp.linspace(0.0, 1.0, 51)
    y = jnp.linspace(-2.0, 2.0, 61)
    f = jnp.exp(-x)
    g = jnp.exp(-jnp.square(y))
    return x, y, f, g, f[:, None] * g[None, :]


class TestCalculateMarginalsOverAxes:
    def test_one_marginal_per_axis(self, separable):
        x, y, _, _, probs = separable

        marginals = calculate_marginals_over_axes(probs, [x, y])

        assert [m.shape for m in marginals] == [(x.size,), (y.size,)]

    def test_a_product_density_marginalises_to_its_factors(self, separable):
        """Integrating out the other axis of ``f(x) g(y)`` leaves ``f`` up to a
        constant, which normalisation then fixes.
        """
        x, y, f, g, probs = separable

        marginal_x, marginal_y = calculate_marginals_over_axes(probs, [x, y])

        np.testing.assert_allclose(
            marginal_x, f / jnp.trapezoid(f, x), rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            marginal_y, g / jnp.trapezoid(g, y), rtol=1e-6, atol=1e-8
        )

    def test_normalised_marginals_integrate_to_one(self, separable):
        x, y, _, _, probs = separable

        marginal_x, marginal_y = calculate_marginals_over_axes(probs, [x, y])

        np.testing.assert_allclose(jnp.trapezoid(marginal_x, x), 1.0, rtol=1e-6)
        np.testing.assert_allclose(jnp.trapezoid(marginal_y, y), 1.0, rtol=1e-6)

    def test_normalisation_can_be_switched_off_per_axis(self, separable):
        """An un-normalised marginal keeps the rate scale the mixture carries."""
        x, y, f, g, probs = separable

        raw_x, normalised_y = calculate_marginals_over_axes(
            probs, [x, y], normalize=[False, True]
        )

        np.testing.assert_allclose(raw_x, f * jnp.trapezoid(g, y), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(jnp.trapezoid(normalised_y, y), 1.0, rtol=1e-6)

    def test_a_zero_density_does_not_divide_by_zero(self, separable):
        """Components can be switched off by a zero rate, and must not produce NaNs."""
        x, y, _, _, _ = separable

        marginals = calculate_marginals_over_axes(jnp.zeros((x.size, y.size)), [x, y])

        for marginal in marginals:
            assert jnp.all(marginal == 0.0)

    def test_three_dimensions(self):
        x = jnp.linspace(0.0, 1.0, 11)
        y = jnp.linspace(0.0, 2.0, 13)
        z = jnp.linspace(0.0, 3.0, 17)
        probs = jnp.ones((x.size, y.size, z.size))

        marginals = calculate_marginals_over_axes(probs, [x, y, z])

        assert [m.shape for m in marginals] == [(11,), (13,), (17,)]
        for marginal, domain in zip(marginals, (x, y, z)):
            np.testing.assert_allclose(
                marginal,
                jnp.full_like(domain, 1.0 / (domain[-1] - domain[0])),
                rtol=1e-6,
            )


class TestCalculateDistLayouts:
    def test_scalar_marginals_map_to_their_own_column(self):
        assert calculate_dist_layouts([0, 1, 2]) == [(0,), (1,), (2,)]

    def test_a_bivariate_marginal_maps_to_its_two_columns(self):
        """``JointDistribution`` records a vector marginal as the half-open column range
        ``(k, k + event_dim)``; the layout is the pair of columns it spans.
        """
        assert calculate_dist_layouts([(0, 2)]) == [(0, 1)]

    def test_matches_a_real_joint_distribution(self):
        joint = JointDistribution(
            dist.Normal(0.0, 1.0),
            dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
            dist.Normal(0.0, 1.0),
        )

        layouts = calculate_dist_layouts(list(joint.shaped_values))

        assert layouts == [(0,), (1, 2), (3,)]
        assert joint.event_shape == (4,)


class TestRemoveComovingVolumeFactor:
    def test_divides_out_the_volume_and_time_dilation(self):
        z = jnp.linspace(0.1, 2.0, 25)
        cosmo = default_cosmology()
        density = jnp.exp(-z)

        corrected = remove_comoving_volume_factor(density, z)

        np.testing.assert_allclose(
            corrected, density * (1.0 + z) / cosmo.dVcdz(z), rtol=1e-6
        )

    def test_inverts_multiplying_the_factor_in(self):
        z = jnp.linspace(0.1, 2.0, 25)
        cosmo = default_cosmology()
        underlying = jnp.power(1.0 + z, 2.7)
        observed = underlying * cosmo.dVcdz(z) / (1.0 + z)

        np.testing.assert_allclose(
            remove_comoving_volume_factor(observed, z), underlying, rtol=1e-6
        )

    def test_a_vanishing_volume_element_gives_zero(self):
        """``dVc/dz`` vanishes at ``z = 0``, where the division is masked out."""
        z = jnp.asarray([0.0, 0.5])

        corrected = remove_comoving_volume_factor(jnp.ones(2), z)

        assert corrected[0] == 0.0
        assert corrected[1] > 0.0

    def test_shape_is_preserved(self):
        z = jnp.linspace(0.1, 1.0, 8)

        assert remove_comoving_volume_factor(jnp.ones((4, 8)), z).shape == (4, 8)


class TestPlotWithIntervals:
    def test_draws_a_median_line_and_a_band(self):
        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
        xx = np.linspace(0.0, 1.0, 16)
        yy = np.outer(np.linspace(0.5, 1.5, 200), xx)

        plot_with_intervals(ax, xx, yy, PlotStyle(color="C0", label="powerlaw"))

        (line,) = ax.get_lines()
        assert line.get_label() == "powerlaw"
        np.testing.assert_allclose(line.get_ydata(), np.median(yy, axis=0))
        assert len(ax.collections) == 1

    def test_the_band_is_the_90_percent_interval(self):
        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
        xx = np.linspace(0.0, 1.0, 8)
        yy = np.outer(np.linspace(0.0, 1.0, 1001), np.ones_like(xx))

        plot_with_intervals(ax, xx, yy, PlotStyle(color="C1", label="band"))

        vertices = ax.collections[0].get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose(vertices.min(), 0.05, atol=1e-3)
        np.testing.assert_allclose(vertices.max(), 0.95, atol=1e-3)

    def test_style_kwargs_reach_the_artists(self):
        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
        xx = np.linspace(0.0, 1.0, 8)
        yy = np.outer(np.linspace(0.5, 1.5, 32), xx)
        style = PlotStyle(
            color="C2",
            label="gaussian",
            line_plot_kwargs={"linewidth": 4.0, "linestyle": "--"},
            fill_between_kwargs={"alpha": 0.75},
        )

        plot_with_intervals(ax, xx, yy, style)

        (line,) = ax.get_lines()
        assert line.get_linewidth() == 4.0
        assert line.get_linestyle() == "--"
        assert ax.collections[0].get_alpha() == 0.75

    def test_an_explicit_colour_overrides_the_style_colour(self):
        """``color`` inside the kwargs wins, and must not be passed twice."""
        from matplotlib import colors, pyplot as plt

        _, ax = plt.subplots()
        xx = np.linspace(0.0, 1.0, 8)
        yy = np.outer(np.linspace(0.5, 1.5, 32), xx)
        style = PlotStyle(
            color="C3",
            label="override",
            line_plot_kwargs={"color": "black"},
            fill_between_kwargs={"color": "red"},
        )

        plot_with_intervals(ax, xx, yy, style)

        (line,) = ax.get_lines()
        assert colors.to_hex(line.get_color()) == colors.to_hex("black")

    def test_the_style_dictionaries_are_not_mutated(self):
        """The same ``PlotStyle`` is reused across components, so popping ``color`` out
        of it would silently change later plots.
        """
        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
        xx = np.linspace(0.0, 1.0, 8)
        yy = np.outer(np.linspace(0.5, 1.5, 32), xx)
        style = PlotStyle(
            color="C4",
            label="reused",
            line_plot_kwargs={"color": "black", "linewidth": 1.0},
            fill_between_kwargs={"color": "red", "alpha": 0.2},
        )

        plot_with_intervals(ax, xx, yy, style)

        assert style.line_plot_kwargs == {"color": "black", "linewidth": 1.0}
        assert style.fill_between_kwargs == {"color": "red", "alpha": 0.2}


class TestSaveResultsToHDF5:
    @pytest.fixture
    def saved(self, tmp_path):
        """Write a two-component, one-parameter result and hand back what went in."""
        path = str(tmp_path / "marginals.hdf5")
        domain_cfg = {"mass_1_source": (5.0, 50.0, 7)}
        constants = {"mmin_pl_0": 5.0}
        # ``beta_pl_0`` is an alias of ``alpha_pl_0``: both read column 0.
        variables_index = {"alpha_pl_0": 0, "beta_pl_0": 0, "log_rate_0": 1}
        samples = np.arange(10.0).reshape(5, 2)
        batched_results = [
            [np.full((5, 7), 1.0)],
            [np.full((5, 7), 2.0)],
        ]

        save_results_to_hdf5(
            constants=constants,
            variables_index=variables_index,
            samples=samples,
            batched_results=batched_results,
            parameters=["mass_1_source"],
            domain_cfg=domain_cfg,
            filepath=path,
        )
        return path, constants, variables_index, samples

    def test_samples_round_trip(self, saved):
        path, _, _, samples = saved

        result = ps.PopulationResult(path)

        np.testing.assert_allclose(result.get_hyperparameter_samples(), samples)

    def test_one_dataset_per_component_and_parameter(self, saved):
        path, _, _, _ = saved
        result = ps.PopulationResult(path)

        for i, expected in enumerate((1.0, 2.0)):
            positions, rates = result.get_rates_on_grids(f"component_{i}_mass_1_source")
            np.testing.assert_allclose(
                positions, np.linspace(5.0, 50.0, 7).reshape(1, -1)
            )
            np.testing.assert_allclose(rates, np.full((5, 7), expected))

    def test_aliased_parameters_collapse_to_one_hyperparameter_name(self, saved):
        """Two names sharing a sampled column occupy one popsummary slot, named after
        the alphabetically first of them.
        """
        path, _, _, _ = saved

        names = list(ps.PopulationResult(path).get_metadata("hyperparameters"))

        assert names == ["alpha_pl_0", "log_rate_0"]

    def test_constants_and_index_are_attached_to_the_samples(self, saved):
        """The report reconstructs the model from these two attributes alone."""
        path, constants, variables_index, _ = saved

        attrs = read_attrs_from_hdf5(path, "/posterior/hyperparameter_samples")

        assert attrs["constants"] == constants
        assert attrs["variables_index"] == variables_index
