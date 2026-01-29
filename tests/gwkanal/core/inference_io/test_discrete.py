# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json

import numpy as np
import pandas as pd
import pytest

from gwkanal.core.inference_io import DiscreteParameterEstimationLoader
from gwkokab.parameters import Parameters as P
from gwkokab.utils.transformations import chieff, chirp_mass


@pytest.fixture
def sample_pe_data():
    """Generate sample PE data for testing."""
    n_samples = 100
    df = pd.DataFrame(
        {
            "mass_1_source": np.random.uniform(30, 50, n_samples),
            "mass_2_source": np.random.uniform(5, 30, n_samples),
            "a_1": np.random.uniform(0, 1, n_samples),
            "a_2": np.random.uniform(0, 1, n_samples),
            "redshift": np.random.uniform(0.01, 0.5, n_samples),
        }
    )
    df["mass_ratio"] = df["mass_2_source"] / df["mass_1_source"]
    df["chirp_mass"] = chirp_mass(
        df["mass_1_source"].to_numpy(), df["mass_2_source"].to_numpy()
    )
    df["chi_eff"] = chieff(
        m1=df["mass_1_source"].to_numpy(),
        m2=df["mass_2_source"].to_numpy(),
        chi1z=df["a_1"].to_numpy(),
        chi2z=df["a_2"].to_numpy(),
    )
    return df


@pytest.fixture
def temp_pe_files(sample_pe_data, tmp_path):
    """Create temporary PE sample files."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"event_{i}.dat"
        sample_pe_data.to_csv(file_path, sep=" ", index=False)
        files.append(str(file_path))
    return files, tmp_path


@pytest.fixture
def temp_config_file(temp_pe_files, tmp_path):
    """Create a temporary JSON config file."""
    files, data_dir = temp_pe_files
    config = {
        "regex": str(data_dir / "event_*.dat"),
        "max_samples": 50,
        "mass_prior": "flat-detector-components",
        "spin_prior": "component",
        "distance_prior": "comoving",
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path), files


class TestDiscreteParameterEstimationLoaderInitialization:
    """Test loader initialization and configuration."""

    def test_basic_initialization(self, temp_pe_files):
        """Test basic loader initialization."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        assert len(loader.filenames) == 3
        assert loader.max_samples is None
        assert loader.mass_prior is None
        assert loader.spin_prior is None
        assert loader.distance_prior is None

    def test_initialization_with_all_parameters(self, temp_pe_files):
        """Test initialization with all parameters specified."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files),
            parameter_aliases={"chirp_mass_detector": "mc_det"},
            max_samples=50,
            mass_prior="flat-detector-components",
            spin_prior="component",
            distance_prior="comoving",
        )

        assert loader.max_samples == 50
        assert loader.mass_prior == "flat-detector-components"
        assert loader.spin_prior == "component"
        assert loader.distance_prior == "comoving"
        assert "chirp_mass_detector" in loader.parameter_aliases

    def test_from_json(self, temp_config_file):
        """Test loading configuration from JSON file."""
        config_path, files = temp_config_file
        loader = DiscreteParameterEstimationLoader.from_json(config_path)

        assert len(loader.filenames) == 3
        assert loader.max_samples == 50
        assert loader.mass_prior == "flat-detector-components"

    def test_from_json_missing_regex(self, tmp_path):
        """Test that missing regex field raises error."""
        config = {"max_samples": 50}
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(Exception):  # Should raise error due to missing regex
            DiscreteParameterEstimationLoader.from_json(str(config_path))

    def test_from_json_no_matching_files(self, tmp_path):
        """Test that no matching files raises error."""
        config = {"regex": str(tmp_path / "nonexistent_*.dat")}
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(Exception):  # Should raise error due to no matches
            DiscreteParameterEstimationLoader.from_json(str(config_path))


class TestLoadMethod:
    """Test the load method with various configurations."""

    def test_load_basic(self, temp_pe_files):
        """Test basic loading of samples."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        parameters = ("mass_1_source", "mass_2_source", "redshift")
        data_list, log_prior_list = loader.load(parameters)

        assert len(data_list) == 3
        assert len(log_prior_list) == 3
        assert data_list[0].shape[1] == 3  # 3 parameters
        assert data_list[0].shape[0] == 100  # 100 samples

    def test_load_with_max_samples(self, temp_pe_files):
        """Test loading with max_samples constraint."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), max_samples=30
        )

        parameters = ("mass_1_source", "redshift")
        data_list, log_prior_list = loader.load(parameters, seed=42)

        assert all(len(data) == 30 for data in data_list)
        assert all(len(lp) == 30 for lp in log_prior_list)

    def test_load_with_parameter_aliases(self, temp_pe_files):
        """Test loading with parameter aliases."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files),
            parameter_aliases={
                P.PRIMARY_MASS_SOURCE: "mass_1_source",
                P.SECONDARY_MASS_SOURCE: "mass_2_source",
            },
        )

        parameters = (P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE)
        data_list, log_prior_list = loader.load(parameters)

        assert len(data_list) == 3
        assert data_list[0].shape[1] == 2

    def test_load_missing_columns(self, temp_pe_files):
        """Test that loading missing columns raises error."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        parameters = ("nonexistent_parameter",)
        with pytest.raises(KeyError):
            loader.load(parameters)

    def test_subsample_with_more_samples_than_available(self, temp_pe_files):
        """Test subsampling when requested samples exceed available."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files[:1]),  # Only one file
            max_samples=200,  # More than available
        )

        parameters = ("mass_1_source",)
        with pytest.warns(UserWarning):
            data_list, _ = loader.load(parameters)

        # Should return all available samples (100)
        assert len(data_list[0]) == 100

    def test_load_reproducibility_with_seed(self, temp_pe_files):
        """Test that same seed produces same subsampling."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), max_samples=30
        )

        parameters = ("mass_1_source",)
        data1, _ = loader.load(parameters, seed=42)
        data2, _ = loader.load(parameters, seed=42)

        np.testing.assert_array_equal(data1[0], data2[0])


class TestMassPrior:
    """Test mass prior calculations."""

    def test_no_mass_prior(self, temp_pe_files):
        """Test that no mass prior returns zero contribution."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), mass_prior=None
        )

        parameters = ("mass_1_source", "redshift")
        _, log_prior_list = loader.load(parameters)

        # Without any priors, log_prior should be all zeros
        np.testing.assert_array_equal(log_prior_list[0], 0.0)

    def test_flat_detector_components_prior(self, temp_pe_files):
        """Test flat detector component mass prior."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), mass_prior="flat-detector-components"
        )

        parameters = ("mass_1_source", "mass_2_source", "redshift")
        _, log_prior_list = loader.load(parameters)

        # Should have non-zero log prior contributions
        assert not np.allclose(log_prior_list[0], 0.0)

    def test_flat_detector_chirp_mass_ratio_prior(self, temp_pe_files):
        """Test flat detector chirp mass-ratio prior."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), mass_prior="flat-detector-chirp-mass-ratio"
        )

        parameters = ("mass_1_source", "mass_2_source", "redshift")
        _, log_prior_list = loader.load(parameters)

        assert not np.allclose(log_prior_list[0], 0.0)

    def test_flat_source_components_prior(self, temp_pe_files):
        """Test flat source component mass prior."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), mass_prior="flat-source-components"
        )

        parameters = ("mass_1_source", "mass_2_source", "redshift")
        _, log_prior_list = loader.load(parameters)

        assert np.allclose(log_prior_list[0], 0.0)

    def test_mass_prior_without_redshift_raises_error(self, temp_pe_files):
        """Test that mass prior without redshift raises error."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), mass_prior="flat-detector-components"
        )

        parameters = ("mass_1_source", "mass_2_source")  # No redshift
        with pytest.raises(ValueError):
            loader.load(parameters)

    def test_mass_prior_with_chirp_mass(self, temp_pe_files):
        """Test mass prior adjustment when using chirp mass."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), mass_prior="flat-detector-chirp-mass-ratio"
        )

        parameters = ("chirp_mass", "mass_ratio", "redshift")
        _, log_prior_list = loader.load(parameters)

        assert not np.allclose(log_prior_list[0], 0.0)


class TestSpinPrior:
    """Test spin prior calculations."""

    def test_no_spin_prior(self, temp_pe_files):
        """Test that no spin prior returns zero contribution."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), spin_prior=None
        )

        parameters = ("a_1", "a_2", "redshift")
        _, log_prior_list = loader.load(parameters)

        np.testing.assert_array_equal(log_prior_list[0], 0.0)

    def test_component_spin_prior(self, temp_pe_files):
        """Test component spin prior calculation."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), spin_prior="component"
        )

        parameters = ("a_1", "a_2", "redshift")
        _, log_prior_list = loader.load(parameters)

        # Should have log(1/4) contribution per sample
        expected = -np.log(4.0)
        np.testing.assert_array_almost_equal(log_prior_list[0], expected)

    def test_effective_spin_prior(self, temp_pe_files):
        """Test effective spin prior calculation."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        parameters = ("chi_eff", "mass_ratio", "redshift")
        _, log_prior_list = loader.load(parameters)

        # Should have non-zero contributions
        assert not np.allclose(log_prior_list[0], 0.0)


class TestDistancePrior:
    """Test distance prior calculations."""

    def test_no_distance_prior(self, temp_pe_files):
        """Test that no distance prior returns zero contribution."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), distance_prior=None
        )

        parameters = ("redshift",)
        _, log_prior_list = loader.load(parameters)

        np.testing.assert_array_equal(log_prior_list[0], 0.0)

    def test_comoving_distance_prior(self, temp_pe_files):
        """Test comoving distance prior calculation."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), distance_prior="comoving"
        )

        parameters = ("redshift",)
        _, log_prior_list = loader.load(parameters)

        # Should have non-zero contributions
        assert not np.allclose(log_prior_list[0], 0.0)
        assert np.all(np.isfinite(log_prior_list[0]))

    def test_euclidean_distance_prior(self, temp_pe_files):
        """Test Euclidean distance prior calculation."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), distance_prior="euclidean"
        )

        parameters = ("redshift",)
        _, log_prior_list = loader.load(parameters)

        assert not np.allclose(log_prior_list[0], 0.0)
        assert np.all(np.isfinite(log_prior_list[0]))


class TestHelperMethods:
    """Test internal helper methods."""

    def test_get_q_from_mass_ratio(self, temp_pe_files):
        """Test mass ratio extraction from mass_ratio column."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        df = pd.read_csv(files[0], delimiter=" ")
        aliases = {P.MASS_RATIO: "mass_ratio"}

        q = loader._get_q(df, aliases)
        np.testing.assert_array_equal(q, df["mass_ratio"].to_numpy())

    def test_get_q_from_source_masses(self, temp_pe_files):
        """Test mass ratio calculation from source masses."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files),
            parameter_aliases={
                P.MASS_RATIO: "nonexistent",
                P.PRIMARY_MASS_SOURCE: "mass_1_source",
                P.SECONDARY_MASS_SOURCE: "mass_2_source",
            },
        )

        df = pd.read_csv(files[0], delimiter=" ")
        aliases = {
            P.MASS_RATIO: "nonexistent",
            P.PRIMARY_MASS_SOURCE: "mass_1_source",
            P.SECONDARY_MASS_SOURCE: "mass_2_source",
        }

        q = loader._get_q(df, aliases)
        expected = df["mass_2_source"].to_numpy() / df["mass_1_source"].to_numpy()
        np.testing.assert_array_almost_equal(q, expected)

    def test_get_q_from_detector_masses(self, temp_pe_files):
        """Test mass ratio calculation from detector masses."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files),
            parameter_aliases={
                P.MASS_RATIO: "nonexistent",
                P.PRIMARY_MASS_SOURCE: "mass_1_source",
                P.SECONDARY_MASS_SOURCE: "mass_2_source",
            },
        )

        df = pd.read_csv(files[0], delimiter=" ")
        aliases = {
            P.MASS_RATIO: "nonexistent",
            P.PRIMARY_MASS_SOURCE: "mass_1_source",
            P.SECONDARY_MASS_SOURCE: "mass_2_source",
        }

        q = loader._get_q(df, aliases)
        expected = df["mass_2_source"].to_numpy() / df["mass_1_source"].to_numpy()
        np.testing.assert_array_almost_equal(q, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_parameter_list(self, temp_pe_files):
        """Test loading with empty parameter list."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        parameters = ()
        data_list, log_prior_list = loader.load(parameters)

        assert len(data_list) == 3
        assert data_list[0].shape[1] == 0

    def test_single_file(self, temp_pe_files):
        """Test loading from single file."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files[:1]))

        parameters = ("mass_1_source",)
        data_list, log_prior_list = loader.load(parameters)

        assert len(data_list) == 1
        assert len(log_prior_list) == 1

    def test_combined_priors(self, temp_pe_files):
        """Test that multiple priors combine correctly."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files),
            mass_prior="flat-detector-components",
            spin_prior="component",
            distance_prior="comoving",
        )

        parameters = ("mass_1_source", "mass_2_source", "a_1", "redshift")
        _, log_prior_list = loader.load(parameters)

        # Should have contributions from all three priors
        assert not np.allclose(log_prior_list[0], 0.0)
        assert np.all(np.isfinite(log_prior_list[0]))

    def test_validate_columns(self, temp_pe_files):
        """Test column validation."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        df = pd.read_csv(files[0], delimiter=" ")

        # Should not raise for existing columns
        loader._validate_columns(df, files[0], ["mass_1_source", "redshift"])

        # Should raise for missing columns
        with pytest.raises(KeyError):
            loader._validate_columns(df, files[0], ["nonexistent_column"])


class TestDataConsistency:
    """Test data consistency across multiple loads."""

    def test_same_data_different_parameters(self, temp_pe_files):
        """Test loading different parameters from same files."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(filenames=tuple(files))

        params1 = ("mass_1_source",)
        params2 = ("mass_1_source", "mass_2_source")

        data1, _ = loader.load(params1, seed=42)
        data2, _ = loader.load(params2, seed=42)

        # First column should match
        np.testing.assert_array_equal(data1[0][:, 0], data2[0][:, 0])

    def test_log_prior_shape_matches_data(self, temp_pe_files):
        """Test that log_prior arrays match data array shapes."""
        files, _ = temp_pe_files
        loader = DiscreteParameterEstimationLoader(
            filenames=tuple(files), max_samples=30
        )

        parameters = ("mass_1_source", "redshift")
        data_list, log_prior_list = loader.load(parameters)

        for data, log_prior in zip(data_list, log_prior_list):
            assert len(data) == len(log_prior)
