# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the discrete (per-event posterior samples) data loader.

The loader does three separable jobs: turn a JSON config into a validated model, read
one event file out of whichever dataset it happens to live in, and undo the prior the
original PE run was performed under. The prior weights are the part worth checking
numerically, since a sign error there tilts every posterior downstream, so each one is
compared against the closed form it is meant to implement.
"""

import warnings

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from gwkokab.analysis.core.inference_io import DiscretePELoader
from gwkokab.analysis.core.utils import to_structured
from gwkokab.analysis.utils.common import write_json
from gwkokab.cosmology import default_cosmology
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean._injection_based_helper import (
    aligned_spin_prior,
    chi_effective_prior_from_isotropic_spins,
    primary_mass_to_chirp_mass_jacobian,
)
from gwkokab.utils.exceptions import (
    LoggedFileNotFoundError,
    LoggedKeyError,
    LoggedValueError,
)


DEFAULT_DATASET = "/GWKokabSyntheticDiscretePE/posterior_samples"

COLUMNS = {
    P.PRIMARY_MASS_SOURCE: np.array([30.0, 40.0, 50.0, 60.0]),
    P.SECONDARY_MASS_SOURCE: np.array([15.0, 30.0, 25.0, 30.0]),
    P.REDSHIFT: np.array([0.1, 0.2, 0.3, 0.4]),
    P.EFFECTIVE_SPIN: np.array([0.1, -0.2, 0.3, 0.0]),
    P.CHI_1: np.array([0.2, 0.4, 0.6, 0.8]),
}


@pytest.fixture
def event_file(tmp_path):
    """Factory writing one synthetic event file in the layout the loader expects."""

    def _make(
        name: str = "event_0",
        dataset: str = DEFAULT_DATASET,
        columns=None,
        n_samples: int | None = None,
    ) -> str:
        columns = COLUMNS if columns is None else columns
        data = np.column_stack([
            np.tile(v, n_samples // v.size) if n_samples else v
            for v in columns.values()
        ])
        path = str(tmp_path / f"{name}.hdf5")
        with h5py.File(path, "a") as f:
            # rewriting the same event is how a test asks for a fresh loader
            if dataset in f:
                del f[dataset]
            f.create_dataset(
                dataset, data=to_structured(data, [str(k) for k in columns])
            )
        return path

    return _make


@pytest.fixture
def config(tmp_path):
    """Factory writing a loader config JSON and returning its path."""

    def _make(**overrides) -> str:
        cfg = {"regex": str(tmp_path / "*.hdf5")}
        cfg.update(overrides)
        path = str(tmp_path / "loader.json")
        write_json(path, cfg)
        return path

    return _make


class TestReadFromJson:
    def test_the_regex_is_expanded_and_sorted(self, event_file, config):
        event_file("event_1")
        event_file("event_0")

        loader = DiscretePELoader.read_from_json(config())

        assert [p.stem for p in loader.filenames] == ["event_0", "event_1"]

    def test_a_missing_regex_is_rejected(self, tmp_path):
        path = str(tmp_path / "loader.json")
        write_json(path, {"max_samples": 10})

        with pytest.raises(LoggedKeyError, match="'regex' field is required"):
            DiscretePELoader.read_from_json(path)

    def test_a_regex_matching_nothing_is_rejected(self, tmp_path):
        path = str(tmp_path / "loader.json")
        write_json(path, {"regex": str(tmp_path / "no-such-*.hdf5")})

        with pytest.raises(LoggedFileNotFoundError, match="No files matched"):
            DiscretePELoader.read_from_json(path)

    def test_an_unknown_field_is_rejected(self, event_file, config):
        """``extra="forbid"`` is what turns a typo into an error instead of a silently
        ignored setting.
        """
        event_file()

        with pytest.raises(ValidationError):
            DiscretePELoader.read_from_json(config(max_sample=10))

    def test_an_invalid_prior_name_is_rejected(self, event_file, config):
        event_file()

        with pytest.raises(ValidationError):
            DiscretePELoader.read_from_json(config(default_mass_prior="flat"))

    def test_a_non_positive_max_samples_is_rejected(self, event_file, config):
        event_file()

        with pytest.raises(ValidationError):
            DiscretePELoader.read_from_json(config(max_samples=0))

    def test_a_dataset_without_a_leading_slash_is_repaired(self, event_file, config):
        event_file()

        with pytest.warns(UserWarning, match="does not start with"):
            loader = DiscretePELoader.read_from_json(
                config(default_datasets=["some/dataset"])
            )

        assert loader.default_datasets == ("/some/dataset",)

    def test_an_alternate_dataset_without_a_leading_slash_is_repaired(
        self, event_file, config
    ):
        event_file()

        with pytest.warns(UserWarning, match="does not start with"):
            loader = DiscretePELoader.read_from_json(
                config(alternate_datasets={"event_0": "other/dataset"})
            )

        assert loader.alternate_datasets == {"event_0": "/other/dataset"}

    def test_defaults_are_the_documented_ones(self, event_file, config):
        event_file()

        loader = DiscretePELoader.read_from_json(config())

        assert loader.default_datasets == (DEFAULT_DATASET,)
        assert loader.max_samples is None
        assert loader.default_mass_prior is None
        assert loader.default_spin_prior is None
        assert loader.default_distance_prior is None
        assert loader.parameter_aliases == {}


class TestLoadFile:
    def test_reads_the_structured_dataset_into_named_columns(self, event_file):
        path = event_file()

        df = DiscretePELoader.load_file(path, DEFAULT_DATASET)

        assert list(df.columns) == [str(k) for k in COLUMNS]
        np.testing.assert_allclose(
            df[str(P.PRIMARY_MASS_SOURCE)].to_numpy(), COLUMNS[P.PRIMARY_MASS_SOURCE]
        )

    def test_accepts_a_bare_dataset_name(self, event_file):
        path = event_file()

        df = DiscretePELoader.load_file(path, datasets=DEFAULT_DATASET)

        assert len(df) == 4

    def test_takes_the_first_dataset_that_exists(self, event_file):
        """The default list is a preference order, not a requirement that all exist."""
        path = event_file(dataset="/second")

        df = DiscretePELoader.load_file(path, datasets=("/first", "/second"))

        assert len(df) == 4

    def test_a_file_with_none_of_the_datasets_is_rejected(self, event_file):
        path = event_file(dataset="/somewhere_else")

        with pytest.raises(LoggedKeyError, match="None of the specified datasets"):
            DiscretePELoader.load_file(path, datasets=("/first",))


class TestSubsampling:
    def test_no_max_samples_keeps_everything(self, event_file, config):
        event_file(n_samples=16)
        loader = DiscretePELoader.read_from_json(config())

        (data,), _ = loader.load((P.PRIMARY_MASS_SOURCE,))

        assert data.shape == (16, 1)

    def test_max_samples_truncates(self, event_file, config):
        event_file(n_samples=16)
        loader = DiscretePELoader.read_from_json(config(max_samples=5))

        (data,), (log_prior,) = loader.load((P.PRIMARY_MASS_SOURCE,))

        assert data.shape == (5, 1)
        assert log_prior.shape == (5,)

    def test_subsampling_is_seeded(self, event_file, config):
        event_file(n_samples=16)
        loader = DiscretePELoader.read_from_json(config(max_samples=5))

        (first,), _ = loader.load((P.PRIMARY_MASS_SOURCE,), seed=3)
        (again,), _ = loader.load((P.PRIMARY_MASS_SOURCE,), seed=3)
        (other,), _ = loader.load((P.PRIMARY_MASS_SOURCE,), seed=4)

        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_asking_for_more_samples_than_there_are_warns(self, event_file, config):
        event_file()
        loader = DiscretePELoader.read_from_json(config(max_samples=100))

        with pytest.warns(UserWarning, match="Subsampling skipped"):
            (data,), _ = loader.load((P.PRIMARY_MASS_SOURCE,))

        assert data.shape == (4, 1)


class TestLoad:
    def test_one_entry_per_event_in_filename_order(self, event_file, config):
        event_file("event_0")
        event_file("event_1")
        loader = DiscretePELoader.read_from_json(config())

        data, log_prior = loader.load((P.PRIMARY_MASS_SOURCE, P.REDSHIFT))

        assert len(data) == len(log_prior) == 2
        assert all(d.shape == (4, 2) for d in data)

    def test_columns_come_back_in_the_requested_order(self, event_file, config):
        event_file()
        loader = DiscretePELoader.read_from_json(config())

        (data,), _ = loader.load((P.REDSHIFT, P.PRIMARY_MASS_SOURCE))

        np.testing.assert_allclose(data[:, 0], COLUMNS[P.REDSHIFT])
        np.testing.assert_allclose(data[:, 1], COLUMNS[P.PRIMARY_MASS_SOURCE])

    def test_a_missing_column_is_rejected(self, event_file, config):
        event_file()
        loader = DiscretePELoader.read_from_json(config())

        with pytest.raises(LoggedValueError, match="missing required columns"):
            loader.load((P.PRECESSING_SPIN,))

    def test_aliases_rename_the_file_columns(self, event_file, config):
        """A real catalogue calls the primary mass something else; the alias maps the
        file's column name onto the parameter the model wants.
        """
        columns = {"m1_src": COLUMNS[P.PRIMARY_MASS_SOURCE]}
        event_file(columns=columns)
        loader = DiscretePELoader.read_from_json(
            config(parameter_aliases={str(P.PRIMARY_MASS_SOURCE): "m1_src"})
        )

        (data,), _ = loader.load((P.PRIMARY_MASS_SOURCE,))

        np.testing.assert_allclose(data[:, 0], COLUMNS[P.PRIMARY_MASS_SOURCE])

    def test_no_priors_means_a_flat_weight(self, event_file, config):
        event_file()
        loader = DiscretePELoader.read_from_json(config())

        _, (log_prior,) = loader.load((P.PRIMARY_MASS_SOURCE, P.REDSHIFT))

        np.testing.assert_allclose(log_prior, np.zeros(4))

    def test_an_alternate_prior_overrides_the_default_for_that_event_only(
        self, event_file, config
    ):
        event_file("event_0")
        event_file("event_1")
        loader = DiscretePELoader.read_from_json(
            config(
                default_spin_prior="component",
                alternate_spin_priors={"event_1": None},
            )
        )

        _, (first, second) = loader.load((P.PRIMARY_MASS_SOURCE, P.REDSHIFT))

        np.testing.assert_allclose(first, np.full(4, -np.log(4.0)))
        np.testing.assert_allclose(second, np.zeros(4))


class TestPriorWeights:
    """Each reweighting must reproduce the log-prior it is documented to remove."""

    def _log_prior(self, event_file, config, parameters, **cfg):
        event_file()
        loader = DiscretePELoader.read_from_json(config(**cfg))
        _, (log_prior,) = loader.load(parameters)
        return log_prior

    def test_comoving_distance_prior(self, event_file, config):
        cosmo = default_cosmology()
        z = COLUMNS[P.REDSHIFT]

        log_prior = self._log_prior(
            event_file,
            config,
            (P.PRIMARY_MASS_SOURCE, P.REDSHIFT),
            default_distance_prior="comoving",
        )

        np.testing.assert_allclose(
            log_prior,
            np.asarray(cosmo.logdVcdz(z)) + np.log(4.0 * np.pi) - np.log1p(z),
            rtol=1e-6,
        )

    def test_euclidean_distance_prior(self, event_file, config):
        cosmo = default_cosmology()
        z = COLUMNS[P.REDSHIFT]

        log_prior = self._log_prior(
            event_file,
            config,
            (P.PRIMARY_MASS_SOURCE, P.REDSHIFT),
            default_distance_prior="euclidean",
        )

        expected = 2.0 * np.log(np.asarray(cosmo.z_to_DL(z))) + np.log(
            np.asarray(cosmo.dDLdz(z))
        )
        np.testing.assert_allclose(log_prior, expected, rtol=1e-6)

    def test_a_distance_prior_without_redshift_is_rejected(self, event_file, config):
        event_file()
        loader = DiscretePELoader.read_from_json(
            config(default_distance_prior="comoving")
        )

        with pytest.raises(LoggedValueError, match="requires Redshift"):
            loader.load((P.PRIMARY_MASS_SOURCE,))

    def test_a_mass_prior_without_redshift_is_rejected(self, event_file, config):
        event_file()
        loader = DiscretePELoader.read_from_json(
            config(default_mass_prior="flat-detector-components")
        )

        with pytest.raises(LoggedValueError, match="requires Redshift"):
            loader.load((P.PRIMARY_MASS_SOURCE,))

    def test_flat_detector_components_mass_prior(self, event_file, config):
        """A prior flat in *detector*-frame components picks up ``(1 + z)^2`` on the way
        to source-frame masses.
        """
        z = COLUMNS[P.REDSHIFT]

        log_prior = self._log_prior(
            event_file,
            config,
            (P.PRIMARY_MASS_SOURCE, P.REDSHIFT),
            default_mass_prior="flat-detector-components",
        )

        np.testing.assert_allclose(log_prior, 2.0 * np.log1p(z), rtol=1e-6)

    def test_flat_detector_chirp_mass_ratio_mass_prior(self, event_file, config):
        z = COLUMNS[P.REDSHIFT]
        m1 = COLUMNS[P.PRIMARY_MASS_SOURCE]
        q = COLUMNS[P.SECONDARY_MASS_SOURCE] / m1

        log_prior = self._log_prior(
            event_file,
            config,
            (P.PRIMARY_MASS_SOURCE, P.REDSHIFT),
            default_mass_prior="flat-detector-chirp-mass-ratio",
        )

        expected = -(
            np.log(m1) - np.log1p(z) + np.log(primary_mass_to_chirp_mass_jacobian(q))
        )
        np.testing.assert_allclose(log_prior, expected, rtol=1e-6)

    def test_component_spin_prior_is_a_constant(self, event_file, config):
        log_prior = self._log_prior(
            event_file,
            config,
            (P.PRIMARY_MASS_SOURCE, P.REDSHIFT),
            default_spin_prior="component",
        )

        np.testing.assert_allclose(log_prior, np.full(4, -np.log(4.0)))

    def test_effective_spin_gets_the_isotropic_prior(self, event_file, config):
        chi_eff = COLUMNS[P.EFFECTIVE_SPIN]
        q = COLUMNS[P.SECONDARY_MASS_SOURCE] / COLUMNS[P.PRIMARY_MASS_SOURCE]

        log_prior = self._log_prior(
            event_file,
            config,
            (P.PRIMARY_MASS_SOURCE, P.REDSHIFT, P.EFFECTIVE_SPIN),
        )

        np.testing.assert_allclose(
            log_prior,
            np.log(chi_effective_prior_from_isotropic_spins(chi_eff, q)),
            rtol=1e-6,
        )

    def test_aligned_spin_components_get_the_aligned_prior(self, event_file, config):
        chi_1 = COLUMNS[P.CHI_1]

        log_prior = self._log_prior(
            event_file, config, (P.PRIMARY_MASS_SOURCE, P.REDSHIFT, P.CHI_1)
        )

        np.testing.assert_allclose(
            log_prior, np.log(aligned_spin_prior(chi_1)), rtol=1e-6
        )

    def test_the_weights_of_several_priors_add_up(self, event_file, config):
        """The three reference priors are independent log-additive terms, so enabling
        them together is the sum of enabling each alone.
        """
        parameters = (P.PRIMARY_MASS_SOURCE, P.REDSHIFT)
        pieces = [
            {"default_mass_prior": "flat-detector-components"},
            {"default_spin_prior": "component"},
            {"default_distance_prior": "comoving"},
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            separately = sum(
                self._log_prior(event_file, config, parameters, **piece)
                for piece in pieces
            )
            together = self._log_prior(
                event_file,
                config,
                parameters,
                **{k: v for piece in pieces for k, v in piece.items()},
            )

        np.testing.assert_allclose(together, separately, rtol=1e-6)
