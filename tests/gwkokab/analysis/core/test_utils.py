# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared plumbing of an analysis run.

Three separable pieces live in this module: the class-level PRNG key every analysis
draws from, the structured-array conversions used to name event columns, and the HDF5
reader/writer pair that the output file and every post-hoc script go through. The HDF5
tests are mostly round-trips, because that pair is the only contract holding the writer
and the readers together.
"""

import json

import h5py
import numpy as np
import pytest
from jax import numpy as jnp, random as jrd

from gwkokab.analysis.core.utils import (
    from_structured,
    IdentitySampleTransformer,
    PRNGKeyMixin,
    read_attrs_from_hdf5,
    read_from_hdf5,
    SampleTransformer,
    to_structured,
    write_to_hdf5,
)
from gwkokab.utils.exceptions import LoggedValueError


@pytest.fixture
def keyed():
    """A throwaway subclass, since ``PRNGKeyMixin`` keeps its key on the *class*."""

    class Keyed(PRNGKeyMixin):
        pass

    return Keyed


class TestPRNGKeyMixin:
    def test_seed_is_recorded(self, keyed):
        keyed.init_rng_seed(37)

        assert keyed().seed == 37

    def test_key_is_split_on_every_access(self, keyed):
        """Two reads must never hand out the same stream."""
        keyed.init_rng_seed(0)
        instance = keyed()

        first, second = instance.rng_key, instance.rng_key

        assert not jnp.array_equal(jrd.uniform(first, (4,)), jrd.uniform(second, (4,)))

    def test_the_key_is_shared_across_instances(self, keyed):
        """The key lives on the class, so two instances advance the same stream."""
        keyed.init_rng_seed(0)
        one, two = keyed(), keyed()

        from_two_instances = [one.rng_key, two.rng_key]

        keyed.init_rng_seed(0)
        instance = keyed()
        from_one_instance = [instance.rng_key, instance.rng_key]

        for a, b in zip(from_two_instances, from_one_instance):
            assert jnp.array_equal(a, b)

    def test_reseeding_reproduces_the_stream(self, keyed):
        keyed.init_rng_seed(11)
        expected = jrd.uniform(keyed().rng_key, (8,))

        keyed.init_rng_seed(11)
        assert jnp.array_equal(jrd.uniform(keyed().rng_key, (8,)), expected)

    def test_different_seeds_give_different_streams(self, keyed):
        keyed.init_rng_seed(1)
        first = jrd.uniform(keyed().rng_key, (8,))

        keyed.init_rng_seed(2)
        assert not jnp.array_equal(jrd.uniform(keyed().rng_key, (8,)), first)

    @pytest.mark.parametrize("seed", [1.0, "3", None])
    def test_non_integer_seed_is_rejected(self, keyed, seed):
        with pytest.raises(LoggedValueError, match="Expected an integer seed"):
            keyed.init_rng_seed(seed)

    def test_negative_seed_is_rejected(self, keyed):
        with pytest.raises(LoggedValueError, match="non-negative"):
            keyed.init_rng_seed(-1)


class TestStructuredArrays:
    def test_round_trip(self):
        data = np.arange(12.0).reshape(4, 3)
        names = ["mass_1_source", "mass_2_source", "redshift"]

        recovered, recovered_names = from_structured(to_structured(data, names))

        np.testing.assert_allclose(recovered, data)
        assert list(recovered_names) == names

    def test_fields_are_named_columns(self):
        data = np.arange(6.0).reshape(3, 2)

        structured = to_structured(data, ["a", "b"])

        np.testing.assert_allclose(structured["a"], data[:, 0])
        np.testing.assert_allclose(structured["b"], data[:, 1])

    def test_fields_are_double_precision(self):
        structured = to_structured(np.zeros((2, 2), dtype=np.float32), ["a", "b"])

        assert all(structured.dtype[name] == np.float64 for name in ("a", "b"))

    def test_single_column(self):
        data = np.linspace(0.0, 1.0, 5).reshape(5, 1)

        recovered, names = from_structured(to_structured(data, ["z"]))

        np.testing.assert_allclose(recovered, data)
        assert list(names) == ["z"]


class TestSampleTransformer:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            SampleTransformer()  # type: ignore[abstract]

    def test_identity_leaves_samples_untouched(self):
        samples = np.arange(12.0).reshape(4, 3)

        transformed = IdentitySampleTransformer().transform(samples)

        np.testing.assert_allclose(transformed, samples)

    def test_identity_jacobian_is_zero_per_sample(self):
        samples = np.arange(12.0).reshape(4, 3)
        transformer = IdentitySampleTransformer()

        log_det = transformer.log_abs_det_jacobian(samples, samples)

        assert log_det.shape == (4,)
        np.testing.assert_allclose(log_det, 0.0)

    def test_default_check_accepts_everything(self):
        samples = np.arange(12.0).reshape(4, 3)

        checked = IdentitySampleTransformer().check(samples, samples)

        assert checked.shape == (4,) and checked.all()


class TestHDF5RoundTrip:
    def test_data_round_trips_through_a_path(self, tmp_path):
        path = str(tmp_path / "out.hdf5")
        data = np.arange(10.0).reshape(5, 2)

        write_to_hdf5(path, "group/samples", data=data)

        np.testing.assert_allclose(read_from_hdf5(path, "group/samples"), data)

    def test_data_round_trips_through_an_open_file(self, tmp_path):
        """Passing a handle must not close it, so a caller can keep writing."""
        path = str(tmp_path / "out.hdf5")
        with h5py.File(path, "a") as f:
            write_to_hdf5(f, "a", data=np.zeros(3))
            write_to_hdf5(f, "b", data=np.ones(3))

        np.testing.assert_allclose(read_from_hdf5(path, "b"), np.ones(3))

    def test_jax_arrays_are_converted(self, tmp_path):
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "samples", data=jnp.arange(6.0))

        np.testing.assert_allclose(read_from_hdf5(path, "samples"), np.arange(6.0))

    def test_writing_twice_replaces_the_dataset(self, tmp_path):
        """Reruns overwrite in place rather than failing on an existing name."""
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "samples", data=np.zeros(4))
        write_to_hdf5(path, "samples", data=np.ones(7))

        np.testing.assert_allclose(read_from_hdf5(path, "samples"), np.ones(7))

    def test_attributes_round_trip(self, tmp_path):
        path = str(tmp_path / "out.hdf5")
        attrs = {
            "n_chains": 4,
            "step_size": 0.1,
            "use_dense_mass": True,
            "sampler_name": "numpyro",
        }

        write_to_hdf5(path, "sampler_cfg", attrs=attrs)

        assert read_attrs_from_hdf5(path, "sampler_cfg") == attrs

    def test_container_attributes_round_trip_as_json(self, tmp_path):
        """``dict``/``list``/``tuple`` values are JSON-encoded on the way in and decoded
        on the way out, which is what lets ``variables_index`` live in an attribute.
        """
        path = str(tmp_path / "out.hdf5")
        attrs = {"variables_index": {"alpha_pl_0": 0, "log_rate_0": 1}, "shape": [2, 3]}

        write_to_hdf5(path, "samples", data=np.zeros((2, 2)), attrs=attrs)
        recovered = read_attrs_from_hdf5(path, "samples")

        assert recovered["variables_index"] == attrs["variables_index"]
        assert recovered["shape"] == [2, 3]

    def test_tuples_come_back_as_lists(self, tmp_path):
        """JSON has no tuple, so the round trip is lossy in exactly this one way."""
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "cfg", attrs={"bounds": (1.0, 2.0)})

        assert read_attrs_from_hdf5(path, "cfg")["bounds"] == [1.0, 2.0]

    def test_none_round_trips(self, tmp_path):
        """An unset config field is stored as an empty dataset, not dropped."""
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "cfg", attrs={"seed": None})

        assert read_attrs_from_hdf5(path, "cfg")["seed"] is None

    def test_attributes_attach_to_an_existing_dataset(self, tmp_path):
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "samples", data=np.zeros(3))
        write_to_hdf5(path, "samples", attrs={"unit": "solar mass"})

        np.testing.assert_allclose(read_from_hdf5(path, "samples"), np.zeros(3))
        assert read_attrs_from_hdf5(path, "samples")["unit"] == "solar mass"

    def test_attributes_alone_create_a_group(self, tmp_path):
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "sampler_cfg/flowMC", attrs={"n_local_steps": 10})

        with h5py.File(path, "r") as f:
            assert isinstance(f["sampler_cfg/flowMC"], h5py.Group)

    def test_a_plain_string_attribute_that_is_not_json_survives(self, tmp_path):
        """Decoding is attempted on every string; a failure must leave it alone."""
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(path, "cfg", attrs={"label": "run-1"})

        assert read_attrs_from_hdf5(path, "cfg")["label"] == "run-1"

    def test_writing_nothing_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Either `data` or `attrs`"):
            write_to_hdf5(str(tmp_path / "out.hdf5"), "samples")

    def test_reading_a_missing_dataset_is_rejected(self, tmp_path):
        path = str(tmp_path / "out.hdf5")
        write_to_hdf5(path, "samples", data=np.zeros(3))

        with pytest.raises(ValueError, match="not found"):
            read_from_hdf5(path, "absent")

    def test_reading_missing_attributes_is_rejected(self, tmp_path):
        path = str(tmp_path / "out.hdf5")
        write_to_hdf5(path, "samples", data=np.zeros(3))

        with pytest.raises(ValueError, match="not found"):
            read_attrs_from_hdf5(path, "absent")

    def test_attributes_read_through_an_open_group(self, tmp_path):
        path = str(tmp_path / "out.hdf5")
        write_to_hdf5(path, "cfg", attrs={"n": 3})

        with h5py.File(path, "r") as f:
            assert read_attrs_from_hdf5(f, "cfg")["n"] == 3

    def test_numpy_scalars_are_narrowed_to_python_types(self, tmp_path):
        """``read_attrs_from_hdf5`` exists so configs come back JSON-serializable."""
        path = str(tmp_path / "out.hdf5")

        write_to_hdf5(
            path,
            "cfg",
            attrs={"n": np.int64(3), "x": np.float32(0.5), "flag": np.bool_(True)},
        )
        recovered = read_attrs_from_hdf5(path, "cfg")

        assert type(recovered["n"]) is int
        assert type(recovered["x"]) is float
        assert type(recovered["flag"]) is bool
        json.dumps(recovered)
