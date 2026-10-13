# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Transport must keep joint rows and apply the reference prior exactly once."""

import json

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from gwkokab.analysis.core.inference_io import (
    DiscretePELoader,
    EccentricityEvolutionConfig,
)
from gwkokab.analysis.core.utils import to_structured
from gwkokab.parameters import Parameters as P
from gwkokab.utils.eccentricity import evolve_eccentricity
from gwkokab.utils.exceptions import LoggedValueError


CONFIG = dict(
    native_frequency_hz=20.0,
    native_frequency_frame="detector",
    native_prior_max=0.9,
    condition_native_max=0.5,
)
DATASET = "/samples"


@pytest.fixture
def event(tmp_path):
    e = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.7])
    z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    mass = np.arange(6) + 20.0
    path = tmp_path / "event.h5"
    with h5py.File(path, "w") as f:
        f[DATASET] = to_structured(np.column_stack((e, z, mass)), ["ecc", "z", "mass"])
    return path, e, z, mass


def loader(path, **kwargs):
    return DiscretePELoader(
        filenames=(path,),
        default_datasets=(DATASET,),
        parameter_aliases={"eccentricity": "ecc", "redshift": "z", "m1_source": "mass"},
        **kwargs,
    )


def test_loader_conversion_prior_and_subsampling_stay_aligned(event):
    path, e, z, mass = event
    cfg = EccentricityEvolutionConfig(**CONFIG)
    obj = loader(path, default_eccentricity_evolution=cfg, max_samples=3)
    data, priors = obj.load((P.ECCENTRICITY, P.REDSHIFT, "m1_source"), seed=42)
    indices = data[0][:, 2].astype(int) - 20
    assert np.all(e[indices] <= 0.5)
    expected, log_jac = evolve_eccentricity(e[indices], 10 / ((1 + z[indices]) * 20))
    np.testing.assert_allclose(data[0][:, 0], expected)
    np.testing.assert_array_equal(data[0][:, 1], z[indices])
    np.testing.assert_allclose(priors[0], log_jac - np.log(0.5))
    repeated, _ = obj.load((P.ECCENTRICITY, P.REDSHIFT, "m1_source"), seed=42)
    np.testing.assert_array_equal(data[0], repeated[0])
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f[DATASET]["ecc"], e)


def test_disabled_behavior_and_native_cut_are_distinct_from_target_cut(event):
    path, e, z, _ = event
    old_data, old_prior = loader(path).load((P.ECCENTRICITY, P.REDSHIFT))
    np.testing.assert_array_equal(old_data[0][:, 0], e)
    np.testing.assert_array_equal(old_prior[0], 0)
    data, prior = loader(path, default_eccentricity_evolution=CONFIG).load((
        P.ECCENTRICITY,
        P.REDSHIFT,
    ))
    assert len(data[0]) == 5
    assert np.any(data[0][:, 0] > 0.5)
    assert np.all(np.isfinite(prior[0]))
    unconditioned = dict(CONFIG, condition_native_max=None)
    data, prior = loader(path, default_eccentricity_evolution=unconditioned).load((
        P.ECCENTRICITY,
        P.REDSHIFT,
    ))
    expected, jac = evolve_eccentricity(e, 10 / ((1 + z) * 20))
    np.testing.assert_allclose(data[0][:, 0], expected)
    np.testing.assert_allclose(prior[0], jac - np.log(0.9))


def test_existing_distance_prior_adds_to_eccentricity_prior(event):
    path, e, z, _ = event
    cfg = dict(CONFIG, condition_native_max=None)
    obj = loader(
        path, default_eccentricity_evolution=cfg, default_distance_prior="comoving"
    )
    disabled = loader(path, default_distance_prior="comoving")
    _, transformed_prior = obj.load((P.ECCENTRICITY, P.REDSHIFT))
    _, native_prior = disabled.load((P.ECCENTRICITY, P.REDSHIFT))
    _, jac = evolve_eccentricity(e, 10 / ((1 + z) * 20))
    np.testing.assert_allclose(
        transformed_prior[0] - native_prior[0], jac - np.log(0.9)
    )


@pytest.mark.parametrize("parameters", [(P.ECCENTRICITY,), (P.REDSHIFT,)])
def test_joint_model_coordinates_are_required(event, parameters):
    obj = loader(event[0], default_eccentricity_evolution=CONFIG)
    with pytest.raises(LoggedValueError, match="eccentricity and redshift"):
        obj.load(parameters)


def test_per_event_coverage_and_common_target_are_required(event):
    path = event[0]
    other = path.with_name("other.h5")
    with pytest.raises(ValidationError, match="every loaded event"):
        DiscretePELoader(
            filenames=(path, other), alternate_eccentricity_evolution={"event": CONFIG}
        )
    with pytest.raises(ValidationError, match="one source frequency"):
        DiscretePELoader(
            filenames=(path, other),
            default_eccentricity_evolution=CONFIG,
            alternate_eccentricity_evolution={
                "other": dict(CONFIG, reference_frequency_source_hz=8)
            },
        )
    with pytest.raises(ValidationError, match="no input files"):
        DiscretePELoader(
            filenames=(path,), alternate_eccentricity_evolution={"typo": CONFIG}
        )
    different = dict(CONFIG, native_frequency_hz=5, native_prior_max=0.5)
    obj = DiscretePELoader(
        filenames=(path, other),
        default_eccentricity_evolution=CONFIG,
        alternate_eccentricity_evolution={"other": different},
    )
    assert obj.alternate_eccentricity_evolution["other"].native_frequency_hz == 5


@pytest.mark.parametrize(
    "change",
    [
        dict(native_frequency_hz=0),
        dict(native_frequency_hz=np.inf),
        dict(native_frequency_frame="unknown"),
        dict(native_prior="log-uniform"),
        dict(native_prior_min=0.9),
        dict(condition_native_max=0.95),
    ],
)
def test_invalid_conventions_rejected(change):
    with pytest.raises(ValidationError):
        EccentricityEvolutionConfig(**(CONFIG | change))


def test_missing_frequency_is_not_inferred(event):
    cfg = {k: v for k, v in CONFIG.items() if k != "native_frequency_hz"}
    with pytest.raises(ValidationError):
        loader(event[0], default_eccentricity_evolution=cfg)


def test_preconverted_dataset_is_rejected(event):
    path = event[0]
    with h5py.File(path, "a") as f:
        f[DATASET].attrs["reference_frequency_source_hz"] = 10.0
    with pytest.raises(LoggedValueError, match="already converted"):
        loader(path, default_eccentricity_evolution=CONFIG).load((
            P.ECCENTRICITY,
            P.REDSHIFT,
        ))


def test_source_frequency_and_guarded_reapplication(event):
    path = event[0]
    df = DiscretePELoader.load_file(path, DATASET)
    cfg = EccentricityEvolutionConfig(
        **(CONFIG | dict(native_frequency_frame="source", native_frequency_hz=10))
    )
    out = cfg.apply(df, "ecc", "z")
    np.testing.assert_allclose(out["ecc"], df["ecc"][:5], atol=1e-14)
    with pytest.raises(LoggedValueError, match="already converted"):
        cfg.apply(out, "ecc", "z")
    with pytest.raises(LoggedValueError, match="requires column"):
        cfg.apply(df.drop(columns=["z"]), "ecc", "z")
    df.loc[0, "z"] = -1
    with pytest.raises(LoggedValueError, match="redshift"):
        cfg.apply(df, "ecc", "z")


def test_loader_provenance_roundtrip(event, tmp_path):
    obj = loader(event[0], default_eccentricity_evolution=CONFIG)
    path = tmp_path / "inference.h5"
    obj.write_eccentricity_provenance(str(path), seed=37)
    with h5py.File(path) as f:
        dumped = json.loads(f["data_loader_cfg"].attrs["json"])
        assert f["data_loader_cfg"].attrs["data_seed"] == 37
    restored = DiscretePELoader.model_validate(dumped)
    assert restored == obj
    path = tmp_path / "disabled.h5"
    loader(event[0]).write_eccentricity_provenance(str(path), seed=37)
    assert not path.exists()


@pytest.mark.parametrize("method", ["peters", "small-e"])
def test_poisson_likelihood_is_coordinate_invariant(event, method):
    from jax import numpy as jnp
    from numpyro.distributions import constraints, Distribution

    from gwkokab.inference.poissonlikelihood_utils import discrete_poisson_likelihood_fn

    path, _, _, _ = event
    # Restrict to low native e so the diagnostic power law remains bound.
    cfg = CONFIG | dict(method=method, condition_native_max=0.2)
    native_df = DiscretePELoader.load_file(path, DATASET)
    native_df = native_df.loc[(native_df["ecc"] > 0) & (native_df["ecc"] <= 0.2)]
    # A zero native eccentricity is also valid; use strictly positive values here
    # for the analytic e^2 population log-density.
    with h5py.File(path, "a") as f:
        del f[DATASET]
        f[DATASET] = to_structured(native_df.to_numpy(), list(native_df.columns))
    native, native_prior = loader(path).load((P.ECCENTRICITY, P.REDSHIFT))
    native_prior[0] = np.full(len(native[0]), -np.log(0.2))
    evolved, evolved_prior = loader(path, default_eccentricity_evolution=cfg).load((
        P.ECCENTRICITY,
        P.REDSHIFT,
    ))

    class Population(Distribution):
        support = constraints.real_vector

        def __init__(self, transported):
            self.transported = transported
            super().__init__(event_shape=(2,), validate_args=False)

        def log_prob(self, values):
            e, z = np.asarray(values)[..., 0], np.asarray(values)[..., 1]
            log_jac = 0.0
            if self.transported:
                e, log_back_jac = evolve_eccentricity(e, ((1 + z) * 20) / 10, method)
                log_jac = -log_back_jac
            return jnp.asarray(np.log(3) + 2 * np.log(e) - 3 * np.log(0.2) + log_jac)

    def evaluate(data, prior, transported):
        return discrete_poisson_likelihood_fn(
            Population(transported),
            lambda *_args, **_kwargs: (0.0, 0.0),
            (jnp.asarray(data[0][None, ...]),),
            (jnp.asarray(prior[0][None, ...]),),
            (jnp.ones((1, len(data[0])), dtype=bool),),
            {"T_obs": 1.0},
            (jnp.array([len(data[0])]),),
            variance_cut_threshold=None,
        )

    np.testing.assert_allclose(
        evaluate(native, native_prior, False),
        evaluate(evolved, evolved_prior, True),
        rtol=1e-12,
    )


def test_mutating_per_event_conventions_cannot_mix_reference_frequencies(event):
    path = event[0]
    obj = loader(path, default_eccentricity_evolution=CONFIG)
    # A copied model or mutated mapping bypasses Pydantic construction validation.
    obj.filenames = (path, path.with_name("other.h5"))
    obj.alternate_eccentricity_evolution["other"] = EccentricityEvolutionConfig(
        **CONFIG
    ).model_copy(update={"reference_frequency_source_hz": 8.0})
    with pytest.raises(ValidationError, match="one source frequency"):
        obj.load((P.ECCENTRICITY, P.REDSHIFT))


def test_mutated_native_prior_is_revalidated_before_loading(event):
    obj = loader(event[0], default_eccentricity_evolution=CONFIG)
    obj.default_eccentricity_evolution = obj.default_eccentricity_evolution.model_copy(
        update={"native_prior": "log-uniform"}
    )
    with pytest.raises(ValidationError):
        obj.load((P.ECCENTRICITY, P.REDSHIFT))


def test_eccentricity_redshift_alias_collision_is_rejected(event):
    obj = loader(event[0], default_eccentricity_evolution=CONFIG)
    obj.parameter_aliases["redshift"] = "ecc"
    with pytest.raises(LoggedValueError, match="distinct columns"):
        obj.load((P.ECCENTRICITY, P.REDSHIFT))


def test_transformed_support_diagnostics_do_not_claim_population_coverage(
    event, tmp_path
):
    obj = loader(
        event[0], default_eccentricity_evolution=(CONFIG | dict(native_frequency_hz=5))
    )
    obj.load((P.ECCENTRICITY, P.REDSHIFT))
    diagnostic = obj.eccentricity_diagnostics["event"]
    assert diagnostic["reference_prior_max_range"][1] < 0.5
    assert diagnostic["native_rows"] == 6
    assert diagnostic["conditioned_rows"] == 5
    assert diagnostic["redshift_sample_range"] == [0.1, 0.5]
    assert "not a population-coverage certificate" in diagnostic["scope"]
    cfg = obj.default_eccentricity_evolution
    lower, upper = cfg.reference_prior_support(np.array([0.0, 0.5, 5.0]))
    assert np.all(lower == 0)
    assert upper[0] < upper[1] < upper[2]
    assert upper[-1] > diagnostic["reference_prior_max_range"][1]
    path = tmp_path / "output.h5"
    obj.write_eccentricity_provenance(str(path), seed=37)
    with h5py.File(path) as f:
        saved = json.loads(
            f["data_loader_cfg"].attrs["eccentricity_support_diagnostics"]
        )
    assert saved == obj.eccentricity_diagnostics
    copied = obj.eccentricity_diagnostics
    copied["event"]["native_rows"] = -1
    assert obj.eccentricity_diagnostics["event"]["native_rows"] == 6


def test_provenance_uses_loaded_conventions_even_if_loader_is_later_disabled(
    event, tmp_path
):
    obj = loader(event[0], default_eccentricity_evolution=CONFIG)
    obj.load((P.ECCENTRICITY, P.REDSHIFT))
    obj.default_eccentricity_evolution = None
    output = tmp_path / "loaded_conventions.h5"
    obj.write_eccentricity_provenance(str(output), seed=37)
    with h5py.File(output) as f:
        saved = json.loads(f["data_loader_cfg"].attrs["json"])
    assert saved["default_eccentricity_evolution"]["method"] == "peters"
