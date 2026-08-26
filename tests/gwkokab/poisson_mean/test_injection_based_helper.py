# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Dict

import h5py
import numpy as np
import pytest

from gwkokab.constants import SECONDS_PER_YEAR
from gwkokab.cosmology import default_cosmology
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean._injection_based_helper import (
    aligned_spin_prior,
    apply_injection_prior,
    get_found_injections,
    load_injection_data,
    primary_mass_to_chirp_mass_jacobian,
)


N = 6
MASS_1 = np.linspace(10.0, 60.0, N)
MASS_2 = np.linspace(5.0, 30.0, N)
REDSHIFT = np.linspace(0.05, 0.9, N)
# in-plane components are non-zero so that |cos_tilt| < 1
SPIN = {
    "spin1x": np.full(N, 0.12),
    "spin1y": np.full(N, 0.16),
    "spin1z": np.full(N, 0.30),
    "spin2x": np.full(N, -0.06),
    "spin2y": np.full(N, 0.08),
    "spin2z": np.full(N, -0.20),
}
A_1 = np.sqrt(SPIN["spin1x"] ** 2 + SPIN["spin1y"] ** 2 + SPIN["spin1z"] ** 2)
A_2 = np.sqrt(SPIN["spin2x"] ** 2 + SPIN["spin2y"] ** 2 + SPIN["spin2z"] ** 2)
COS_TILT_1 = SPIN["spin1z"] / A_1
COS_TILT_2 = SPIN["spin2z"] / A_2


def _write_injections_group(path: str, extra: Dict[str, Any], **attrs: Any) -> str:
    """Write a file in the O3 ``injections``-group layout."""
    with h5py.File(path, "w") as f:
        group = f.create_group("injections")
        group.attrs["total_generated"] = np.int64(attrs.get("total_generated", 100))
        group.attrs["analysis_time_s"] = np.float64(
            attrs.get("analysis_time_years", 1.0) * SECONDS_PER_YEAR
        )
        datasets = {
            "mass1_source": MASS_1,
            "mass2_source": MASS_2,
            "redshift": REDSHIFT,
            **SPIN,
        }
        datasets.update(extra)
        for key, value in datasets.items():
            group.create_dataset(key, data=value)
    return path


def _write_events_dataset(
    path: str, fields: Dict[str, np.ndarray], **attrs: Any
) -> str:
    """Write a file in the ``events`` compound-dataset layout."""
    dtype = np.dtype([(key, value.dtype) for key, value in fields.items()])
    array = np.zeros(N, dtype=dtype)
    for key, value in fields.items():
        array[key] = value
    with h5py.File(path, "w") as f:
        f.create_dataset("events", data=array)
        f.attrs["total_generated"] = np.int64(attrs.get("total_generated", 100))
        for key, value in attrs.get("time_attrs", {}).items():
            f.attrs[key] = np.float64(value)
    return path


# ---------------------------------------------------------------------------
# small analytic helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spin", [-0.9, -0.25, 0.25, 0.9])
def test_aligned_spin_prior(spin):
    np.testing.assert_allclose(aligned_spin_prior(spin), -np.log(abs(spin)) / 2)


def test_aligned_spin_prior_is_even():
    spin = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(aligned_spin_prior(spin), aligned_spin_prior(-spin))


def test_primary_mass_to_chirp_mass_jacobian():
    r"""``m_1 / J(q)`` must reproduce :math:`\mathcal{M} = (m_1 m_2)^{3/5}/(m_1+m_2)^{1/5}`."""
    q = MASS_2 / MASS_1
    chirp_mass = np.power(MASS_1 * MASS_2, 0.6) / np.power(MASS_1 + MASS_2, 0.2)
    np.testing.assert_allclose(
        MASS_1 / primary_mass_to_chirp_mass_jacobian(q), chirp_mass, rtol=1e-12
    )


def test_primary_mass_to_chirp_mass_jacobian_at_equal_masses():
    np.testing.assert_allclose(primary_mass_to_chirp_mass_jacobian(1.0), 2.0**0.2)


# ---------------------------------------------------------------------------
# get_found_injections
# ---------------------------------------------------------------------------


def test_get_found_injections_thresholds_on_ifar():
    ifar = np.asarray([0.1, 1.0, 1.5, 1e4])
    found = get_found_injections({"ifar_gstlal": ifar}, ifar.shape, 1.0, 10.0)
    np.testing.assert_array_equal(found, ifar > 1.0)


def test_get_found_injections_combines_several_pipelines():
    """The mask is the union over every pipeline's inverse false-alarm rate."""
    data = {
        "ifar_gstlal": np.asarray([1e4, 0.1, 0.1]),
        "ifar_pycbc": np.asarray([0.1, 1e4, 0.1]),
    }
    found = get_found_injections(data, (3,), 1.0, 10.0)
    np.testing.assert_array_equal(found, [True, True, False])


def test_get_found_injections_converts_far_to_ifar():
    far = np.asarray([0.1, 2.0, 1e-4])
    data = {"far_gstlal": far}
    found = get_found_injections(data, far.shape, 1.0, 10.0)
    np.testing.assert_array_equal(found, 1.0 / far > 1.0)
    # the derived inverse false alarm rates must not leak back into `data`
    assert list(data) == ["far_gstlal"]


def test_get_found_injections_with_no_ifar_threshold_finds_nothing():
    """``ifar_threshold=None`` is mapped to 1e300, i.e. an unreachable bar."""
    ifar = np.asarray([1e4, 1e10, 1e20])
    found = get_found_injections({"ifar_gstlal": ifar}, ifar.shape, None, 10.0)
    assert not found.any()


@pytest.mark.parametrize(
    "key", ["observed_phase_maximized_snr_net", "observed_snr_net"]
)
def test_get_found_injections_falls_back_to_snr(key):
    snr = np.asarray([5.0, 10.0, 12.0])
    found = get_found_injections({key: snr}, snr.shape, 1.0, 10.0)
    np.testing.assert_array_equal(found, snr > 10.0)


def test_get_found_injections_prefers_phase_maximized_snr():
    data = {
        "observed_phase_maximized_snr_net": np.asarray([12.0, 5.0]),
        "observed_snr_net": np.asarray([5.0, 12.0]),
    }
    found = get_found_injections(data, (2,), 1.0, 10.0)
    np.testing.assert_array_equal(found, [True, False])


def test_get_found_injections_uses_snr_for_o1_o2():
    """O1/O2 injections carry no FAR, so they are recovered through the optimal SNR."""
    data = {
        "ifar_gstlal": np.asarray([0.1, 0.1, 0.1]),
        "name": np.asarray([b"o1", b"o2", b"o3"]),
        "optimal_snr_net": np.asarray([12.0, 5.0, 12.0]),
    }
    found = get_found_injections(data, (3,), 1.0, 10.0)
    np.testing.assert_array_equal(found, [True, False, False])


def test_get_found_injections_uses_semianalytic_snr():
    data = {
        "ifar_gstlal": np.asarray([0.1, 0.1]),
        "semianalytic_observed_phase_maximized_snr_net": np.asarray([12.0, 5.0]),
    }
    found = get_found_injections(data, (2,), 1.0, 10.0)
    np.testing.assert_array_equal(found, [True, False])


def test_get_found_injections_without_usable_keys():
    with pytest.raises(ValueError, match="Cannot find keys to filter"):
        get_found_injections({"mass1_source": MASS_1}, (N,), 1.0, None)


# ---------------------------------------------------------------------------
# load_injection_data
# ---------------------------------------------------------------------------


def test_load_injection_data_injections_layout(tmp_path):
    sampling_pdf = np.linspace(1.0, 2.0, N)
    path = _write_injections_group(
        str(tmp_path / "inj.hdf5"),
        {"sampling_pdf": sampling_pdf, "ifar_gstlal": np.full(N, 1e4)},
        total_generated=1234,
        analysis_time_years=1.5,
    )
    data = load_injection_data(path)

    assert data["total_generated"] == 1234
    np.testing.assert_allclose(data["analysis_time"], 1.5)
    np.testing.assert_allclose(data["mass_1"], MASS_1)
    np.testing.assert_allclose(data["mass_2"], MASS_2)
    np.testing.assert_allclose(data["redshift"], REDSHIFT)
    np.testing.assert_allclose(data["a_1"], A_1)
    np.testing.assert_allclose(data["a_2"], A_2)
    np.testing.assert_allclose(data["cos_tilt_1"], COS_TILT_1)
    np.testing.assert_allclose(data["cos_tilt_2"], COS_TILT_2)
    # spherical spin coordinates pick up a Jacobian of (2 pi a_1 a_2)**2
    np.testing.assert_allclose(
        data["prior"], sampling_pdf * np.square(2 * np.pi * A_1 * A_2)
    )


def test_load_injection_data_applies_the_found_mask(tmp_path):
    ifar = np.asarray([1e4, 0.1, 1e4, 0.1, 1e4, 0.1])
    path = _write_injections_group(
        str(tmp_path / "inj.hdf5"),
        {"sampling_pdf": np.ones(N), "ifar_gstlal": ifar},
    )
    data = load_injection_data(path)

    np.testing.assert_allclose(data["mass_1"], MASS_1[ifar > 1.0])
    # `idx` is deliberately the *unmasked* index range
    np.testing.assert_array_equal(data["idx"], np.arange(N))


def test_load_injection_data_divides_by_weights(tmp_path):
    weights = np.linspace(0.5, 1.5, N)
    sampling_pdf = np.ones(N)
    path = _write_injections_group(
        str(tmp_path / "inj.hdf5"),
        {
            "sampling_pdf": sampling_pdf,
            "ifar_gstlal": np.full(N, 1e4),
            "weights": weights,
        },
    )
    data = load_injection_data(path)
    np.testing.assert_allclose(
        data["prior"], np.square(2 * np.pi * A_1 * A_2) / weights
    )


def test_load_injection_data_raises_when_nothing_is_found(tmp_path):
    path = _write_injections_group(
        str(tmp_path / "inj.hdf5"),
        {"sampling_pdf": np.ones(N), "ifar_gstlal": np.full(N, 0.01)},
    )
    with pytest.raises(ValueError, match="No sensitivity injections pass threshold"):
        load_injection_data(path)


def test_load_injection_data_unknown_layout(tmp_path):
    path = str(tmp_path / "bad.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("something_else", data=np.zeros(3))
    with pytest.raises(KeyError, match="Unable to identify injections"):
        load_injection_data(path)


def _events_fields(**extra: np.ndarray) -> Dict[str, np.ndarray]:
    fields = {
        "mass1_source": MASS_1,
        "mass2_source": MASS_2,
        "z": REDSHIFT,
        **SPIN,
        "ifar_gstlal": np.full(N, 1e4),
    }
    fields.update(extra)
    return fields


def test_load_injection_data_events_layout(tmp_path):
    """The O4 ``events`` layout keeps the redshift under ``z`` instead of
    ``redshift``.
    """
    sampling_pdf = np.linspace(1.0, 2.0, N)
    path = _write_events_dataset(
        str(tmp_path / "events.hdf5"),
        _events_fields(sampling_pdf=sampling_pdf),
        total_generated=777,
        time_attrs={"total_analysis_time": 3.0 * SECONDS_PER_YEAR},
    )
    data = load_injection_data(path)

    assert data["total_generated"] == 777
    np.testing.assert_allclose(data["analysis_time"], 3.0)
    np.testing.assert_allclose(data["redshift"], REDSHIFT)
    np.testing.assert_allclose(
        data["prior"], sampling_pdf * np.square(2 * np.pi * A_1 * A_2)
    )


def test_load_injection_data_combined_lnpdraw(tmp_path):
    """O1+O2+O3+O4a mixture files carry one joint ``lnpdraw_...`` column."""
    key = (
        "lnpdraw_mass1_source_mass2_source_redshift_"
        "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
    )
    lnpdraw = np.linspace(-3.0, -1.0, N)
    path = _write_events_dataset(
        str(tmp_path / "events.hdf5"),
        _events_fields(**{key: lnpdraw}),
        time_attrs={"analysis_time": SECONDS_PER_YEAR},
    )
    data = load_injection_data(path)
    np.testing.assert_allclose(
        data["prior"], np.exp(lnpdraw) * np.square(2 * np.pi * A_1 * A_2)
    )


def test_load_injection_data_factorised_lnpdraw(tmp_path):
    """O4a sensitivity files factorise the draw density and use polar angles."""
    pieces = {
        "lnpdraw_mass1_source": np.full(N, -1.0),
        "lnpdraw_mass2_source_GIVEN_mass1_source": np.full(N, -2.0),
        "lnpdraw_z": np.full(N, -0.5),
        "lnpdraw_spin1_magnitude": np.full(N, -0.25),
        "lnpdraw_spin2_magnitude": np.full(N, -0.75),
        "lnpdraw_spin1_polar_angle": np.full(N, -0.1),
        "lnpdraw_spin2_polar_angle": np.full(N, -0.2),
    }
    path = _write_events_dataset(
        str(tmp_path / "events.hdf5"),
        _events_fields(**pieces),
        time_attrs={"total_analysis_time_1ifo": SECONDS_PER_YEAR},
    )
    data = load_injection_data(path)

    expected = np.exp(np.sum(list(pieces.values()), axis=0))
    expected /= np.sin(np.arccos(COS_TILT_1))
    expected /= np.sin(np.arccos(COS_TILT_2))
    np.testing.assert_allclose(data["prior"], expected)


def test_load_injection_data_zero_analysis_time_falls_back_to_one_month(tmp_path):
    path = _write_events_dataset(
        str(tmp_path / "events.hdf5"),
        _events_fields(sampling_pdf=np.ones(N)),
        time_attrs={"total_analysis_time": 0.0},
    )
    data = load_injection_data(path)
    np.testing.assert_allclose(data["analysis_time"], 1.0 / 12.0)


def test_load_injection_data_without_analysis_time(tmp_path):
    path = _write_events_dataset(
        str(tmp_path / "events.hdf5"),
        _events_fields(sampling_pdf=np.ones(N)),
        time_attrs={},
    )
    with pytest.raises(AttributeError, match="does not provide analysis time"):
        load_injection_data(path)


def test_load_injection_data_far_only_injections_layout(tmp_path):
    """A read-only file carrying only ``far_*`` columns must still be loadable."""
    far = np.asarray([1e-4, 1e-4, 10.0, 1e-4, 10.0, 1e-4])
    path = _write_injections_group(
        str(tmp_path / "inj.hdf5"),
        {"sampling_pdf": np.ones(N), "far_gstlal": far},
    )
    data = load_injection_data(path)
    np.testing.assert_allclose(data["mass_1"], MASS_1[1.0 / far > 1.0])


def test_load_injection_data_without_in_plane_spins(tmp_path):
    """Missing in-plane spin columns default to zero, so ``a_i`` reduces to
    ``|spin_iz|`` -- including when part of the injections are cut.
    """
    ifar = np.asarray([1e4, 0.1, 1e4, 0.1, 1e4, 0.1])
    fields = {
        "mass1_source": MASS_1,
        "mass2_source": MASS_2,
        "redshift": REDSHIFT,
        "spin1z": SPIN["spin1z"],
        "spin2z": SPIN["spin2z"],
        "sampling_pdf": np.ones(N),
        "ifar_gstlal": ifar,
    }
    path = str(tmp_path / "inj.hdf5")
    with h5py.File(path, "w") as f:
        group = f.create_group("injections")
        group.attrs["total_generated"] = np.int64(100)
        group.attrs["analysis_time_s"] = np.float64(SECONDS_PER_YEAR)
        for key, value in fields.items():
            group.create_dataset(key, data=value)

    data = load_injection_data(path)
    found = ifar > 1.0
    np.testing.assert_allclose(data["a_1"], np.abs(SPIN["spin1z"][found]))
    np.testing.assert_allclose(data["a_2"], np.abs(SPIN["spin2z"][found]))
    np.testing.assert_allclose(data["cos_tilt_1"], np.ones(int(found.sum())))
    np.testing.assert_allclose(data["cos_tilt_2"], -np.ones(int(found.sum())))


# ---------------------------------------------------------------------------
# apply_injection_prior
# ---------------------------------------------------------------------------


def _base_data() -> Dict[str, np.ndarray]:
    return {
        P.PRIMARY_MASS_SOURCE.value: MASS_1.copy(),
        P.SECONDARY_MASS_SOURCE.value: MASS_2.copy(),
        P.REDSHIFT.value: REDSHIFT.copy(),
        P.PRIMARY_SPIN_MAGNITUDE.value: A_1.copy(),
        P.SECONDARY_SPIN_MAGNITUDE.value: A_2.copy(),
        P.COS_TILT_1.value: COS_TILT_1.copy(),
        P.COS_TILT_2.value: COS_TILT_2.copy(),
        "prior": np.ones(N),
    }


def test_apply_injection_prior_is_a_no_op_for_source_frame_parameters():
    data = apply_injection_prior(
        _base_data(),
        [P.PRIMARY_MASS_SOURCE.value, P.SECONDARY_MASS_SOURCE.value, P.REDSHIFT.value],
    )
    np.testing.assert_allclose(data["prior"], np.ones(N))


def test_apply_injection_prior_mass_ratio():
    data = apply_injection_prior(_base_data(), [P.MASS_RATIO.value])
    np.testing.assert_allclose(data[P.MASS_RATIO.value], MASS_2 / MASS_1)
    np.testing.assert_allclose(data["prior"], MASS_1)


def test_apply_injection_prior_chirp_mass():
    data = apply_injection_prior(_base_data(), [P.MASS_RATIO.value, P.CHIRP_MASS.value])
    chirp_mass = np.power(MASS_1 * MASS_2, 0.6) / np.power(MASS_1 + MASS_2, 0.2)
    jacobian = primary_mass_to_chirp_mass_jacobian(MASS_2 / MASS_1)
    np.testing.assert_allclose(data[P.CHIRP_MASS.value], chirp_mass, rtol=1e-12)
    np.testing.assert_allclose(data["prior"], MASS_1 * jacobian)


@pytest.mark.parametrize(
    "parameter, magnitude, cos_tilt",
    [
        (P.CHI_1.value, A_1, COS_TILT_1),
        (P.CHI_2.value, A_2, COS_TILT_2),
    ],
)
def test_apply_injection_prior_aligned_spin(parameter, magnitude, cos_tilt):
    data = apply_injection_prior(_base_data(), [parameter])
    aligned = magnitude * cos_tilt
    np.testing.assert_allclose(data[parameter], aligned)
    np.testing.assert_allclose(data["prior"], -np.log(np.abs(aligned)))


def test_apply_injection_prior_detector_frame_masses():
    data = apply_injection_prior(
        _base_data(),
        [
            P.MASS_RATIO.value,
            P.PRIMARY_MASS_DETECTED.value,
            P.SECONDARY_MASS_DETECTED.value,
        ],
    )
    mass_ratio = MASS_2 / MASS_1
    mass_1_detected = MASS_1 * (1.0 + REDSHIFT)
    np.testing.assert_allclose(data[P.PRIMARY_MASS_DETECTED.value], mass_1_detected)
    np.testing.assert_allclose(
        data[P.SECONDARY_MASS_DETECTED.value], mass_1_detected * mass_ratio
    )
    np.testing.assert_allclose(
        data["prior"], MASS_1 / (1.0 + REDSHIFT) / mass_1_detected
    )


def test_apply_injection_prior_detector_frame_chirp_mass():
    data = apply_injection_prior(
        _base_data(),
        [
            P.MASS_RATIO.value,
            P.PRIMARY_MASS_DETECTED.value,
            P.CHIRP_MASS_DETECTOR.value,
        ],
    )
    jacobian = primary_mass_to_chirp_mass_jacobian(MASS_2 / MASS_1)
    np.testing.assert_allclose(
        data[P.CHIRP_MASS_DETECTOR.value], MASS_1 * (1.0 + REDSHIFT) / jacobian
    )
    np.testing.assert_allclose(data["prior"], MASS_1 / (1.0 + REDSHIFT) * jacobian)


def test_apply_injection_prior_detector_frame_chirp_mass_without_detector_masses():
    """Without ``mass_1`` the detector-frame chirp mass is derived from the source
    frame.
    """
    data = apply_injection_prior(
        _base_data(), [P.MASS_RATIO.value, P.CHIRP_MASS_DETECTOR.value]
    )
    jacobian = primary_mass_to_chirp_mass_jacobian(MASS_2 / MASS_1)
    np.testing.assert_allclose(
        data[P.CHIRP_MASS_DETECTOR.value], MASS_1 * (1.0 + REDSHIFT) / jacobian
    )
    np.testing.assert_allclose(data["prior"], MASS_1 * jacobian / (1.0 + REDSHIFT))


def test_apply_injection_prior_luminosity_distance():
    cosmology = default_cosmology()
    data = apply_injection_prior(_base_data(), [P.LUMINOSITY_DISTANCE.value])
    np.testing.assert_allclose(
        data[P.LUMINOSITY_DISTANCE.value], cosmology.z_to_DL(REDSHIFT), rtol=1e-6
    )
    np.testing.assert_allclose(
        data["prior"], 1.0 / cosmology.dDLdz(REDSHIFT), rtol=1e-6
    )


def test_apply_injection_prior_effective_spin():
    with np.errstate(all="ignore"):
        data = apply_injection_prior(_base_data(), [P.EFFECTIVE_SPIN.value])

    chi_eff = (MASS_1 * A_1 * COS_TILT_1 + MASS_2 * A_2 * COS_TILT_2) / (
        MASS_1 + MASS_2
    )
    np.testing.assert_allclose(data[P.EFFECTIVE_SPIN.value], chi_eff, rtol=1e-6)
    # chi_eff branch also materialises the mass ratio it needs
    np.testing.assert_allclose(data[P.MASS_RATIO.value], MASS_2 / MASS_1)
    # p_chi_iso / (1/2)**2 / 1**2, so the reweighting is strictly positive
    assert np.all(data["prior"] > 0.0)


def test_apply_injection_prior_effective_and_precessing_spin():
    with np.errstate(all="ignore"):
        data = apply_injection_prior(
            _base_data(), [P.EFFECTIVE_SPIN.value, P.PRECESSING_SPIN.value]
        )

    mass_ratio = MASS_2 / MASS_1
    chi_p = np.maximum(
        A_1 * np.sqrt(1.0 - COS_TILT_1**2),
        (3.0 + 4.0 * mass_ratio)
        / (4.0 + 3.0 * mass_ratio)
        * mass_ratio
        * A_2
        * np.sqrt(1.0 - COS_TILT_2**2),
    )
    np.testing.assert_allclose(data[P.PRECESSING_SPIN.value], chi_p, rtol=1e-6)
    assert np.all(np.isfinite(data["prior"]))
    assert np.all(data["prior"] > 0.0)
