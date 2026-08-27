# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Jenks bucketing used to batch ragged per-event posteriors.

``pad_and_stack`` reorders events by size, groups them into buckets, then zero-pads each
bucket to its largest member. Padding is only safe because a companion mask says which
rows are real, so nearly every test here checks the *pair*: the stacked array and its
mask must describe the same events, and no event may be lost or altered by the
reordering.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from gwkokab.analysis.utils.jenks import pad_and_stack
from gwkokab.utils.exceptions import LoggedValueError


def _unpadded_rows(stacked, mask):
    """Recover the original arrays from a bucket, dropping the padded tail of each
    row.
    """
    return [np.asarray(row)[np.asarray(m)] for row, m in zip(stacked, mask)]


def _as_sorted_tuples(arrays):
    return sorted(tuple(np.asarray(a).ravel().tolist()) for a in arrays)


@pytest.fixture
def ragged():
    """Five one-dimensional arrays whose sizes are deliberately not unique."""
    return [np.arange(n, dtype=float) + 100.0 * n for n in (1, 5, 2, 9, 5)]


def test_returns_one_list_per_sequence_plus_the_masks(ragged):
    out = pad_and_stack(ragged, list(ragged), n_buckets=2)

    assert len(out) == 3
    assert all(len(buckets) == 2 for buckets in out)


def test_every_event_survives_the_reordering(ragged):
    """Bucketing is a permutation: the same arrays come back, only regrouped."""
    *stacked, masks = pad_and_stack(ragged, n_buckets=2)

    recovered = [
        row
        for bucket, mask in zip(stacked[0], masks)
        for row in _unpadded_rows(bucket, mask)
    ]

    assert _as_sorted_tuples(recovered) == _as_sorted_tuples(ragged)


def test_padding_is_zero_and_the_mask_marks_it(ragged):
    stacked, masks = pad_and_stack(ragged, n_buckets=1)

    bucket, mask = stacked[0], masks[0]
    assert bucket.shape == (len(ragged), max(a.size for a in ragged))
    assert mask.shape == bucket.shape
    assert jnp.all(jnp.where(mask, 0.0, bucket) == 0.0)
    assert np.asarray(mask).sum() == sum(a.size for a in ragged)


def test_mask_is_a_left_aligned_prefix(ragged):
    """Real samples occupy the head of every row, padding the tail."""
    _, masks = pad_and_stack(ragged, n_buckets=2)

    for mask in masks:
        for row in np.asarray(mask):
            assert not np.any(row[:-1] < row[1:]), "mask is not monotone non-increasing"


def test_sequences_stay_aligned(ragged):
    """Two sequences reordered together must keep event *i* of one beside event *i* of
    the other, which is what makes the shared mask meaningful.
    """
    others = [-a for a in ragged]

    first, second, masks = pad_and_stack(ragged, others, n_buckets=2)

    for bucket_a, bucket_b, mask in zip(first, second, masks):
        rows_a = _unpadded_rows(bucket_a, mask)
        rows_b = _unpadded_rows(bucket_b, mask)
        for row_a, row_b in zip(rows_a, rows_b):
            np.testing.assert_allclose(row_b, -row_a)


def test_buckets_are_ordered_by_size(ragged):
    _, masks = pad_and_stack(ragged, n_buckets=2)

    widths = [mask.shape[1] for mask in masks]
    assert widths == sorted(widths)


def test_identical_sizes_collapse_to_one_bucket():
    """Nothing to gain from splitting when every array is the same length."""
    arrays = [np.ones(4) * i for i in range(6)]

    stacked, masks = pad_and_stack(arrays)

    assert len(stacked) == 1
    assert stacked[0].shape == (6, 4)
    assert jnp.all(masks[0])


def test_more_buckets_than_unique_sizes_is_clamped():
    """Asking for more buckets than there are distinct sizes cannot fail; it warns and
    falls back to one bucket per distinct size.
    """
    arrays = [np.ones(n) for n in (3, 3, 7, 7)]

    stacked, _ = pad_and_stack(arrays, n_buckets=4)

    assert len(stacked) == 2


def test_automatic_bucket_count_beats_a_single_bucket(ragged):
    """With the default threshold the elbow heuristic must split a spread-out set."""
    auto, _ = pad_and_stack(ragged)
    single, _ = pad_and_stack(ragged, n_buckets=1)

    assert len(auto) > 1
    padded_auto = sum(bucket.size for bucket in auto)
    assert padded_auto < single[0].size


def test_a_generous_threshold_gives_a_single_bucket(ragged):
    """A threshold no split can beat degenerates to one bucket."""
    stacked, _ = pad_and_stack(ragged, threshold=100.0)

    assert len(stacked) == 1


def test_multidimensional_arrays_pad_only_the_leading_axis():
    arrays = [np.ones((n, 3, 2)) for n in (2, 4)]

    stacked, masks = pad_and_stack(arrays, n_buckets=1)

    assert stacked[0].shape == (2, 4, 3, 2)
    assert masks[0].shape == (2, 4)


def test_empty_sequences_return_empty_buckets():
    assert pad_and_stack([], []) == ([], [], [])


@pytest.mark.parametrize("threshold", [-0.1, 100.1])
def test_threshold_out_of_range_is_rejected(threshold, ragged):
    with pytest.raises(LoggedValueError, match="Threshold must be between 0 and 100"):
        pad_and_stack(ragged, threshold=threshold)


def test_no_sequences_is_rejected():
    with pytest.raises(LoggedValueError, match="cannot be empty"):
        pad_and_stack()


def test_mismatched_lengths_are_rejected(ragged):
    with pytest.raises(LoggedValueError, match="same length"):
        pad_and_stack(ragged, ragged[:-1])


@pytest.mark.parametrize("n_buckets", [0, -1])
def test_non_positive_bucket_count_is_rejected(n_buckets, ragged):
    """Too *many* buckets is clamped, but fewer than one has no sensible reading."""
    with pytest.raises(LoggedValueError, match="must be between 1 and"):
        pad_and_stack(ragged, n_buckets=n_buckets)


def test_jax_and_numpy_inputs_agree(ragged):
    from_numpy, _ = pad_and_stack(ragged, n_buckets=2)
    from_jax, _ = pad_and_stack([jnp.asarray(a) for a in ragged], n_buckets=2)

    for a, b in zip(from_numpy, from_jax):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))
