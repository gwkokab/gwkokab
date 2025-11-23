# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""Bucketing and padding utilities using Jenks' Natural Breaks algorithm.

This module provides functionality to partition sequences of arrays into buckets based
on their sizes using Jenks' Natural Breaks algorithm, then pad and stack these arrays
within each bucket for uniform shape. This is useful for managing memory usage when
processing large datasets.
"""

from collections.abc import Sequence
from typing import Optional, Tuple, TypeVar, Union

import jenkspy
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array
from loguru import logger

from gwkokab.utils.tools import error_if


__all__ = ["pad_and_stack"]

T = TypeVar("T", bound=Array | np.ndarray | int)


def _loss(bucket: Sequence[int]) -> float:
    r"""Calculate the loss for a given bucket of data.

    .. math::
        \text{loss} = 1 - \frac{\text{sum}(\mathrm{bucket})}{\text{count}(\mathrm{bucket}) \cdot \max(\mathrm{bucket})}

    This calculates the relative unused space in the bucket.

    Parameters
    ----------
    bucket : Sequence[int]
        A sequence of integers representing a bucket of data (typically array sizes).

    Returns
    -------
    float
        The loss percentage (multiplied by 100) for the bucket, calculated as described above.
    """
    if not bucket:
        return 0.0

    max_size = max(bucket)

    if max_size == 0:
        return 0.0

    n = len(bucket)
    total_size = sum(bucket)

    return (1.0 - total_size / (n * max_size)) * 100.0


def _total_loss(subsets: Sequence[Sequence[int]]) -> float:
    """Calculate the total loss for a set of data subsets (buckets).

    Parameters
    ----------
    subsets : Sequence[Sequence[int]]
        A sequence of buckets, where each bucket is a sequence of sizes.

    Returns
    -------
    float
        The total loss percentage.
    """
    total_space = sum(len(subset) * max(subset) for subset in subsets if subset)
    total_used = sum(sum(subset) for subset in subsets if subset)

    # Bug fix: Avoid division by zero if total_space is 0 (i.e., all arrays had size 0).
    if total_space == 0:
        return 0.0

    # The original calculation was: (total_space - total_used) / total_space * 100.0
    return (1.0 - total_used / total_space) * 100.0


def _right_most_index(data: Sequence[int], value: int) -> int:
    """Using Binary Search to find the rightmost index of a value in a sorted list.

    This function has been simplified. The `data` should be sorted before calling.

    Parameters
    ----------
    data : Sequence[int]
        Sorted list of integers.
    value : int
        The value to find the rightmost index of.

    Returns
    -------
    int
        The index of the last occurrence of the value.
    """
    idx = np.searchsorted(data, value, side="right")

    error_if(
        idx == 0 or data[idx - 1] != value,  # type: ignore
        msg=f"Value {value} not found in the list.",
    )

    # The index of the last occurrence is idx - 1.
    return int(idx - 1)


def _jenks_natural_breaks(
    data: Sequence[int], n_buckets: int, verbose: bool = True
) -> Sequence[int]:
    """Calculate Jenks' Natural Breaks for a given data set and number of buckets.

    Parameters
    ----------
    data : Sequence[int]
        A sequence of integers representing the data to be bucketed. Must be sorted in
        ascending order.
    n_buckets : int
        The number of buckets to create from the data.
    verbose : bool, optional
        If True, logs the total loss and losses for each bucket. Default is True.

    Returns
    -------
    Sequence[int]
        A sequence of indices representing the rightmost index of each bucket's break
        value in the sorted data.
    """
    error_if(
        n_buckets < 1 or n_buckets > len(data),
        msg=f"Number of buckets {n_buckets} must be between 1 and {len(data)}",
    )

    # Optimization: if n_buckets == len(data), each element is its own bucket,
    # but the logic for breaks still holds.
    if n_buckets == 1:
        # Breaks for 1 bucket: min(data) and max(data)
        breaks = [data[0], data[-1]]
    else:
        # data must be sorted for `jenkspy.jenks_breaks`
        breaks = jenkspy.jenks_breaks(data, n_classes=n_buckets)

    rightmost_indices = [
        _right_most_index(data, break_value) for break_value in breaks[1:]
    ]

    end_indices = [idx + 1 for idx in rightmost_indices]

    indexes = [0] + end_indices

    if indexes[-1] != len(data):
        indexes[-1] = len(data)

    return indexes


def _get_subsets(data: Sequence[T], indexes: Sequence[int]) -> Sequence[Sequence[T]]:
    """Get the subset of data based on the provided indexes.

    Parameters
    ----------
    data : Sequence[T]
        A sequence of data from which subsets will be created.
    indexes : Sequence[int]
        A sequence of indices that define the boundaries of the subsets,
        i.e., [start0, end0, end1, ..., endk], where data[endi:end(i+1)] is a subset.

    Returns
    -------
    Sequence[Sequence[T]]
        A sequence of subsets of the data.
    """
    return [data[indexes[i] : indexes[i + 1]] for i in range(len(indexes) - 1)]


def _partition_data_for_bucket(
    data: Sequence[int], n_buckets: int, verbose: bool = True
) -> tuple[Sequence[int], float]:
    """Partition the data into buckets using Jenks' Natural Breaks.

    Parameters
    ----------
    data : Sequence[int]
        A sequence of integers representing the data to be partitioned. **Must be sorted.**
    n_buckets : int
        The number of buckets to create from the data.
    verbose : bool, optional
        If True, logs the total loss and losses for each bucket, by default True

    Returns
    -------
    tuple[Sequence[int], float]
        A tuple containing:
        - A sequence of indices representing the boundaries for slicing the data.
        - The total loss percentage for the partitioned data.
    """
    indexes = _jenks_natural_breaks(data, n_buckets, verbose=verbose)
    subsets = _get_subsets(data, indexes)

    error_if(not subsets, msg="Partitioning resulted in no subsets.")

    total_loss = _total_loss(subsets)
    if verbose:
        logger.info("Total loss of all buckets: {:.4f}%", total_loss)

        losses = [_loss(subset) for subset in subsets]
        logger.info(
            "Losses for each bucket: {}", ", ".join(f"{loss:.4f}%" for loss in losses)
        )

    return indexes, total_loss


def _partition_data(
    data: Sequence[int], n_buckets: Optional[int] = None, threshold: float = 3.0
) -> Sequence[int]:
    """Determine the optimal bucketing indices for the data."""
    if len(data) == 0:
        return [0]

    # Pre-sort the data, as it's required by jenkspy and _partition_data_for_bucket
    # Note: data is already sorted by the caller `pad_and_stack`

    n_unique = len(set(data))

    if n_unique == 1:
        logger.debug(
            "All elements in the data are identical in size. Returning single bucket."
        )
        return [0, len(data)]

    if n_buckets is not None:
        if n_unique < n_buckets:
            logger.warning(
                f"Number of unique sizes ({n_unique}) is less than the requested number of buckets ({n_buckets}). "
                f"Reducing number of buckets to {n_unique}."
            )
            n_buckets = n_unique
        # User specified the number of buckets
        indexes, _ = _partition_data_for_bucket(data, n_buckets, verbose=True)
        return indexes

    # Automatic bucket selection
    total_losses = []
    n_buckets_count = 1

    # Calculate loss for n=1 to n=n_unique
    while n_buckets_count <= n_unique:
        try:
            # Bug fix: use len(data) as the upper limit, not some arbitrary break.
            _, total_loss = _partition_data_for_bucket(
                data, n_buckets_count, verbose=False
            )
            total_losses.append(total_loss)
        except ValueError:
            # jenkspy can fail if n_classes is too high or data is problematic
            break
        n_buckets_count += 1

    # Cannot calculate diff if there's only one loss value (n_buckets=1)
    if len(total_losses) <= 1:
        logger.warning("Only one loss value calculated. Defaulting to 1 bucket.")
        return [0, len(data)]

    np_losses = np.array(total_losses)

    d1 = np.diff(np_losses)
    loss_reduction = -d1

    below_thresh_idx = np.where(loss_reduction < threshold)[0]

    if len(below_thresh_idx) == 0:
        n_buckets = len(total_losses)
        logger.warning(
            f"No suitable number of buckets found with the given threshold ({threshold}%). "
            f"Using max calculated buckets: {n_buckets}."
        )
    else:
        n_buckets = int(below_thresh_idx[0] + 1)

    error_if(
        n_buckets < 1,
        msg="Calculated number of buckets is less than 1. This should not happen.",
    )

    logger.info(
        f"By first derivative of total loss with {threshold}% threshold, "
        f"recommended number of buckets is {n_buckets}."
    )

    indexes, _ = _partition_data_for_bucket(data, n_buckets, verbose=True)

    return indexes


def _pad_and_stack_bucket(bucket: Sequence[Union[Array, np.ndarray]]) -> Array:
    """Pad a bucket of arrays to the size of the largest array in the bucket and stack
    them.

    Parameters
    ----------
    bucket : Sequence[Union[Array, np.ndarray]]
        A sequence of arrays to pad and stack.

    Returns
    -------
    Array
        A stacked array where each array in the bucket is padded to the size of the
        largest array in the bucket. The shape of the returned array is (n_arrays, max_size,
        ...), where n_arrays is the number of arrays in the bucket and max_size is the size
        of the largest array in the bucket.
    """
    if not bucket:
        return jnp.zeros((0, 0))  # Return a 2D array of size 0

    max_size = max(arr.shape[0] for arr in bucket)

    bucket_padded = [
        jnp.pad(
            jnp.asarray(b),
            ((0, max_size - b.shape[0]),) + ((0, 0),) * (b.ndim - 1),
            mode="constant",
            constant_values=0,
        )
        for b in bucket
    ]
    return jnp.stack(bucket_padded, axis=0)


def _bucket_mask(bucket: Sequence[Union[Array, np.ndarray]]) -> Array:
    """Create a mask for a bucket of arrays.

    Parameters
    ----------
    bucket : Sequence[Union[Array, np.ndarray]]
        A sequence of arrays to create a mask for.

    Returns
    -------
    Array
        A boolean array where True indicates a valid element and False indicates a padded
        element. The shape of the returned array is (n_arrays, max_size), where
        n_arrays is the number of arrays in the bucket and max_size is the size of the
        largest array in the bucket.
    """
    if not bucket:
        return jnp.zeros((0, 0), dtype=bool)

    max_size = max(arr.shape[0] for arr in bucket)
    masks = [
        jnp.pad(
            jnp.ones(arr.shape[0], dtype=bool),
            (0, max_size - arr.shape[0]),
            constant_values=False,
        )
        for arr in bucket
    ]
    return jnp.stack(masks, axis=0)


def pad_and_stack(
    *arrays: Sequence[Union[Array, np.ndarray]],
    n_buckets: Optional[int] = None,
    threshold: float = 3.0,
) -> Tuple[Sequence[Array], ...]:
    """Pad and stack multiple arrays into buckets.

    Parameters
    ----------
    *arrays : Sequence[Union[Array, np.ndarray]]
        Variable number of array sequences to be bucketed. All sequences must
        have the same length.
    n_buckets : Optional[int]
        The number of buckets to create from the data. If None, the function will
        partition the data into buckets based on the sizes of the arrays.
    threshold : float
        if :code:`n_buckets` is None, this value is used to determine the maximum size of
        the buckets.

    Returns
    -------
    Tuple[Sequence[Array], ...]
        A sequence of lists of padded arrays, where each list corresponds to an original
        array sequence, and the elements of the inner list are the stacked buckets.
        The last element of the returned tuple is the list of mask arrays (one mask array
        per bucket).
    """
    error_if(
        not (0.0 <= threshold <= 100.0),
        msg="Threshold must be between 0 and 100.",
    )
    error_if(
        not arrays,
        msg="Input array sequences cannot be empty.",
    )
    n_total_elements = len(arrays[0])
    error_if(
        not all(n_total_elements == len(arr) for arr in arrays),
        msg="All arrays must have the same length.",
    )

    if n_total_elements == 0:
        # Return a tuple of empty lists, plus an empty list for masks
        return tuple([] for _ in range(len(arrays) + 1))

    # Get the size of the first dimension of the arrays for bucketing
    # The padding is applied to the first dimension.
    index_and_size = [
        (i, sum(arr[i].size for arr in arrays)) for i in range(n_total_elements)
    ]
    index_and_size = sorted(index_and_size, key=lambda x: x[1])
    sizes = [size for _, size in index_and_size]
    shuffle_indices = [i for i, _ in index_and_size]

    # Bug fix: convert to a list of lists/sequences before slicing in _get_subsets
    shuffled_arrays = [[arrays_j[i] for i in shuffle_indices] for arrays_j in arrays]

    # sizes is already sorted (required by _partition_data)
    subset_indices = _partition_data(
        sizes, n_buckets, threshold
    )  # [0, end1, end2, ..., len(data)]

    # subsets_arrays[i] is a sequence of buckets for the i-th input array sequence
    subsets_arrays = [_get_subsets(sa, subset_indices) for sa in shuffled_arrays]

    # padded_subsets[i] is a sequence of stacked arrays for the i-th input array sequence (one element per bucket)
    padded_subsets = [
        [_pad_and_stack_bucket(s) for s in subset] for subset in subsets_arrays
    ]

    # Calculate masks based on the first sequence's subsets, as all arrays in a bucket have the same dimensions
    masks = [_bucket_mask(s) for s in subsets_arrays[0]]

    # The return format is a tuple of:
    # ( [bucket1_arr1, bucket2_arr1, ...], [bucket1_arr2, bucket2_arr2, ...], ..., [mask1, mask2, ...] )
    return tuple(padded_subsets + [masks])
