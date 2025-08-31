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
from typing import Optional, TypeVar, Union

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
        \text{loss} = \frac{n \cdot \max(\mathrm{bucket}) - \text{sum}(\mathrm{bucket})}{n \cdot \max(\mathrm{bucket})} \times 100

    Parameters
    ----------
    bucket : Sequence[int]
        A sequence of integers representing a bucket of data.

    Returns
    -------
    float
        The loss percentage for the bucket, calculated as described above.
    """

    total = sum(bucket)
    return (len(bucket) * max(bucket) - total) / (len(bucket) * max(bucket)) * 100.0


def _total_loss(subsets: Sequence[Sequence[int]]) -> float:
    total_loss = (
        (
            sum(len(subset) * max(subset) for subset in subsets)
            - sum(sum(subset) for subset in subsets)
        )
        / sum(len(subset) * max(subset) for subset in subsets)
        * 100.0
    )
    return total_loss


def _right_most_index(data: Sequence[int], value: int) -> int:
    """Using Binary Search to find the rightmost index of a value in a sorted list.

    Parameters
    ----------
    data : Sequence[int]
        Sorted list of integers.
    value : int
        The value to find the rightmost index of.

    Returns
    -------
    int
        The rightmost index of the value in the list, or -1 if not found.

    Raises
    ------
    ValueError
        If the value is not found in the list or if the right index is out of bounds.
    """
    low, high = 0, len(data) - 1
    result = -1
    while low <= high:
        mid = (low + high) >> 1
        if data[mid] == value:
            result = mid
            low = mid + 1
        elif value < data[mid]:
            high = mid - 1
        else:
            low = mid + 1
    error_if(result == -1, msg=f"Value {value} not found in the list.")
    return result


def _jenks_natural_breaks(
    data: Sequence[int], n_buckets: int, verbose: bool = True
) -> Sequence[int]:
    """Calculate Jenks' Natural Breaks for a given data set and number of buckets.

    Parameters
    ----------
    data : Sequence[int]
        A sequence of integers representing the data to be bucketed.
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
    if n_buckets == 1:
        return [0, len(data) - 1]
    breaks = jenkspy.jenks_breaks(data, n_classes=n_buckets)
    n_unique_breaks = len(set(breaks))
    if n_unique_breaks != n_buckets + 1:
        if n_unique_breaks != n_buckets + 1:
            msg = f"{n_buckets} buckets requested, but less than {n_buckets + 1} unique breaks found. "
            msg += "Consider halving the number of buckets or using a different method."
            if verbose:
                logger.error(msg)
            raise ValueError(msg)
    indexes = [_right_most_index(data, break_value) for break_value in breaks]
    return indexes


def _get_subsets(data: Sequence[T], indexes: Sequence[int]) -> Sequence[Sequence[T]]:
    """Get the subset of data based on the provided indexes.

    Parameters
    ----------
    data : Sequence[T]
        A sequence of data from which subsets will be created.
    indexes : Sequence[int]
        A sequence of indices that define the boundaries of the subsets.

    Returns
    -------
    Sequence[Sequence[T]]
        A sequence of subsets of the data, where each subset corresponds to the
        indices defined in `indexes`.
    """
    return [data[indexes[i] : indexes[i + 1]] for i in range(len(indexes) - 2)] + [
        data[indexes[-2] : indexes[-1] + 1]
    ]


def _partition_data_for_bucket(
    data: Sequence[int], n_buckets: int, verbose: bool = True
) -> tuple[Sequence[int], float]:
    """Partition the data into buckets using Jenks' Natural Breaks.

    Parameters
    ----------
    data : Sequence[int]
        A sequence of integers representing the data to be partitioned.
    n_buckets : int
        The number of buckets to create from the data.
    verbose : bool, optional
        If True, logs the total loss and losses for each bucket, by default True

    Returns
    -------
    tuple[Sequence[int], float]
        A tuple containing:
        - A sequence of indices representing the rightmost index of each bucket's break
          value in the sorted data.
        - The total loss percentage for the partitioned data.
    """
    indexes = _jenks_natural_breaks(data, n_buckets, verbose=verbose)
    subsets = _get_subsets(data, indexes)
    total_loss = _total_loss(subsets)
    if verbose:
        logger.info("Total loss of all buckets: {:.4f}%", _total_loss(subsets))

        losses = [_loss(subset) for subset in subsets]
        logger.info(
            "Losses for each bucket: {}", ", ".join(f"{loss:.4f}%" for loss in losses)
        )

    return indexes, total_loss


def _partition_data(
    data: Sequence[int], n_buckets: Optional[int] = None, threshold: float = 3.0
) -> Sequence[int]:
    if len(set(data)) == 1:
        logger.debug(
            "All elements in the data are identical in size. Returning single bucket."
        )
        return [0, len(data) - 1]
    if n_buckets is not None:
        indexes, total_loss = _partition_data_for_bucket(data, n_buckets)
        return indexes

    total_losses = []
    n_buckets_count = 1
    while True:
        try:
            _, total_loss = _partition_data_for_bucket(
                data, n_buckets_count, verbose=False
            )
        except ValueError:
            break
        total_losses.append(total_loss)
        n_buckets_count += 1

    d1 = np.diff(total_losses)
    loss_reduction = -d1

    below_thresh_idx = np.where(loss_reduction < threshold)[0]
    error_if(
        len(below_thresh_idx) == 0,
        msg="No suitable number of buckets found with the given threshold. "
        "Consider increasing the threshold.",
    )

    n_buckets = int(below_thresh_idx[0] + 1)  # +1 because diff shifts index
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
        largest array in the bucket. The shape of the returned array is (n_buckets, max_size,
        ...), where n_buckets is the number of arrays in the bucket and max_size is the size
        of the largest array in the bucket.
    """
    max_size = max(arr.shape[0] for arr in bucket)
    bucket_padded = [
        jnp.pad(b, ((0, max_size - b.shape[0]),) + ((0, 0),) * (b.ndim - 1))
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
        element. The shape of the returned array is (n_buckets, max_size), where
        n_buckets is the number of arrays in the bucket and max_size is the size of the
        largest array in the bucket.
    """
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
    n_buckets: Optional[int],
    threshold: float,
) -> Sequence[Sequence[Array]]:
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
    Sequence[Sequence[Array]]
        A sequence of padded arrays, where each array corresponds to a bucket.
        The last element of the sequence is a mask indicating which elements are valid
        and which are padded.
    """
    error_if(
        not (0.0 <= threshold <= 100.0),
        msg="Threshold must be between 0 and 100.",
    )
    error_if(
        not all(len(arrays[0]) == len(arr) for arr in arrays),
        msg="All arrays must have the same length.",
    )
    index_and_size = [(i, arr.size) for i, arr in enumerate(arrays[0])]
    index_and_size = sorted(index_and_size, key=lambda x: x[1])
    sizes = [size for _, size in index_and_size]
    shuffle_indices = [i for i, _ in index_and_size]
    shuffled_arrays = [[arrays_j[i] for i in shuffle_indices] for arrays_j in arrays]

    subset_indices = _partition_data(sizes, n_buckets, threshold)

    subsets_arrays = [_get_subsets(sa, subset_indices) for sa in shuffled_arrays]

    padded_subsets = [
        [_pad_and_stack_bucket(s) for s in subset] for subset in subsets_arrays
    ]
    masks = [_bucket_mask(s) for s in subsets_arrays[0]]

    return padded_subsets + [masks]
