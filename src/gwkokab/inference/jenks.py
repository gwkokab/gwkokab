# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Optional, TypeVar

import jenkspy
import numpy as np
from jaxtyping import Array
from loguru import logger

from ..utils.tools import error_if


__all__ = ["partition_data", "get_subsets"]

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
            sum([len(subset) * max(subset) for subset in subsets])
            - sum([sum(subset) for subset in subsets])
        )
        / sum([len(subset) * max(subset) for subset in subsets])
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
    breaks = jenkspy.jenks_breaks(data, n_classes=n_buckets)
    n_unique_breaks = len(set(breaks))
    if n_unique_breaks != n_buckets + 1:
        msg = f"{n_buckets} buckets requested, but less than {n_buckets + 1} unique breaks found. "
        ("Consider halving the number of buckets or using a different method.",)
        if verbose:
            logger.error(msg)
        raise ValueError(msg)
    indexes = [_right_most_index(data, break_value) for break_value in breaks]
    return indexes


def get_subsets(data: Sequence[T], indexes: Sequence[int]) -> Sequence[Sequence[T]]:
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
    return [data[indexes[i] : indexes[i + 1]] for i in range(len(indexes) - 1)] + [
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
        If True, logs the total loss and losses for each bucketby,  default True

    Returns
    -------
    tuple[Sequence[int], float]
        A tuple containing:
        - A sequence of indices representing the rightmost index of each bucket's break
          value in the sorted data.
        - The total loss percentage for the partitioned data.
    """
    indexes = _jenks_natural_breaks(data, n_buckets)
    subsets = get_subsets(data, indexes)
    total_loss = _total_loss(subsets)
    if verbose:
        logger.info("Total loss: {:.4f}%", _total_loss(subsets))

        losses = [_loss(subset) for subset in subsets]
        logger.info(
            "Losses for each bucket: {}", ", ".join(f"{loss:.4f}%" for loss in losses)
        )

    return indexes, total_loss


def partition_data(
    data: Sequence[int], n_buckets: Optional[int] = None, threshold: float = 3.0
) -> Sequence[int]:
    if n_buckets is not None:
        indexes, total_loss = _partition_data_for_bucket(data, n_buckets)
        return indexes

    total_losses = []
    list_of_indexes = []
    n_buckets_count = 1
    while True:
        try:
            indexes, total_loss = _partition_data_for_bucket(
                data, n_buckets_count, verbose=False
            )
        except ValueError:
            break
        total_losses.append(total_loss)
        list_of_indexes.append(indexes)
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

    return list_of_indexes[n_buckets]
