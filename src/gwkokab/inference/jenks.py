# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence

import jenkspy

from ..utils.tools import error_if


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


def jenks_natural_breaks(data: Sequence[int], n_buckets: int) -> Sequence[int]:
    """Calculate Jenks' Natural Breaks for a given data set and number of buckets.

    Parameters
    ----------
    data : Sequence[int]
        A sequence of integers representing the data to be bucketed.
    n_buckets : int
        The number of buckets to create from the data.

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
    sorted_data = sorted(data)
    breaks = jenkspy.jenks_breaks(sorted_data, n_classes=n_buckets)
    n_unique_breaks = len(set(breaks))
    error_if(
        n_unique_breaks != n_buckets + 1,
        msg=f"{n_buckets} buckets requested, but less than {n_buckets + 1} unique breaks found. "
        "Consider halving the number of buckets or using a different method.",
    )
    indexes = [_right_most_index(sorted_data, break_value) for break_value in breaks]
    return indexes
