# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from typing import Tuple

import jax
from jaxtyping import Array
from loguru import logger


def error_if(cond: bool, err: type[Exception] = ValueError, msg: str = "") -> None:
    """Raise an error if condition is met.

    Reference: utils of `interpax <https://github.com/f0uriest/interpax>`_.

    Parameters
    ----------
    cond : bool
        The condition to check.
    err : Exception, optional
        The error to raise, by default ValueError
    msg : str, optional
        The message to include with the error, by default ""

    Raises
    ------
    err
        The error raised if the condition is met.
    """
    if cond:
        logger.error(msg)
        raise err(msg)


def warn_if(cond: bool, err: type[Warning] = UserWarning, msg: str = "") -> None:
    """Raise a warning if condition is met.

    Reference: utils of `interpax <https://github.com/f0uriest/interpax>`_.

    Parameters
    ----------
    cond : bool
        The condition to check.
    err : Warning, optional
        The warning to raise, by default UserWarning
    msg : str, optional
        The message to include with the warning, by default ""
    """
    if cond:
        logger.warning(msg)
        warnings.warn(msg, err)


def batch_and_remainder(x: Array, batch_size: int) -> Tuple[Array, Array]:
    """Calculated batch and remainder of an array given a batch size.

    Copied from JAX codebase.

    Parameters
    ----------
    x : Array
        Array of interest
    batch_size : int
        batch size

    Returns
    -------
    Tuple[Array, Array]
        batched array and remainder
    """
    leaves, treedef = jax.tree_util.tree_flatten(x)

    scan_leaves = []
    remainder_leaves = []

    for leaf in leaves:
        num_batches, _ = divmod(leaf.shape[0], batch_size)
        total_batch_elems = num_batches * batch_size
        scan_leaves.append(
            leaf[:total_batch_elems].reshape(num_batches, batch_size, *leaf.shape[1:])
        )
        remainder_leaves.append(leaf[total_batch_elems:])

    scan_tree = treedef.unflatten(scan_leaves)
    remainder_tree = treedef.unflatten(remainder_leaves)
    return scan_tree, remainder_tree
