# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import jax
from jaxtyping import Array


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
