# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from typing import Callable, Dict, Optional, Tuple, TypeVar

import jax
from jaxtyping import Array
from loguru import logger


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def fetch_first_matching_value(dictionary: Dict[_KT, _VT], *keys: _KT) -> Optional[_VT]:
    """Get the first value in the dictionary that matches one of the keys.

    Parameters
    ----------
    dictionary : Dict[_KT, _VT]
        The dictionary to search.
    keys : _KT
        The keys to search for.

    Returns
    -------
    Optional[_VT]
        The value of the first key that is found in the dictionary, or None if no key is
        found.
    """
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    return None


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


def batched_map(
    f: Callable[[Array], Array], xs: Array, *, batch_size: Optional[int] = None
) -> Array:
    """Same as :func:`jax.lax.map` but without the :func:`jax.vmap`. :code:`f` is
    applied on the leading dimension of the input array.

    Parameters
    ----------
    f : Callable[[Array], Array]
        The function to apply to each element.
    xs : Array
        The input array.
    batch_size : Optional[int], optional
        The batch size for processing, by default None

    Returns
    -------
    Array
        The output array.
    """
    if batch_size is not None:
        scan_xs, remainder_xs = batch_and_remainder(xs, batch_size)
        g = lambda _, x: ((), f(x))
        _, scan_ys = jax.lax.scan(g, (), scan_xs)
        remainder_ys = f(remainder_xs)
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        ys = jax.tree_util.tree_map(
            lambda x, y: jax.lax.concatenate([flatten(x), y], dimension=0),
            scan_ys,
            remainder_ys,
        )
    else:
        g = lambda _, x: ((), f(x))
        _, ys = jax.lax.scan(g, (), xs)
    return ys
