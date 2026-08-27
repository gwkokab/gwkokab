# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformed distribution that keeps the transformed support."""

from numpyro.distributions import TransformedDistribution

from ..constraints import transform_constraint


class ExtendedSupportTransformedDistribution(TransformedDistribution):
    r"""A transformed distribution that keeps its base distribution's support.

    NumPyro reports the support of a transformed distribution as the *codomain* of the
    final transform, which is generally wider than the image of the base distribution's
    support. This subclass instead pushes the base support through the transform chain
    with :func:`~gwkokab.models.constraints.transform_constraint`, so a constrained base
    distribution keeps its constraint after the change of variables.

    This matters for the mass models, where a change of mass coordinates must preserve a
    sandwich constraint such as :math:`m_\min \leq m_2 \leq m_1 \leq m_\max` rather than
    relaxing it to the whole plane.
    """

    @property
    def support(self):
        """The base distribution's support, pushed through the transform chain.

        Returns
        -------
        constraints.Constraint
            The image of ``base_dist.support`` under ``transforms``.
        """
        return transform_constraint(self.base_dist.support, self.transforms)
