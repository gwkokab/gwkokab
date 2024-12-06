``gwkokab.models``
==================

.. automodule:: gwkokab.models

Models
------

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    FlexibleMixtureModel
    GaussianSpinModel
    IndependentSpinOrientationGaussianIsotropic
    MassGapModel
    NDistribution
    NPowerlawMGaussian
    NSmoothedPowerlawMSmoothedGaussian
    PowerlawPrimaryMassRatio
    SmoothedGaussianPrimaryMassRatio
    SmoothedPowerlawPrimaryMassRatio
    Wysocki2019MassModel

Redshift
^^^^^^^^

.. automodule:: gwkokab.models.redshift

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    PowerlawRedshift

Utilities
---------

.. automodule:: gwkokab.models.utils

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    doubly_truncated_power_law_cdf
    doubly_truncated_power_law_icdf
    doubly_truncated_power_law_log_prob
    JointDistribution
    ScaledMixture

Constraints
-----------

.. automodule:: gwkokab.models.constraints

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    all_constraint
    any_constraint
    decreasing_vector
    increasing_vector
    mass_ratio_mass_sandwich
    mass_sandwich
    positive_decreasing_vector
    positive_increasing_vector
    strictly_decreasing_vector
    strictly_increasing_vector

Transformations
---------------

We have used :class:`numpyro.distributions.transforms.Transform` to define
transformations between different parameterizations of the same model. This allows us to
sample from a distribution with a simple parameterization and then transform the samples
to the desired parameterization. The transformations are defined in the following
classes.

.. automodule:: gwkokab.models.transformations

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    ComponentMassesAndRedshiftToDetectedMassAndRedshift
    ComponentMassesToTotalMassAndMassRatio
    ComponentMassesToChirpMassAndDelta
    ComponentMassesToChirpMassAndSymmetricMassRatio
    ComponentMassesToMassRatioAndSecondaryMass
    ComponentMassesToPrimaryMassAndMassRatio
    DeltaToSymmetricMassRatio
    PrimaryMassAndMassRatioToComponentMassesTransform
    SourceMassAndRedshiftToDetectedMassAndRedshift
