``gwkokab.models``
==================

.. automodule:: gwkokab.models

Models
------

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    BrokenPowerLawMassModel
    GaussianSpinModel
    IndependentSpinOrientationGaussianIsotropic
    MultiPeakMassModel
    NDistribution
    PowerLawPeakMassModel
    PowerLawPrimaryMassRatio
    TruncatedPowerLaw
    Wysocki2019MassModel

Utilities
---------

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    JointDistribution

Constraints
-----------

.. automodule:: gwkokab.models.constraints

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    decreasing_vector
    greater_than_equal_to
    increasing_vector
    less_than_equal_to
    mass_ratio_mass_sandwich
    mass_sandwich
    positive_decreasing_vector
    positive_increasing_vector
    strictly_decreasing_vector
    strictly_increasing_vector
    unique_intervals

Transformations
---------------

We have used :class:`numpyro.distributions.transforms.Transform` to define transformations between different parameterizations of the same model. This allows us to sample from a distribution with a simple parameterization and then transform the samples to the desired parameterization. The transformations are defined in the following classes.

.. automodule:: gwkokab.models.transformations

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    ComponentMassesToChirpMassAndDelta
    ComponentMassesToChirpMassAndSymmetricMassRatio
    DeltaToSymmetricMassRatio
    PrimaryMassAndMassRatioToComponentMassesTransform
