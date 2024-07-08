Numpyro Transformations
=======================

We have used :class:`numpyro.distributions.transforms.Transform` to define transformations between different parameterizations of the same model. This allows us to sample from a distribution with a simple parameterization and then transform the samples to the desired parameterization. The transformations are defined in the following classes.

Component Masses to Chirp Mass and Symmetric Mass Ratio
-------------------------------------------------------
.. autoclass:: gwkokab._src.models.utils.transformations.ComponentMassesToChirpMassAndSymmetricMassRatio
    :no-index:

Component Masses to Chirp Mass and Delta
----------------------------------------
.. autoclass:: gwkokab._src.models.utils.transformations.ComponentMassesToChirpMassAndDelta
    :no-index:

Delta to Symmetric Mass Ratio
-----------------------------
.. autoclass:: gwkokab._src.models.utils.transformations.DeltaToSymmetricMassRatio
    :no-index:

Primary Mass and Mass Ratio to Component Masses
-----------------------------------------------
.. autoclass:: gwkokab._src.models.utils.transformations.PrimaryMassAndMassRatioToComponentMassesTransform
    :no-index: