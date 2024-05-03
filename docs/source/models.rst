General Remarks
===============
Models in GWKokab are inherited from the `numpyro.distributions.Distribution <https://num.pyro.ai/en/stable/distributions.html>`__ class. This allows for easy sampling and evaluation of the models. The models are also compatible with the `NumPyro <https://num.pyro.ai/en/stable/index.html>`__ library, which is a probabilistic programming library built on top of `JAX <https://jax.readthedocs.io/en/latest/>`__. This means that every model present in `NumPyro <https://num.pyro.ai/en/stable/index.html>`__ which is inherited from `numpyro.distributions.Distribution <https://num.pyro.ai/en/stable/distributions.html>`__ can be used in GWKokab. We have implemented a few models which are not present in `NumPyro <https://num.pyro.ai/en/stable/index.html>`__ and are specific to GWKokab. These models are listed below.

Mass Models
===========

Broken Power Law Mass Model
---------------------------

.. autoclass:: gwkokab.models.brokenpowerlawmassmodel.BrokenPowerLawMassModel
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Multi Peak Mass Model
---------------------------

.. autoclass:: gwkokab.models.multipeakmassmodel.MultiPeakMassModel
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Power Law + Peak Mass Model
---------------------------

.. autoclass:: gwkokab.models.powerlawpeakmassmodel.PowerLawPeakMassModel
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Power Law + Primary Mass Ratio
------------------------------

.. autoclass:: gwkokab.models.powerlawprimarymassratio.PowerLawPrimaryMassRatio
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Wysocki2019MassModel
--------------------

.. autoclass:: gwkokab.models.wysocki2019massmodel.Wysocki2019MassModel
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Spin Models
===========

Effective Spin Gaussian Model
-----------------------------

.. autoclass:: gwkokab.models.gaussianchieff.GaussianChiEff
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Gaussain Spin Model
-------------------

.. autoclass:: gwkokab.models.gaussianspinmodel.GaussianSpinModel
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Independent Spin Orientation Gaussian Isotropic
-----------------------------------------------

.. autoclass:: gwkokab.models.independentspinorientationgaussianisotropic.IndependentSpinOrientationGaussianIsotropic
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Precessing Spin Gaussian Model
------------------------------

.. autoclass:: gwkokab.models.gaussianchip.GaussianChiP
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

.. Eccentricity Models
.. ===================

.. Redshift Models
.. ===============

Miscellaneous Models
====================

Truncated Power Law
-------------------

.. autoclass:: gwkokab.models.truncpowerlaw.TruncatedPowerLaw
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical


Utils
=====

Constraints
-----------

NumPyro uses constraints to ensure the samples provided to :func:`log_prob` are with in the valid range. NumPyro has pre-defined constraints as `numpyro.distributions.constraints <https://num.pyro.ai/en/stable/distributions.html#constraints>`__ module. GWKokab has implemented a few constraints which are not present in `NumPyro <https://num.pyro.ai/en/stable/index.html>`__. These constraints are listed below.

.. autodata:: gwkokab.models.utils.constraints.greater_than_equal_to
.. autodata:: gwkokab.models.utils.constraints.less_than_equal_to
.. autodata:: gwkokab.models.utils.constraints.mass_sandwich
.. autodata:: gwkokab.models.utils.constraints.mass_ratio_mass_sandwich


Joint Distribution
------------------
.. autoclass:: gwkokab.models.utils.jointdistribution.JointDistribution
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: alphabetical

Smoothing Kernel
----------------

.. autofunction:: gwkokab.models.utils.smoothing.smoothing_kernel