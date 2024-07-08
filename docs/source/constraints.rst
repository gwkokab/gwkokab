Constraints
===========
Numpyro provides a set of constraints (:class:`numpyro.distributions.constraints.Constraint`) that can be used to restrict the support of a distribution. We have defined couple of constraints which are not available in Numpyro.

.. attention::
    Uptill the release of :code:`gwkokab-0.0.1`, :code:`less_than_equal_to` and :code:`greater_than_equal_to` were not available in Numpyro. They have been added in the nightly version in the `PR #1822 <https://github.com/pyro-ppl/numpyro/pull/1822>`_ and `PR #1793 <https://github.com/pyro-ppl/numpyro/pull/1793>`_.

.. autodata:: gwkokab._src.models.utils.constraints.decreasing_vector
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.greater_than_equal_to
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.increasing_vector
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.less_than_equal_to
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.mass_ratio_mass_sandwich
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.mass_sandwich
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.positive_decreasing_vector
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.positive_increasing_vector
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.strictly_decreasing_vector
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.strictly_increasing_vector
    :no-value:
.. autodata:: gwkokab._src.models.utils.constraints.unique_intervals
    :no-value:
