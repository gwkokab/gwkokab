Scalar Transformations
======================

Scalar Transformations are versatile functions that accept scalar values (:class:`int`, :class:`float`) or scalar arrays (:class:`numpy.ndarray`, :class:`jax.Array`) to perform various transformations. These functions simplify code by consolidating common operations into four categories:

- Mass Transformations
- Spin Transformations
- Heterogeneous Transformations
- Coordinate Transformations

Mass Transformations
--------------------

Mass Transformations involve functions that deal with various calculations related to mass in physical systems. These functions help in deriving important parameters like reduced mass, symmetric mass ratio, and total mass, among others. They are essential for simplifying complex mass-related computations in multi-body systems.

Chirp Mass
^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.chirp_mass
    :no-index:

Chirp Mass and Delta M to Component Masses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.Mc_delta_to_m1_m2
    :no-index:

Chirp Mass and Symmetric Mass Ratio to Component Masses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.Mc_eta_to_m1_m2
    :no-index:

Component Masses to Chirp Mass and Symmetric Mass Ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_to_Mc_eta
    :no-index:

Delta M
^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.delta_m
    :no-index:

Delta M to Symmetric Mass Ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.delta_m_to_symmetric_mass_ratio
    :no-index:

Mass Ratio
^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.mass_ratio
    :no-index:

Ordering of Component Masses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_ordering
    :no-index:

Primary Mass and Mass Ratio to Secondary Mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_q_to_m2
    :no-index:

Product of Component Masses
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_times_m2
    :no-index:

Secondary Mass and Mass Ratio to Primary Mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m2_q_to_m1
    :no-index:

Symmetric Mass Ratio
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.symmetric_mass_ratio
    :no-index:

Symmetric Mass Ratio to Delta M
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.symmetric_mass_ratio_to_delta_m
    :no-index:

Reduced Mass
^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.reduced_mass
    :no-index:

Total Mass
^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.total_mass
    :no-index:

Total Mass and Mass Ratio to Component Masses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.M_q_to_m1_m2
    :no-index:



Spin Transformations
--------------------

Spin Transformations encompass functions that calculate spin-related properties in physical systems. These functions simplify the process of deriving effective spin, component spins, and other spin-related parameters, making it easier to analyze the rotational dynamics of objects.

Spin Magnitudes to Effective Spin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.chi_costilt_to_chiz
    :no-index:



Heterogeneous Transformations
----------------------------
Heterogeneous Transformations include functions that combine various types of data, such as mass, spin magnitudes, and angles, to derive complex parameters. These transformations are crucial for integrating multiple aspects of physical systems to obtain comprehensive results.

Chirp Mass, Delta M, Effective Spin, and Minus Spin to Aligned Spins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.Mc_delta_chieff_chiminus_to_chi1z_chi2z
    :no-index:

Component Masses and Aligned Spins to Effective Spin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_chi1z_chi2z_to_chieff
    :no-index:

Component Masses and Aligned Spins to Minus Spin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_chi1z_chi2z_to_chiminus
    :no-index:

Component Masses, Effective Spin, and Minus Spin to Aligned Spins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_chieff_chiminus_to_chi1z_chi2z
    :no-index:

Component Masses, Spin Magnitudes, and Tilt Angles to Effective Spin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_chi1_chi2_costilt1_costilt2_to_chieff
    :no-index:

Component Masses, Spin Magnitudes, and Tilt Angles to Minus Spin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus
    :no-index:

Detected Mass and Redshift to Source Mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m_det_z_to_m_source
    :no-index:

Source Mass and Redshift to Detected Mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.m_source_z_to_m_det
    :no-index:





Coordinate Transformations
--------------------------
Coordinate Transformations consist of functions that convert between different coordinate systems, such as Cartesian, polar, and spherical coordinates. These transformations are essential for simplifying spatial computations and visualizations, allowing for easier manipulation and understanding of geometric data.

Cartesian to Polar
^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.cart_to_polar
    :no-index:

Cartesian to Spherical
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.cart_to_spherical
    :no-index:

Polar to Cartesian
^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.polar_to_cart
    :no-index:

Spherical to Cartesian
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gwkokab._src.utils.transformations.spherical_to_cart
    :no-index:
