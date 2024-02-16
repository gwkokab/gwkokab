Writing the configuration file
==============================
GWKokab uses configuration files as an interface to the user. The configuration file is written in the :code:`ini` format. There are different types of configuration files for the different tasks that GWKokab can perform.

Synthetic Data Generation
-------------------------

The configuration file is divided into the following sections:

General
^^^^^^^
This section contains the general configuration for the synthetic data generation. The following options are available:

- :code:`size` : The size of the synthetic data. This is the number of events in the synthetic data.
- :code:`error_size` : The size of the error in the synthetic data. This is the number of events in the error data.
- :code:`root_container` : The root container for the synthetic data. This is the name of the root container for the synthetic data.
- :code:`event_filename` : The name of the file that contains the synthetic data. It is assumed to be a formatted string where the event number can be inserted, for example :code:`event_{}.dat`.
- :code:`config_filename` : The name of the file that contains the configuration for the models that will be used to generate the synthetic data.
- :code:`num_realizations` : The number of realizations of the synthetic data that will be generated.
- :code:`extra_size` : The size of the extra data. This is to over sample the data to account for the selection effect. Defaults to 1500.
- :code:`extra_error_size` : The size of the error in the extra data. This is to over sample the data to account for the selection effect. Defaults to 1000.

.. warning:: This section is mandatory.

An example of the general section is as follows:

.. code-block:: ini

    [general]
    size = 100
    error_size = 5000
    root_container = syn_data
    event_filename = event_{}.dat
    config_filename = configuration.dat
    num_realizations = 5
    extra_size = 15000
    extra_error_size = 10000


Selection effect
^^^^^^^^^^^^^^^^
Synthetic data is just a sample of the provided distributions, to account it for the real physics VT selection effect is applied. This section contains the configuration for the selection effect. The following options are available:

- :code:`vt_filename` : The name of the file that contains the weights for the selection effect. Supported file formats are :code:`.h5` and :code:`.hdf5`.

An example of the selection effect section is as follows:

.. code-block:: ini

    [selection_effect]
    vt_filename = mass_vt.hdf5

Models
^^^^^^

This section contains the configuration for the models that will be used to generate the synthetic data. It is not a specific section but a class of section that is repeated for each model. The following options are available:

- :code:`model` : The name of the model that will be used to generate the synthetic data.
- :code:`params` : The parameters for the model. This is a dictionary that contains the parameters for the model.
- :code:`config_vars` : Values we want to save in the configuration file. This is a list of tuple of strings. First string in the tuple is the name of the variable as it is :code:`params`, the second string is the name by which we want to save it.
- :code:`col_names` : The names of the columns in the synthetic data. This is a list of strings.
- :code:`error_type` : The type of the error that will be used to generate the synthetic data.
- :code:`error_params` : The parameters for the error. This is a dictionary that contains the parameters for the error.

.. note:: section name of each model should contain the keyword :code:`model`.

Available models can be found in `models <https://gwkokab.readthedocs.io/en/latest/models.html>`__. Some example of the model section is as follows:

.. code-block:: ini

    [beta_model]
    model = Beta
    params = {'concentration1': 1.8, 'concentration0': 0.9}
    config_vars = [('concentration1', 'alpha_1'), ('concentration0', 'beta_1')]
    col_names = ['x']
    error_type = truncated_normal
    error_params = {'scale': 0.5, 'lower': 0.0, 'upper': 1.0, }

    [truncnorm_model]
    model = TruncatedNormal
    params = {'loc': 0.0, 'scale': 1.0, 'low': 0.0, 'high': 1.0, }
    config_vars = [('scale', 'sigma')]
    col_names = ['y']
    error_type = truncated_normal
    error_params = {'scale': 0.1, 'lower': 0.0, 'upper': 1.0,}

Plots
^^^^^
Sometimes people want to see the plots of the synthetic data. This section contains the configuration for the plots that will be generated. The following options are available:

- :code:`injs` : This is a list of string. It contains the list of the names of quantities that will be plotted for the injected data. The names should be the same as in :code:`col_names` in the synthetic data.
- :code:`posts` : This is a list of string. It contains the list of the names of quantities that will be plotted for the posterior samples. The names should be the same as in :code:`col_names` in the synthetic data.

.. note:: We can only plot 2D and 3D scatter plots.

An example of the plots section is as follows:

.. code-block:: ini

    injs = [['x', 'y'], ['x', 'y', 'z']]
    posts = [['x', 'y'], ['u', 'v']]


Example
^^^^^^^
Lets run the process of synthetic data generation for the following configuration file:

.. code-block:: ini

    [general]
    size = 100
    error_size = 5000
    root_container = syn_data
    event_filename = event_{}.dat
    config_filename = configuration.dat
    num_realizations = 5
    extra_size = 15000
    extra_error_size = 10000

    [selection_effect]
    vt_filename = mass_vt.hdf5

    [mass_model]
    model = Wysocki2019MassModel
    params = {'alpha_m': 0.8, 'k': 0, 'mmin': 10.0, 'mmax': 50.0, 'Mmax': 100.0,}
    config_vars = [('alpha_m', 'alpha'), ('mmin', 'mass_min'), ('mmax', 'mass_max')]
    col_names = ['m1_source', 'm2_source']
    error_type = banana

    [spin1_model]
    model = Beta
    params = {'concentration1': 1.8, 'concentration0': 0.9}
    config_vars = [('concentration1', 'alpha_1'), ('concentration0', 'beta_1')]
    col_names = ['a1']
    error_type = truncated_normal
    error_params = {'scale': 0.5, 'lower': 0.0, 'upper': 1.0, }

    [spin2_model]
    model = Beta
    params = {'concentration1': 1.8, 'concentration0': 0.9}
    config_vars = [('concentration1', 'alpha_2'), ('concentration0', 'beta_2')]
    col_names = ['a2']
    error_type = truncated_normal
    error_params = {'scale': 0.5, 'lower': 0.0, 'upper': 1.0, }

    [ecc_model]
    model = TruncatedNormal
    params = {'loc': 0.0, 'scale': 0.05, 'low': 0.0, 'high': 1.0, }
    config_vars = [('scale', 'sigma_ecc')]
    col_names = ['ecc']
    error_type = truncated_normal
    error_params = {'scale': 0.1, 'lower': 0.0, 'upper': 1.0,}

    [plots]
    injs = [['m1_source', 'm2_source'], ['m1_source', 'm2_source', 'ecc']]
    posts = [['m1_source', 'm2_source'], ['a1', 'a2']]



VT Generation
-------------

Inference
---------