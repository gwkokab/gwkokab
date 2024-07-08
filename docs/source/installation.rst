Installation
============

GWKokab_ is available on PyPI_ and can be easily installed using pip. For optimal setup, it is recommended to install GWKokab_ in a virtual environment. You can install GWKokab_ with the following command:

.. code-block:: bash

    pip install --upgrade gwkokab


To access the latest development version, you can clone the repository or use:

.. code-block:: bash
    
    pip install --upgrade git+https://github.com/gwkokab/gwkokab


If you plan to leverage CUDA_ for enhanced performance, you'll need to install a specific version of JAX_ with this command:

.. code-block:: bash

    pip install --upgrade "jax[cuda12]"


.. _GWKokab: www.github.com/gwkokab/gwkokab
.. _JAX: www.github.com/google/jax
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _PyPI: https://pypi.org/project/gwkokab/