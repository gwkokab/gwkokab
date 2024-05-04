Installing GWKokab
==================

GWKokab is available on PyPI, and can be installed using pip. It is recommended to install GWKokab in a virtual environment. If you do not have the `venv` module installed, you may install it by doing the following

.. code-block:: bash

    pip install --upgrade venv
    python -m venv jvenv
    source jvenv/bin/activate


You may then install GWKokab by doing

.. code-block:: bash

    pip install --upgrade gwkokab


You may install the bleeding edge version by cloning this repo, or doing

.. code-block:: bash
    
    pip install --upgrade git+https://github.com/gwkokab/gwkokab


If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

.. code-block:: bash

    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
