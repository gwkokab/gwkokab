Installation
============

GWKokab_ is available on PyPI_ and can be easily installed using pip. For optimal
setup, it is recommended to install GWKokab_ in a virtual environment. You can
install GWKokab_ with the following command:

.. tab-set::

    .. tab-item:: Stable Release 📦

        .. code-block:: bash
        
            pip install --upgrade gwkokab

    .. tab-item:: Nightly 🍺

        .. code-block:: bash
            
            pip install --upgrade git+https://github.com/gwkokab/gwkokab



Depending on the accelerator you have, you can install the appropriate version of JAX_
with the following command:

.. tab-set::

    .. tab-item:: CPU 🐢

        GWKokab assumes default accelerator is CPU, so you do not need to install JAX_ 
        separately. However, if you want to install JAX_ separately, you can do so with
        the following command:

        .. code-block:: bash
        
            pip install -U jax
    
    .. tab-item:: GPU 🚀

        If you plan to leverage CUDA_ for enhanced performance, you'll need to install
        a specific version of JAX_ with this command:

        .. code-block:: bash
        
            pip install -U "jax[cuda12]"
    
    .. tab-item:: TPU ⚡

        .. code-block:: bash
        
            pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

If you are not a Linux user, you may check the support for you platform in the
`JAX supported platforms
<https://jax.readthedocs.io/en/latest/installation.html#supported-platforms>`_.


.. _GWKokab: https://github.com/gwkokab/gwkokab
.. _JAX: https://github.com/google/jax
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _PyPI: https://pypi.org/project/gwkokab/