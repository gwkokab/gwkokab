GWKokab is available on PyPI and can be installed using pip.

!!! tip
    It is recommended to install GWKokab in a virtual environment. If you do not have the `venv` module installed, you may install it by doing the following

    === "venv"

        ```bash
        pip install --upgrade venv
        python -m venv gwkenv
        source gwkenv/bin/activate
        ```

    === "conda"

        ```bash
        conda create -n gwkenv python=3.11
        ```

    === "mamba"

        ```bash
        mamba create -n gwkenv python=3.11
        ```

You may then install GWKokab by doing

```bash
pip install --upgrade gwkokab
```

You may install the bleeding edge version by cloning this repo or doing

```bash
pip install --upgrade git+https://github.com/gwkokab/gwkokab
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
