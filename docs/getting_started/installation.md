GWKokab is available on PyPI and can be installed using pip.

!!! tip
    It is recommended to install GWKokab in a virtual environment.

    === "venv"

        ```bash
        pip install --upgrade venv
        python -m venv gwkenv
        source gwkenv/bin/activate
        ```

    === "conda"

        You can download and install Anaconda from [here](https://www.anaconda.com/download/).

        ```bash
        conda create -n gwkenv python=3.11
        conda activate gwkenv
        ```

    === "mamba"

        You can download and install Mamba from [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

        ```bash
        mamba create -n gwkenv python=3.11
        mamaba activate gwkenv
        ```

You may then install GWKokab by doing

=== "User (stable)"

    ```bash
    pip install --upgrade gwkokab
    ```

=== "Developer (bleeding edge)"

    ```bash
    pip install --upgrade git+https://github.com/gwkokab/gwkokab
    ```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
