::: gwkokab.models.utils.jointdistribution
::: gwkokab.models.utils.smoothing

## Constraints

NumPyro uses constraints to ensure the samples provided to `log_prob` are within the valid range. NumPyro has pre-defined constraints as [`numpyro.distributions.constraints`](https://num.pyro.ai/en/stable/distributions.html#constraints) module. GWKokab has implemented a few constraints that are not present in [`NumPyro`](https://num.pyro.ai/en/stable/index.html). These constraints are listed below.

::: gwkokab.models.utils.constraints
    options:
        show_if_no_docstring: true
        filters: ["!__all__", "!_[A-Z]+"]
