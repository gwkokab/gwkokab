NumPyro uses constraints to ensure the samples provided to `log_prob` are within the valid range. NumPyro has pre-defined constraints as [`numpyro.distributions.constraints`](https://num.pyro.ai/en/stable/distributions.html#constraints) module. GWKokab has implemented a few constraints that are not present in [`NumPyro`](https://num.pyro.ai/en/stable/index.html). These constraints are listed below.

## API

Following are the attributes to call for the constraints. For further details see [pytorch/issues/50616](https://github.com/pytorch/pytorch/issues/50616)

::: gwkokab.models.utils.constraints
    options:
        show_if_no_docstring: true
        filters: ["!__all__", "!_[A-Z]+"]

## Internal Constraints Objects

These are the internal constraints objects that are used in the constraints.

::: gwkokab.models.utils.constraints
    options:
        show_if_no_docstring: false
        filters: ["!__all__"]

## How to define a new constraint

To define a new constraint, you can use the `numpyro.distributions.constraints.Constraint` class. The following methods are required to be implemented in the new constraint class.

```python
from numpyro.distributions.constraints import Constraint


class _MyConstraint(Constraint):
    def __init__(self, ...):
        ...

    def __call__(self, x):
        ...

    def tree_flatten(self):
        ...
```
