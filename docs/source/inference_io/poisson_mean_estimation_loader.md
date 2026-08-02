# `PoissonMeanEstimationLoader`

Every hierarchical likelihood in GWKokab carries the term $-\mu(\Lambda)$, the expected
number of detections under hyperparameters $\Lambda$:

$$
\mu(\Lambda) = T_{\mathrm{obs}} \iint \rho(\theta, z \mid \Lambda)
\frac{\mathrm{d}V_c}{\mathrm{d}z}\frac{1}{1+z} p_{\det}(\theta; z)\,
\mathrm{d}\theta\,\mathrm{d}z
$$

Here $p_{\det}(\theta; z)$ is the probability that the detector network finds a source
with parameters $\theta$ at redshift $z$, and $T_{\mathrm{obs}}$ is the total observing
time. If redshift is not a parameter of the population model, or factors out of it, the
redshift integral can be done once in advance and folded into the **sensitive spacetime
volume** $\mathrm{VT}(\theta)$:

$$
\mu(\Lambda) = T_{\mathrm{obs}} \int \rho(\theta \mid \Lambda) \underbrace{\int
\frac{\mathrm{d}V_c}{\mathrm{d}z}\frac{1}{1+z} p_{\det}(\theta; z)\,\mathrm{d}z
}_{\mathrm{VT}(\theta)}\,\mathrm{d}\theta
$$

`PoissonMeanEstimationLoader` decides how that integral is estimated. Four strategies are
available — two neural surrogates, one injection-based, and an escape hatch — and the JSON
picks one with a **discriminator field**, `estimator_type`. Which of the two forms above
you are working in decides between the first two.

The same file is used by the analyses (`--pmean-cfg`) and by the synthetic-catalogue
generators. Using the *same* file on both sides is not optional: if the selection function
that produced a catalogue differs from the one used to analyse it, the inferred merger
rate is biased.

## Reading one

Unlike the other interfaces in this section, `read_from_json` needs two extra arguments
that cannot come from a file — a JAX PRNG key and the tuple of parameters the population
model works in:

```python
import jax.random as jrd
from gwkokab.analysis.core.inference_io import PoissonMeanEstimationLoader

loader = PoissonMeanEstimationLoader.read_from_json(
    "pmean_cfg.json",
    jrd.key(0),
    ("mass_1_source", "mass_2_source", "redshift", "chi_eff"),
)

log_selection_fn, poisson_mean_estimator, pmean_kwargs = loader.get_estimators()
```

Parsing the JSON is cheap; the expensive work — loading a neural network, reading an
injection set — happens in `get_estimators`, which returns a three-tuple:

`log_selection_fn`
: $\ln \mathrm{VT}(\theta)$ or $\ln p_{\det}(\theta)$ as a callable, or `None` when the
  estimator has no such notion. The injection-based estimator returns `None`.

`poisson_mean_estimator`
: A callable taking the population model plus the keyword arguments below, and returning
  the pair $(\mu, \mathrm{Var}[\mu])$. The variance is what `--variance-cut-threshold`
  tests against.

`pmean_kwargs`
: The keyword arguments the estimator needs, so the caller does not have to know which
  variant it received:

  ```python
  mean, variance = poisson_mean_estimator(model, **pmean_kwargs)
  ```

  For the neural estimators this is `{"T_obs": time_scale}`; for the injection estimator it
  is `{"samples": ..., "log_weights": ..., "T_obs": analysis_time_years}`, with the
  observing time read from the injection file rather than from your configuration.

## Choosing an `estimator_type`

`estimator_type` is **required** and selects which set of remaining keys is valid. Give an
unknown value and Pydantic reports the four it accepts.

| `estimator_type` | Uses | When |
| --- | --- | --- |
| `"neural_vt"` | An MLP surrogate for $\mathrm{VT}(\theta)$ | Redshift is absent from the model or factorises out |
| `"neural_pdet"` | An MLP surrogate for $p_{\det}(\theta; z)$ | Redshift is part of the population model |
| `"injection"` | A search-pipeline injection campaign | Reproducing an LVK-style analysis |
| `"custom"` | Your own Python function | None of the above fits |

`filename` is required by all four.

## `neural_vt` — sensitive spacetime volume

When redshift factors out, $\mathrm{VT}(\theta)$ can be tabulated once and a small
multilayer perceptron trained to reproduce it — a fast, differentiable surrogate that
avoids interpolation. GWKokab then evaluates

$$
\hat{\mu}(\Lambda) = T_{\mathrm{obs}} \sum_{k=1}^{K}\mathcal{R}_{k}
\left\langle Z_k \hat{\mathrm{VT}}(\theta_i) \right\rangle_{\theta_i \sim \rho^*_k(\theta \mid \Lambda)}
$$

where the sum runs over the $K$ components of the population model, $\mathcal{R}_k$ is the
rate of the $k$-th component, $\rho^*_k$ is its normalised density and $Z_k$ whatever
factor was needed to normalise it. The average is taken by drawing `num_samples` points
from each component and pushing them through the network.

```json
{
    "estimator_type": "neural_vt",
    "filename": "../neural_vt_1_200_1000_ecc_matters.hdf5",
    "time_scale": 248.0,
    "num_samples": 2000,
    "batch_size": 1000
}
```

`filename`
: **Required.** An HDF5 file holding a trained MLP, in the format written by
  [`gwkokab.utils.train.train_regressor`](../examples/training_mlp.md). It carries a
  `names` dataset listing the parameters the network expects; those must all be present in
  the `parameters` tuple passed to `read_from_json`, otherwise:

  ```text
  LoggedValueError: Model in ... expects parameters [...], but received [...]. Missing: {...}
  ```

  The order need not match — the estimator permutes columns for you.

`time_scale`
: Default `1.0`. The observing time $T_{\mathrm{obs}}$, in whatever units the network's
  $\mathrm{VT}$ was trained in. This multiplies $\mu$ directly, so it also sets the units
  of the inferred merger rate; getting it wrong rescales the rate posterior and nothing
  else.

`num_samples`
: Default `1000`. Monte Carlo draws per mixture component per likelihood evaluation. Too
  few makes $\mu(\Lambda)$ noisy and the likelihood surface rough, which both samplers
  dislike. The reported variance is the diagnostic.

`batch_size`
: Default `null`, meaning evaluate all draws in one go. Set it to bound peak memory when
  `num_samples` is large; it only affects how
  [`jax.lax.map`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html) chunks the
  work, never the result.

## `neural_pdet` — probability of detection

Identical in shape, but the network models $p_{\det}(\theta; z)$ directly, so redshift
stays in the population model and the draws are over $(\theta, z)$ jointly:

$$
\hat{\mu}(\Lambda) = T_{\mathrm{obs}} \sum_{k=1}^{K}\mathcal{R}_{k}
\left\langle Z_k \hat{p}_{\det}(\theta_i; z_i)
\right\rangle_{\theta_i, z_i \sim \rho^*_k(\theta, z \mid \Lambda)}
$$

The estimator additionally folds in the normalisation of a
[`PowerlawRedshiftModel`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/models/index.html)
component when it finds one, so the redshift distribution does not have to be normalised
by hand.

```json
{
    "estimator_type": "neural_pdet",
    "filename": "../neural_pdet_with_TaylorF2Ecc_uniform_injections.hdf5",
    "time_scale": 1.0,
    "num_samples": 1500,
    "batch_size": 150
}
```

The fields and their defaults are exactly those of `neural_vt`.

## `injection` — sensitivity injections

Given an injection campaign with $N_{\mathrm{inj}}$ injections drawn from a known density
$p_{\mathrm{inj}}$, of which $N_{\det}$ are recovered,

$$
\hat{\mu}(\Lambda) = \frac{T_{\mathrm{obs}}}{N_{\mathrm{inj}}}
\sum_{i=1}^{N_{\det}} \frac{\rho(\theta_i, z_i \mid \Lambda)}{p_{\mathrm{inj}}(\theta_i, z_i)}
$$

```json
{
    "estimator_type": "injection",
    "filename": "../mixture-semi_o1_o2-real_o3_o4a-cartesian_spins.hdf",
    "batch_size": 20000,
    "far_cut": 1.0,
    "snr_cut": 10.0
}
```

`filename`
: **Required.** An injection file in the LVK O3 format, i.e. with either an `injections`
  group or an `events` group. The O1+O2+O3 and O1+O2+O3+O4a mixture releases and the O4a
  sensitivity injections are all recognised; see
  [LIGO-T2100377](https://dcc.ligo.org/LIGO-T2100377),
  [LIGO-T2400110](https://dcc.ligo.org/LIGO-T2400110) and
  [LIGO-T2400073](https://dcc.ligo.org/LIGO-T2400073). Cartesian injection spins are
  converted to GWKokab's spherical convention with the appropriate Jacobian.

`far_cut`
: Default `1.0`. False-alarm-rate threshold in ${\rm yr}^{-1}$; an injection counts as
  found when its FAR is below this. Internally it is applied as an inverse-FAR threshold
  of $1/\texttt{far\_cut}$ years.

`snr_cut`
: Default `10.0`. Network SNR threshold, used for the O1/O2 injections where the search
  pipelines report no FAR.

`batch_size`
: Default `null`. Chunk size for evaluating the population model over the found
  injections. Injection sets are large — tens of thousands of rows is normal — so this is
  the field most likely to be the difference between a run that fits in device memory and
  one that does not.

`time_scale` is **not** accepted here: the analysis time comes from the injection file's
metadata, so $\mu$ is automatically in the file's units (years).

```{admonition} Both cuts must match the catalogue
:class: important

The thresholds define which events "count" as detections. They have to be the same
thresholds that selected the events you are analysing, or the selection function describes
a different experiment from the one that produced your data. How many injections survived
is logged at debug level, and is worth a glance:

`Found 12345 out of 2000000 injections with FAR < 1.0 and SNR > 10.0`
```

## `custom` — your own estimator

When none of the above fits, point at a Python file:

```json
{
    "estimator_type": "custom",
    "python_module_path": "../my_estimator.py",
    "filename": "../my_sensitivity_data.hdf5",
    "kwargs": {
        "batch_size": 1000,
        "time_scale": 248.0,
        "anything_else": "value"
    }
}
```

All four keys are required, `kwargs` included — pass `{}` if your estimator needs nothing
extra. Note that free-floating extra keys are **not** accepted at the top level; anything
your function needs beyond `filename` goes inside `kwargs`, because the top-level model
forbids unknown fields.

The module is loaded by path and must define a function named exactly
`custom_poisson_mean_estimator`, called as
`custom_poisson_mean_estimator(key, parameters, filename, **kwargs)` and returning the
same three-tuple that `get_estimators` hands back:

```python
import jax.numpy as jnp


def custom_poisson_mean_estimator(key, parameters, filename, **kwargs):
    time_scale = kwargs["time_scale"]

    # ... load whatever `filename` holds, build your estimator ...

    def _poisson_mean(model, T_obs):
        mean = ...      # your estimate of mu(Lambda)
        variance = ...  # its Monte Carlo variance, or 0.0
        return mean, variance

    return None, _poisson_mean, {"T_obs": time_scale}
```

The first element is a log-selection function or `None`; the second is the estimator; the
third is the keyword-argument dictionary that the analysis will splat back into it. Since
the estimator is called inside a JIT-compiled likelihood, it must be written in JAX and be
traceable.

A module that loads but has no such function is rejected before sampling starts:

```text
LoggedValueError: The custom module must have a 'custom_poisson_mean_estimator' function.
```

## Common errors

| Message | Cause |
| --- | --- |
| `ValidationError: Input tag 'x' found using 'estimator_type' does not match any of the expected tags` | Unrecognised `estimator_type`. |
| `ValidationError: ... Field required` | A required key is missing — `filename` for all four, `python_module_path` and `kwargs` for `custom`. |
| `ValidationError: ... Extra inputs are not permitted` | A key that belongs to a different `estimator_type`, e.g. `time_scale` on `injection`, or an extra argument that should have gone inside `kwargs`. |
| `LoggedValueError: Model in ... expects parameters [...] Missing: {...}` | The network was trained on parameters the population model does not provide. |
| `LoggedImportError: Could not load spec for module at ...` | `python_module_path` is wrong. |
| `LoggedValueError: The custom module must have a 'custom_poisson_mean_estimator' function.` | The module has no entry point of that name. |

## See also

- [Training a Neural Network to Estimate Sensitivity](../examples/training_mlp.md) — how to
  produce the file that `neural_vt` and `neural_pdet` read.
- [Detector Sensitivity](../examples/ecc_plus_spin/simulating_a_catalogue.md#detector-sensitivity) —
  a `neural_vt` configuration inside a complete worked example.
