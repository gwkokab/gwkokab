# `AnalyticalGWalkPELoader`

`AnalyticalGWalkPELoader` reads events that have been reduced to a **multivariate normal
summary** — a mean vector, a covariance matrix and a bounding box — rather than a cloud of
posterior samples. It is the data side of the
[Analytical GWalk likelihood](../examples/ecc_plus_spin/inference_from_gaussian_summaries.md),
where each event's likelihood is approximated as

$$
\mathcal{L}_n(\theta) \propto \mathcal{N}(\theta \mid \mu_n, \Sigma_n),
\qquad
\theta \in [\ell_n, u_n]
$$

and Monte Carlo draws are taken from that Gaussian rather than from stored samples. The
number of draws per event becomes an analysis knob instead of a property of the data, and
an event costs a $D \times D$ matrix instead of thousands of rows.

Analyses receive it through `--data-loader-cfg`.

## The minimal configuration

Exactly as for the [discrete loader](./discrete_pe_loader.md), one key is required:

```json
{
    "regex": "../data/event_*.hdf5"
}
```

```python
from gwkokab.analysis.core.inference_io import AnalyticalGWalkPELoader

loader = AnalyticalGWalkPELoader.read_from_json("data_loader_cfg.json")
data = loader.load(("mass_1_source", "mass_2_source", "redshift", "chi_eff"))
```

`load` returns a dictionary of five lists, each with one entry per event, in the sorted
file order:

`mean`
: $\mu_n$, flattened, shape `(D,)`.

`cov`
: $\Sigma_n$, shape `(D, D)`.

`lower_bound`, `upper_bound`
: The two columns of the file's `limits` dataset, shape `(D,)` each. Draws falling outside
  this box are rejected and redrawn, so no sample wanders outside the region the original
  posterior actually covered.

`scale`
: A per-coordinate rescaling used to precondition the draw, shape `(D,)`. Defaults to ones.

`load` also accepts a `seed` argument for signature compatibility with the discrete
loader, but the Analytical GWalk loader does no sub-sampling and ignores it.

:::{admonition} `D` comes from the file, not from `parameters`
:class: warning

This is the one behaviour that surprises people. Unlike the discrete loader, this loader
does **not** slice the file down to the parameters you asked for — it returns $\mu_n$ and
$\Sigma_n$ in full, exactly as stored. The `parameters` argument is used only to *check*
that the requested names appear in the file's `coords` attribute, and a mismatch is a
warning, not an error.

The consequence is that the file's `coords` must list exactly the parameters your
analysis uses, **in the same order**. A summary fitted over five coordinates cannot be
used for a four-parameter analysis by simply asking for four; regenerate it with
`synthetic_analytical_gwalk_pe --coords ...` restricted to the coordinates you want.
:::

## The file layout

Each event file holds one HDF5 **group per waveform**, and the group holds the summary:

```text
event_0.hdf5
└── GWKokabSyntheticAnalyticalGWalkPE      # the group named by `default_waveform`
    ├── @coords                       # attribute: coordinate names, length D
    ├── mu                            # (D,)     mean vector
    ├── cov                           # (D, D)   covariance matrix
    ├── limits                        # (D, 2)   [lower, upper] per coordinate
    └── scale                         # (D,)     optional, defaults to ones
```

This is what `synthetic_analytical_gwalk_pe` writes. Groups from other pipelines work equally
well as long as they follow the same layout, which is what `default_waveform` and
`alternate_waveforms` are for.

A missing `scale` dataset is not an error — the loader substitutes ones and warns. A
non-positive entry *is* an error, since `scale` divides:

```text
LoggedValueError: File '../data/event_3.hdf5' contains non-positive scale values, which are invalid.
```

## Fields

### `regex`

**Required.** A glob pattern, with exactly the semantics described for the
[discrete loader](./discrete_pe_loader.md#regex): matches are sorted, that order is the
event order, and matching nothing raises `LoggedFileNotFoundError`. Event names — the keys
of `alternate_waveforms` — are file stems.

### `default_waveform`

Default `"GWKokabSyntheticAnalyticalGWalkPE"`. The name of the HDF5 group holding the summary.
Unlike the discrete loader's `default_datasets` this is a **single string**, not a list of
candidates, and it is a group name rather than a path.

```json
{
    "regex": "../data/*.h5",
    "default_waveform": "IMRPhenomXPHM"
}
```

If the group is missing, the loader lists what the file does contain:

```text
LoggedKeyError: Waveform 'IMRPhenomXPHM' not found in file '../data/GW150914.h5'.
Available waveforms: SEOBNRv4PHM, NRSur7dq4
```

### `alternate_waveforms`

Default `{}`. Per-event override mapping event name to a group name, for a catalogue where
one or two events are best described by a different waveform:

```json
{
    "regex": "../data/*.h5",
    "default_waveform": "IMRPhenomXPHM",
    "alternate_waveforms": {
        "GW190521": "NRSur7dq4"
    }
}
```

### `parameter_aliases`

Default `{}`. Maps the GWKokab parameter name (key) to the name used in the file's
`coords` attribute (value), in the same direction as the discrete loader. Because this
loader only validates rather than selects columns, aliases here affect the warning check
only — they do not reorder or rename anything in the returned arrays.

### `transform_module_path`

Default `null`, meaning the identity. This is the interesting field.

A Gaussian is a poor description of a posterior in coordinates where the posterior is
skewed, but often an excellent one after a change of variables — fitting in
$\ln m_1$ rather than $m_1$, say. `transform_module_path` points at a Python file that
declares that change of variables, so the Gaussian can live in well-behaved coordinates
while the population model keeps working in physical ones.

The module must define a class named exactly `Transform`, subclassing
[`SampleTransformer`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/analysis/core/utils/index.html),
and constructible with no arguments:

```python
import numpy as np
from gwkokab.analysis.core.utils import SampleTransformer


class Transform(SampleTransformer):
    """Gaussian fitted in log-primary-mass; model wants primary mass."""

    def transform(self, samples: np.ndarray) -> np.ndarray:
        out = np.copy(samples)
        out[..., 0] = np.exp(samples[..., 0])
        return out

    def log_abs_det_jacobian(
        self, samples: np.ndarray, transformed_samples: np.ndarray
    ) -> np.ndarray:
        return samples[..., 0]

    def check(
        self, samples: np.ndarray, transformed_samples: np.ndarray
    ) -> np.ndarray:
        # optional; drop draws that leave the physical region
        return transformed_samples[..., 0] > 0.0
```

`transform` maps a draw from the Gaussian into the coordinates the population model
evaluates in. `log_abs_det_jacobian` returns the log absolute determinant of that map's
Jacobian, $\ln\left|\partial\,\texttt{transformed}/\partial\,\texttt{samples}\right|$, one
value per draw; it is added to the model log-density, so it is the $\ln J_{n,k}$ term of
the Analytical GWalk likelihood. `check` is optional — it defaults to accepting everything — and
lets you reject draws that are numerically valid but physically meaningless; rejected
draws are redrawn alongside those that fall outside `limits`.

```json
{
    "regex": "../data/*.h5",
    "transform_module_path": "../transforms/log_mass.py"
}
```

The path is loaded with
[`importlib`](https://docs.python.org/3/library/importlib.html#importlib.util.spec_from_file_location)
as a standalone file, so it does not need to be installed or importable. Two failure modes
degrade to the identity transform *with a warning rather than an error*, which is worth
knowing about because the analysis will otherwise look like it is working:

- the module loads but has no `Transform` attribute;
- the module is empty.

A path that cannot be loaded at all raises `LoggedImportError`.

## A complete configuration

```json
{
    "regex": "../data/*.h5",
    "default_waveform": "IMRPhenomXPHM",
    "alternate_waveforms": {
        "GW190521": "NRSur7dq4"
    },
    "transform_module_path": "../transforms/log_mass.py",
    "parameter_aliases": {}
}
```

A template is produced with

```bash
gwk_analytical_gwalk_data_loader_cfg_template -o data_loader_cfg.json
```

Note that the template does not include `parameter_aliases`; add it by hand if you need
it.

## Reading one file without a configuration

`load_file` is a classmethod returning a `NamedTuple`, which is the quickest way to check
what a summary actually contains — in particular the `coords` order that everything else
depends on:

```python
from gwkokab.analysis.core.inference_io import AnalyticalGWalkPELoader

d = AnalyticalGWalkPELoader.load_file(
    "../data/event_0.hdf5", waveform_name="GWKokabSyntheticAnalyticalGWalkPE"
)
print(d.coords)          # ['mass_1_source', 'mass_2_source', 'redshift', 'chi_eff']
print(d.mu, d.cov.shape) # the summary itself
print(d.limits)          # (D, 2)
```

## Discrete or Analytical GWalk?

Both loaders read the same event files and are wired into the same analyses, so the choice
is about the likelihood, not the data:

| | Discrete | Analytical GWalk |
| --- | --- | --- |
| Per-event storage | $M_n \times D$ samples | $\mu_n$, $\Sigma_n$, `limits` |
| Monte Carlo draws | fixed by the data | an analysis knob |
| Reference prior | reweighted by the loader | not handled; must already be divided out |
| Non-Gaussian posteriors | exact | not captured |
| Missing requested column | error | warning |

The Analytical GWalk method is exactly as good as the Gaussian approximation.
`synthetic_analytical_gwalk_pe` reports the Jensen–Shannon divergence between the true and
approximated marginals so you can check; posteriors that are multimodal or railing against
a prior boundary should stay with the discrete loader.

## Common errors

| Message | Cause |
| --- | --- |
| `LoggedKeyError: Config error: 'regex' field is required.` | The `regex` key is absent. |
| `LoggedFileNotFoundError: No files matched the regex pattern: ...` | The glob matched nothing. |
| `LoggedKeyError: Waveform '...' not found in file '...'` | Wrong `default_waveform`; the message lists the groups present. |
| `LoggedValueError: File '...' contains non-positive scale values` | A zero or negative entry in the `scale` dataset. |
| `LoggedImportError: Could not load spec for module at ...` | `transform_module_path` does not point at a loadable Python file. |
| `UserWarning: The custom module must have a 'Transform' method.` | The module loaded but defines no `Transform` class; the identity transform is used. |
| `UserWarning: File '...' is missing required columns: {...}` | A requested parameter is not in `coords`. Not fatal, but almost certainly a mistake. |

## See also

- [Population Inference from Gaussian Summaries](../examples/ecc_plus_spin/inference_from_gaussian_summaries.md) —
  this loader inside a complete analysis, compared against the discrete method.
- [`DiscretePELoader`](./discrete_pe_loader.md) — the sample-based alternative.
