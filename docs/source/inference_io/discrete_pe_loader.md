# `DiscretePELoader`

`DiscretePELoader` turns a glob pattern into a list of events, reads each event's
posterior samples out of an HDF5 file, optionally sub-samples them, and computes the log
of the reference prior that the original parameter-estimation run assumed. It is the data
side of the [discrete likelihood](../examples/ecc_plus_spin/inference_from_posterior_samples.md),
where each event contributes a Monte Carlo sum over its samples

$$
\ln\mathcal{L}(\Lambda) \propto
\sum_{n=1}^{N}\ln\left[\frac{1}{M_n}\sum_{k=1}^{M_n}
\frac{\rho(\theta_{n,k}\mid\Lambda)}{\pi_n(\theta_{n,k})}\right]
$$

and the loader's job is to produce both $\theta_{n,k}$ and $\ln\pi_n(\theta_{n,k})$.

Analyses receive it through `--data-loader-cfg`.

## The minimal configuration

One key is required:

```json
{
    "regex": "../data/event_*.hdf5"
}
```

That is a complete, working configuration. It says: every file matching the pattern is one
event, read its samples from the default dataset, keep all of them, and assume a flat
reference prior.

```python
from gwkokab.analysis.core.inference_io import DiscretePELoader

loader = DiscretePELoader.read_from_json("data_loader_cfg.json")
data, log_ref_priors = loader.load(
    ("mass_1_source", "mass_2_source", "redshift", "chi_eff")
)
```

`load` returns two lists of length $N$, one entry per event. `data[n]` has shape
`(M_n, len(parameters))` with columns in exactly the order you asked for, and
`log_ref_priors[n]` has shape `(M_n,)`. The number of samples $M_n$ may differ between
events; the analysis handles that by
[bucketing](../examples/ecc_plus_spin/inference_from_posterior_samples.md#buckets).

Downstream, the likelihood computes `model_log_prob - log_ref_priors`, so a value of zero
means "flat reference prior, nothing to divide out".

## Fields

### `regex`

**Required.** Despite the name this is a **glob** pattern, not a regular expression — it
is passed to [`glob.glob`](https://docs.python.org/3/library/glob.html#glob.glob), so
`*`, `?` and `[0-9]` work but `.*` and `+` do not. Matches are sorted, and that sorted
order is the event order used everywhere downstream.

If the pattern matches nothing, the loader raises immediately rather than proceeding with
an empty analysis:

```text
LoggedFileNotFoundError: No files matched the regex pattern: ../data/event_*.hdf5
```

Recursive `**` is not enabled, so it behaves as a single `*`. To pull events from a tree
of directories, either list them under one directory or use an explicit pattern such as
`../data/*/posterior.hdf5`.

```{admonition} Event names
:class: note

Several fields below are keyed by *event name*, which is the file's stem: the basename
with directory and extension stripped. `../data/GW150914_result.hdf5` has the event name
`GW150914_result`. Get the name wrong and the override simply does not apply: the loader
falls back to the default and logs which prior or dataset it used, so the run log is where
to check.
```

### `default_datasets`

Default `["/GWKokabSyntheticDiscretePE/posterior_samples"]`. HDF5 paths to try, **in
order of preference**; the first one present in a given file wins. This is what makes a
single configuration work across a heterogeneous catalogue: list every convention your
files might use and let the loader pick.

```json
{
    "regex": "../data/*.h5",
    "default_datasets": [
        "/PublicationSamples/posterior_samples",
        "/C01:Mixed/posterior_samples",
        "/GWKokabSyntheticDiscretePE/posterior_samples"
    ]
}
```

The dataset must be an HDF5 **structured** (compound) array; its field names become the
column names the loader matches `parameter_aliases` against. A leading `/` is added for
you if you leave it out, with a warning. Note that this field must be a JSON *list* even
when it holds a single entry.

If none of the listed datasets is found the loader tells you what the file actually
contains:

```text
LoggedKeyError: None of the specified datasets ('/PublicationSamples/posterior_samples',)
found in file '../data/GW150914.h5'. Available datasets: ['C01:IMRPhenomXPHM', 'history']
```

### `alternate_datasets`

Default `{}`. Per-event override mapping event name to a **single** dataset path (a
string, not a list). When an event name appears here, `default_datasets` is not consulted
at all for that event.

```json
{
    "regex": "../data/*.h5",
    "default_datasets": ["/PublicationSamples/posterior_samples"],
    "alternate_datasets": {
        "GW190521": "/NRSur7dq4/posterior_samples",
        "GW170817": "/IMRPhenomPv2NRT_lowSpin_posterior/posterior_samples"
    }
}
```

### `parameter_aliases`

Default `{}`. Maps **the parameter name GWKokab uses** to **the column name in your
files**. The key is the GWKokab name, the value is the file's name — this direction is
worth committing to memory, since getting it backwards produces a confusing "missing
required columns" error naming parameters you thought you had provided.

```json
{
    "regex": "../data/*.h5",
    "parameter_aliases": {
        "mass_1_source": "mass_1_source_frame",
        "mass_2_source": "mass_2_source_frame",
        "chi_eff": "chi_effective"
    }
}
```

Aliases apply both to the columns you request in `load` and to the columns the prior
reweighting needs internally, so a catalogue with non-standard names only has to be
declared once. The GWKokab names are the values of
[`gwkokab.parameters.Parameters`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/parameters/index.html);
the ones the prior machinery may look for are `chirp_mass`, `chirp_mass_detector`,
`chi_1`, `chi_2`, `chi_eff`, `chi_p`, `mass_ratio`, `mass_1`, `mass_1_source`, `a_1`,
`a_2`, `mass_2`, `mass_2_source` and `redshift`.

A parameter that is requested but absent from a file is fatal:

```text
LoggedValueError: File '../data/GW150914.h5' is missing required columns: {'chi_p'}
```

### `max_samples`

Default `null`, meaning keep every sample. Set it to a positive integer to sub-sample each
event down to at most that many samples. The cost of the discrete likelihood is linear in
the total number of samples, so this is the main knob for trading accuracy against memory
and speed — useful when a few events in the catalogue have far more samples than the rest
and inflate the padded array everyone else has to fit into.

```json
{
    "regex": "../data/*.h5",
    "max_samples": 5000
}
```

Sub-sampling is without replacement and deterministic: event $i$ uses `seed + i`, where
`seed` is the analysis seed (`--seed`). The same seed therefore reproduces the same
sub-sample. An event that already has fewer than `max_samples` samples is left untouched,
with a warning.

```{admonition} Sub-sampling is not free of consequence
:class: caution

The Monte Carlo estimate of each event's marginal likelihood becomes noisier as $M_n$
falls, and that noise can pull the sampler towards spurious likelihood spikes. If you
sub-sample aggressively, consider also setting `--variance-cut-threshold` on the analysis
so that hyperparameter values with an unreliable estimate are penalised.
```

## Reference priors

The remaining six fields all serve one purpose: reconstructing $\pi_n$, the prior the
original parameter-estimation run sampled under, so it can be divided out. Each is a
`default_*` applying to every event, plus an `alternate_*` mapping event names to
overrides. The logic is inspired by
[`gwpopulation_pipe.data_collection.evaluate_prior`](https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe/).

All three default to `null`, which means **a flat reference prior with no Jacobian**.
That is correct for synthetic catalogues generated by `synthetic_discrete_pe`, where no PE
prior was ever imposed. It is essentially never correct for a real catalogue.

### `default_mass_prior`, `alternate_mass_priors`

One of `null`, `"flat-detector-components"`, `"flat-detector-chirp-mass-ratio"` or
`"flat-source-components"`.

`"flat-detector-components"`
: The PE run was flat in the *detector-frame* component masses $(m_1^{\det}, m_2^{\det})$,
  which is the LVK default. Converting to source-frame masses contributes $2\ln(1+z)$.

`"flat-detector-chirp-mass-ratio"`
: The PE run was flat in detector-frame chirp mass and mass ratio
  $(\mathcal{M}^{\det}, q)$. The Jacobian to component masses is removed.

`"flat-source-components"`
: The PE run was flat in the source-frame component masses; no redshift factor is needed.

Whenever a mass prior is set, additional Jacobians are applied for the parameterisation
*you* are inferring in: requesting `mass_ratio` alongside a primary mass adds $\ln m_1$,
and requesting `chirp_mass` or `chirp_mass_detector` adds the chirp-mass Jacobian. The
mass ratio itself is taken from a `mass_ratio` column if present and otherwise computed
from the component masses, so you do not have to supply it. With the prior left at `null`
none of this happens and the contribution is exactly zero.

```{admonition} Mass and distance priors need redshift
:class: important

Any non-`null` mass or distance prior requires `redshift` to be among the parameters
passed to `load`, since the source/detector frame conversion depends on it. Otherwise you
get `LoggedValueError: Mass prior reweighting requires Redshift.` or
`LoggedValueError: Distance prior requires Redshift.` respectively.
```

### `default_spin_prior`, `alternate_spin_priors`

Either `null` or `"component"`. `"component"` means the PE run used a prior uniform in each
of the two component spins over $[-1, 1]$, contributing $\ln\tfrac{1}{2}$ per component and
so $-\ln 4$ in total.

Independently of this setting, spin *parameterisation* Jacobians are applied based on what
you ask for. Requesting `chi_eff` inserts the effective-spin prior induced by isotropic
component spins, following
[`tcallister/effective-spin-priors`](https://github.com/tcallister/effective-spin-priors);
requesting `chi_eff` and `chi_p` together inserts the joint
$(\chi_{\mathrm{eff}}, \chi_p)$ prior derived by Iwaya et al. (2024); requesting `chi_1`
or `chi_2` inserts the aligned spin prior $p(\chi) = -\tfrac{1}{2}\ln|\chi|$ for each.

### `default_distance_prior`, `alternate_distance_priors`

One of `null`, `"comoving"` or `"euclidean"`.

`"comoving"`
: The PE run was uniform in comoving volume and source-frame time, contributing
  $\ln\frac{\mathrm{d}V_c}{\mathrm{d}z} + \ln 4\pi - \ln(1+z)$.

`"euclidean"`
: The PE run was $\propto d_L^2$, the LVK default, contributing
  $2\ln d_L + \ln\frac{\mathrm{d}d_L}{\mathrm{d}z}$.

Both use GWKokab's [default cosmology](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/cosmology/index.html).

### Overrides, and the difference between "absent" and `null`

The `alternate_*` dictionaries distinguish a key that is missing from a key whose value is
`null`. A missing key falls back to the default; a key set explicitly to `null` overrides
the default with *no* prior. This is how you exempt one event from a catalogue-wide
setting:

```json
{
    "regex": "../data/*.h5",
    "default_mass_prior": "flat-detector-components",
    "default_spin_prior": "component",
    "default_distance_prior": "euclidean",
    "alternate_distance_priors": {
        "GW170817": "comoving"
    },
    "alternate_mass_priors": {
        "GW190521": null
    }
}
```

Here every event gets the LVK defaults, `GW170817` is reweighted from a comoving-volume
prior instead, and `GW190521` — whose samples came from a re-analysis with a flat mass
prior already divided out — gets no mass reweighting at all.

Each event logs the priors it ended up using, so a run's log is the quickest way to
confirm the overrides landed where you intended.

## A complete configuration

```json
{
    "regex": "../data/*.h5",
    "max_samples": 5000,
    "default_datasets": [
        "/PublicationSamples/posterior_samples",
        "/C01:Mixed/posterior_samples"
    ],
    "alternate_datasets": {
        "GW190521": "/NRSur7dq4/posterior_samples"
    },
    "parameter_aliases": {
        "chi_eff": "chi_effective"
    },
    "default_mass_prior": "flat-detector-components",
    "default_spin_prior": "component",
    "default_distance_prior": "euclidean",
    "alternate_mass_priors": {},
    "alternate_spin_priors": {},
    "alternate_distance_priors": {}
}
```

A template containing every key at its default is produced with

```bash
gwk_discrete_data_loader_cfg_template -o data_loader_cfg.json
```

## Reading one file without a configuration

`load_file` is a classmethod and needs no loader instance, which makes it handy for
inspecting a file before writing the configuration around it:

```python
from gwkokab.analysis.core.inference_io import DiscretePELoader

df = DiscretePELoader.load_file(
    "../data/event_0.hdf5", "/GWKokabSyntheticDiscretePE/posterior_samples"
)
print(df.columns.tolist())
print(df.describe())
```

It returns a `pandas.DataFrame` and accepts either a single dataset path or a tuple of
candidates, with the same first-match-wins semantics as `default_datasets`.

## Common errors

| Message | Cause |
| --- | --- |
| `LoggedKeyError: Config error: 'regex' field is required.` | The `regex` key is absent. |
| `LoggedFileNotFoundError: No files matched the regex pattern: ...` | The glob matched nothing; check the pattern and the working directory. |
| `LoggedKeyError: None of the specified datasets ... found in file` | Wrong `default_datasets`; the message lists what the file does contain. |
| `LoggedValueError: File '...' is missing required columns: {...}` | A requested parameter has no column; add a `parameter_aliases` entry. |
| `LoggedValueError: Mass prior reweighting requires Redshift.` | A mass prior is set but `redshift` is not among the analysis parameters. |
| `LoggedValueError: Distance prior requires Redshift.` | Likewise for a distance prior. |
| `ValidationError: ... Extra inputs are not permitted` | A misspelled key. |

## See also

- [Population Inference from Posterior Samples](../examples/ecc_plus_spin/inference_from_posterior_samples.md) —
  this loader inside a complete analysis.
- [`AnalyticalPELoader`](./analytical_pe_loader.md) — the same role for Gaussian summaries
  instead of samples.
