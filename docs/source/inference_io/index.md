# Inference I/O

Every GWKokab analysis is assembled from a handful of JSON files. Nothing about an
analysis is hard-coded: which event files to read, how to undo the parameter-estimation
prior, how the expected number of detections is computed, and how the sampler is driven
are all described by JSON and parsed by
[`gwkokab.analysis.core.inference_io`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/analysis/core/inference_io/index.html).

This section documents each of those interfaces from the point of view of the file on
disk: what keys it accepts, what they mean, what the defaults are, and what happens when
you get one wrong. The [worked examples](../examples/index.md) show these files in the context
of a complete analysis; the pages here are the reference you reach for when writing one.

## The interfaces

| Interface | Describes | Read with | CLI flag |
| --- | --- | --- | --- |
| [`DiscretePELoader`](./discrete_pe_loader.md) | Event posterior samples | `read_from_json(path)` | `--data-loader-cfg` |
| [`AnalyticalPELoader`](./analytical_pe_loader.md) | Event Gaussian summaries | `read_from_json(path)` | `--data-loader-cfg` |
| [`PoissonMeanEstimationLoader`](./poisson_mean_estimation_loader.md) | Selection function / $\mu(\Lambda)$ | `read_from_json(path, key, parameters)` | `--pmean-cfg` |
| [`SamplerConfig`](./sampler_config.md) | Which sampler, and how | `read_from_json(path)` | `--sampler-cfg` |
| [`NumpyroGlobalConfig`](./sampler_config.md#numpyro-numpyroglobalconfig) | NUTS kernel and MCMC driver | via `SamplerConfig` | `--sampler-cfg` |
| [`FlowMCGlobalConfig`](./sampler_config.md#flowmc-flowmcglobalconfig) | flowMC local sampler and normalizing flow | via `SamplerConfig` | `--sampler-cfg` |

The last two are the two things `SamplerConfig` can return; they are documented on the
same page, along with the `NumpyroNUTSSamplerConfig` and `NumpyroMCMCConfig` sub-blocks.

An analysis such as `discrete_n_pls_m_gs` is handed all four at once:

```bash
discrete_n_pls_m_gs \
    --data-loader-cfg data_loader_cfg.json \
    --prior-cfg       prior_cfg.json \
    --pmean-cfg       pmean_cfg.json \
    --sampler-cfg     sampler_cfg.json \
    ...
```

The fifth file, `prior_cfg.json`, is not part of `inference_io`; it is documented in
[Priors](../examples/ecc_plus_spin/inference_from_posterior_samples.md#priors).

## Conventions shared by every configuration

These hold for all of the files documented in this section, and knowing them removes most
of the surprises.

**Every interface is a [Pydantic](https://docs.pydantic.dev/latest/) model.** The JSON is
not merely loaded, it is *validated*. Types are coerced where that is unambiguous (a JSON
list becomes a tuple or a NumPy array, an integer becomes a float), and rejected where it
is not.

**Unknown keys are an error, not a warning.** Every model is declared with
[`extra="forbid"`](https://docs.pydantic.dev/latest/concepts/models/#extra-data), so a
misspelled key raises a `ValidationError` naming the offending field rather than being
silently ignored. This is deliberate: a typo in `max_samples` that was quietly dropped
would change your results without telling you.

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for DiscretePELoader
typo_field
  Extra inputs are not permitted [type=extra_forbidden, ...]
```

**Omitted keys take their documented default.** Only a small number of keys are actually
required — `regex` for the data loaders, `estimator_type` and `filename` for the Poisson
mean loader, `sampler_name` for the sampler. Everything else has a default, and the
minimal configuration for most analyses is two or three lines long.

**JSON is strict JSON.** Comments, trailing commas and single quotes are not accepted.
Parsing is done by
[`gwkokab.analysis.utils.common.read_json`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/analysis/utils/common/index.html),
which converts both a missing file and a malformed one into the same `ValueError`:

```text
ValueError: Error loading configuration: Expecting ',' delimiter: line 4 column 5 (char 82)
```

**Relative paths are resolved against the working directory**, not against the
configuration file. The examples in this documentation keep the shared configurations one
level up and run each analysis from its own sub-directory, which is why their paths begin
with `../`. Absolute paths are always safe.

**Templates can be generated.** Three commands write out a fully populated file with
every key present at its default, which is often quicker than starting from scratch:

```bash
gwk_discrete_data_loader_cfg_template   -o data_loader_cfg.json
gwk_analytical_data_loader_cfg_template -o data_loader_cfg.json
gwk_numpyro_cfg_template                -o sampler_cfg.json
gwk_flowMC_cfg_template                 -o sampler_cfg.json
```

## Reading a configuration by hand

Nothing forces you to go through a CLI. Each interface is importable and can be pointed
at a JSON file directly, which is the fastest way to check that a configuration does what
you meant before committing to a multi-hour run:

```python
from gwkokab.analysis.core.inference_io import DiscretePELoader

loader = DiscretePELoader.read_from_json("data_loader_cfg.json")
print(loader.filenames)          # which files the glob actually matched
print(loader.max_samples)        # what the loader resolved every field to

data, log_ref_priors = loader.load(("mass_1_source", "mass_2_source", "redshift"))
print(len(data), data[0].shape)
```

Printing the model itself, or `loader.model_dump()`, shows every field including the ones
you left out — the most direct way to see the defaults that are in play.

```{toctree}
:maxdepth: 1

discrete_pe_loader
analytical_pe_loader
poisson_mean_estimation_loader
sampler_config
```
