# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Requires `uv` and GNU `make`. `make install` always uninstalls first, then installs editable with `GWKOKAB_DEV_BUILD=1`.

```bash
make install PIP_FLAGS=--upgrade EXTRA=cpu GROUP=dev,test,doc   # EXTRA: cpu|cuda12|cuda13|tpu; GROUP: dev|test|doc
prek run --all-files                        # lint/format (prek, NOT pre-commit; config in prek.toml)
pytest tests                                # full suite
pytest tests/gwkokab/models/test_distributions.py -k test_dist_shape   # one file / one test
make doc                                    # reinstalls, then sphinx html into docs/build
pytest docs --doctest-glob='*.md' --doctest-glob='*.rst'  # doctests in docs (CI runs this)
GWKOKAB_DEV_BUILD=0 uv build                # release build (omits git-hash version suffix)
```

Note: `CONTRIBUTING.md` shows `pytest -n auto`, but `pytest-xdist` is not a declared dependency — drop `-n auto` unless you install it.

CI (`.github/workflows/ci.yml`) runs prek → tests (py3.11 & 3.13) → docs, with `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`, `XLA_FLAGS=--xla_force_host_platform_device_count=4`. CI installs with `uv sync --extra cpu --group dev --group test` and runs `uv run pytest tests`, not `make install`. `pyproject.toml` sets `filterwarnings = ["error", ...]`, so a new warning anywhere in the import chain fails tests; the test job has a 10-minute timeout, so keep new tests CPU-cheap.

## Architecture

Three top-level pieces:

- `src/gwkokab/` — the library: NumPyro/JAX population models, likelihoods, selection-function (Poisson mean) estimators, cosmology, parameter algebra.
- `src/gwkokab/analysis/` — the CLI drivers that assemble a full hierarchical inference run from JSON config, plus `inference_io` (Pydantic config parsing).
- `src/gwkokab_scripts/` — standalone post-hoc utility CLIs (plotting, HDF5 surgery, flowMC/NumPyro tuning helpers).

Every user-facing entry point is a console script declared in `pyproject.toml`'s `[project.scripts]`; there is no `__main__`.

### The analysis matrix

An analysis is the product of three independent axes, composed by multiple inheritance:

1. **Model family** (`analysis/<family>/`): `n_pls_m_gs`, `ecc_matters`, `multisource`, `subpopulation`. Each has a `common.py` defining `<Family>Core` with two key properties — `parameters` (the event coordinates read from data) and `model_parameters` (the flat list of population hyper-parameter names).
2. **Data representation** (`analysis/core/`): `discrete_base.py` (per-event posterior samples, bucketed) or `analytical_gwalk_base.py` (per-event Gaussian summaries — mean/covariance — resampled). Selected by which module you run: `discrete_<family>` vs `analytical_gwalk_<family>`.
3. **Sampler backend** (`analysis/core/`): `flowMC_base.py` or `numpyro_base.py`. Chosen *at runtime* from `sampler_cfg.json`'s `sampler_name`, not by the entry point.

So `analysis/n_pls_m_gs/discrete.py` defines `NPowerlawMGaussianDiscreteAnalysis(NPowerlawMGaussianCore, DiscreteBase)` and then two trivial subclasses that mix in `FlowMCBase` or `NumpyroBase`. The sampler mixin supplies `driver()`; the data-representation base supplies `run()` and `read_data()`. `main()` picks the class after reading the sampler config.

Likelihoods are picked by `inference/factory.py::get_likelihood_fn(sampler_name, analysis_type)`, which maps the 2×2 to the four modules in `inference/`. Those wrap the pure math in `inference/poissonlikelihood_utils.py` (`discrete_poisson_likelihood_fn` / `analytical_gwalk_poisson_likelihood_fn`): the inhomogeneous-Poisson log-likelihood `-μ(Λ) + Σ_n log Σ_k [p(ω_k|Λ)/π_k]`, with optional variance tapering when `--variance-cut-threshold` is set.

`AnalysisBase.run()` sequence: `classify_model_parameters()` → `read_data()` → build the Poisson-mean estimator → build `logpdf` via `likelihood_fn` → `driver()`.

### Priors: regex, aliases, constants, lazy

`prior_cfg.json` is **not** parameter-name-keyed; keys are regexes matched against `model_parameters` by `analysis/utils/regex.py::match_all`, then processed in `analysis/utils/priors.py::get_processed_priors`. A value can be:

- a dict with `"dist"` → an instantiated NumPyro/GWKokab prior (`{"dist": "Uniform", "low": 0.0, "high": 10.0}`),
- a number → a **constant** (frozen with `lax.stop_gradient`, not sampled),
- a string naming another parameter → an **alias** (tied parameter, shares one sampled dimension),
- a dict whose *field values* are strings → a **lazy prior**: becomes a `jax.tree_util.Partial` plus a dependency map, so a prior's hyper-parameters are themselves sampled.

`analysis_base.py::_classify_model_parameters` splits these four kinds, topologically sorts lazy dependencies (cycle-checked), and builds either a `JointDistribution` or a `LazyJointDistribution` over the sorted variable names. Sampled dimensions are ordered by `sorted(variables.keys())` — that ordering is what `variables_index` in the output HDF5 records, and it is how chains map back to parameter names.

Hyper-parameter names follow `<name>_<component>_<index>`. The component tag is family-specific: `pl`/`g` (power-law, Gaussian) for `n_pls_m_gs` and `ecc_matters`; `spl`/`bpl`/`gpl`/`gg` for `multisource`; `spl`/`bpl`/`gpl` for `subpopulation`. So `alpha_pl_0`, `mmax_pl_1`, `spin_1z_loc_g_0`, `log_rate_2`. `analysis/utils/common.py::expand_arguments` generates the indexed forms.

### Configuration I/O

Four JSON files, all Pydantic models with `extra="forbid"` (a typo raises `ValidationError` rather than being ignored) — see `analysis/core/inference_io/` and `docs/source/inference_io/`:

| Flag | Model | Purpose |
| --- | --- | --- |
| `--data-loader-cfg` | `DiscretePELoader` / `AnalyticalGWalkPELoader` | which event files, how to undo the PE prior |
| `--prior-cfg` | (plain JSON, see above) | population priors |
| `--pmean-cfg` | `PoissonMeanEstimationLoader` | selection function; `estimator_type` ∈ `neural_vt`, `neural_pdet`, `injection`, `custom` |
| `--sampler-cfg` | `SamplerConfig` → `NumpyroGlobalConfig` \| `FlowMCGlobalConfig` | which sampler and all its knobs |

The `gwk_*_cfg_template` console scripts dump starter versions of these.

`--pmean-cfg`'s `estimator_type` is a Pydantic discriminator picking one loader in `inference_io/_poisson_mean.py`, whose `get_estimators()` calls into `poisson_mean/`: `neural_vt` → `poisson_mean_from_neural_vt`, `neural_pdet` → `poisson_mean_from_neural_pdet`, `injection` → `poisson_mean_from_sensitivity_injections` (with `far_cut`/`snr_cut`). `custom` loads `python_module_path` by file path and calls its `custom_poisson_mean_estimator(key, parameters, filename, **kwargs)`.

### Output

Everything lands in one HDF5 file named by `analysis/utils/literals.py` (`inference_data.hdf5`): `constants`, `variables_index`, `sampler_cfg/*`, `samples`, and `chains/chain_<n>`. Sampler configs round-trip (`write_to_hdf5` / `read_from_hdf5`), so post-hoc scripts reconstruct the run from the file alone. `gwk_report` executes `analysis/report/template_report.ipynb` through papermill to produce an HTML report.

### Models

Models are NumPyro `Distribution` subclasses. Central abstraction is `models/utils/_scaledmixture.py::ScaledMixture` — a mixture whose components carry `log_scales` (log rates); the Poisson-mean estimators expect a `ScaledMixture` so rates and shapes are inferred jointly.

Hybrid models (`models/hybrids/`) are built compositionally: `_ncombination.py` holds `create_*` factories (one per physical parameter block), which `_npowerlawmgaussian.py`, `_multisource.py`, `_subpopulation.py` combine into a per-component `JointDistribution` and then into a `ScaledMixture`. Adding a physical parameter to a family means touching four places: a `create_*` factory, the `_build_*_distributions` wiring, the family's `model_parameters`/`parameters` properties in `analysis/<family>/common.py`, and its `--add-*` CLI flag in `model_arg_parser`.

`models/constraints.py` and `models/transformations.py` add custom NumPyro constraints/transforms (mass sandwiches, ordered vectors, mass-coordinate changes) and register them with `biject_to` via `@_transform_to_*` dispatch.

### Parameters and coordinate changes

`parameters.py` defines the `Parameters` str-enum (imported everywhere as `from gwkokab.parameters import Parameters as P`) and a `RelationMesh`: a rule graph over GW parameters (`(m1, m2) → chirp_mass`, `(chi1, cos_tilt_1) → spin_1z`, `z ↔ luminosity_distance`, …) with a `resolve()` that derives every reachable parameter from whatever is present. That is what backs the `--derive-parameters` flag on the synthetic-data CLIs. Pure formulas live in `utils/transformations.py`.

### Logging and errors

Uses `loguru`, configured in `analysis/utils/logger.py`; CLIs call `log_info(start=True)` before anything else. Raise the auto-logging exceptions from `utils/exceptions.py` (`LoggedValueError`, `LoggedKeyError`, `LoggedTypeError`, `LoggedUserWarning`, …) rather than bare built-ins — they log at error/warning level on construction, with loguru `{}`-style formatting args.

Runtime env vars: `GWKOKAB_LOG_LEVEL` (default `TRACE`), `GWKOKAB_LOG_DIR` (`./logs`), `GWKOKAB_LOG_FILE`, `GWKOKAB_DEBUG` (adds source paths to log lines), `GWKOKAB_DEFAULT_COSMOLOGY` (default `Planck15`), `GWKOKAB_DEV_BUILD` (appends git hash to `__version__`).

### End-to-end workflow

See `docs/source/examples/ecc_plus_spin/` for a complete worked run: `synthetic_events_<family>` (draw a population) → `synthetic_discrete_pe` or `synthetic_analytical_gwalk_pe` (add measurement error, write per-event files; the error models are `errors.py`'s `banana_error` / `truncated_normal_error` / `mock_spin_error`, wired per-parameter by `synthetic_pe.py::error_function_registry`) → `discrete_<family>` / `analytical_gwalk_<family>` (inference) → `gwk_report`.

## Testing

`tests/` mirrors `src/` one-to-one (`tests/gwkokab/models/mass/test_models.py` ↔ `src/gwkokab/models/mass/_models.py`). Several packages carry same-named modules, so test basenames collide; `--import-mode=importlib` in `addopts` is what makes that legal — do not drop it.

Shared fixtures live in five conftests and are worth reusing before writing new setup:

- `tests/gwkokab/conftest.py` — `linear_model_file` (writes an HDF5 network that is exactly an affine map, so Poisson means have closed forms), `injections_file`.
- `models/conftest.py` — `pytree_roundtrip` (flatten/unflatten a distribution, i.e. what a `jit`/`vmap` boundary does), `trapz_1d`, `trapz_nd` (normalization checks).
- `models/hybrids/conftest.py` — `hyper_parameters`: feed it a family's `core.model_parameters` and get a physically sensible value for every name, keyed by *role* (the part before the component tag), so it stays valid as component counts change. Passing `core.model_parameters` also catches drift between the analysis layer and the model factory.
- `inference/conftest.py` — `mixture`, `normal_mixture`, `estimator`, `model_factory` (recording doubles), `discrete_data`, `analytical_data`.
- `gwkokab_scripts/conftest.py` — `run_main(main, *argv)` drives a console script's `main()` through a fake `sys.argv`; `samples_file`/`compound_file` fabricate the two `inference_data.hdf5` layouts. Autouse fixtures force the `Agg` backend and isolate pyplot state.

Style is `pytest.mark.parametrize` plus plain asserts, with `chex` for array/shape assertions in the numerics-heavy files. Because warnings are errors, a dependency that warns at *import* must be pre-imported quietly in a conftest (see the `glasbey` note in `gwkokab_scripts/conftest.py`).

## Conventions

- Every source file starts with `# Copyright 2023 The GWKokab Authors` and `# SPDX-License-Identifier: Apache-2.0`.
- `__init__.py` re-exports use the explicit `from .x import y as y` form (required for the strict re-export rules the docs/type-checkers rely on). Public modules are also re-exported as `from . import x as x`.
- numpydoc-style docstrings; docformatter wraps at 88. Ruff: line length 88, double quotes, isort with `order-by-type = false`, `combine-as-imports`, two blank lines after imports.
- Test files must be named `test_*.py` (enforced by the `name-tests-test` hook).
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `build:`).
- prek hooks also enforce: HTTPS-only URLs in markdown/yaml, no nested-bracket link pattern (the `no-bracket-links` hook in `prek.toml`), codespell, gitleaks, and JSON-schema validation of workflow/dependabot/readthedocs files.
- New CLI ⇒ add it to `[project.scripts]` in `pyproject.toml`.
