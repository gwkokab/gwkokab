# `SamplerConfig`

GWKokab ships two samplers, NumPyro's NUTS and [flowMC](https://github.com/kazewong/flowMC).
Which one runs is decided entirely by the contents of the file passed to `--sampler-cfg`;
the analysis command is byte-for-byte identical either way. `SamplerConfig` is the small
factory that reads that file, looks at one key, and hands back the right configuration
object.

It is not a model of its own — you never instantiate it — but a pair of static methods:

```python
from gwkokab.analysis.core.inference_io import SamplerConfig

cfg = SamplerConfig.read_from_json("sampler_cfg.json")
print(type(cfg).__name__)   # NumpyroGlobalConfig  or  FlowMCGlobalConfig
```

The two objects it can return, [`NumpyroGlobalConfig`](#numpyro-numpyroglobalconfig) and
[`FlowMCGlobalConfig`](#flowmc-flowmcglobalconfig), are documented in full further down
this page.

## The discriminator

`sampler_name` is the only key `SamplerConfig` looks at, and it is **required**:

```json
{
    "sampler_name": "numpyro"
}
```

```json
{
    "sampler_name": "flowMC"
}
```

Either of those two lines is a complete, valid sampler configuration — every other key
takes its default. `"numpyro"` produces a `NumpyroGlobalConfig`, `"flowMC"` a
`FlowMCGlobalConfig`, and the rest of the file is then validated against that model's
fields. Because the two field sets are disjoint, a key that belongs to the other sampler
is reported as an unexpected extra rather than quietly ignored — which is the usual
symptom of copying half a configuration between the two.

Note the capitalisation: `flowMC`, not `flowmc`. Values are matched exactly.

```text
ValidationError: 1 validation error for tagged-union[NumpyroGlobalConfig,FlowMCGlobalConfig]
  Input tag 'flowmc' found using 'sampler_name' does not match any of the
  expected tags: 'numpyro', 'flowMC'
```

Omitting `sampler_name` entirely gives `Unable to extract tag using discriminator
'sampler_name'`.

## Which sampler?

Both are driven by the same likelihood and produce the same
[output file](../examples/ecc_plus_spin/inference_from_posterior_samples.md#output), so
switching is a one-key edit and running both is a genuinely useful cross-check.

**NumPyro / NUTS** is gradient-based and self-tuning: with `adapt_step_size` and
`adapt_mass_matrix` on, the warm-up phase discovers the geometry by itself, and $\hat{R}$
and the effective sample size give an unambiguous verdict on whether the run converged.
Reach for it first. Its weakness is strongly curved or multimodal posteriors, where NUTS
can crawl even with a dense mass matrix.

**flowMC** trains a normalizing flow on the chains and proposes from it, which lets it
jump across a curved or multimodal posterior that NUTS would have to walk around. The
price is that it has many more knobs, and one of them — `condition_matrix` — genuinely has
to be tuned to your problem before the sampler works well.

---

## NumPyro: `NumpyroGlobalConfig`

`NumpyroGlobalConfig` describes a run of the No U-Turn Sampler. Selected by
`"sampler_name": "numpyro"`, it has exactly three keys:

```json
{
    "sampler_name": "numpyro",
    "kernel": {},
    "mcmc": {}
}
```

`kernel` configures the NUTS kernel itself — step size, mass matrix, tree depth — and is
validated by `NumpyroNUTSSamplerConfig`. `mcmc` configures the driver around it — how many
samples, how many chains, how they are mapped onto devices — and is validated by
`NumpyroMCMCConfig`. Both blocks are optional; omitting them accepts every default.

GWKokab passes these through to
[`numpyro.infer.NUTS`](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS) and
[`numpyro.infer.MCMC`](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.mcmc.MCMC),
whose documentation is the authority on what each parameter does; the notes below are
about what they mean for a population-inference run.

```python
from gwkokab.analysis.core.inference_io import SamplerConfig

cfg = SamplerConfig.read_from_json("sampler_cfg.json")
print(cfg.mcmc.num_samples, cfg.kernel.dense_mass)
```

`NumpyroGlobalConfig.read_from_json` works identically when you already know the file
describes a NumPyro run.

### The `kernel` block

```json
{
    "sampler_name": "numpyro",
    "kernel": {
        "step_size": 1.0,
        "inverse_mass_matrix": null,
        "adapt_step_size": true,
        "adapt_mass_matrix": true,
        "dense_mass": true,
        "target_accept_prob": 0.8,
        "max_tree_depth": 8,
        "find_heuristic_step_size": false,
        "forward_mode_differentiation": false,
        "regularize_mass_matrix": true
    }
}
```

`step_size`
: Default `1.0`, must be positive. The initial size of a step taken by the Verlet
  integrator. With `adapt_step_size` on this is only a starting point, so it rarely needs
  changing.

`adapt_step_size`
: Default `true`. Adapt the step size during warm-up by dual averaging. Leave it on unless
  you are deliberately reusing a tuned step size.

`adapt_mass_matrix`
: Default `true`. Adapt the mass matrix during warm-up using Welford's algorithm. Leave it
  on for the same reason.

`dense_mass`
: Default `false`. **The field most worth thinking about.** `false` gives a diagonal mass
  matrix, which assumes the hyperparameters are uncorrelated after scaling. Population
  posteriors are usually not: a power-law slope and a mass cut-off, or $\alpha$ and
  $\beta$ in a mass model, are typically strongly correlated, and a diagonal mass matrix
  makes NUTS crawl through the resulting ridge. Setting `true` uses a full-rank matrix,
  which costs $O(D^2)$ memory and a more expensive warm-up but usually pays for itself.

  A middle ground is a **block** structure, given as a list of lists of parameter names.
  Each inner list is one dense block; everything not mentioned stays diagonal:

  ```json
  {"dense_mass": [["alpha_pl_0", "beta_pl_0"], ["mmin_pl_0", "mmax_pl_0"]]}
  ```

  The names are the model's hyperparameter names, the same ones used as keys in
  `prior_cfg.json` after regex expansion — `variables_index` in an output file lists them.
  If you have a pilot run but no intuition for the geometry,

  ```bash
  gwk_numpyro_dense_mass_structure inference_data.hdf5 --corr-threshold 0.5
  ```

  suggests blocks from the posterior sample correlations.

`inverse_mass_matrix`
: Default `null`, i.e. the identity. A starting value for the inverse mass matrix, which
  `adapt_mass_matrix` then refines. Accepts a JSON list — turned into a NumPy array — or a
  dictionary keyed by tuples of site names for a structured block matrix. Chiefly useful
  for restarting from a previous run's adapted matrix.

`target_accept_prob`
: Default `0.8`, in $(0, 1]$. The acceptance probability dual averaging aims for. Raising
  it towards `0.9`–`0.95` forces a smaller step size: slower, but the standard first
  response to divergences.

`max_tree_depth`
: Default `8`. The doubling scheme builds at most $2^{\texttt{max\_tree\_depth}}$ leapfrog
  steps per sample, so raising this is exponentially expensive. Hitting the cap regularly
  is a symptom of bad geometry — fix `dense_mass` first. A two-element list `[d1, d2]`
  sets different caps for the warm-up and post-warm-up phases.

`find_heuristic_step_size`
: Default `false`. Re-estimate a step size heuristically at the start of each adaptation
  window.

`forward_mode_differentiation`
: Default `false`, i.e. reverse-mode. GWKokab likelihoods have many more inputs than
  outputs, so reverse mode is the right default; forward mode exists for models whose
  control flow (`jax.lax.while_loop`, `jax.lax.fori_loop`) supports nothing else.

`regularize_mass_matrix`
: Default `true`. Regularises the adapted mass matrix, which matters most with
  `dense_mass: true` and few warm-up samples, where the empirical covariance can be
  near-singular.

### The `mcmc` block

```json
{
    "sampler_name": "numpyro",
    "mcmc": {
        "num_warmup": 5000,
        "num_samples": 7000,
        "num_chains": 4,
        "thinning": 1,
        "chain_method": "parallel",
        "progress_bar": true,
        "progress_rate": null,
        "jit_model_args": true
    }
}
```

`num_warmup`
: Default `1000`. Warm-up iterations, discarded. This is where step size and mass matrix
  adapt, so an under-warmed run shows up as divergences or a step size driven to zero, not
  as an obviously short chain. Population runs commonly want several thousand.

`num_samples`
: Default `2000`. Retained samples **per chain**. The `samples` dataset in the output has
  `num_chains * num_samples` rows.

`num_chains`
: Default `1`. Independent chains. More than one is what makes $\hat{R}$ meaningful, so
  four is a sensible floor for a run you intend to trust.

`chain_method`
: Default `"parallel"`. One of `"parallel"`, `"sequential"` or `"vectorized"`; the three
  are compared in their own section below.

`thinning`
: Default `1`, i.e. no thinning. `2` keeps every other post-warm-up sample. Thinning
  discards information and is worth it only when the output size is a real constraint.

`progress_bar`
: Default `true`. Turn it off for batch or cluster jobs, where the redraws bloat log files.

`progress_rate`
: Default `null`, meaning updates every 5 % of iterations when there are more than 20.
  Set an integer for a fixed number of iterations per update.

`jit_model_args`
: Default `true`. Compiles the potential energy as a function of the model arguments, so a
  second `run` on a different dataset of the same shape does not recompile. Note that it
  has no effect when `num_chains > 1` with `chain_method: "parallel"`.

### `chain_method` in detail

- `"parallel"` maps chains onto XLA devices with
  [`jax.pmap`](https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html). Fastest when you
  have the devices, but the number of chains run in one batch is capped by how many are
  visible — on a single-CPU host that is one, and NumPyro falls back to running them in
  sequence.
- `"sequential"` runs one chain after another. Slowest, lowest memory.
- `"vectorized"` vectorises all chains onto one device. Usually the right choice for
  multiple chains on a single CPU or GPU.

```{admonition} Exposing more devices on one host
:class: tip

To actually run four chains in parallel on a single-CPU host, set

    export XLA_FLAGS=--xla_force_host_platform_device_count=4

before launching, or switch to `"vectorized"`.
```

### A complete NumPyro configuration

The one used by the [worked examples](../examples/ecc_plus_spin/inference_from_posterior_samples.md#numpyro):

```json
{
    "kernel": {
        "step_size": 1.0,
        "inverse_mass_matrix": null,
        "adapt_step_size": true,
        "adapt_mass_matrix": true,
        "dense_mass": true,
        "target_accept_prob": 0.8,
        "max_tree_depth": 8,
        "find_heuristic_step_size": false,
        "forward_mode_differentiation": false,
        "regularize_mass_matrix": true
    },
    "mcmc": {
        "num_warmup": 5000,
        "num_samples": 7000,
        "num_chains": 4,
        "thinning": 1,
        "chain_method": "parallel",
        "progress_bar": true,
        "jit_model_args": true
    },
    "sampler_name": "numpyro"
}
```

### Reading the NumPyro diagnostics

After each batch of chains NumPyro prints a summary table. Two columns decide whether the
configuration was adequate:

- $\hat{R}$ close to $1$ for every hyperparameter. Anything above roughly $1.01$ means the
  chains have not mixed — raise `num_warmup`, or `num_samples`, or fix the geometry.
- `n_eff`, the effective sample size. A few hundred per hyperparameter is usable; tens
  are not.

Divergences, or a step size adapted down towards zero, almost always mean the mass matrix
has not captured the geometry. In order of what to try: set `dense_mass: true`, raise
`target_accept_prob`, raise `num_warmup`.

`gwk_report -i inference_data.hdf5 -o report.html` renders trace plots, marginals and
convergence diagnostics from the output file alone.

---

## flowMC: `FlowMCGlobalConfig`

[flowMC](https://github.com/kazewong/flowMC) augments a local MCMC sampler with global
proposals drawn from a normalizing flow trained on the chains themselves. That combination
handles the curved, correlated posteriors typical of population inference well, at the
cost of considerably more knobs than NUTS.

`FlowMCGlobalConfig` is selected by `"sampler_name": "flowMC"`. Unlike the NumPyro
configuration it is flat — no `kernel` or `mcmc` sub-blocks — and every key other than
`sampler_name` is optional:

```json
{
    "sampler_name": "flowMC"
}
```

```python
from gwkokab.analysis.core.inference_io import SamplerConfig

cfg = SamplerConfig.read_from_json("sampler_cfg.json")
print(type(cfg).__name__, cfg.n_chains)
```

:::{admonition} Read it through `SamplerConfig`
:class: warning

`FlowMCGlobalConfig.read_from_json` currently mis-forwards its argument to Pydantic and
raises

```text
TypeError: BaseModel.model_validate() got an unexpected keyword argument 'sampler_name'
```

`SamplerConfig.read_from_json` reads flowMC files correctly and is what every GWKokab
analysis calls, so use it for both samplers.
:::

### How a flowMC run is shaped

A flowMC run is two phases of nested loops, and the fields divide up along the same lines.

```text
n_training_loops  ×  ( n_local_steps local moves  →  n_global_steps flow proposals
                                                  →  n_epochs of flow training )
n_production_loops × ( n_local_steps local moves  →  n_global_steps flow proposals )
```

All of that happens in `n_chains` chains at once. The samples you keep come from the
production phase, so

$$
N_{\mathrm{samples}} = \texttt{n\_chains} \times \texttt{n\_production\_loops}
\times \texttt{n\_local\_steps}
$$

before thinning — 20 chains, 40 production loops and 300 local steps give 240 000 samples.

```bash
gwk_flowMC_info sampler_cfg.json --n-dims 8
```

summarises what a given configuration implies before you commit to running it: how many
samples each loop contributes, when the rolling training history saturates and the flow
starts forgetting early loops, and a set of heuristic suggestions for `history_window`,
`n_max_examples`, `batch_size`, the loop counts and the flow architecture. It reads the
raw JSON, so it needs a fully populated file rather than one relying on defaults.

### Sampling geometry

`n_chains`
: Default `10`. Chains run simultaneously. The flow is trained on all of them, so more
  chains means better training data as well as more samples; 20 is a reasonable starting
  point for a ten-ish-dimensional problem.

`n_training_loops`
: Default `15`. Loops spent training the flow. Too few and the global proposals never
  become useful; the `loss` dataset in the output should have flattened by the end.

`n_production_loops`
: Default `40`. Loops spent producing samples, after the flow is trained.

`n_local_steps`
: Default `300`. Local-sampler steps per loop.

`n_global_steps`
: Default `20`. Flow proposals per loop.

`chain_batch_size`
: Default `0`, meaning process all chains at once. A positive value processes them in
  batches, trading speed for peak memory — the first thing to reach for when a run does
  not fit on the device.

`local_thinning`, `global_thinning`
: Both default `1`. Thinning applied to local-sampler steps and to flow proposals
  respectively.

`n_NFproposal_batch_size`
: Default `10`. Batch size used when generating proposals from the flow.

### Local sampler

`local_sampler_name`
: Default `"hmc"`. Either `"hmc"` (Hamiltonian Monte Carlo) or `"mala"` (Metropolis-adjusted
  Langevin). HMC is the better default for smooth population likelihoods.

`step_size`
: Default `0.01`, must be positive. The local sampler's step size. Together with
  `condition_matrix` this determines the local acceptance rate, which is the number to
  watch: near zero means steps are too large, near one means they are too small.

`n_leapfrog`
: Default `10`. Leapfrog steps per HMC trajectory. Ignored by MALA.

`condition_matrix`
: Default `1.0`. Either a positive scalar or a **one-dimensional** array of positive
  numbers, one per hyperparameter, giving the per-dimension conditioning of the HMC
  trajectories. See below — this is the field worth getting right.

### `condition_matrix`, at more length

```{admonition} The field that decides whether flowMC works
:class: important

The entries should be the variances of the target along each direction. Population
posteriors routinely span four orders of magnitude between the tightest and loosest
hyperparameter, and with the scalar default the sampler takes steps that are
simultaneously far too large in one direction and far too small in another. A stalled
flowMC run is more often this than anything else.
```

The entries are in the **alphabetical hyperparameter order** that GWKokab uses internally
and that every output product follows; `variables_index` in an output file spells the
ordering out. Measure them with a short pilot run:

```bash
gwk_diag_condition_matrix inference_data.hdf5
```

which prints the per-dimension sample variances, ready to paste in.

Validation catches a two-dimensional array, a non-positive entry, and a non-positive
scalar. It does **not** check the length against the number of hyperparameters, because
the configuration is parsed before the model is built — a wrong-length array is accepted
here and fails later, inside flowMC, with a much less helpful message. Count the entries
against `variables_index` yourself.

### Normalizing flow

The flow is a stack of coupling layers using rational-quadratic splines.

`rq_spline_n_layers`
: Default `10`. Coupling blocks in the flow. More layers model a more contorted target and
  cost more to train.

`rq_spline_hidden_units`
: Default `[128, 128]`. Widths of the conditioning network inside each coupling block.

`rq_spline_n_bins`
: Default `8`. Bins per spline transformation.

`rq_spline_range`
: Default `[-10.0, 10.0]`. The interval over which the splines are active; outside it the
  transformation is the identity. Hyperparameters whose posterior lies well outside this
  box — a mass cut-off around 80, say — get no benefit from the flow, so either widen the
  range or rescale.

`learning_rate`
: Default `0.001`. Adam learning rate for training the flow.

`batch_size`
: Default `1000`. Samples per training batch.

`n_epochs`
: Default `4`. Training epochs per training loop.

`n_max_examples`
: Default `100000`. Cap on the training history kept in memory.

`history_window`
: Default `500`. Size of the rolling window of recent chain history used for training.

`verbose`
: Default `false`. Print progress and loss metrics.

### A complete flowMC configuration

The one used by the [worked examples](../examples/ecc_plus_spin/inference_from_posterior_samples.md#flowmc),
for an eight-hyperparameter model:

```json
{
    "chain_batch_size": 0,
    "n_chains": 20,
    "batch_size": 5000,
    "history_window": 900,
    "n_epochs": 7,
    "n_max_examples": 100000,
    "n_NFproposal_batch_size": 10,
    "n_global_steps": 20,
    "n_local_steps": 300,
    "global_thinning": 1,
    "local_thinning": 1,
    "n_production_loops": 40,
    "n_training_loops": 30,
    "local_sampler_name": "hmc",
    "step_size": 0.01,
    "n_leapfrog": 10,
    "condition_matrix": [
        0.092886444,
        0.15304805,
        0.00029991572,
        0.038987259,
        3.7471383,
        0.18542411,
        0.0011376989,
        0.00084687465
    ],
    "learning_rate": 0.001,
    "rq_spline_hidden_units": [
        128,
        128
    ],
    "rq_spline_n_bins": 8,
    "rq_spline_n_layers": 10,
    "rq_spline_range": [
        -10.0,
        10.0
    ],
    "verbose": false,
    "sampler_name": "flowMC"
}
```

Note the spread in `condition_matrix`: four orders of magnitude between
$\sigma_\epsilon$ at $3\times10^{-4}$ and $m_{\mathrm{max}}$ at $3.7$.

### Tuning a flowMC run

flowMC gives no $\hat{R}$-style verdict, so read the output file instead. It contains three
diagnostics beyond the samples:

`loss`
: The flow's training loss. It should flatten by the end of the training phase. If it is
  still falling, raise `n_training_loops` or `n_epochs`.

`acceptances/local/prod`
: Local-sampler acceptance during production. Near zero means `step_size` is too large or
  `condition_matrix` is badly scaled; near one means the steps are too small to explore.
  Aim for the middle.

`acceptances/global/prod`
: Flow-proposal acceptance. Persistently low here, with a healthy local acceptance and a
  converged loss, means the flow has not learned the target — try more layers, a wider
  `rq_spline_range`, or more training loops.

The order to tune in is `condition_matrix` first, then `step_size`, then the flow
hyperparameters. A pilot run of a few loops is enough to measure the first two.

---

## Reading from HDF5

Every completed run writes the configuration it actually used into `inference_data.hdf5`
under the `sampler_cfg` group, so a result file is self-describing. `SamplerConfig` reads
that back with the same dispatch on `sampler_name`:

```python
from gwkokab.analysis.core.inference_io import SamplerConfig

cfg = SamplerConfig.read_from_hdf5("inference_data.hdf5")
print(cfg.model_dump())
```

This is how to recover the settings of a run whose JSON has been lost or edited since, and
it accepts a path, an open `h5py.File`, or a `h5py.Group`. An HDF5 file whose
`sampler_cfg/@sampler_name` is neither value raises

```text
LoggedValueError: Unsupported or missing sampler_name '...' in HDF5 file.
```

Both concrete classes have their own `read_from_hdf5`, which additionally *checks* that
the file came from the sampler you expected:

```python
from gwkokab.analysis.core.inference_io import FlowMCGlobalConfig, NumpyroGlobalConfig

NumpyroGlobalConfig.read_from_hdf5("numpyro_run.hdf5")   # LoggedValueError on a flowMC file
FlowMCGlobalConfig.read_from_hdf5("flowMC_run.hdf5")     # LoggedValueError on a NumPyro file
```

```text
LoggedValueError: Expected sampler_name 'numpyro', but got 'flowMC'
```

The NumPyro configuration is stored across two attribute groups, `sampler_cfg/kernel` and
`sampler_cfg/mcmc`; the flowMC one is flat, in `sampler_cfg` itself.

Writing is `cfg.write_to_hdf5(path)` — except for flowMC, which also needs the
dimensionality, since that is not part of the configuration:

```python
cfg = SamplerConfig.read_from_json("sampler_cfg.json")
cfg.write_to_hdf5("inference_data.hdf5", n_dims=8)
```

Omitting `n_dims` there raises `LoggedValueError: n_dims must be specified when writing
FlowMC configuration to HDF5.` Reading drops `n_dims` again, so the round trip returns an
object equal to the original.

## Templates

Both samplers have a template generator that writes out every key at its default:

```bash
gwk_numpyro_cfg_template -o sampler_cfg.json
gwk_flowMC_cfg_template  -o sampler_cfg.json
```

## See also

- [Sampler Configurations](../examples/ecc_plus_spin/inference_from_posterior_samples.md#sampler-configurations) —
  both files in the context of a real analysis.
- [NumPyro MCMC documentation](https://num.pyro.ai/en/stable/mcmc.html) — the upstream
  reference for every `kernel` and `mcmc` field.
- [flowMC](https://github.com/kazewong/flowMC) — the upstream reference for the flowMC
  sampler and its hyperparameters.
