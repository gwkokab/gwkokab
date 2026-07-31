# Population Inference from Gaussian Summaries (Analytical Method)

## Introduction

The [discrete method](./inference_from_posterior_samples.md) evaluates each event's contribution to the
likelihood as a sum over its posterior samples. That is exact up to Monte Carlo noise,
but it means every likelihood evaluation touches every sample of every event: for $N$
events with $M$ samples each, the cost scales as $N \times M$, and the samples must be
resident on the device for the whole run.

The **analytical** method replaces each event's posterior with the multivariate normal
fitted by `synthetic_analytical_pe` in the
[first tutorial](./simulating_a_catalogue.md#summarising-the-posteriors-analytically). An event
is then described by a mean vector and a covariance matrix instead of thousands of
samples, and the number of Monte Carlo draws used per event becomes a knob of the
analysis rather than a property of the data.

This tutorial runs that analysis with both samplers, and then puts all four runs — two
methods $\times$ two samplers — next to each other and against the injected truth. It
assumes you have read the previous two tutorials; the model, priors and sensitivity
configuration are carried over unchanged.

## The Likelihood

Write the per-event marginal likelihood as

$$
\int \mathcal{L}_n(\theta)\, \rho(\theta\mid\Lambda)\, \mathrm{d}\theta,
\qquad
\mathcal{L}_n(\theta) \propto \mathcal{N}(\theta \mid \mu_n, \Sigma_n)
$$

where $\mu_n$ and $\Sigma_n$ are the mean and covariance stored in the event file. Since
the approximate likelihood is a distribution we can sample from directly, the natural
Monte Carlo estimate draws $\theta_{n,k} \sim \mathcal{N}(\mu_n, \Sigma_n)$ and averages
$\rho$ over those draws. The normalisation of $\mathcal{N}$ and the $1/K$ from the
average are independent of $\Lambda$, so they drop out of the posterior and GWKokab
simply computes

$$
\ln\mathcal{L}(\Lambda) = -\mu(\Lambda) + N \ln T_{\mathrm{obs}} +
\sum_{n=1}^{N}\ln\sum_{k=1}^{K}
\rho\!\left(\theta_{n,k}\mid\Lambda\right) J_{n,k},
\qquad \theta_{n,k} \sim \mathcal{N}(\mu_n, \Sigma_n)
$$

Two details matter in practice.

- The draws are **rejected and redrawn** until they fall inside the per-event bounding
  box stored in `limits`, so no sample can wander outside the region the original
  posterior actually covered. Events whose Gaussian is wide compared to that box
  therefore take longer to set up.
- $J_{n,k}$ is the Jacobian of an optional coordinate transformation, allowing the
  Gaussian to be fitted in coordinates that are more nearly normal than the ones the
  population model uses. With the default identity transform $J_{n,k} = 1$, which is
  what we use here: the summaries were fitted directly in
  $(m_1, m_2, \chi_{1,z}, \chi_{2,z}, \epsilon)$.

The draws are made once, before sampling starts, and reused for every likelihood
evaluation. This keeps the likelihood smooth in $\Lambda$ — which both samplers depend on
— but it also means the answer is conditioned on that particular set of draws, so `K`
should be large enough that the estimate is stable.

```{admonition} When is this a good idea?
:class: important

The method is exactly as good as the Gaussian approximation to each event's posterior.
`synthetic_analytical_pe` reports the Jensen–Shannon divergence between the true and
approximated marginals precisely so that you can check this; for this data set it is of
order $10^{-3}$ nats, and the two methods agree closely as a result. Strongly
non-Gaussian or multimodal posteriors — railing against a prior boundary, for instance —
will not be captured, and the discrete method should be preferred there.
```

## Configuration

Three of the four configuration files are shared verbatim with the discrete analysis:

- [`pmean_cfg.json`](./simulating_a_catalogue.md#detector-sensitivity) — the selection function
  must be the same one used to generate the catalogue.
- [`prior_cfg.json`](./inference_from_posterior_samples.md#priors) — the same eight free
  hyperparameters, in the same alphabetical order.
- `sampler_cfg.json` — same structure as before; see
  [Sampler Configurations](./inference_from_posterior_samples.md#sampler-configurations).

The fourth,
[`data_loader_cfg.json`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/data_loader_cfg.json),
is also shared, because both loaders need only a glob pattern:

```json
{
    "regex": "../data/event_*.hdf5"
}
```

The analytical loader reads a different part of each file, though. Its optional fields
are

`default_waveform`
: Name of the HDF5 group holding the summary. Defaults to
  `GWKokabSyntheticAnalyticalPE`, which is what `synthetic_analytical_pe` writes.
  Per-event overrides go in `alternate_waveforms`.

`transform_module_path`
: Path to a Python module defining a `Transform` class (a subclass of
  `SampleTransformer`) that maps the coordinates the Gaussian was fitted in to the
  coordinates the population model expects, together with the Jacobian $J_{n,k}$ above.
  Defaults to `null`, i.e. the identity.

`parameter_aliases`
: Mapping from model parameter names to the coordinate names used in the files.

As with the discrete loader, a fully populated template is available:

```bash
gwk_analytical_data_loader_cfg_template -o data_loader_cfg.json
```

The flowMC configuration differs from the discrete one only in `condition_matrix`, which
was re-tuned on a pilot run of *this* likelihood
([`analytical_flowMC/sampler_cfg.json`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/analytical_flowMC/sampler_cfg.json)):

```json
    "condition_matrix": [
        0.054446193,
        0.095683258,
        0.00032538989,
        0.03905882,
        3.0177744,
        0.19384819,
        0.0010991484,
        0.00065443988
    ],
```

The NumPyro configuration
([`analytical_numpyro/sampler_cfg.json`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/analytical_numpyro/sampler_cfg.json))
is identical to the discrete one — NUTS adapts its own mass matrix during warm-up, so
nothing needs re-tuning.

## Running the Analysis

The command is `analytical_n_pls_m_gs`, from
[`analytical_numpyro/analysis.sh`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/analytical_numpyro/analysis.sh):

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR="$HOME/jax_cache" \
    GWKOKAB_LOG_FILE="analytical.log" \
    analytical_n_pls_m_gs \
    --n-pl 1 \
    --n-g 0 \
    --n-samples 10000 \
    --seed $RANDOM \
    --data-loader-cfg "../data_loader_cfg.json" \
    --prior-cfg "../prior_cfg.json" \
    --pmean-cfg "../pmean_cfg.json" \
    --sampler-cfg "./sampler_cfg.json" \
    --add-truncated-normal-spin-z \
    --add-eccentricity-mixture

gwk_report -i inference_data.hdf5 -o analytical_numpyro_report.html
```

[`analytical_flowMC/analysis.sh`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/analytical_flowMC/analysis.sh)
is the same apart from the report name. Run each from its own directory:

```bash
cd analytical_numpyro && bash analysis.sh
cd ../analytical_flowMC && bash analysis.sh
```

Compared with the discrete command, the model flags, the four configuration files and the
seed are unchanged. Two things differ:

- `--n-samples` is the $K$ of the likelihood above: the number of multivariate normal
  draws per event. It defaults to `1000`; we use `10000`.
- There is no `--n-buckets` or `--threshold`. Every event contributes exactly `K` draws,
  so the data is already rectangular and no bucketing is needed. This is one of the
  practical attractions of the method.

### Output

Alongside the familiar `inference_data.hdf5` — same layout as in the
[discrete tutorial](./inference_from_posterior_samples.md#output) — the run writes
`analytical_samples.hdf5`, containing the draws it will reuse for every likelihood
evaluation:

```text
analytical_samples.hdf5
├── event_00
│   ├── samples              # draws in the coordinates the Gaussian was fitted in
│   ├── transformed_samples  # the same draws in model coordinates
│   └── ln_offsets           # ln J for each draw (zero for the identity transform)
├── event_01
│   └── ...
└── ...
```

Keeping this file makes a run reproducible: the same draws, the same likelihood.

```{admonition} A harmless warning
:class: note

The loader looks for an optional per-event `scale` dataset that lets you condition badly
scaled covariance matrices before sampling. `synthetic_analytical_pe` does not write one,
so you will see `'scale' dataset not found ... Defaulting to ones.` once per event. That
is the intended behaviour for this data set.
```

## Comparing the Methods

We now have four runs:

```text
ecc_plus_spin
├── discrete_flowMC/inference_data.hdf5
├── discrete_numpyro/inference_data.hdf5
├── analytical_flowMC/inference_data.hdf5
└── analytical_numpyro/inference_data.hdf5
```

### Hyperparameter posteriors

[`overplot.py`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/overplot.py)
reads the `samples` dataset of two of them and draws a corner plot against the injected
truth:

```python
from gwkokab.analysis.core.utils import read_from_hdf5

DISCRETE = read_from_hdf5("discrete_flowMC/inference_data.hdf5", "samples")
ANALYTICAL = read_from_hdf5("analytical_flowMC/inference_data.hdf5", "samples")
TRUE_VALUES = [1.0, 0.0, 0.15, 4.0, 50.0, 5.0, 0.0, 0.4]
```

The `TRUE_VALUES` list is in the alphabetical hyperparameter order the samplers use,
$(\alpha, \beta, \sigma_\epsilon, \ln\mathcal{R}_0, m_{\mathrm{max}}, m_{\mathrm{min}},
\mu_{\chi_z}, \sigma_{\chi_z})$; if you are ever unsure, read it off the
`variables_index` attributes of the file itself.

```bash
python overplot.py
```

```{image} figs/overplot.png
:alt: Corner plot of the eight hyperparameters for the discrete and analytical methods, both sampled with flowMC, against the injected truth.
:width: 100%
```

All eight hyperparameters are recovered, and the two methods produce posteriors of very
similar width and orientation, including the pronounced $\alpha$–$\beta$ and
$\ln\mathcal{R}_0$–$\alpha$ correlations. The residual offsets — $m_{\mathrm{max}}$
peaking a little above $50\,M_\odot$, $\sigma_{\chi_z}$ a little below $0.4$ — appear in
*both* methods, which is the useful observation: they are properties of this particular
finite catalogue seen through this particular selection function, not artefacts of either
likelihood. $m_{\mathrm{max}}$ in particular is constrained only by the handful of
heaviest events, and is pushed upwards by their measurement uncertainty.

Running the same comparison across all four runs with
[`overplot_different_samplers.py`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/overplot_different_samplers.py)

```bash
python overplot_different_samplers.py
```

```{image} figs/overplot_flowMC_numpyro.png
:alt: Corner plot of the eight hyperparameters for all four runs, two methods times two samplers, against the injected truth.
:width: 100%
```

separates the two effects. NumPyro and flowMC land on the same posterior for a given
method, so the sampler is not the limiting factor; the visible spread between the four
contours is dominated by the choice of method, and even that is small compared with the
statistical width of each posterior. The largest single displacement is in
$\ln\mathcal{R}_0$ for the analytical/flowMC run, and it is well inside the $1\sigma$
contour of the others.

### Population marginals

Corner plots compare hyperparameters; what we usually want to show is the *population*
they imply. That is a two-step process.

First, evaluate the model on a grid for a subset of posterior samples with
[`generate_probs.py`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/generate_probs.py):

```python
from gwkokab.analysis.n_pls_m_gs.common import NPowerlawMGaussianCore
from gwkokab.analysis.utils.marginals import generate_marginal_probs
from gwkokab.parameters import Parameters as P

generate_marginal_probs(
    model_meta_cls=NPowerlawMGaussianCore,
    input_file_path=f"{analyses_dir}/inference_data.hdf5",
    output_file_path=f"{analyses_dir}/marginal_probs.hdf5",
    domain_cfg={
        P.PRIMARY_MASS_SOURCE: (1.0, 60.0, 500),
        P.SECONDARY_MASS_SOURCE: (1.0, 60.0, 500),
        P.PRIMARY_SPIN_Z: (-1.0, 1.0, 200),
        P.SECONDARY_SPIN_Z: (-1.0, 1.0, 200),
        P.ECCENTRICITY: (0.0, 1.0, 200),
    },
    max_samples=10_000,
    batch_size=100,
)
```

Each entry of `domain_cfg` is a `(low, high, n_points)` triple. For every one of
`max_samples` hyperparameter samples the joint density is built on that grid and
integrated down to one dimension at a time, in batches of `batch_size` to keep memory
bounded. The result, `marginal_probs.hdf5`, is written in the
[`popsummary`](https://pypi.org/project/popsummary/) format:

```text
marginal_probs.hdf5
└── posterior
    ├── hyperparameter_samples                      # the samples used, plus constants
    └── rates_on_grids
        ├── component_0_mass_1_source
        │   ├── positions                           # the grid
        │   └── rates                               # one marginal per sample
        ├── component_0_mass_2_source
        ├── component_0_spin_1z
        ├── component_0_spin_2z
        └── component_0_eccentricity
```

Second, plot those marginals with their credible bands using
[`marginal_plots.py`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/marginal_plots.py):

```python
from gwkokab.analysis.utils.marginals import plot_marginal_with_intervals, PlotStyle

plot_marginal_with_intervals(
    ax=ax_m1,
    filename=f"{analysis.path}/marginal_probs.hdf5",
    parameter=P.PRIMARY_MASS_SOURCE,
    component_idxs=[0],
    scale=lambda p: np.exp(p["log_rate_0"]),
    style=PlotStyle(color=..., label=...),
)
```

`plot_marginal_with_intervals` draws the median marginal as a line and the 5th–95th
percentile range as a band. `component_idxs` selects which mixture components to add
together — there is only one here — and `scale` converts a normalised density into a rate
density; passing `np.exp(p["log_rate_0"])` turns $p(m_1)$ into
$\mathrm{d}\mathcal{R}/\mathrm{d}m_1$, propagating the rate posterior into the band. The
spin and eccentricity panels omit `scale` and stay normalised.

```bash
python generate_probs.py
python marginal_plots.py
```

```{image} figs/marginals.png
:alt: Marginal rate densities in primary mass, secondary mass, aligned spin and eccentricity for all four runs, with the injected distributions overlaid.
:width: 100%
```

The dashed black curves are the injected distributions, computed directly from
`PowerlawPrimaryMassRatio` and the two truncated normals with the true hyperparameters.
All four analyses track them over the full range, and — as in the corner plots — where
they deviate, they deviate together: the secondary mass rate is slightly under-predicted
above $\sim 30\,M_\odot$, and the recovered spin distribution is marginally narrower than
the truth, the direct consequence of the small $\sigma_{\chi_z}$ offset seen earlier.

## Summary

| | Discrete | Analytical |
| --- | --- | --- |
| Per-event data | all posterior samples | mean, covariance, bounding box |
| Cost per likelihood call | $\sum_n M_n$ density evaluations | $N \times K$ density evaluations |
| Needs bucketing | yes, events have unequal sample counts | no |
| Approximation | Monte Carlo noise only | Gaussian posteriors, plus Monte Carlo noise |
| Fails when | too few samples per event | posteriors are non-Gaussian |

For this catalogue the two agree to well within their statistical uncertainties, which
is exactly the result you want before trusting the cheaper method on a larger data set.
The check to run first on new data is always the Jensen–Shannon divergence reported by
`synthetic_analytical_pe`: if the Gaussian describes the posteriors, the analytical
method buys you speed for free; if it does not, no amount of sampling will repair it.

---

## See Also

- [Simulating a Gravitational-Wave Catalogue](./simulating_a_catalogue.md)
- [Population Inference from Posterior Samples](./inference_from_posterior_samples.md)
- [Expected Number of Detections and Sensitivity Estimation](../sensitivity.md)
