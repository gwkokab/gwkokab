# Simulating a Gravitational-Wave Catalogue

## Introduction

This is the first of three tutorials that walk through a complete, self-contained
population-inference study in GWKokab. We will

1. define a population of binary black holes (BBHs) whose component masses, aligned
   spins and orbital eccentricities are all drawn from a known model,
2. draw a synthetic catalogue of *detected* events from it, given a detector
   sensitivity,
3. attach realistic measurement uncertainties to every event, producing mock posterior
   samples ("discrete" parameter estimates),
4. summarise those posterior samples by a multivariate normal ("Analytical GWalk" parameter
   estimates).

The two later tutorials then recover the population hyperparameters from this data set:

- [Population Inference from Posterior Samples](./inference_from_posterior_samples.md)
  uses the posterior samples directly.
- [Population Inference from Gaussian Summaries](./inference_from_gaussian_summaries.md)
  uses the Gaussian summaries, and compares the two methods.

Because the truth is known by construction, every number recovered later can be checked
against the value we put in here.

Everything in this tutorial is driven by pre-defined command line interfaces (CLIs) that
ship with GWKokab; no Python needs to be written. The three commands are collected in
[`synthetic_data.sh`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/synthetic_data.sh),
and are explained one at a time below.

## The Population Model

GWKokab defines niche[^1] models as subclasses of
[`numpyro.distributions.Distribution`](https://num.pyro.ai/en/stable/distributions.html#numpyro.distributions.distribution.Distribution);
everything else is imported directly from NumPyro. The primary mass and mass ratio are
jointly modelled by
[`PowerlawPrimaryMassRatio`](https://gwkokab.readthedocs.io/en/latest/autoapi/gwkokab/models/mass/index.html#gwkokab.models.mass.PowerlawPrimaryMassRatio),

$$
    p(m_1,q\mid \alpha, \beta, m_{\mathrm{min}}, m_{\mathrm{max}}) \propto
    m_1^{-\alpha} q^{\beta}, \qquad
    m_{\mathrm{min}} \leq m_1 \leq m_{\mathrm{max}}, \quad
    \frac{m_{\mathrm{min}}}{m_1} \leq q \leq 1
$$

Population inference in this example is carried out in component masses, so we change
variables from $(m_1, q)$ to $(m_1, m_2)$ using $q = m_2/m_1$. The Jacobian of that map
is $m_1$, hence

$$
    p(m_1,m_2\mid \alpha, \beta, m_{\mathrm{min}}, m_{\mathrm{max}}) =
    \frac{1}{m_1}p(m_1,q\mid \cdot) \propto m_1^{-(\alpha+1)} q^{\beta},
    \qquad m_{\mathrm{min}} \leq m_2 \leq m_1 \leq m_{\mathrm{max}}
$$

Both aligned spin components are drawn independently from the same truncated normal
distribution on $[-1, 1]$,

$$
    p(\chi_{i,z}\mid \mu_{\chi_z}, \sigma_{\chi_z}) \propto
    \exp\left[-\frac{(\chi_{i,z} - \mu_{\chi_z})^2}{2\sigma_{\chi_z}^2}\right],
    \qquad -1 \leq \chi_{i,z} \leq 1, \quad i \in \{1, 2\}
$$

and the orbital eccentricity at the reference frequency is a truncated normal on
$[0, 1]$ pinned at zero, so that $\sigma_{\epsilon}$ alone controls how eccentric the
population is,

$$
    p(\epsilon\mid \sigma_{\epsilon}) \propto
    \exp\left[-\frac{\epsilon^2}{2\sigma_{\epsilon}^2}\right],
    \qquad 0 \leq \epsilon \leq 1
$$

All five parameters are independent, and the whole population is scaled by a single
merger rate $\mathcal{R}_0$:

$$
\rho(m_1,m_2, \chi_{1,z}, \chi_{2,z}, \epsilon\mid\Lambda) = \mathcal{R}_0\,
p(m_1,m_2\mid \alpha, \beta, m_{\mathrm{min}}, m_{\mathrm{max}})\,
p(\chi_{1,z}\mid \mu_{\chi_z}, \sigma_{\chi_z})\,
p(\chi_{2,z}\mid \mu_{\chi_z}, \sigma_{\chi_z})\,
p(\epsilon\mid \sigma_{\epsilon})
$$

where $\Lambda = \{\mathcal{R}_0, \alpha, \beta, m_{\mathrm{min}}, m_{\mathrm{max}},
\mu_{\chi_z}, \sigma_{\chi_z}, \sigma_{\epsilon}\}$ is the vector of population
hyperparameters we will later try to recover.

````{admonition} Why $\rho$ and not $p$?
:class: note

The merger rate $\mathcal{R}_0$ destroys the normalisation, so $\rho$ is a *rate
density* rather than a probability density. This is deliberate: the number of detections
is what carries the information about $\mathcal{R}_0$.
````

### Model Specification

The `n_pls_m_gs` family of CLIs builds a mixture of $N_{\mathrm{pl}}$
`PowerlawPrimaryMassRatio` components and $N_{\mathrm{g}}$ bivariate truncated normal
components in mass, each with its own merger rate:

$$
\rho(m_1, m_2 \mid \Lambda) =
\sum_{i=0}^{N_{\mathrm{pl}}-1} \frac{\mathcal{R}_i}{m_1}
p(m_1, q\mid \alpha_i,\beta_i,m_{\mathrm{min},\mathrm{pl},i},m_{\mathrm{max},\mathrm{pl},i}) +
\sum_{j=0}^{N_{\mathrm{g}}-1} \mathcal{R}_{j+N_{\mathrm{pl}}}
\mathcal{N}_{[\cdot,\cdot]}(m_1 \mid \mu_{1,j}, \sigma_{1,j})
\mathcal{N}_{[\cdot,\cdot]}(m_2 \mid \mu_{2,j}, \sigma_{2,j})
$$

Additional parameters (spins, eccentricity, redshift, sky location, …) are switched on
with `--add-*` flags and are multiplied into every mixture component. Our model needs
exactly one power-law component and no Gaussian component, plus aligned spins and
eccentricity, which corresponds to

```bash
--n-pl 1 --n-g 0 --add-truncated-normal-spin-z --add-eccentricity-mixture
```

`--add-truncated-normal-spin-z` adds an independent truncated normal for `spin_1z` and
`spin_2z`. `--add-eccentricity-mixture` adds a **two**-component truncated normal
mixture,

$$
p(\epsilon \mid \cdot) = (1-\zeta)\,
\mathcal{N}_{[\ell_1, h_1]}(\epsilon\mid \mu_1, \sigma_1) + \zeta\,
\mathcal{N}_{[\ell_2, h_2]}(\epsilon\mid \mu_2, \sigma_2)
$$

which reduces to the single truncated normal of the previous section when the mixing
weight $\zeta$ is fixed to $0$. The second component is then unreachable, but its
parameters still have to be supplied because the distribution object is always
constructed.

Model parameters follow the naming pattern
`<parameter_name>_<model_parameter_name>_<component_type>_<component_index>`, where
`component_type` is `pl` for a power-law component and `g` for a Gaussian one. Mass
model parameters (`alpha_pl_0`, `beta_pl_0`, `mmin_pl_0`, `mmax_pl_0`) and the merger
rates (`log_rate_0`, given as a natural logarithm) are exceptions to that rule.

Values are supplied through a JSON file, where **every key is a regular expression**
matched against the parameter names the model asks for. That keeps the file short when
several parameters share a value. Our
[`model.json`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/model.json)
is

```json
{
    "N_g": 0,
    "N_pl": 1,
    "log_rate_0": 4.0,
    "alpha_pl_[0-9]+": 1.0,
    "beta_pl_[0-9]+": 0.0,
    "mmax_pl_[0-9]+": 50.0,
    "mmin_pl_[0-9]+": 5.0,
    "spin_[1-2]z_high_(g|pl)_[0-9]+": 1.0,
    "spin_[1-2]z_low_(g|pl)_[0-9]+": -1.0,
    "spin_[1-2]z_loc_(g|pl)_[0-9]+": 0.0,
    "spin_[1-2]z_scale_(g|pl)_[0-9]+": 0.4,
    "eccentricity_comp[1-2]_high_(g|pl)_[0-9]+": 1.0,
    "eccentricity_comp[1-2]_low_(g|pl)_[0-9]+": 0.0,
    "eccentricity_comp[1-2]_loc_(g|pl)_[0-9]+": 0.0,
    "eccentricity_comp[1-2]_scale_(g|pl)_[0-9]+": 0.15,
    "eccentricity_zeta_(g|pl)_[0-9]+": 0.0
}
```

which corresponds to the following injected truth:

| Symbol                | Key                             | Value  |
| --------------------- | ------------------------------- | ------ |
| $\ln\mathcal{R}_0$    | `log_rate_0`                    | $4.0$  |
| $\alpha$              | `alpha_pl_0`                    | $1.0$  |
| $\beta$               | `beta_pl_0`                     | $0.0$  |
| $m_{\mathrm{min}}$    | `mmin_pl_0`                     | $5.0$  |
| $m_{\mathrm{max}}$    | `mmax_pl_0`                     | $50.0$ |
| $\mu_{\chi_z}$        | `spin_1z_loc_pl_0`              | $0.0$  |
| $\sigma_{\chi_z}$     | `spin_1z_scale_pl_0`            | $0.4$  |
| $\sigma_{\epsilon}$   | `eccentricity_comp1_scale_pl_0` | $0.15$ |
| $\zeta$               | `eccentricity_zeta_pl_0`        | $0.0$  |

````{admonition} N_pl and N_g in the JSON file
:class: tip

`N_pl` and `N_g` are listed for readability only. The CLI always overrides them with the
values of `--n-pl` and `--n-g`, so keep the two in sync to avoid confusing yourself
later.
````

## Detector Sensitivity

Drawing a *detected* population requires a model of the detectors' selection function.
GWKokab expresses this through the expected number of detections,

$$
\mu(\Lambda) = T_{\mathrm{obs}} \int \rho(\theta \mid \Lambda)\,
\mathrm{VT}(\theta)\, \mathrm{d}\theta
$$

and offers several estimators for it. This example uses a small multilayer perceptron
trained to evaluate the sensitive spacetime volume $\mathrm{VT}(\theta)$ over
$(m_1, m_2, \chi_{1,z}, \chi_{2,z})$. The estimator is configured in
[`pmean_cfg.json`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/pmean_cfg.json):

```json
{
    "estimator_type": "neural_vt",
    "filename": "../neural_calibrated_vt_aLIGO140MpcT1800545_BBH+BNS+NSBH_fine_m1_m2_s1z_s2z.hdf5",
    "time_scale": 1.0,
    "num_samples": 2000,
    "batch_size": 2000
}
```

`time_scale` is $T_{\mathrm{obs}}$, `num_samples` is the number of Monte Carlo draws used
to evaluate the integral above and `batch_size` controls how many of them are pushed
through the network at once. See
[`PoissonMeanEstimationLoader`](../../inference_io/poisson_mean_estimation_loader.md) for
these fields in full and for the three other estimator types — a $p_{\det}$ surrogate, an
injection campaign, or your own function — and
[Training a Multilayer Perceptron](../training_mlp.md) for how such a network is
produced.

```{admonition} Relative paths in configuration files
:class: warning

`filename` is resolved relative to the **working directory of the command**, not
relative to the JSON file. The analyses in the next two tutorials are launched from
sub-directories, so either keep a copy of the `VT` file where each command runs, or
simply use an absolute path.
```

## Generating a Synthetic Catalogue

We now have everything needed for the first command:

```bash
JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GWKOKAB_LOG_FILE="synthetic_events.log" \
    synthetic_events_n_pls_m_gs \
    --seed $RANDOM \
    --n-pl 1 \
    --n-g 0 \
    --pmean-cfg pmean_cfg.json \
    --model-params model.json \
    --add-truncated-normal-spin-z \
    --add-eccentricity-mixture \
    --derive-parameters
```

- `--seed` seeds the random number generator; use a fixed integer instead of `$RANDOM`
  if you want the catalogue to be reproducible.
- `--model-params` points at the JSON file described above.
- `--pmean-cfg` points at the sensitivity configuration.
- `--derive-parameters` computes every parameter that can be derived from the sampled
  ones, using GWKokab's relation mesh. From $(m_1, m_2)$ it derives the chirp mass,
  symmetric mass ratio, mass ratio, total mass and so on. We need this because the
  measurement uncertainty in the next step is applied in
  $(\mathcal{M}_c, \eta)$ coordinates.
- `--output-filename` defaults to `synthetic_events.hdf5`.
- `--n-buffer-events` (default `10000`) is the size of the pool of events drawn from the
  population *before* selection effects are applied. If the number of detections comes
  out close to the buffer size the CLI will warn you; increase it in that case.

The generator draws `n_buffer_events` events from $\rho$, evaluates $\mathrm{VT}$ for
each of them, and resamples without replacement in proportion to those weights. The
number of surviving events is itself a Poisson draw with mean $\mu(\Lambda)$, so the
catalogue size varies from seed to seed.

The environment variables are optional. `JAX_PLATFORMS=cpu` keeps this cheap step off
the GPU, the two `XLA_PYTHON_CLIENT_*` variables stop JAX from pre-allocating the whole
device memory, and `GWKOKAB_LOG_FILE` redirects the (rather verbose) log to a file.

The result is a single HDF5 file:

```text
synthetic_events.hdf5
├── events            # detected population, one row per event
├── indices           # mixture component each detected event came from
├── buffer_events     # the full pre-selection pool
├── buffer_indices    # mixture component of each buffer event
├── resample_indices  # which buffer events were selected
└── resample_prob     # VT weights used for the resampling
```

`events` and `buffer_events` are structured arrays whose field names are also stored in
the file-level attribute `parameters`. Keeping the buffer around is useful: comparing
`buffer_events` with `events` shows the selection function at work. A quick look at any
field of either dataset is one command away:

```bash
gwk_hist -i synthetic_events.hdf5 -d events -v mass_1_source -o m1.png --density
```

`-i` takes several files at once, which is handy for overlaying realizations generated
with different seeds.

## Adding Measurement Uncertainties

Real catalogues do not come with true parameter values, they come with posterior
samples. The second command turns each injection into a mock posterior.

Masses are perturbed with the *banana* error model of section 3 of
{cite:p}`10.1093/mnras/stw2883`, which adds noise in chirp mass and symmetric mass ratio,

$$
\mathcal{M}_{c} = \mathcal{M}_{c}^{T}
\left[1+\beta_{\mathcal{M}_c}\frac{12}{\rho_{\mathrm{SNR}}}\left(r_{0}+r\right)\right],
\qquad
\eta = \eta^{T}
\left[1+0.03\frac{12}{\rho_{\mathrm{SNR}}}\left(r_{0}^{\prime}+r^{\prime}\right)\right]
$$

where $r_0, r, r_0^{\prime}, r^{\prime}$ are standard normal draws ($r_0$ and
$r_0^{\prime}$ once per event, $r$ and $r^{\prime}$ once per sample) and the widths are
tuned by `scale_Mc` and `scale_eta`. The characteristic banana-shaped degeneracy in the
$m_1$-$m_2$ plane comes out of the change of variables back to component masses. Spins
and eccentricity get a truncated normal error of width `scale`, reflected back into
their physical bounds when a sample falls outside. Every error scales as
$1/\rho_{\mathrm{SNR}}$, and the SNR of each event is drawn from the usual
$p(\rho_{\mathrm{SNR}}) \propto \rho_{\mathrm{SNR}}^{-4}$ distribution above a threshold
of $9$, so bright events end up with tighter posteriors.

The widths live in
[`error_params.json`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/error_params.json),
whose keys are again regular expressions:

```json
{
    "spin_[1-2]z_high": 1.0,
    "spin_[1-2]z_low": -1.0,
    "spin_[1-2]z_scale": 0.1,
    "eccentricity_high": 1.0,
    "eccentricity_low": 0.0,
    "eccentricity_scale": 0.1,
    "scale_Mc": 1.0,
    "scale_eta": 1.0
}
```

The command is

```bash
JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GWKOKAB_LOG_FILE="discrete.log" \
    synthetic_discrete_pe \
    --seed $RANDOM \
    --filename synthetic_events.hdf5 \
    --error-params error_params.json \
    --size 5000 \
    --coords=chirp_mass,symmetric_mass_ratio,spin_1z,spin_2z,eccentricity \
    --derive-parameters
```

- `--filename` is the catalogue produced by the previous step.
- `--size` is the number of posterior samples generated per event.
- `--coords` selects the coordinates the error model is applied in. The banana error is
  registered against the *pair* `chirp_mass, symmetric_mass_ratio`, so both have to be
  listed for it to trigger; `spin_1z`, `spin_2z` and `eccentricity` each get the
  truncated normal error.
- `--derive-parameters` converts the perturbed $(\mathcal{M}_c, \eta)$ samples back to
  component masses (and everything else derivable) so that the posteriors are expressed
  in the same coordinates as the population model.
- `--dataset` (default `events`) can be set to `buffer_events` to generate posteriors
  for the pre-selection pool instead.

Output goes to a `data/` directory, one HDF5 file per event:

```text
data
├── event_00.hdf5
├── event_01.hdf5
└── ...
```

Each file carries the event's SNR in the attribute `rho`, the list of parameters in
`parameters`, and a group named after the waveform:

```text
event_00.hdf5
└── GWKokabSyntheticDiscretePE
    ├── approximant
    ├── injection_data      # the true values, for reference
    └── posterior_samples   # the mock posterior, shape (size, n_parameters)
```

```{admonition} Samples containing NaNs
:class: note

The banana error can push $\mathcal{M}_c$ or $\eta$ outside their physical range; those
samples are dropped, so individual events may end up with slightly fewer than `--size`
samples. If *every* sample of an event is invalid the CLI warns and writes no file for
it.
```

## Summarising the Posteriors Analytically

The third command reads each event file back, fits a multivariate normal to its
posterior samples and stores the summary alongside them. This is what the analytical
GWalk analysis will consume instead of the samples themselves.

```bash
for event_file in $(ls data/event_*.hdf5); do

    JAX_PLATFORMS=cpu \
        XLA_PYTHON_CLIENT_ALLOCATOR=platform \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        GWKOKAB_LOG_FILE="$event_file.log" \
        synthetic_analytical_gwalk_pe "$event_file" \
        --coords=mass_1_source,mass_2_source,spin_1z,spin_2z,eccentricity \
        --seed $RANDOM
done
```

`--coords` picks the sub-space to summarise; here it is exactly the five parameters of
the population model. `--discrete-waveform` (default `GWKokabSyntheticDiscretePE`)
selects the group the samples are read from.

The summary is written back into the same file under a new group:

```text
event_00.hdf5
├── GWKokabSyntheticDiscretePE
│   └── ...
└── GWKokabSyntheticAnalyticalGWalkPE
    ├── approximant
    ├── mu       # sample mean, shape (n_coords,)
    ├── std      # marginal standard deviations
    ├── cov      # covariance matrix, shape (n_coords, n_coords)
    ├── cor      # correlation matrix
    └── limits   # per-coordinate (min, max) of the samples
```

To tell you how good the Gaussian approximation is, the CLI draws a fresh set of samples
from the fitted normal and reports the Jensen-Shannon (JS) divergence between the true
and the approximated one-dimensional marginals. The per-coordinate values are stored in
the group attribute `js_divs`, together with `mean_js_div`, `min_js_div` and
`max_js_div`, and they are printed to the log:

```text
Mean JS divergence: 0.001782
Min  JS divergence: 0.000544
Max  JS divergence: 0.003248
```

JS divergence is measured in nats here and is bounded by $\ln 2 \approx 0.693$, so values
of order $10^{-3}$ mean the Gaussian is an excellent description of these mock
posteriors. Keep an eye on this number when you move to real data: the Analytical GWalk method
is only as good as this approximation.

## Putting It Together

All three steps are collected in
[`synthetic_data.sh`](https://github.com/kokabsc/gwkokab/blob/main/docs/source/examples/ecc_plus_spin/synthetic_data.sh).
Run it from the `ecc_plus_spin` directory:

```bash
bash synthetic_data.sh
```

When it finishes you should have

```text
ecc_plus_spin
├── synthetic_events.hdf5
├── synthetic_events.log
├── discrete.log
└── data
    ├── event_00.hdf5
    ├── event_01.hdf5
    └── ...
```

Every event file now contains both a discrete and an Analytical GWalk representation of the
same posterior, which is precisely what lets the next two tutorials run the two
inference methods on identical data.

---

## Next Steps

- [Population Inference from Posterior Samples](./inference_from_posterior_samples.md)
- [Population Inference from Gaussian Summaries](./inference_from_gaussian_summaries.md)

## References

```{bibliography}
:filter: docname in docnames
```

[^1]: A niche model is a model that is specific to population inference of compact
      binary coalescences and not available in NumPyro.
