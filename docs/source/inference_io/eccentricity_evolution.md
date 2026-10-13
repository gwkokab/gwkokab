# Eccentricity at a common source-frame frequency

`DiscretePELoader` can optionally transport each **native** eccentricity sample to a
common source-frame quadrupole GW frequency and transform its PE reference prior in the
same operation. The likelihood already divides by that prior; no likelihood rewrite,
extra posterior weights, or separately converted sample files are needed.

This is a **quick global ansatz**: identify the model-native EOB eccentricity with a
Newtonian eccentricity and use orbit-averaged quadrupole evolution. It does not make
SEOBNRv5EHM and TEOBResumS-Dali eccentricities physically identical, evolve anomaly,
include spin/strong-field corrections, or compute a waveform-based eccentricity.

## Explicit native conventions

Set `default_eccentricity_evolution` only when every event has the same audited native
frequency and prior. Mixed catalogs should use `alternate_eccentricity_evolution`, keyed
by **full input filename stem**, with a complete configuration for every loaded event.
Missing events, misspelled override keys, or different target frequencies/methods fail
validation rather than mix eccentricity definitions silently. Conventions are revalidated
at load time as well, including changes made through mutable mappings or copied models.
Eccentricity and redshift must map to distinct input columns. Leaving both fields absent
retains the previous loader behavior.

Example for an event whose native PE prior was uniform on [0, 0.9], conditioned on
native e <= 0.5, at detector-frame 20 Hz:

```json
{
    "regex": "native_samples/event_*.h5",
    "default_datasets": ["/samples"],
    "default_eccentricity_evolution": {
        "native_frequency_hz": 20.0,
        "native_frequency_frame": "detector",
        "reference_frequency_source_hz": 10.0,
        "method": "peters",
        "native_prior": "uniform",
        "native_prior_min": 0.0,
        "native_prior_max": 0.9,
        "condition_native_max": 0.5
    },
    "alternate_eccentricity_evolution": {
        "event_low_frequency": {
            "native_frequency_hz": 5.0,
            "native_frequency_frame": "detector",
            "reference_frequency_source_hz": 10.0,
            "method": "peters",
            "native_prior_min": 0.0,
            "native_prior_max": 0.5,
            "condition_native_max": 0.5
        }
    }
}
```

Replace the dataset, filenames and conventions with the actual run settings, and retain
the existing mass, spin, distance-prior and parameter-alias settings. This example is not
a complete population-analysis configuration. The loader currently supports **uniform
native eccentricity priors only**; other prior names are rejected.

`native_frequency_hz` and `native_frequency_frame` are required: the loader does not
infer them from `f_start`, `f_ref`, the likelihood cutoff, or a catalog-wide default.
Audit the waveform interface: the inspected TEOB GWSignal wrapper uses `f22_start` for
its eccentric initial conditions, while the inspected SEOB interface uses `f22_ref`.
These coincide in the audited O3 runs, but are separate parameters in general. A generic
PESummary `f_start` can reflect the likelihood cutoff instead of the waveform start.

The target frequency defaults to **10 Hz source frame**, and the method defaults to
`peters`. Explicit `native_frequency_frame: "source"` is also supported; it omits the
redshift rescaling of the native frequency. The population's requested coordinates must
include both `eccentricity` and `redshift`, retaining their induced dependence. Their
column names honor `parameter_aliases`. No redshift is regenerated from distance or a
new cosmology; the mass/redshift columns remain untouched.

## Transformation and the denominator prior

For each joint row with redshift z,

$$
R=\frac{f_*}{(1+z)f_{0,\mathrm{det}}}.
$$

The full Peters inverse solves

$$
\frac{g(e_*)}{g(e_0)}=R^{-2/3},\qquad
 g(e)=\frac{e^{12/19}}{1-e^2}
       \left(1+\frac{121}{304}e^2\right)^{870/2299}.
$$

See Peters, Physical Review 136, B1224 (1964),
[DOI](https://doi.org/10.1103/PhysRev.136.B1224), ADS key `1964PhRv..136.1224P`.
The NumPy helper `gwkokab.utils.eccentricity.evolve_eccentricity` solves in log(e),
including an exact circular limit and very low eccentricities, before data enter JAX.
Solutions whose returned floating-point eccentricity fails the Peters relation are
rejected, including extremely near-unity inversions that cannot be resolved reliably.
It returns `(e_reference, log_inverse_jacobian)` with

$$
J=\left|\frac{\partial e_0}{\partial e_*}\right|_z
  =\frac{[d\ln g/de](e_*)}{[d\ln g/de](e_0)}.
$$

The alternative `method: "small-e"` uses e* = e0 R^(-19/18) and log J = (19/18) log R.
It fails with an actionable error if the transported value reaches unity; values are
never clipped. Choose the full Peters prescription for the whole analysis as a backstop
when the power law becomes unreliable, rather than switching methods sample by sample.

For a native uniform prior on [a, b], the loader adds

$$
\ln\pi_e(e_*\mid z)=\ln J-\ln(b-a)
$$

to the existing mass/spin/distance log reference-prior factors. This is a conditional
factor, not a replacement for the entire PE prior. No Jacobian is applied again as a
posterior sample weight. Support is the image of the native interval under the map and
therefore depends on z, even when a and b are fixed.

If `condition_native_max` is set, the loader first removes whole native rows above that
bound, normalizes the native prior on the reduced interval, and then transports and
subsamples whole joint rows with the existing deterministic seed. The native cut is
**not** a source-reference cut. Update population-support hyperparameters deliberately:
keeping an old `eccentricity_high = 0.5` after transport imposes a different restriction.
A source-reference population may extend only where the event PE likelihoods are
actually covered. For example, native e<=0.5 at detector 5 Hz and z=0 maps to an
upper eccentricity below 0.5 at source 10 Hz. Widening the source population bound to 1
does not supply the missing event likelihood outside that transformed interval.
Importance sampling then computes a truncated integral and can bias population shape
or rate. A correct Jacobian does not repair missing support.

Before inference, audit the candidate population's **full joint eccentricity/redshift
support** against every event's transported PE support. Use a covered population
hypothesis and consistent selection treatment, or obtain broader PE samples/reruns.
This loader cannot certify coverage for arbitrary population model families. Native
conditioning is a declared restriction, not evidence that the event likelihood is
physically zero outside that interval. Conditioning also changes event likelihood
normalization; use the same conditioned hypothesis consistently in comparisons.

`EccentricityEvolutionConfig.reference_prior_support(redshift)` returns transported
native-prior bounds for a supplied redshift array. Use it over the population's redshift
domain, not just the PE posterior. After loading, `loader.eccentricity_diagnostics`
reports each file's native/conditioned row counts, observed redshift range, and
transformed prior-bound ranges over that observed range. These diagnostics are also
saved under `data_loader_cfg` as `eccentricity_support_diagnostics` JSON. **Posterior-
range diagnostics are not a population-coverage certificate**: likelihood tails may
extend beyond the observed redshift range. A run must resolve this audit before using
population support outside the covered domain.

The HDF5 input is never changed. The transformed column exists only in the loader's
copy. The configured conventions, native file paths and data seed are saved as JSON in
`inference_data.hdf5` under `data_loader_cfg` attributes when a discrete analysis runs.

## Prepared files and double conversion

Use either native PESummary release datasets or the preserved `native_posterior_samples`
table of a prepared conversion file. The latter still contains native eccentricity and
is valid input to this feature. Do not feed the prepared `derived_samples` table through
the conversion again: the loader rejects its source-frequency metadata, an existing
`eccentricity_source_ref` column, or an already transformed DataFrame. Arbitrary external
files without such metadata cannot be identified reliably; retain honest provenance and
never relabel an already transformed eccentricity column as native.

## Zee's O3/O4 analyses and selection effects

The active `discrete_multisource` and `discrete_subpopulation` analyses need only an
updated `--data-loader-cfg`, their intended event selection, and deliberate source-
reference population support. The conversion works for either sampler backend. The
Gaussian-summary/analytical loaders are not changed: the nonlinear Peters map cannot be
applied to a Gaussian summary by changing its mean alone.

The independently converted O3 reference products and audited manifest are at CIT:

```text
/home/richard.oshaughnessy/tmp_Zee/O3_eccentric_referee_comment3/sample_index.json
/home/richard.oshaughnessy/tmp_Zee/O3_eccentric_referee_comment3/revised_source10Hz_peters
```

They cover **76 O3 events per waveform**, whereas the active BBH input list includes
153 events spanning O1/O2, O3 and O4a. Audit the remaining inputs' native conventions
before configuring a full-catalog run. Do not assume that every event uses 20 Hz or a
uniform prior with the same upper bound. When translating the manifest into per-file
configurations, match event/model identities and verify source-file provenance, not
just directory order. The required fields come from `f_native_detector_hz`,
`native_e_prior_min`, `native_e_prior_max`, and the explicit detector-frame convention.

The existing `ignore_ecc_from_mu.py` custom selection estimator removes eccentricity
and uses circular sensitivity injections. That **eccentricity-independent selection
approximation remains unchanged** and requires no eccentricity injection Jacobian.
This patch does not make it an eccentric selection calculation. Its hard-coded marginal
index should be checked whenever the population parameter order changes.

For eccentricity-dependent injections, use the same transport and transform the
injection sampling density by J; this is not automatically enabled by the PE-loader
option. For a neural selection model trained at a native eccentricity reference, use a
validated coordinate wrapper or retrain it. Passing e* directly to an estimator trained
on e0 is inconsistent. The PE option does not assert or silently alter selection-model
conventions.

## Branch and checks

This implementation starts from upstream `main` at `1ee362a`; upstream no longer
publishes `dev`. The fork's old `dev` is left untouched. Zee's older analysis settings
must also account for the upstream flowMC 0.6.1 and sampler-config changes when upgrading;
this feature does not promise an otherwise unchanged old development environment.

Tests cover reference-prior normalization, round trips, numerical inverse Jacobians,
circular/very-low-e limits, whole-row alignment and conditioning, unchanged disabled
behavior, per-event coverage, double-conversion rejection and likelihood coordinate
invariance. Real O3 checks compare the internal loader against independently generated
Peters products for both waveform models. This is validation of the implementation,
not a completed population rerun or a claim of astrophysical robustness.

Validation used CIT CPU mode with Python 3.13.13, JAX/jaxlib 0.11.2, NumPyro 0.22.0,
flowMC 0.6.1 and `JAX_ENABLE_X64=1`: 289 eccentricity, inference-I/O and likelihood tests passed, including adversarial
configuration, aliasing, numerical-resolution and provenance regressions.
The real-event cross-checks retained 77,688 GW190521 SEOB rows and 79,956 GW190527
TEOB rows after native e<=0.5 conditioning and matched both evolved e and log prior.
