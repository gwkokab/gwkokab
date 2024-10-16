# Change log

Best view [here](https://gwkokab.readthedocs.io/en/latest/changelog.html).

## gwkokab 0.1.1

### Breaking changes

- Following parameters are renamed:
  - `COS_TILE_1`
  - `COS_TILE_2`
  - `TILE_1`
  - `TILE_2`
- Most assisting commands have been shifter from using index to column names.
- Following models have been depreciated,
  - `MultiSourceModel`
  - `MultiSpinModel`
  - `TruncatedPowerLaw`

### New features

- New models:
  - `FlexibleMixtureModel`
  - `MassGapModel`
  - `NPowerLawMGaussian`
- New transforms:
  - `ComponentMassesAndRedshiftToDetectedMassAndRedshift`
  - `ComponentMassesToMassRatioAndSecondaryMass`
  - `ComponentMassesToPrimaryMassAndMassRatio`
  - `ComponentMassesToTotalMassAndMassRatio`
  - `SourceMassAndRedshiftToDetectedMassAndRedshift`
- scaling factor in vt, during inference and synthetic data generation.
- Debug mode for inference.

### New Design

- Introduced a refined base class for smoothed mass distribution models, enhancing functionality and maintainability.

### Deprecations

Following has been deprecated:

- `genie_ecc_matters`
- `genie_powerlaw_primary_mass_ratio`
- `genie_single_gaussians`
- `sage_ecc_matters`
- `sage_powerlaw_primary_mass_ratio`

## gwkokab 0.1.0 (July 27, 2024)

### Breaking changes

- `gwkokab.inference.BayesianHierarchicalModel` is deprecated in favor of `gwkokab.inference.PoissonLikelihood`
- `gwkokab.priors.UnnormalizedUniform` deprecated in favor of `numpyro.distributions.ImproperUniform`.
- Trivial errors are removed from `gwkokab.errors` module.
  - `normal_error`
  - `truncated_normal_error`
  - `uniform_error`
- After the release of [numpyro-0.15.1](https://github.com/pyro-ppl/numpyro/releases/tag/0.15.1), `less_than_equals_to` and `greater_than_equals_to` constraints are removed from `gwkokab.constraints` module.
- The directory containing individual injections is removed from the synthetic data generation process.
- `gwkokab.population.PopFactory` is removed from the public API; instead, an instance of the class is provided in the `gwkokab.population` module.
- `gwkokab.population.NoisePopInfo`, `gwkokab.population.PopInfo`, and `gwkokab.population.run_noise_factory` are removed.
- `gwkokab.inference.BayesianHierarchicalModel` in favor of `gwkokab.inference.PoissonLikelihood`
- `gwkokab.inference.ModelPack` removed.

### New features

- Constraints related to closed intervals, for details, see [PR#125](https://github.com/gwkokab/gwkokab/pull/125).
- Bijective transformations on different mass coordinates, for details, see [PR#125](https://github.com/gwkokab/gwkokab/pull/125).
- Wrapper for scaling models.
- Model registration and retrieval. `gwkokab.population.popmodel_magazine` and `gwkokab.population.error_magazine` are introduced to register the population model and error models, respectively.
- New models:
  - `MultiSourceModel`
  - `MultiSpinModel`
  - `NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment`
- New API for inference, compatible with `gwkokab.inference.flowMChandler`
  - `gwkokab.inference.PoissonLikelihood`
  - `gwkokab.inference.Bake`
- Progress Bar for the synthetic data generation process.

## gwkokab 0.0.1 (Jul 5, 2024)

- Initial release.
