# Change log

Best view [here](https://gwkokab.readthedocs.io/en/latest/changelog.html).

## gwkokab 0.0.2

### Breaking changes

* `gwkokab.priors.UnnormalizedUniform` deprecated in favor of `numpyro.distributions.ImproperUniform`.
* Trivial errors are removed from `gwkokab.errors` module.
  * `normal_error`
  * `truncated_normal_error`
  * `uniform_error`

### New features

* Constraints related to closed intervals, for details, see [PR#125](https://github.com/gwkokab/gwkokab/pull/125).
* Bijective transformations on different mass coordinates, for details, see [PR#125](https://github.com/gwkokab/gwkokab/pull/125).

## gwkokab 0.0.1

* Initial release.
