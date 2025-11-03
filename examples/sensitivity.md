# Expected Number of Detections and Sensitivity Estimation

In order to calculate the expected number of detections $\mu(\Lambda)$, sensitivity
of the detectors is also required. Mathematically, it is expressed as:

$$
\mu(\Lambda) = T_{\mathrm{obs}} \int \int \rho(\theta,z | \Lambda)
\frac{dV_c}{dz} \frac{1}{1+z}p_{\mathrm{det}}(\theta;z) d\theta dz
$$

If redshift is not a parameter in the population model or can be factored out, then

$$
\mu(\Lambda) = T_{\mathrm{obs}} \int \rho(\theta | \Lambda) \underbrace{\int
\frac{dV_c}{dz} \frac{1}{1+z}p_{\mathrm{det}}(\theta;z) dz}_{\mathrm{VT}(\theta)} d\theta
$$

where, $\mathrm{VT}(\theta)$ is called Sensitive Spacetime Volume of a network with
probability of detection $p_{\mathrm{det}}(\theta;z)$ for a source with parameters
$\theta$ at redshift $z$. $T_{\mathrm{obs}}$ is the total observation time.

GWKokab has three major methods to estimate the number of detections.

## Probability of Detection

To avoid interpolation, GWKokab trains a small Multilayer Perceptron (MLP) model to
estimate the probability of detection $p_{\mathrm{det}}(\theta;z)$. These trained models
serves as a fast surrogate to estimate the probability of detection for a given
source parameters $\theta$ and redshift $z$. For a multi-source population model, the
overall probability of detection is calculated by averaging over the individual source
probabilities. If $\hat{p}_{\mathrm{det}}(\theta;z)$ is the neural network estimate of
the probability of detection, then for $K$ sources, the overall probability of detection
is given by:

$$
\hat{\mu}(\Lambda) = T_{\mathrm{obs}} \sum_{k=1}^{K}\mathcal{R}_{k}\left<Z_k\hat{p}_{\mathrm{det}}(\theta_i;z)\right>_{\theta_i,z\sim \rho^*_k(\theta,z|\Lambda)}
$$

where, $Z_k$ is any factor required to normalize $\rho_k$, and $\rho^*_k$ is the normalized $\rho_k$.

An example configuration for probability of detection estimator is shown below:

```json
{
    "estimator_type": "neural_pdet",
    "filename": "neural_pdet_with_TaylorF2Ecc_uniform_injections.hdf5",
    "time_scale": 1.0,
    "num_samples": 1500,
    "batch_size": 150
}
```

Here, the configuration specifies the use of a neural network to estimate the probability of detection.
`"estimator_type"` indicates the type of estimator, `"filename"` is the path to the trained neural network model, `"time_scale"` is the observation time (in appropriate units), `"num_samples"` is the number of samples to draw from the population model, and `"batch_size"` is the size of a batch to evaluate the neural network.

## Sensitive Spacetime Volume

If population model is independent of redshift or redshift can be factored out, then
one can integrate out the redshift dependence to calculate the sensitive spacetime volume
$\mathrm{VT}(\theta)$, and train a MLP model to estimate $\mathrm{VT}(\theta)$ directly.

$$
\hat{\mu}(\Lambda) = T_{\mathrm{obs}} \int \rho(\theta | \Lambda) \hat{\mathrm{VT}}(\theta) d\theta = T_{\mathrm{obs}} \sum_{k=1}^{K}\mathcal{R}_{k}\left<Z_k\hat{\mathrm{VT}}(\theta_i)\right>_{\theta_i\sim \rho^*_k(\theta|\Lambda)}
$$

where, $Z_k$ is any factor required to normalize $\rho_k$, and $\rho^*_k$ is the normalized $\rho_k$.

An example configuration for sensitive spacetime volume estimator is shown below:

```json
{
    "estimator_type": "neural_vt",
    "filename": "neural_vt_1_200_1000_ecc_matters.hdf5",
    "time_scale": 248.0,
    "num_samples": 2000,
    "batch_size": 1000
}
```

Here, the configuration specifies the use of a neural network to estimate the sensitive spacetime volume.
`"estimator_type"` indicates the type of estimator, `"filename"` is the path to the trained neural network model, `"time_scale"` is the observation time (in appropriate units), `"num_samples"` is the number of samples to draw from the population model, and `"batch_size"` is the size of a batch to evaluate the neural network.

## Injection Based Method

Expected number of detections can also be estimated given a set of injections and their
sampling probability density. If we have $N_{\mathrm{inj}}$ injections with parameters
$\{\theta_i,z_i\}_{i=1}^{N_{\mathrm{inj}}}$, and out of these $N_{\mathrm{det}}$ injections
are detected by the search pipeline, then the expected number of detections is given by:

$$
\hat{\mu}(\Lambda) = \frac{T_{\mathrm{obs}}}{N_{\mathrm{inj}}} \sum_{i=1}^{N_{\mathrm{det}}} \frac{\rho(\theta_i,z_i|\Lambda)}{p_{\mathrm{inj}}(\theta_i,z_i)}
$$

An example configuration for o1o2o3o4a injection based estimator is shown below:

```json
{
    "estimator_type": "injection",
    "filename": "mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf",
    "batch_size": 20000,
    "snr_cut": 10.0,
    "far_cut": 1.0
}
```

Here, the configuration specifies the use of an injection-based method to estimate the expected number of detections. `"estimator_type"` indicates the type of estimator, `"filename"` is the path to the injection data file, `"batch_size"` is the size of a batch to process injections, `"snr_cut"` is the signal-to-noise ratio threshold for detection, and `"far_cut"` is the false alarm rate threshold for detection.

## Custom Estimator

If none of the above estimators fit your needs, you can pass a custom estimator by
implementing a function named `custom_poisson_mean_estimator` in a python module and
that has a signature like below:

```python
from typing import Callable, Sequence, Optional
from jaxtyping import PRNGKeyArray, ArrayLike
from gwkokab.models.utils import ScaleMixture

def custom_poisson_mean_estimator(
    key: PRNGKeyArray,
    parameters: Sequence[str],
    filename: str,
    batch_size: Optional[int],
    **kwargs) -> Callable[[ScaleMixture], ArrayLike]: ...
```

Its configuration would look like:

```json
{
    "estimator_type": "custom",
    "module": "my_custom_module.py",
    "filename": "my_custom_file.hdf5",
    "batch_size": 1000,
    "other_param_1": "value_1",
    "other_param_2": 10
}
```
