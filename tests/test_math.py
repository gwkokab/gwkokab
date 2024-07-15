import numpy as np
import pytest

from gwkokab.utils.math import (
    beta_dist_concentrations_to_mean_variance,
    beta_dist_mean_variance_to_concentrations,
)


@pytest.mark.parametrize("alpha", np.random.uniform(1, 200, 5))
@pytest.mark.parametrize("beta", np.random.uniform(1, 200, 5))
def test_beta_dist1(alpha, beta):
    mean, variance = beta_dist_concentrations_to_mean_variance(alpha, beta)
    alpha_, beta_ = beta_dist_mean_variance_to_concentrations(mean, variance)
    print(alpha, alpha_)
    print(beta, beta_)
    assert np.allclose(alpha, alpha_)
    assert np.allclose(beta, beta_)


@pytest.mark.parametrize("mean", np.random.uniform(0, 1, 5))
@pytest.mark.parametrize("var", np.random.uniform(0, 0.25, 5))
def test_beta_dist2(mean, var):
    alpha, beta = beta_dist_mean_variance_to_concentrations(mean, var)
    mean_, var_ = beta_dist_concentrations_to_mean_variance(alpha, beta)
    assert np.allclose(mean, mean_)
    assert np.allclose(var, var_)
