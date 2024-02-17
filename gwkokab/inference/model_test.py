#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpyro
import numpyro.distributions as dist


def model(X, y):
    alpha = numpyro.sample("alpha", dist.Uniform(-3, 3))  # .expand([total_events]))
    m_min = numpyro.sample("m_min", dist.Uniform(5, 15))  # .expand([total_events]))
    m_max = numpyro.sample("m_max", dist.Uniform(30, 70))  # .expand([total_events]))

    mu = alpha * (X[:, 0] - m_min) / (m_max - m_min)
    sigma = 0.1
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
