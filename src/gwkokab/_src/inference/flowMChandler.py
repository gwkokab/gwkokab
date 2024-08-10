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

from typing_extensions import Any, Callable

from flowMC.nfmodel.base import NFModel
from flowMC.nfmodel.realNVP import RealNVP  # noqa F401
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline  # noqa F401
from flowMC.proposal.base import ProposalBase
from flowMC.proposal.flowHMC import flowHMC  # noqa F401
from flowMC.proposal.Gaussian_random_walk import GaussianRandomWalk  # noqa F401
from flowMC.proposal.HMC import HMC  # noqa F401
from flowMC.proposal.MALA import MALA  # noqa F401
from flowMC.Sampler import Sampler  # noqa F401
from jaxtyping import Array

from .utils import save_data_from_sampler


class flowMChandler(object):
    r"""Handler class for running flowMC."""

    def __init__(
        self,
        logpdf: Callable,
        local_sampler_kwargs: dict[str, Any],
        nf_model_kwargs: dict[str, Any],
        sampler_kwargs: dict[str, Any],
        data_dump_kwargs: dict[str, Any],
        initial_position: Array,
        data: dict,
    ) -> None:
        self.logpdf = logpdf
        self.local_sampler_kwargs = local_sampler_kwargs
        self.nf_model_kwargs = nf_model_kwargs
        self.sampler_kwargs = sampler_kwargs
        self.data_dump_kwargs = data_dump_kwargs
        self.initial_position = initial_position
        self.data = data

    def make_local_sampler(self) -> ProposalBase:
        """Make a local sampler based on the given arguments.

        :raises ValueError: If the sampler is not recognized.
        :return: A local sampler.
        """
        sampler_name = self.local_sampler_kwargs["sampler"]
        if sampler_name not in ["flowHMC", "GaussianRandomWalk", "HMC", "MALA"]:
            raise ValueError("Invalid sampler")
        del self.local_sampler_kwargs["sampler"]
        return eval(sampler_name)(self.logpdf, **self.local_sampler_kwargs)

    def make_nf_model(self) -> NFModel:
        """Make a normalizing flow model based on the given arguments.

        :raises ValueError: If the model is not recognized.
        :return: A normalizing flow model.
        """
        model_name = self.nf_model_kwargs["model"]
        if model_name not in ["RealNVP", "MaskedCouplingRQSpline"]:
            raise ValueError("Invalid model")
        del self.nf_model_kwargs["model"]
        return eval(model_name)(**self.nf_model_kwargs)

    def make_sampler(self) -> Sampler:
        """Make a sampler based on the given arguments.

        :return: A sampler.
        """
        return Sampler(
            local_sampler=self.make_local_sampler(),
            nf_model=self.make_nf_model(),
            **self.sampler_kwargs,
        )

    def run(self) -> None:
        """Run the flowMC sampler and save the data."""
        sampler = self.make_sampler()
        sampler.sample(self.initial_position, self.data)
        save_data_from_sampler(sampler, **self.data_dump_kwargs)
