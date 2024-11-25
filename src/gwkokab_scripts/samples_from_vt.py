# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

import h5py
import numpyro
from jax import numpy as jnp, random as jrd
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from gwkokab.vts import NeuralVT, save_model


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Samples from VT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script runs a MCMC sampler on VT to generate samples.",
    )

    vt_group = parser.add_argument_group("VT Options")

    vt_group.add_argument(
        "--filename",
        help="path to the VT file",
        required=True,
        type=str,
    )
    vt_group.add_argument(
        "--output",
        help="output file path",
        required=True,
        type=str,
    )
    vt_group.add_argument(
        "--vt-params",
        help="name of the VT parameters in order along with the minimum and maximum values for their uniform priors",
        required=True,
        nargs=3,
        type=str,
        action="append",
    )

    nuts_group = parser.add_argument_group("NUTS Options")

    nuts_group.add_argument(
        "--target-accept-prob",
        help="target acceptance probability",
        default=0.90,
        type=float,
    )

    mcmc_group = parser.add_argument_group("MCMC Options")

    mcmc_group.add_argument(
        "--num-warmup",
        help="number of warmup samples",
        default=5000,
        type=int,
    )
    mcmc_group.add_argument(
        "--num-chains",
        help="number of chains",
        default=4,
        type=int,
    )
    mcmc_group.add_argument(
        "--num-samples",
        help="number of samples",
        default=1000,
        type=int,
    )
    mcmc_group.add_argument(
        "--chain-method",
        help="chain method",
        choices=["parallel", "sequential", "vectorized"],
        default="parallel",
    )
    mcmc_group.add_argument(
        "--jit-model-args",
        help="If set, this will compile the potential energy computation as a function of model arguments.",
        action="store_true",
    )
    mcmc_group.add_argument(
        "--seed",
        help="PRNG seed",
        default=37,
        type=int,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    filename = args.filename
    if not filename.endswith(".hdf5"):
        raise ValueError("This script only supports Neural VTs.")

    output = args.output
    if not output.endswith(".hdf5"):
        raise ValueError("This script only supports HDF5 output files.")

    vt_params_names = []
    min_vals = []
    max_vals = []
    for param, min_val, max_val in args.vt_params:
        if not min_val < max_val:
            raise ValueError(f"{param} has min_val >= max_val.")
        vt_params_names.append(param)
        min_vals.append(float(min_val))
        max_vals.append(float(max_val))

    nvt = NeuralVT(vt_params_names, filename)

    logVT = nvt.get_logVT()

    def model():
        params = numpyro.sample(
            "params",
            dist.Uniform(
                jnp.asarray(min_vals), jnp.asarray(max_vals), validate_args=True
            ),
        )
        numpyro.factor("log_vt_fac", logVT(params))

    nuts = NUTS(model, target_accept_prob=args.target_accept_prob)

    mcmc = MCMC(
        nuts,
        num_warmup=args.num_warmup,
        num_chains=args.num_chains,
        num_samples=args.num_samples,
        jit_model_args=args.jit_model_args,
        chain_method=args.chain_method,
    )

    mcmc.run(jrd.PRNGKey(args.seed))

    mcmc.print_summary()

    samples = mcmc.get_samples()

    if filename != output:
        # save model weights if the output file is different from the input file
        save_model(
            filename=output,
            model=nvt.model,
            names=[vt_params_names[i] for i in nvt.shuffle_indices],
        )
    with h5py.File(output, "a") as f:
        samples_group = f.create_group("samples")
        for i, param in enumerate(vt_params_names):
            samples_group.create_dataset(param, data=samples["params"][..., i])
