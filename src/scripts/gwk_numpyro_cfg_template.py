# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Dict


def get_numpyro_cfg() -> Dict[str, Any]:
    """Defines the configuration for a NumPyro-based sampler. This structure is kept
    nested to logically separate kernel parameters from MCMC execution parameters.

    Resources
    ---------

    - MCMC Class: https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mcmc.MCMC
    - NUTS Kernel: https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS
    """
    return {
        "kernel": {
            "step_size": 1.0,
            "inverse_mass_matrix": None,
            "adapt_step_size": True,
            "adapt_mass_matrix": True,
            "dense_mass": False,
            "target_accept_prob": 0.8,
            "max_tree_depth": 8,
            "find_heuristic_step_size": False,
            "forward_mode_differentiation": False,
            "regularize_mass_matrix": True,
        },
        "mcmc": {
            "num_warmup": 1000,
            "num_samples": 2000,
            "num_chains": 10,
            "thinning": 1,
            "chain_method": "parallel",
            "progress_bar": True,
            "jit_model_args": False,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Creates a template config for NumPyro NUTS sampler and MCMC"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="numpyro_cfg_template.json",
        help="Output JSON filename",
    )
    args = parser.parse_args()

    cfg = get_numpyro_cfg()

    import json

    with open(args.output, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"Successfully generated NumPyro configuration at: {args.output}")
    print("\nResources for these parameters:")
    print(" - MCMC: https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.mcmc.MCMC")
    print(" - NUTS: https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS")
