# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict


def get_cfg() -> OrderedDict:
    """Defines the configuration as a flat OrderedDict.

    Keys are defined in logical blocks to maintain readability.
    """

    from typing import Any

    cfg: OrderedDict[str, Any] = OrderedDict()

    cfg["chain_batch_size"] = 0
    cfg["n_chains"] = 10

    cfg["batch_size"] = 1000
    cfg["history_window"] = 500
    cfg["n_epochs"] = 4
    cfg["n_max_examples"] = 100_000

    cfg["n_NFproposal_batch_size"] = 10

    cfg["n_global_steps"] = 20
    cfg["n_local_steps"] = 300

    cfg["global_thinning"] = 1
    cfg["local_thinning"] = 1

    cfg["n_production_loops"] = 40
    cfg["n_training_loops"] = 15

    cfg["local_sampler_name"] = "hmc"
    cfg["step_size"] = 0.01
    cfg["n_leapfrog"] = 5
    cfg["mass_matrix"] = 1.0

    cfg["learning_rate"] = 1e-3
    cfg["rq_spline_hidden_units"] = [128, 128]
    cfg["rq_spline_n_bins"] = 8
    cfg["rq_spline_n_layers"] = 10
    cfg["rq_spline_range"] = [-10.0, 10.0]

    cfg["verbose"] = False

    return cfg


def format_json_with_gaps(cfg: OrderedDict) -> str:
    """Converts dict to JSON and injects newlines before specific keys to create visual
    logical blocks for human readers.
    """
    # Keys that represent the start of a new logical section
    section_headers = (
        "batch_size",
        "n_NFproposal_batch_size",
        "n_global_steps",
        "global_thinning",
        "n_production_loops",
        "local_sampler_name",
        "learning_rate",
        "verbose",
    )

    import json

    lines = json.dumps(cfg, indent=4).splitlines()
    formatted_lines = []

    for line in lines:
        # Check if any section header is the key in the current line
        if any(f'"{header}"' in line for header in section_headers):
            formatted_lines.append("")  # Add a spacer line
        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Creates a template config for flowMC")
    parser.add_argument(
        "--output",
        "-o",
        default="flowMC_cfg_template.json",
        help="Output JSON filename",
    )
    args = parser.parse_args()

    # Get the flat config and format it with logical spacing
    sampler_cfg = get_cfg()
    formatted_output = format_json_with_gaps(sampler_cfg)

    with open(args.output, "w") as f:
        f.write(formatted_output)

    print(f"Successfully generated flowMC configuration at: {args.output}")
