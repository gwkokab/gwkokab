# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse
import subprocess
import sys
from pathlib import Path

import papermill as pm

from gwkokab.analysis.utils.literals import INFERENCE_OUTPUT_FILENAME


def generate_report():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from a dynamic data file."
    )
    parser.add_argument(
        "-i",
        "--input-data",
        default=INFERENCE_OUTPUT_FILENAME,
        help="Path to the input data file to be processed.",
    )
    parser.add_argument(
        "-o",
        "--output-html",
        default="report.html",
        help="Path for the generated HTML file (default: report.html).",
    )

    args = parser.parse_args()

    MODULE_DIR = Path(__file__).resolve().parent
    notebook_path = MODULE_DIR / "template_report.ipynb"
    output_notebook = (
        Path(args.output_html).resolve().parent / "executed_temporary.ipynb"
    )

    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Error: Template notebook not found at {notebook_path}"
        )

    input_data_path = Path(args.input_data).resolve()

    try:
        # Pass the verified absolute path to papermill
        pm.execute_notebook(
            str(notebook_path),
            str(output_notebook),
            parameters=dict(inference_data_file=str(input_data_path)),
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--no-input",
                "--to",
                "html",
                "--output",
                str(Path(args.output_html).resolve()),
                str(output_notebook),
            ],
            check=True,
        )

    except Exception as e:
        print(f"An error occurred during generation: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if output_notebook.exists():
            output_notebook.unlink()
