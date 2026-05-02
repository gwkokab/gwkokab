# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def create_grid(data: dict[str, str], query: str, label_text: str):
    """Filters, clusters, and builds a collection of Rich tables."""
    from rich.table import Table

    # 1. Filter and Group
    grouped: dict[str, list[str]] = {}
    for param, val in data.items():
        if query.lower() in param.lower():
            val_str = str(val)
            grouped.setdefault(val_str, []).append(param)

    if not grouped:
        return [f"[dim italic]No matches found for '{query}'[/dim italic]"]

    # 2. Sort Logic (Numeric then Alpha)
    sorted_keys = sorted(
        grouped.keys(),
        key=lambda k: (0, float(k)) if k.replace(".", "", 1).isdigit() else (1, k),
    )

    # 3. Create a Table for each cluster
    renderables = []
    for key in sorted_keys:
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column("Label", style="bold magenta", width=12, justify="right")
        table.add_column("Params", style="bright_white")

        # First row shows the label, subsequent rows are empty on the left
        params = grouped[key]
        for i, p in enumerate(params):
            label = f"{label_text} {key}" if i == 0 else ""
            table.add_row(label, f"• {p}")

        renderables.append(table)

    return renderables


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description="Search and display parameter mappings and constants."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Search term to filter parameters (default: show all)",
    )
    args = parser.parse_args()

    query = args.query

    import json

    from rich.columns import Columns
    from rich.console import Console

    console = Console()

    try:
        with open("constants.json", "r") as f:
            constants = json.load(f)
        with open("nf_samples_mapping.json", "r") as f:
            mapping = json.load(f)
    except FileNotFoundError as e:
        console.print(
            f"[bold red]Error:[/bold red] Could not find {e.filename}", style="red"
        )
        return

    console.print("[bold cyan]MAPPING[/bold cyan] [dim](By Index)[/dim]")
    console.print("—" * 40, style="dim")
    mapping_tables = create_grid(mapping, query, "IDX")
    console.print(Columns(mapping_tables, equal=True, expand=True))

    console.print("\n\n[bold cyan]CONSTANTS[/bold cyan] [dim](By Value)[/dim]")
    console.print("—" * 40, style="dim")
    constants_tables = create_grid(constants, query, "VAL")
    console.print(Columns(constants_tables, equal=True, expand=True))
