# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate the JSON configuration."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path_obj, "r") as f:
        cfg = json.load(f)

    if "bundle_config" not in cfg:
        raise ValueError("Config file must contain a 'bundle_config' key.")

    return cfg["bundle_config"]


def infer_n_dims(bundle_config: dict, cli_n_dims: Optional[int]) -> int:
    """Infer n_dims from CLI argument or config (mass_matrix length)."""
    if cli_n_dims is not None:
        return cli_n_dims

    if "n_dims" in bundle_config:
        try:
            return int(bundle_config["n_dims"])
        except Exception:
            pass

    mm = bundle_config.get("mass_matrix", None)
    if isinstance(mm, list):
        return len(mm)

    return 0


def generate_heuristics(
    n_dims: int,
    n_chains: int,
    kept_total_per_loop: int,
) -> Dict[str, Any]:
    """Generate heuristic suggestions for FlowMC parameters."""
    if n_dims == 0:
        return {}

    if n_dims <= 15:
        history_loops = 5
        n_max_target = 20_000
        hidden_units = [64, 64]
        n_layers = 4
    elif n_dims <= 30:
        history_loops = 6
        n_max_target = 30_000
        hidden_units = [128, 128]
        n_layers = 6
    else:
        history_loops = 8
        n_max_target = 40_000
        hidden_units = [256, 256]
        n_layers = 8

    kept_per_chain = kept_total_per_loop // max(n_chains, 1)
    history_window = history_loops * kept_per_chain
    candidate_capacity = history_window * n_chains
    n_max_suggest = min(n_max_target, candidate_capacity)

    if n_max_suggest > 0:
        bs_raw = int(0.5 * n_max_suggest / 8)
        batch_size = max(1024, min(bs_raw, 10_000))
    else:
        batch_size = 4096

    training_loops = max(4 * history_loops, 24)
    production_loops = max(2 * history_loops, training_loops // 2)

    return {
        "history_window": history_window,
        "history_loops": history_loops,
        "n_max_examples": n_max_suggest,
        "candidate_capacity": candidate_capacity,
        "batch_size": batch_size,
        "hidden_units": hidden_units,
        "n_layers": n_layers,
        "training_loops": training_loops,
        "production_loops": production_loops,
    }


def plot_history(
    loops: List[int], candidates: List[int], n_train: List[int], save_path: str
):
    """Generate a matplotlib plot of the training history."""
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        loops,
        candidates,
        label="Candidate Pool (History)",
        linewidth=2,
        color="#3498db",
    )
    ax.plot(
        loops,
        n_train,
        label="Effective NF Training Set",
        linewidth=2,
        linestyle="--",
        color="#e74c3c",
    )

    ax.set_xlabel("Training Loop")
    ax.set_ylabel("Number of Samples")
    ax.set_title("FlowMC Training Dynamics", fontsize=14)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# Rich Display Functions
# -------------------------------------------------------------------------


def print_header(title: str):
    console.print(
        Panel(Text(title, justify="center", style="bold white"), style="bold blue")
    )


def print_section(title: str):
    console.rule(f"[bold cyan]{title}")


def display_config_table(data: List[tuple]):
    # Updated: Value column is right justified
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow", justify="right")

    for row in data:
        table.add_row(row[0], str(row[1]))
    console.print(table)


def display_diagnostics_panel(info_dict: Dict[str, str], memory_notes: List[str]):
    """Display the detailed sample breakdown and memory notes in a clean panel."""
    # Updated: expand=False prevents massive gaps; justify="right" for numbers
    grid = Table.grid(expand=False, padding=(0, 2))
    grid.add_column(style="white", justify="left")
    grid.add_column(style="bold yellow", justify="right")

    for key, value in info_dict.items():
        grid.add_row(key, value)

    # 2. Add the specific memory notes
    note_text = Text()
    if memory_notes:
        note_text.append("\n\n")  # Spacing
        for note in memory_notes:
            if "Step count where forgetting matters:" in note:
                parts = note.split(":", 1)
                note_text.append("• " + parts[0] + ":", style="cyan")
                note_text.append(parts[1] + "\n", style="bold green")
            elif ":" in note:
                parts = note.split(":", 1)
                note_text.append("• " + parts[0] + ":", style="cyan")
                note_text.append(parts[1] + "\n", style="white")
            else:
                note_text.append("• " + note + "\n", style="white")

    content = Group(grid, note_text)

    console.print(
        Panel(content, title="Detailed Dynamics & Diagnostics", border_style="blue")
    )


def display_loop_table(rows: List[tuple], max_loops: int):
    # Updated: Numeric columns are strictly right justified
    table = Table(title="Training Loop Simulation", box=box.MINIMAL_HEAVY_HEAD)
    table.add_column("Loop", justify="right", style="cyan")
    table.add_column("Total Collected", justify="right")
    table.add_column("Pool Size (History)", justify="right", style="green")
    table.add_column("NF Train Size", justify="right", style="magenta")
    table.add_column("Status", justify="left")  # Text stays left

    for row in rows:
        status = row[4]
        status_style = "dim"
        if "Forgetting" in status:
            status_style = "bold red"
        elif "Saturated" in status:
            status_style = "yellow"

        table.add_row(
            str(row[0]),
            f"{row[1]:,}",
            f"{row[2]:,}",
            f"{row[3]:,}",
            Text(status, style=status_style),
        )
    console.print(table)


def display_suggestions(current: dict, suggested: dict):
    # Updated: Numeric columns are right justified
    table = Table(title="Heuristic Tuning Suggestions", box=box.ROUNDED)
    table.add_column("Parameter", style="bold white")
    table.add_column("Current Config", style="red", justify="right")
    table.add_column("Recommended", style="green", justify="right")

    params = [
        ("History Window", current["history_window"], suggested["history_window"]),
        ("N Max Examples", current["n_max_examples"], suggested["n_max_examples"]),
        ("Batch Size", current["batch_size"], suggested["batch_size"]),
        ("Training Loops", current["n_training_loops"], suggested["training_loops"]),
        (
            "Production Loops",
            current["n_production_loops"],
            suggested["production_loops"],
        ),
    ]

    for label, curr, sugg in params:
        c_str = f"{curr:,}" if isinstance(curr, int) else str(curr)
        s_str = f"{sugg:,}" if isinstance(sugg, int) else str(sugg)
        if curr == sugg:
            c_style = "green"
        else:
            c_style = "yellow"
        table.add_row(
            label, Text(c_str, style=c_style), Text(s_str, style="bold green")
        )

    console.print(table)

    arch_panel = Panel(
        f"Hidden Units: {suggested['hidden_units']}\nLayers: {suggested['n_layers']}",
        title="Suggested Architecture",
        border_style="green",
        width=50,
    )
    console.print(arch_panel)


# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Summarize FlowMC training configuration."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="flowMC_config.json",
        help="Path to config.json (default: flowMC_config.json)",
    )
    parser.add_argument(
        "--max-loops-table", type=int, default=-1, help="Max loops to display in table"
    )
    parser.add_argument("--plot", action="store_true", help="Generate history plot")
    parser.add_argument(
        "--plot-path", default="flowmc_history_plot.png", help="Plot filename"
    )
    parser.add_argument(
        "--n-dims", type=int, default=None, help="Force override n_dims"
    )
    args = parser.parse_args()

    # 1. Load Data
    try:
        bc = load_config(args.filename)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    # 2. Extract Parameters
    n_chains = int(bc["n_chains"])
    n_local = int(bc["n_local_steps"])
    local_thin = int(bc["local_thinning"])
    n_global = int(bc["n_global_steps"])
    global_thin = int(bc["global_thinning"])
    history_window = int(bc["history_window"])
    n_max = int(bc["n_max_examples"])
    n_train_loops = int(bc["n_training_loops"])
    n_prod_loops = int(bc.get("n_production_loops", 0))
    batch_size = int(bc["batch_size"])
    n_epochs = int(bc["n_epochs"])

    # 3. Derived Math
    n_dims = infer_n_dims(bc, args.n_dims)
    kept_local = n_local // max(local_thin, 1)
    kept_global = n_global // max(global_thin, 1)

    # Per Chain Calculations
    kept_per_chain = kept_local + kept_global

    # Total (All Chains) Calculations
    kept_local_total = kept_local * n_chains
    kept_global_total = kept_global * n_chains
    kept_total_per_loop = kept_per_chain * n_chains
    history_capacity = history_window * n_chains

    # Memory / History Saturation Logic
    if kept_per_chain > 0 and history_window > 0:
        loops_to_fill = math.ceil(history_window / kept_per_chain)
        effective_depth = history_window / kept_per_chain
    else:
        loops_to_fill = 0
        effective_depth = 0.0

    # Epoch Indices for Forgetting
    first_forget_loop = loops_to_fill + 1 if loops_to_fill > 0 else 0

    if first_forget_loop > 0:
        forget_epoch_0 = loops_to_fill * n_epochs
        forget_epoch_1 = forget_epoch_0 + 1

        # Calculate exact sampler steps at the forgetting boundary
        # Formula: (local_steps * loops) + (global_steps * loops)
        steps_local_total = loops_to_fill * n_local
        steps_global_total = loops_to_fill * n_global
        forget_step = steps_local_total + steps_global_total

        # Format string for display
        forget_step_str = (
            f"{forget_step:,} "
            f"({loops_to_fill} loops × {n_local} local) + "
            f"({loops_to_fill} loops × {n_global} global)"
        )
    else:
        forget_epoch_0 = 0
        forget_epoch_1 = 0
        forget_step_str = "N/A"

    # 4. Render Summary
    print_header("FlowMC Configuration Analysis")

    # Basic Config Table
    config_data = [
        ("Config Path", args.filename),
        ("Dimensions (n_dims)", n_dims if n_dims else "[red]Unknown[/red]"),
        ("Chains", n_chains),
        ("Epochs per Loop", n_epochs),
    ]

    if "mass_matrix" not in bc:
        console.print(
            Panel(
                "[bold yellow]Warning:[/bold yellow] 'mass_matrix' missing. Run diagnostics to estimate.",
                border_style="yellow",
            )
        )

    print_section("Configuration Overview")
    display_config_table(config_data)

    # 5. Detailed Diagnostics
    diagnostics_info = {
        "Total NF candidate capacity (window)": f"{history_capacity:,} samples",
        "Kept LOCAL samples per loop": f"{kept_local_total:,}",
        "Kept GLOBAL samples per loop": f"{kept_global_total:,}",
        "TOTAL new samples per loop": f"{kept_total_per_loop:,}",
    }

    # Constructing the note strings
    diag_notes = []
    # Note 1: Training subset logic
    if n_max > 0 and n_max < history_capacity:
        diag_notes.append(
            f"Training Subset: NF trains on a random subset of up to {n_max:,} samples from the visible history each loop."
        )
    elif n_max >= history_capacity:
        diag_notes.append(
            "Training Subset: NF trains on ALL visible samples each loop (n_max >= history capacity)."
        )

    # Note 2: Saturation depth
    diag_notes.append(
        f"History saturation: {loops_to_fill} loop(s) (effective depth ~{effective_depth:.2f} loops)"
    )

    # Note 3: Forgetting indices (Epochs AND Steps)
    if first_forget_loop > 0:
        diag_notes.append(
            f"First epoch index where forgetting matters: {forget_epoch_0} (0-based), {forget_epoch_1} (1-based)"
        )
        diag_notes.append(f"Step count where forgetting matters: {forget_step_str}")

    print_section("History & Memory Dynamics")
    display_diagnostics_panel(diagnostics_info, diag_notes)

    # 6. Build Loop Table
    if args.max_loops_table < 0:
        limit = (
            min(n_train_loops, first_forget_loop + 2)
            if first_forget_loop
            else n_train_loops
        )
    else:
        limit = min(args.max_loops_table, n_train_loops)

    loop_rows = []
    plot_loops = []
    plot_cands = []
    plot_train = []

    for L in range(1, limit + 1):
        total_collected = kept_total_per_loop * L
        candidates = (
            min(total_collected, history_capacity)
            if history_capacity > 0
            else total_collected
        )
        n_train = min(candidates, n_max) if n_max > 0 else candidates

        status = []
        if L == loops_to_fill:
            status.append("Saturated")
        if L >= first_forget_loop and first_forget_loop > 0:
            status.append("Forgetting")

        status_str = ", ".join(status) if status else "-"

        loop_rows.append((L, total_collected, candidates, n_train, status_str))

        plot_loops.append(L)
        plot_cands.append(candidates)
        plot_train.append(n_train)

    if args.plot and limit < n_train_loops:
        for L in range(limit + 1, n_train_loops + 1):
            total_collected = kept_total_per_loop * L
            candidates = (
                min(total_collected, history_capacity)
                if history_capacity > 0
                else total_collected
            )
            n_train = min(candidates, n_max) if n_max > 0 else candidates
            plot_loops.append(L)
            plot_cands.append(candidates)
            plot_train.append(n_train)

    display_loop_table(loop_rows, limit)

    # 7. Suggestions
    print_section("Optimization")
    if n_dims > 0:
        suggestions = generate_heuristics(n_dims, n_chains, kept_total_per_loop)
        current_conf = {
            "history_window": history_window,
            "n_max_examples": n_max,
            "batch_size": batch_size,
            "n_training_loops": n_train_loops,
            "n_production_loops": n_prod_loops,
        }
        suggestions_disp = suggestions.copy()
        display_suggestions(current_conf, suggestions_disp)
    else:
        msg = "Cannot generate suggestions: 'n_dims' is unknown.\nPass --n_dims or ensure 'mass_matrix' is in config."
        console.print(f"[italic dim]{msg}[/italic dim]")

    # 8. Plotting
    if args.plot:
        plot_history(plot_loops, plot_cands, plot_train, args.plot_path)
        console.print(
            f"\n[bold green]✓[/bold green] Plot saved to: [underline]{args.plot_path}[/underline]"
        )
