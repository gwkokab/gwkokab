#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_samples(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        data = np.load(path)
        return {k: np.asarray(data[k]).reshape(-1) for k in data.files}

    if ext in [".dat", ".txt"]:
        with open(path, "r") as f:
            header = f.readline().replace("#", "").split()

        arr = np.loadtxt(path)

        if arr.ndim == 1:
            arr = arr[None, :]

        if len(header) != arr.shape[1]:
            raise ValueError(
                f"Header has {len(header)} columns but data has {arr.shape[1]} columns."
            )

        return {name: arr[:, i] for i, name in enumerate(header)}

    raise ValueError("Supported formats: .npz, .dat, .txt")


def split_chains(values, num_chains):
    values = np.asarray(values).reshape(-1)

    n_total = len(values)
    n_per_chain = n_total // num_chains
    n_used = n_per_chain * num_chains

    if n_per_chain < 10:
        raise ValueError("Not enough samples per chain.")

    return values[:n_used].reshape(num_chains, n_per_chain)


def split_single_chain_for_rhat(chain):
    chain = np.asarray(chain).reshape(-1)

    half = len(chain) // 2
    used = 2 * half

    if half < 10:
        return None

    return chain[:used].reshape(2, half)


def autocorr_1d(x, max_lag=200):
    x = np.asarray(x)
    x = x - np.mean(x)

    if np.var(x) == 0:
        return np.zeros(max_lag + 1)

    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / corr[0]

    return corr[: max_lag + 1]


def compute_rhat(chains):
    m, n = chains.shape

    if m < 2:
        return np.nan

    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    B = n * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)

    if W == 0:
        return np.nan

    var_hat = ((n - 1) / n) * W + (B / n)

    return float(np.sqrt(var_hat / W))


def compute_ess(chains, max_lag=1000):
    m, n = chains.shape
    ess_total = 0.0

    for c in range(m):
        ac = autocorr_1d(chains[c], max_lag=min(max_lag, n - 1))

        tau = 1.0

        for k in range(1, len(ac)):
            if ac[k] <= 0:
                break
            tau += 2.0 * ac[k]

        ess_total += n / tau

    return float(ess_total)


def diagnose_parameter(values, num_chains, max_lag):
    chains = split_chains(values, num_chains)
    x = chains.reshape(-1)

    if num_chains == 1:
        split_for_rhat = split_single_chain_for_rhat(x)
        rhat = compute_rhat(split_for_rhat) if split_for_rhat is not None else np.nan
        rhat_type = "split-chain R_hat"
    else:
        rhat = compute_rhat(chains)
        rhat_type = "R_hat"

    ess = compute_ess(chains, max_lag=max_lag)

    return {
        "chains": chains,
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ess": ess,
        "rhat": rhat,
        "rhat_type": rhat_type,
    }


def diagnostic_label(stats):
    rhat = stats["rhat"]
    rhat_type = stats["rhat_type"]

    if np.isnan(rhat):
        return f"ESS={stats['ess']:.1f}, {rhat_type}=N/A"

    return f"ESS={stats['ess']:.1f}, {rhat_type}={rhat:.3f}"


def safe_filename(name):
    return (
        name
        .replace("/", "_")
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(" ", "_")
    )


def save_trace_plot(param, stats, outdir):
    chains = stats["chains"]

    plt.figure(figsize=(11, 5))

    for c in range(chains.shape[0]):
        plt.plot(chains[c], lw=0.7, alpha=0.85, label=f"chain {c + 1}")

    plt.xlabel("sample")
    plt.ylabel(param)
    plt.title(f"Trace: {param} | {diagnostic_label(stats)}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"trace_{safe_filename(param)}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def save_hist_plot(param, stats, outdir):
    chains = stats["chains"]

    plt.figure(figsize=(7, 5))

    for c in range(chains.shape[0]):
        plt.hist(
            chains[c],
            bins=50,
            density=True,
            histtype="step",
            lw=1.2,
            label=f"chain {c + 1}",
        )

    plt.xlabel(param)
    plt.ylabel("density")
    plt.title(f"Histogram: {param} | {diagnostic_label(stats)}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"hist_{safe_filename(param)}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def save_autocorr_plot(param, stats, outdir, max_lag):
    chains = stats["chains"]

    plt.figure(figsize=(8, 5))

    for c in range(chains.shape[0]):
        ac = autocorr_1d(chains[c], max_lag=min(max_lag, chains.shape[1] - 1))
        plt.plot(np.arange(len(ac)), ac, lw=1.0, label=f"chain {c + 1}")

    plt.axhline(0.0, lw=0.8)
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title(f"Autocorr: {param} | {diagnostic_label(stats)}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"autocorr_{safe_filename(param)}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def write_summary(all_stats, outdir, num_chains):
    path = os.path.join(outdir, "diagnostic_summary.txt")

    with open(path, "w") as f:
        f.write("MCMC Diagnostic Summary\n")
        f.write("=======================\n\n")

        f.write("Benchmarks:\n")
        f.write("- Ideal R_hat: < 1.01\n")
        f.write("- Acceptable R_hat: < 1.05\n")
        f.write("- Problematic R_hat: > 1.05\n")
        f.write("- Good n_eff: > 1000\n")
        f.write("- Acceptable n_eff: > 400-500\n")
        f.write("- Weak n_eff: 100-300\n")
        f.write("- Bad n_eff: < 100\n")
        f.write("- Very bad n_eff: < 50\n")
        f.write(
            "- Autocorrelation should decay toward zero within the plotted lag range.\n\n"
        )

        if num_chains == 1:
            f.write(
                "NOTE: One real chain was provided. The reported R_hat is split-chain "
                "R_hat, computed by splitting the single chain into two halves. "
                "This is weaker than true multi-chain R_hat.\n\n"
            )
        else:
            f.write(
                "NOTE: Multiple chains were provided. The reported R_hat is the "
                "standard between-chain/within-chain Gelman-Rubin diagnostic.\n\n"
            )

        f.write(f"Number of chains assumed: {num_chains}\n\n")

        for param, stats in all_stats.items():
            rhat = stats["rhat"]
            rhat_str = "N/A" if np.isnan(rhat) else f"{rhat:.6g}"

            f.write(f"Parameter: {param}\n")
            f.write(f"  mean   = {stats['mean']:.6g}\n")
            f.write(f"  std    = {stats['std']:.6g}\n")
            f.write(f"  median = {stats['median']:.6g}\n")
            f.write(f"  5%     = {stats['q05']:.6g}\n")
            f.write(f"  95%    = {stats['q95']:.6g}\n")
            f.write(f"  min    = {stats['min']:.6g}\n")
            f.write(f"  max    = {stats['max']:.6g}\n")
            f.write(f"  n_eff  = {stats['ess']:.6g}\n")
            f.write(f"  {stats['rhat_type']} = {rhat_str}\n\n")

    print(f"Saved summary: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic tool for MCMC posterior samples produced by "
            "NumPyro, PyMC, emcee, or similar samplers.\n\n"
            "This script computes effective sample size (n_eff), "
            "R-hat diagnostics, autocorrelation statistics, and "
            "optionally generates trace, histogram, and "
            "autocorrelation plots for posterior parameters."
        ),
        epilog=(
            "Examples:\n"
            "----------\n"
            "1) Generate only diagnostic summary:\n"
            "   python trace.py samples.dat --num-chains 1\n\n"
            "2) Generate autocorrelation plots:\n"
            "   python trace.py samples.dat "
            "--num-chains 1 --plot-autocorrelation\n\n"
            "3) Generate trace and histogram plots:\n"
            "   python trace.py samples.dat "
            "--num-chains 4 --plot-trace --plot-histogram\n\n"
            "4) Diagnose only selected parameters:\n"
            "   python trace.py samples.dat "
            "--params log_rate lambda_0 alpha1_bpl_0\n\n"
            "Notes:\n"
            "------\n"
            "- If only one chain is provided, the script computes "
            "split-chain R-hat by splitting the chain into two halves.\n"
            "- If multiple chains are provided, standard Gelman-Rubin "
            "R-hat is computed.\n"
            "- Effective sample size (n_eff) is estimated using "
            "autocorrelation time.\n"
            "- Posterior sample files must be in .dat, .txt, or .npz format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "path",
        help=(
            "Path to posterior sample file. Supported formats:\n"
            "  - .dat : whitespace-separated text file with header\n"
            "  - .txt : whitespace-separated text file with header\n"
            "  - .npz : NumPy compressed archive"
        ),
    )

    parser.add_argument(
        "--outdir",
        default="diagnostic_plots",
        help=("Directory where diagnostic summary and plots will be saved."),
    )

    parser.add_argument(
        "--num-chains",
        type=int,
        default=1,
        help=(
            "Number of REAL MCMC chains contained in the input file.\n"
            "Use 1 if the file contains samples from only one chain.\n"
            "Use the true number of chains if chains were concatenated "
            "together in the file."
        ),
    )

    parser.add_argument(
        "--max-lag",
        type=int,
        default=200,
        help=(
            "Maximum lag used when computing autocorrelation "
            "functions and estimating effective sample size."
        ),
    )

    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help=(
            "Optional list of parameter names to diagnose.\n"
            "If omitted, diagnostics are computed for all parameters "
            "found in the posterior sample file."
        ),
    )

    parser.add_argument(
        "--plot-trace",
        action="store_true",
        help=(
            "Generate trace plots for posterior parameters.\n"
            "Trace plots help assess chain mixing and stationarity."
        ),
    )

    parser.add_argument(
        "--plot-histogram",
        action="store_true",
        help=(
            "Generate posterior histogram plots.\n"
            "Histograms visualize marginal posterior distributions."
        ),
    )

    parser.add_argument(
        "--plot-autocorrelation",
        action="store_true",
        help=(
            "Generate autocorrelation plots.\n"
            "Autocorrelation plots help evaluate sampling efficiency "
            "and correlation length."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    samples = load_samples(args.path)

    if args.params is None:
        params = list(samples.keys())
    else:
        params = [p for p in args.params if p in samples]

    print(f"Loaded {len(samples)} parameters.")
    print(f"Diagnosing {len(params)} parameters.")
    print(f"Assuming num_chains = {args.num_chains}")
    print(f"Output directory: {args.outdir}")

    all_stats = {}

    for param in params:
        print(f"Processing {param}")

        stats = diagnose_parameter(
            samples[param],
            num_chains=args.num_chains,
            max_lag=args.max_lag,
        )

        all_stats[param] = stats

        if args.plot_trace:
            save_trace_plot(param, stats, args.outdir)

        if args.plot_histogram:
            save_hist_plot(param, stats, args.outdir)

        if args.plot_autocorrelation:
            save_autocorr_plot(param, stats, args.outdir, args.max_lag)

    write_summary(all_stats, args.outdir, args.num_chains)

    print("Done.")


if __name__ == "__main__":
    main()
