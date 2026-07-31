import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cirq
import numpy as np

from alias_sampler_cirq import build_alias_sampler_circuit
from selectcopy import format_compact_resource_report, loglog_slope


def _parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Sweep full alias-sampler resources and export artifacts."
    )
    parser.add_argument(
        "--num-entries-list",
        default="4,8,16,32,64,128,256,512,1024",
        help="Comma-separated database sizes to sweep.",
    )
    parser.add_argument(
        "--keep-bits-list",
        default="1,2,3,4,5",
        help="Comma-separated keep-bit widths to sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for the deterministic sample tables.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output") / "full_alias_sampler",
        help="Root directory for logs, plots, and QASM exports.",
    )
    return parser


def make_alias_sampler_tables(num_entries, keep_bits, seed=0):
    """Create deterministic alias and keep tables for a scaling sweep."""
    rng = np.random.default_rng(seed)
    alias_values = rng.integers(0, num_entries, size=num_entries).tolist()
    keep_values = rng.integers(0, 2**keep_bits, size=num_entries).tolist()
    return alias_values, keep_values


def choose_alias_sampler_widths(num_entries, keep_bits, alias_gap=2):
    """Return the enforced alias width and threshold width for the sampler."""
    addr_bits = max(1, (num_entries - 1).bit_length())
    alias_bits = addr_bits
    threshold_bits = keep_bits
    return alias_bits, threshold_bits


def run_stamp():
    """Return a compact timestamp suitable for run-unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def make_run_output_dir(root, stamp):
    """Create the per-run output directory for full-alias artifacts."""
    out_dir = Path(root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def format_range(values):
    """Format a set of widths or counts as a compact inclusive range."""
    unique = sorted(set(values))
    if not unique:
        return "n/a"
    if len(unique) == 1:
        return str(unique[0])
    return f"{unique[0]}-{unique[-1]}"


def summarize_register_widths(results):
    """Collect the register-width ranges present in a scaling sweep."""
    return {
        "address": [row["register_widths"]["address"] for row in results],
        "threshold": [row["register_widths"]["threshold"] for row in results],
        "keep": [row["register_widths"]["keep"] for row in results],
        "alias": [row["register_widths"]["alias"] for row in results],
        "compare": [row["register_widths"]["compare"] for row in results],
        "qrom_q": [row["register_widths"]["qrom_q"] for row in results],
        "qrom_r": [row["register_widths"]["qrom_r"] for row in results],
    }


def summarize_keep_bit_slopes(results, group_key, metric_key):
    """Fit one log-log slope per keep-bit slice."""
    slopes = {}
    for keep_bits in sorted({row["keep_bits"] for row in results}):
        rows = sorted(
            [row for row in results if row["keep_bits"] == keep_bits],
            key=lambda row: row["N"],
        )
        sizes = [row["N"] for row in rows]
        values = [row[group_key][metric_key] for row in rows]
        slopes[keep_bits] = loglog_slope(sizes, values)
    return slopes


def print_console_summary(results):
    """Print a slope and register-width summary for the sweep."""
    register_widths = summarize_register_widths(results)
    print(
        "Register widths: "
        f"addr={format_range(register_widths['address'])} bits, "
        f"threshold={format_range(register_widths['threshold'])} bits, "
        f"keep={format_range(register_widths['keep'])} bits, "
        f"alias={format_range(register_widths['alias'])} bits, "
        f"qrom_q={format_range(register_widths['qrom_q'])} bits, "
        f"qrom_r={format_range(register_widths['qrom_r'])} bits"
    )
    for group_key, metric_key, label in [
        ("gate_metrics", "total_gate_count", "Total gates"),
        ("gate_metrics", "toffoli_count", "Toffolis"),
        ("dirty_budget", "total_dirty_ancillas", "Dirty ancillas"),
    ]:
        slopes = summarize_keep_bit_slopes(results, group_key, metric_key)
        slope_text = ", ".join(
            f"keep_bits={keep_bits}: {slope:.2f}"
            for keep_bits, slope in slopes.items()
        )
        print(f"{label} slopes: {slope_text}")


def sweep_full_alias_sampler_scaling(num_entries_list, keep_bits_list, seed=0):
    """Build full alias-sampler circuits across a parameter grid."""
    results = []
    for num_entries in num_entries_list:
        for keep_bits in keep_bits_list:
            alias_bits, threshold_bits = choose_alias_sampler_widths(
                num_entries=num_entries,
                keep_bits=keep_bits,
            )
            alias_values, keep_values = make_alias_sampler_tables(
                num_entries=num_entries,
                keep_bits=keep_bits,
                seed=seed + 97 * num_entries + 13 * keep_bits,
            )
            circuit, regs = build_alias_sampler_circuit(
                alias_values,
                keep_values,
                keep_bits=keep_bits,
                threshold_bits=threshold_bits,
                alias_bits=alias_bits,
                clean_intermediates=True,
                use_measurement_uncompute=True,
            )
            result = {
                "N": num_entries,
                "keep_bits": keep_bits,
                "alias_bits": alias_bits,
                "threshold_bits": threshold_bits,
                "qrom_lambda": regs["qrom_lambda"],
                "register_widths": regs["register_widths"],
                "gate_metrics": regs["gate_metrics"],
                "dirty_budget": regs["dirty_budget"],
                "moment_count": len(circuit),
                "circuit": circuit,
            }
            results.append(result)
    return results


def format_qasm_filename(row, stamp):
    """Build a filename that encodes the circuit's register parameters."""
    widths = row["register_widths"]
    return (
        f"full_alias_sampler_"
        f"N{row['N']}_"
        f"addr{widths['address']}_"
        f"th{widths['threshold']}_"
        f"keep{widths['keep']}_"
        f"alias{widths['alias']}_"
        f"qromq{widths['qrom_q']}_"
        f"qromr{widths['qrom_r']}_"
        f"lam{row['qrom_lambda']}_"
        f"{stamp}.qasm"
    )


def export_qasm_files(results, out_dir, stamp):
    """Write one QASM file per generated circuit."""
    written = []
    for row in results:
        qasm_path = out_dir / format_qasm_filename(row, stamp)
        qasm_str = cirq.qasm(row["circuit"], args=cirq.QasmArgs(version="3.0"))
        qasm_path.write_text(qasm_str, encoding="utf-8")
        written.append(qasm_path)
    return written


def plot_full_alias_sampler_scaling(results, out_path=None):
    """Plot total gates, Toffoli count, and dirty-ancilla budget."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot scaling results.") from exc

    if out_path is None:
        stamp = run_stamp()
        out_path = Path("output") / "full_alias_sampler" / stamp / f"full_alias_sampler_scaling_{stamp}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def plot_positive_log_series(ax, rows, group_key, metric_key, label):
        """Plot only positive points on a log-log axis.

        Returns True if any points were drawn.
        """
        positive_rows = [
            row for row in rows if row[group_key][metric_key] > 0 and row["N"] > 0
        ]
        if not positive_rows:
            return False

        sizes = [row["N"] for row in positive_rows]
        values = [row[group_key][metric_key] for row in positive_rows]
        ax.loglog(sizes, values, marker="o", linewidth=2, label=label)
        return True

    keep_bit_values = sorted({row["keep_bits"] for row in results})
    metric_keys = [
        ("gate_metrics", "total_gate_count", "Total gates", "Count"),
        ("gate_metrics", "toffoli_count", "Toffolis", "Count"),
        ("dirty_budget", "total_dirty_ancillas", "Dirty ancillas", "Qubits"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    axes = axes.ravel()

    register_summary = None
    for ax, (group_key, metric_key, title, ylabel) in zip(axes, metric_keys):
        plotted_any = False
        register_widths = {
            "address": [],
            "threshold": [],
            "keep": [],
            "alias": [],
            "compare": [],
            "qrom_q": [],
            "qrom_r": [],
        }
        for keep_bits in keep_bit_values:
            rows = sorted(
                [row for row in results if row["keep_bits"] == keep_bits],
                key=lambda row: row["N"],
            )
            sizes = [row["N"] for row in rows]
            values = [row[group_key][metric_key] for row in rows]
            slope = loglog_slope(sizes, values)
            plotted_any |= plot_positive_log_series(
                ax,
                rows,
                group_key,
                metric_key,
                f"keep_bits={keep_bits}, slope={slope:.2f}",
            )
            for row in rows:
                reg = row["register_widths"]
                for key in register_widths:
                    register_widths[key].append(reg[key])
        slope_summary_parts = []
        for keep_bits in keep_bit_values:
            rows = sorted(
                [row for row in results if row["keep_bits"] == keep_bits],
                key=lambda row: row["N"],
            )
            slope = loglog_slope(
                [row["N"] for row in rows],
                [row[group_key][metric_key] for row in rows],
            )
            slope_summary_parts.append(f"k{keep_bits}={slope:.2f}")
        slope_summary = ", ".join(slope_summary_parts)
        ax.set_title(f"{title}\nslopes: {slope_summary}")
        ax.set_xlabel("Database size N")
        ax.set_ylabel(ylabel)
        if plotted_any:
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
        else:
            ax.text(
                0.5,
                0.5,
                "All values are zero",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0].legend(fontsize=8, title="Keep bits")
    register_summary = (
        f"addr={format_range(register_widths['address'])} bits, "
        f"threshold={format_range(register_widths['threshold'])} bits, "
        f"keep={format_range(register_widths['keep'])} bits, "
        f"alias={format_range(register_widths['alias'])} bits, "
        f"qrom_q={format_range(register_widths['qrom_q'])} bits, "
        f"qrom_r={format_range(register_widths['qrom_r'])} bits"
    )
    fig.suptitle(
        f"Full Alias Sampler Resource Scaling\nregisters: {register_summary}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


class Tee:
    """Write text to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    num_entries_list = _parse_csv_ints(args.num_entries_list)
    keep_bits_list = _parse_csv_ints(args.keep_bits_list)
    stamp = run_stamp()
    output_dir = make_run_output_dir(args.output_root, stamp)
    log_path = output_dir / f"full_alias_sampler_scaling_{stamp}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        try:
            results = sweep_full_alias_sampler_scaling(
                num_entries_list=num_entries_list,
                keep_bits_list=keep_bits_list,
                seed=123,
            )

            print("=== Full Alias Sampler Scaling Study ===")
            print_console_summary(results)
            print("\n--- Per-instance resource summary ---")
            for row in results:
                metrics = row["gate_metrics"]
                print(
                    f"N={row['N']:>3}  keep_bits={row['keep_bits']:>2}  "
                    f"alias_bits={row['alias_bits']:>2}  "
                    f"lambda={row['qrom_lambda']:>2}  "
                    + format_compact_resource_report(metrics)
                )

            qasm_paths = export_qasm_files(results, output_dir, stamp)
            print("\n--- QASM exports ---")
            for path in qasm_paths:
                print(path)

            output_path = plot_full_alias_sampler_scaling(
                results,
                out_path=output_dir / f"full_alias_sampler_scaling_{stamp}.png",
            )
            print(f"Plot written to {output_path}")
            print(f"Log written to {log_path}")
        finally:
            sys.stdout = original_stdout
