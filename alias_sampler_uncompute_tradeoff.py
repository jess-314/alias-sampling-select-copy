import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from matplotlib.lines import Line2D

from alias_sampler_cirq import build_alias_sampler_circuit


def make_alias_sampler_tables(num_entries, bit_width, seed=0):
    """Create deterministic alias and keep tables for a sweep."""
    rng = np.random.default_rng(seed)
    alias_values = rng.integers(0, num_entries, size=num_entries).tolist()
    keep_values = rng.integers(0, 2**bit_width, size=num_entries).tolist()
    return alias_values, keep_values


def sweep_uncompute_tradeoffs(num_entries_list, seed=0):
    """Collect gate/ancilla tradeoff data for both uncompute paths."""
    rows = []
    for num_entries in num_entries_list:
        bit_width = max(2, (num_entries - 1).bit_length())
        alias_values, keep_values = make_alias_sampler_tables(
            num_entries=num_entries,
            bit_width=bit_width,
            seed=seed + 97 * num_entries + 13 * bit_width,
        )
        for clean_intermediates, mode in [
            (True, "reverse"),
            (False, "measurement"),
        ]:
            _, regs = build_alias_sampler_circuit(
                alias_values,
                keep_values,
                keep_bits=bit_width,
                alias_bits=bit_width,
                clean_intermediates=clean_intermediates,
                use_measurement_uncompute=(mode == "measurement"),
            )
            dirty_budget = regs["dirty_budget"]
            rows.append(
                {
                    "N": num_entries,
                    "bit_width": bit_width,
                    "mode": mode,
                    "dirty_ancillas": dirty_budget["total_dirty_ancillas"],
                    "shared_dirty_scratch": dirty_budget["shared_dirty_scratch"],
                    "result_latches": dirty_budget["result_latches"],
                    "latch_fraction": dirty_budget["result_latches"]
                    / dirty_budget["total_dirty_ancillas"],
                    "gate_count": regs["gate_metrics"]["total_gate_count"],
                    "toffoli_count": regs["gate_metrics"]["toffoli_count"],
                    "lambda": regs["qrom_lambda"],
                }
            )
    return rows


def plot_uncompute_tradeoffs(rows, out_path="alias_sampler_uncompute_tradeoff.png"):
    """Plot gate-vs-ancilla tradeoffs for the two uncompute paths."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot tradeoffs.") from exc

    mode_styles = {
        "reverse": {"linestyle": "-", "marker": "o", "label": "Reverse uncompute"},
        "measurement": {
            "linestyle": "--",
            "marker": "s",
            "label": "Measurement-based uncompute",
        },
    }
    mode_colors = {
        "reverse": "#2f5597",
        "measurement": "#cc6b2c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    ax_tradeoff, ax_budget, ax_latch = axes

    for mode in ("reverse", "measurement"):
        mode_rows = sorted(
            [row for row in rows if row["mode"] == mode],
            key=lambda row: row["N"],
        )
        xs = [row["dirty_ancillas"] for row in mode_rows]
        ys = [row["gate_count"] for row in mode_rows]
        style = mode_styles[mode]
        color = mode_colors[mode]
        ax_tradeoff.plot(
            xs,
            ys,
            color=color,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=5,
            label=style["label"],
        )
        ax_budget.plot(
            [row["N"] for row in mode_rows],
            [row["dirty_ancillas"] for row in mode_rows],
            color=color,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=5,
            label=style["label"],
        )
        ax_latch.plot(
            [row["N"] for row in mode_rows],
            [100.0 * row["latch_fraction"] for row in mode_rows],
            color=color,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=5,
            label=style["label"],
        )

    ax_tradeoff.set_title("Gate vs ancilla tradeoff", fontsize=14)
    ax_tradeoff.set_xlabel("Dirty ancillas", fontsize=12)
    ax_tradeoff.set_ylabel("Total gate count", fontsize=12)
    ax_tradeoff.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_tradeoff.set_xscale("log")
    ax_tradeoff.set_yscale("log")

    ax_budget.set_title("Dirty ancilla budget vs database size", fontsize=14)
    ax_budget.set_xlabel("Database size N", fontsize=12)
    ax_budget.set_ylabel("Total dirty ancillas", fontsize=12)
    ax_budget.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_budget.set_xscale("log")
    ax_budget.set_yscale("log")
    ax_budget.legend(fontsize=8, title="Uncompute path", loc="upper left")

    ax_latch.set_title("Branch wiring latch share", fontsize=14)
    ax_latch.set_xlabel("Database size N", fontsize=12)
    ax_latch.set_ylabel("Latch share of dirty ancillas (%)", fontsize=12)
    ax_latch.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_latch.set_xscale("log")
    ax_latch.set_ylim(0, 100)
    ax_latch.legend(fontsize=8, title="Uncompute path", loc="upper right")

    # Build a compact legend for the uncompute modes.
    mode_handles = []
    for mode in ("reverse", "measurement"):
        mode_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=mode_styles[mode]["linestyle"],
                marker=mode_styles[mode]["marker"],
                linewidth=2,
                label=mode_styles[mode]["label"],
            )
        )

    legend1 = ax_tradeoff.legend(
        handles=mode_handles,
        loc="upper left",
        title="Uncompute path",
        fontsize=8,
    )
    ax_tradeoff.add_artist(legend1)
    fig.suptitle("Alias Sampler Uncompute Tradeoffs", fontsize=15)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    num_entries_list = [4, 5, 8, 16, 32, 64, 128, 256, 512]

    rows = sweep_uncompute_tradeoffs(
        num_entries_list=num_entries_list,
        seed=123,
    )

    print("=== Alias Sampler Uncompute Tradeoff Study ===")
    for row in rows:
        print(
            f"N={row['N']:>3}  bits={row['bit_width']:>2}  mode={row['mode']:<11}  "
            f"dirty={row['dirty_ancillas']:>3}  "
            f"latches={row['result_latches']:>2}  "
            f"latch_share={100.0 * row['latch_fraction']:>5.1f}%  "
            f"gate_count={row['gate_count']:>5}  "
            f"toffoli_count={row['toffoli_count']:>5}  "
            f"lambda={row['lambda']:>2}"
        )

    output_path = plot_uncompute_tradeoffs(rows)
    print(f"Plot written to {output_path}")
