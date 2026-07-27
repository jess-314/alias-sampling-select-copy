import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from matplotlib.lines import Line2D

from qualtran.surface_code import (
    AlgorithmSummary,
    CompactDataBlock,
    FifteenToOne,
    PhysicalCostModel,
    PhysicalParameters,
    QECScheme,
)
from qualtran.resource_counting import GateCounts

from alias_sampler_cirq import build_alias_sampler_circuit
from full_alias_sampler_scaling import (
    choose_alias_sampler_widths,
    format_range,
    make_alias_sampler_tables,
    make_run_output_dir,
    run_stamp,
)
from selectcopy import loglog_slope


def make_logical_summary(circuit, gate_metrics):
    """Convert the alias sampler counts into a Qualtran AlgorithmSummary."""
    logical_gates = GateCounts(toffoli=gate_metrics["toffoli_count"])
    return AlgorithmSummary(
        n_algo_qubits=len(circuit.all_qubits()),
        n_logical_gates=logical_gates,
    )


def build_fixed_distance_model(data_d):
    """Use Qualtran's compact surface-code model with a fixed code distance."""
    return PhysicalCostModel.make_beverland_et_al(data_d=data_d)


def build_budgeted_distance_model(data_d):
    """Build the same model, but with a code distance chosen from a budget."""
    return PhysicalCostModel(
        physical_params=PhysicalParameters.make_beverland_et_al(),
        data_block=CompactDataBlock(data_d=data_d),
        factory=FifteenToOne(d_X=9, d_Z=3, d_m=3),
        qec_scheme=QECScheme.make_beverland_et_al(),
    )


def estimate_budgeted_distance(logical_summary, total_error_budget=1e-2):
    """Pick a conservative code distance from the logical Toffoli budget.

    Qualtran's `GateCounts.total_t_count()` applies the default 4 T per Toffoli
    conversion, so this stays aligned with the library's physical-cost model.
    """
    qec_scheme = QECScheme.make_beverland_et_al()
    physical_error = PhysicalParameters.make_beverland_et_al().physical_error
    qualtran_t_equiv_count = max(1, logical_summary.n_logical_gates.total_t_count())
    per_t_budget = total_error_budget / qualtran_t_equiv_count
    return qec_scheme.code_distance_from_budget(physical_error, per_t_budget)


def sweep_alias_sampler_physical_scaling(num_entries_list, keep_bits_list, seed=0):
    """Build logical and physical resource estimates for the alias sampler."""
    rows = []
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
            logical_summary = make_logical_summary(circuit, regs["gate_metrics"])
            fixed_model = build_fixed_distance_model(data_d=15)
            fixed_phys_qubits = fixed_model.n_phys_qubits(logical_summary)

            budgeted_d = estimate_budgeted_distance(logical_summary)
            budgeted_model = build_budgeted_distance_model(data_d=budgeted_d)
            budgeted_phys_qubits = budgeted_model.n_phys_qubits(logical_summary)

            rows.append(
                {
                    "N": num_entries,
                    "keep_bits": keep_bits,
                    "alias_bits": alias_bits,
                    "threshold_bits": threshold_bits,
                    "register_widths": regs["register_widths"],
                    "gate_metrics": regs["gate_metrics"],
                    "logical_qubits": logical_summary.n_algo_qubits,
                    "logical_toffoli_count": logical_summary.n_logical_gates.toffoli,
                    "qualtran_t_equiv_count": logical_summary.n_logical_gates.total_t_count(),
                    "budgeted_d": budgeted_d,
                    "fixed_phys_qubits": fixed_phys_qubits,
                    "budgeted_phys_qubits": budgeted_phys_qubits,
                    "circuit": circuit,
                }
            )
    return rows


def format_scale_summary(rows):
    """Return a compact textual summary of the logical and physical ranges."""
    logical_qubits = [row["logical_qubits"] for row in rows]
    toffoli_counts = [row["logical_toffoli_count"] for row in rows]
    code_distances = [row["budgeted_d"] for row in rows]
    return (
        f"logical_qubits={format_range(logical_qubits)}  "
        f"logical_toffolis={format_range(toffoli_counts)}  "
        f"budgeted_d={format_range(code_distances)}"
    )


def print_console_summary(rows):
    """Print slopes and resource ranges for the scaling sweep."""
    print(f"Logical / code-distance ranges: {format_scale_summary(rows)}")
    for metric_key, label in [
        ("logical_qubits", "Logical qubits"),
        ("logical_toffoli_count", "Logical Toffolis"),
        ("fixed_phys_qubits", "Fixed-distance physical qubits"),
        ("budgeted_phys_qubits", "Budgeted physical qubits"),
    ]:
        slopes = {}
        for keep_bits in sorted({row["keep_bits"] for row in rows}):
            keep_rows = sorted(
                [row for row in rows if row["keep_bits"] == keep_bits],
                key=lambda row: row["N"],
            )
            slopes[keep_bits] = loglog_slope(
                [row["N"] for row in keep_rows],
                [row[metric_key] for row in keep_rows],
            )
        slope_text = ", ".join(
            f"keep_bits={keep_bits}: {slope:.2f}" for keep_bits, slope in slopes.items()
        )
        print(f"{label} slopes: {slope_text}")


def plot_alias_sampler_physical_scaling(rows, out_path=None):
    """Plot logical and physical scaling curves side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot scaling results.") from exc

    if out_path is None:
        out_path = f"alias_sampler_physical_scaling_{run_stamp()}.png"

    keep_bit_values = sorted({row["keep_bits"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)
    ax_logical, ax_distance, ax_physical = axes

    logical_qubit_handle = Line2D(
        [0],
        [0],
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2,
        label="Solid: logical qubits",
    )
    logical_t_handle = Line2D(
        [0],
        [0],
        color="#ff7f0e",
        linestyle="--",
        marker="s",
        linewidth=2,
        label="Dashed: logical Toffoli count",
    )
    code_distance_handle = Line2D(
        [0],
        [0],
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2,
        label="Solid: budgeted code distance",
    )
    fixed_phys_handle = Line2D(
        [0],
        [0],
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2,
        label="Solid: fixed-distance physical qubits",
    )
    budgeted_phys_handle = Line2D(
        [0],
        [0],
        color="#ff7f0e",
        linestyle="--",
        marker="s",
        linewidth=2,
        label="Dashed: budgeted physical qubits",
    )

    def asymptotic_shape(sizes, kind):
        """Return a normalized asymptotic reference curve."""
        sizes = np.asarray(sizes, dtype=float)
        safe_sizes = np.maximum(sizes, 2.0)
        if kind == "sqrt_log":
            values = np.sqrt(safe_sizes) * np.log2(safe_sizes)
        elif kind == "N_log2":
            values = safe_sizes * np.log2(safe_sizes) ** 2
        elif kind == "log":
            values = np.log2(safe_sizes)
        elif kind == "sqrt_log3":
            values = np.sqrt(safe_sizes) * np.log2(safe_sizes) ** 3
        else:
            raise ValueError(f"Unknown asymptotic shape: {kind}")
        return values / values[0]

    def add_theory_guide(ax, sizes, kind, anchor_value, label):
        """Overlay a faint normalized reference curve on the current axes."""
        guide = anchor_value * asymptotic_shape(sizes, kind)
        ax.loglog(
            sizes,
            guide,
            color="#666666",
            linestyle=":",
            linewidth=2.2,
            alpha=0.8,
            label="_nolegend_",
        )

    def add_panel_text(ax, lines, loc=(0.03, 0.97), va="top"):
        """Add a boxed annotation in axes coordinates."""
        ax.text(
            loc[0],
            loc[1],
            "\n".join(lines),
            transform=ax.transAxes,
            va=va,
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.9,
            },
        )

    for keep_bits in keep_bit_values:
        keep_rows = sorted(
            [row for row in rows if row["keep_bits"] == keep_bits],
            key=lambda row: row["N"],
        )
        sizes = [row["N"] for row in keep_rows]

        logical_qubits = [row["logical_qubits"] for row in keep_rows]
        toffoli_counts = [row["logical_toffoli_count"] for row in keep_rows]
        budgeted_ds = [row["budgeted_d"] for row in keep_rows]
        fixed_phys = [row["fixed_phys_qubits"] for row in keep_rows]
        budgeted_phys = [row["budgeted_phys_qubits"] for row in keep_rows]

        logical_qubit_slope = loglog_slope(sizes, logical_qubits)
        toffoli_count_slope = loglog_slope(sizes, toffoli_counts)
        fixed_phys_slope = loglog_slope(sizes, fixed_phys)
        budgeted_phys_slope = loglog_slope(sizes, budgeted_phys)

        ax_logical.loglog(
            sizes,
            logical_qubits,
            marker="o",
            linewidth=2,
            label=f"keep_bits={keep_bits}, slope={logical_qubit_slope:.2f}",
        )
        ax_logical.loglog(
            sizes,
            toffoli_counts,
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"Toffolis, k={keep_bits}, slope={toffoli_count_slope:.2f}",
        )
        if keep_bits == keep_bit_values[0]:
            add_theory_guide(
                ax_logical,
                sizes,
                "sqrt_log",
                logical_qubits[0],
                "Theory: O(sqrt(N) log N)",
            )
            add_theory_guide(
                ax_logical,
                sizes,
                "N_log2",
                toffoli_counts[0],
                "Theory: O(N log^2 N) Toffolis",
            )

        ax_distance.loglog(
            sizes,
            budgeted_ds,
            marker="o",
            linewidth=2,
            label=f"keep_bits={keep_bits}, slope={loglog_slope(sizes, budgeted_ds):.2f}",
        )
        if keep_bits == keep_bit_values[0]:
            add_theory_guide(
                ax_distance,
                sizes,
                "log",
                budgeted_ds[0],
                "Theory: O(log N)",
            )

        ax_physical.loglog(
            sizes,
            fixed_phys,
            marker="o",
            linewidth=2,
            label=f"fixed d=15, k={keep_bits}, slope={fixed_phys_slope:.2f}",
        )
        ax_physical.loglog(
            sizes,
            budgeted_phys,
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"budgeted d, k={keep_bits}, slope={budgeted_phys_slope:.2f}",
        )
        if keep_bits == keep_bit_values[0]:
            add_theory_guide(
                ax_physical,
                sizes,
                "sqrt_log",
                fixed_phys[0],
                "Theory: O(sqrt(N) log N)",
            )
            add_theory_guide(
                ax_physical,
                sizes,
                "sqrt_log3",
                budgeted_phys[0],
                "Theory: O(sqrt(N) log^3 N)",
            )

    logical_keep_slopes = [
        loglog_slope(
            [row["N"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
            [row["logical_qubits"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
        )
        for keep_bits in keep_bit_values
    ]
    toffoli_keep_slopes = [
        loglog_slope(
            [row["N"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
            [row["logical_toffoli_count"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
        )
        for keep_bits in keep_bit_values
    ]
    fixed_keep_slopes = [
        loglog_slope(
            [row["N"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
            [row["fixed_phys_qubits"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
        )
        for keep_bits in keep_bit_values
    ]
    budgeted_keep_slopes = [
        loglog_slope(
            [row["N"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
            [row["budgeted_phys_qubits"] for row in sorted([r for r in rows if r["keep_bits"] == keep_bits], key=lambda row: row["N"])],
        )
        for keep_bits in keep_bit_values
    ]

    ax_logical.set_title("Logical resources")
    ax_logical.set_xlabel("Database size N")
    ax_logical.set_ylabel("Count")
    ax_logical.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_logical.legend(handles=[logical_qubit_handle, logical_t_handle], fontsize=8)
    add_panel_text(
        ax_logical,
        [
            "Theory:",
            "qubits O(sqrt(N) log N)",
            "Toffolis O(N log^2 N)",
            f"Actual fit: qubits ~ N^{min(logical_keep_slopes):.2f}-{max(logical_keep_slopes):.2f}",
            f"Actual fit: Toffolis ~ N^{min(toffoli_keep_slopes):.2f}-{max(toffoli_keep_slopes):.2f}",
        ],
        loc=(0.52, 0.04),
        va="bottom",
    )

    ax_distance.set_title("Budgeted code distance")
    ax_distance.set_xlabel("Database size N")
    ax_distance.set_ylabel("Surface-code distance d")
    ax_distance.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_distance.legend(handles=[code_distance_handle], fontsize=8)
    add_panel_text(
        ax_distance,
        [
            "Theory:",
            "distance O(log N)",
            "from a fixed error budget",
            "Actual: integer step function",
        ],
    )

    ax_physical.set_title("Physical qubit footprint")
    ax_physical.set_xlabel("Database size N")
    ax_physical.set_ylabel("Physical qubits")
    ax_physical.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_physical.legend(handles=[fixed_phys_handle, budgeted_phys_handle], fontsize=8)
    add_panel_text(
        ax_physical,
        [
            "Theory:",
            "fixed d O(sqrt(N) log N)",
            "budgeted d O(sqrt(N) log^3 N)",
            f"Actual fit: fixed ~ N^{min(fixed_keep_slopes):.2f}-{max(fixed_keep_slopes):.2f}",
            f"Actual fit: budgeted ~ N^{min(budgeted_keep_slopes):.2f}-{max(budgeted_keep_slopes):.2f}",
        ],
        loc=(0.52, 0.04),
        va="bottom",
    )

    fig.suptitle(
        "Alias Sampler Logical-to-Physical Scaling with Qualtran",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def write_model_note(out_dir, rows):
    """Write a short text note explaining the physical model choices."""
    note = (
        "Physical model summary\n"
        "\n"
        "1. Logical counts come from the alias-sampler circuit: the algorithm qubit\n"
        "   count is the number of qubits in the Cirq circuit, and the logical\n"
        "   non-Clifford cost is the Toffoli count.\n"
        "2. The fixed-distance curve uses Qualtran's Beverland-et-al physical-cost\n"
        "   model with data_d=15, which isolates the surface-code encoding overhead.\n"
        "3. The budgeted curve uses the same Qualtran ingredients, but chooses the\n"
        "   code distance from a fixed per-algorithm failure budget distributed over\n"
        "   the logical Toffoli count. Qualtran then converts Toffolis to T\n"
        "   gates using its default 4 T per Toffoli convention. That makes the\n"
        "   physical footprint grow when the logical non-Clifford load grows.\n"
        "\n"
        f"Observed ranges: {format_scale_summary(rows)}\n"
    )
    note_path = out_dir / "note.txt"
    note_path.write_text(note, encoding="utf-8")
    return note_path


if __name__ == "__main__":
    num_entries_list = [4, 8, 16, 32, 64, 128, 256, 512]
    keep_bits_list = [1, 2, 3, 4, 5]
    stamp = run_stamp()
    output_dir = make_run_output_dir(f"{stamp}_physical")

    rows = sweep_alias_sampler_physical_scaling(
        num_entries_list=num_entries_list,
        keep_bits_list=keep_bits_list,
        seed=123,
    )

    print("=== Alias Sampler Physical Scaling Study ===")
    print_console_summary(rows)
    print("\n--- Per-instance resource summary ---")
    for row in rows:
        print(
            f"N={row['N']:>3}  keep_bits={row['keep_bits']:>2}  "
            f"addr={row['register_widths']['address']:>2}  "
            f"logical_qubits={row['logical_qubits']:>3}  "
            f"toffolis={row['logical_toffoli_count']:>5}  "
            f"d_budget={row['budgeted_d']:>2}  "
            f"phys_fixed={row['fixed_phys_qubits']:>5}  "
            f"phys_budgeted={row['budgeted_phys_qubits']:>5}"
        )

    note_path = write_model_note(output_dir, rows)
    print(f"\nModel note written to {note_path}")

    output_path = plot_alias_sampler_physical_scaling(
        rows,
        out_path=output_dir / f"alias_sampler_physical_scaling_{stamp}.png",
    )
    print(f"Plot written to {output_path}")
