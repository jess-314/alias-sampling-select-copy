import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from alias_sampler_cirq import build_alias_sampler_circuit


def make_alias_sampler_tables(num_entries, keep_bits, seed=0):
    """Create deterministic alias and keep tables for the schematic example."""
    rng = np.random.default_rng(seed)
    alias_values = rng.integers(0, num_entries, size=num_entries).tolist()
    keep_values = rng.integers(0, 2**keep_bits, size=num_entries).tolist()
    return alias_values, keep_values


def draw_grouped_alias_sampler_schematic(out_path="alias_sampler_grouped_steps.png"):
    """Render a grouped block schematic for one SelectCopy QROM instance."""
    num_entries = 32
    keep_bits = 5
    alias_bits = 5
    threshold_bits = 5
    lambda_param = 8

    alias_values, keep_values = make_alias_sampler_tables(
        num_entries=num_entries,
        keep_bits=keep_bits,
        seed=20260710,
    )
    _circuit, regs = build_alias_sampler_circuit(
        alias_values,
        keep_values,
        keep_bits=keep_bits,
        threshold_bits=threshold_bits,
        alias_bits=alias_bits,
        lambda_param=lambda_param,
        clean_intermediates=True,
        use_measurement_uncompute=True,
    )

    dirty = regs["dirty_budget"]["shared_dirty_scratch"]
    register_rows = [
        ("addr[5]", "addr"),
        ("threshold[5]", "threshold"),
        ("keep_reg[5]", "keep"),
        ("alias_reg[5]", "alias"),
        ("out[5]", "out"),
        ("sample_branch", "branch"),
        (f"shared_scratch[{dirty}]", "scratch"),
        ("cmp_z", "cmp"),
    ]

    phase_blocks = [
        {
            "title": "1. Load keep table",
            "subtitle": "addr -> keep_reg + shared_scratch",
            "rows": {"addr", "keep", "scratch"},
            "color": "#9ecae1",
            "x": 1.30,
            "w": 1.95,
        },
        {
            "title": "2. Compare",
            "subtitle": "threshold vs keep_reg -> cmp_z",
            "rows": {"threshold", "keep", "scratch", "cmp"},
            "color": "#fdae6b",
            "x": 3.45,
            "w": 1.75,
        },
        {
            "title": "3. Load alias table",
            "subtitle": "addr -> alias_reg + shared_scratch",
            "rows": {"addr", "alias", "scratch"},
            "color": "#a1d99b",
            "x": 5.45,
            "w": 1.95,
        },
        {
            "title": "4. Select and copy",
            "subtitle": "cmp_z -> sample_branch; out <- addr or alias_reg",
            "rows": {"addr", "alias", "out", "branch"},
            "color": "#c4b5fd",
            "x": 7.55,
            "w": 2.00,
        },
        {
            "title": "5. MBU cleanup",
            "subtitle": "measure dirty outputs, classically fix phase, reset scratch",
            "rows": {"addr", "keep", "alias", "scratch", "branch", "cmp"},
            "color": "#f2b6c6",
            "x": 9.85,
            "w": 1.95,
        },
    ]

    y_positions = {key: i for i, (_label, key) in enumerate(register_rows[::-1])}
    max_y = len(register_rows) - 1

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.87, bottom=0.08)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(-1.0, max_y + 1.25)
    ax.axis("off")

    # Title block.
    fig.suptitle(
        "Grouped SelectCopy QROM Schematic",
        fontsize=20,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.93,
        "Example: N=32, keep_bits=5, alias_bits=5, lambda=8. Measurement-based uncompute is enabled.",
        ha="center",
        va="top",
        fontsize=11,
    )

    # Register wires.
    for idx, (label, key) in enumerate(register_rows[::-1]):
        y = idx
        ax.hlines(y, 0.55, 11.85, color="#606060", linewidth=1.0, alpha=0.75)
        ax.text(
            0.30,
            y,
            label,
            ha="right",
            va="center",
            fontsize=11,
            family="monospace",
        )

    # A thin timeline rail above the blocks.
    ax.hlines(max_y + 0.45, 1.15, 11.85, color="#888888", linewidth=1.2, alpha=0.8)

    def draw_block(block, idx):
        touched = [y_positions[row] for row in block["rows"]]
        y0 = min(touched) - 0.42
        y1 = max(touched) + 0.42
        patch = FancyBboxPatch(
            (block["x"], y0),
            block["w"],
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=2.0,
            edgecolor=block["color"],
            facecolor=block["color"],
            alpha=0.18,
        )
        ax.add_patch(patch)
        ax.text(
            block["x"] + 0.06,
            y1 - 0.12,
            block["title"],
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            block["x"] + block["w"] / 2,
            (y0 + y1) / 2 - 0.04,
            textwrap.fill(block["subtitle"], width=28),
            ha="center",
            va="center",
            fontsize=10.6,
            family="sans-serif",
        )
        ax.text(
            block["x"] + block["w"] - 0.04,
            y0 + 0.02,
            f"{idx + 1}",
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#444444",
        )

    for idx, block in enumerate(phase_blocks):
        draw_block(block, idx)

    ax.text(0.56, -0.65, "Five grouped operations: keep-load, compare, alias-load, controlled copy, measurement-based cleanup.",
            fontsize=10.5, color="#333333", ha="left", va="center")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = draw_grouped_alias_sampler_schematic()
    print(f"Wrote {path}")
