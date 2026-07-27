import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from selectcopy import build_select_copy_qrom, summarize_select_copy_layout


def make_sample_database(num_entries, bitsize, seed=0, density=0.5):
    """Create a deterministic pseudo-random database for the schematic."""
    rng = np.random.default_rng(seed)
    data = (rng.random((num_entries, bitsize)) < density).astype(int)
    return data.tolist()


def draw_forked_grouped_selectcopy_schematic(
    out_path="selectcopy_pretty_print_forked.png",
):
    """Render a grouped SelectCopy QROM schematic with one block per big step."""
    num_entries = 16
    bitsize = 4
    lambda_param = 4

    data = make_sample_database(num_entries, bitsize, seed=20260727, density=0.5)
    circuit = build_select_copy_qrom(
        data,
        lambda_param=lambda_param,
        use_measurement_uncompute=True,
        use_reverse_uncompute=False,
    )
    layout = summarize_select_copy_layout(data, lambda_param)

    dirty_blocks = len(layout["dirty_ancillas"])
    dirty_bits = len(layout["dirty_ancillas"][0]) if layout["dirty_ancillas"] else 0

    register_rows = [
        ("q_reg[2]", "q"),
        ("r_reg[2]", "r"),
        ("target[4]", "t"),
        (f"dirty[{dirty_blocks}x{dirty_bits}]", "dirty"),
        ("w_reg[3]", "w"),
        ("phase_work[1]", "phase"),
    ]

    y_positions = {key: i for i, (_label, key) in enumerate(register_rows[::-1])}
    max_y = len(register_rows) - 1

    phase_blocks = [
        {
            "title": "1. Sel1 lookup",
            "subtitle": "q_reg -> target + dirty",
            "rows": {"q", "t", "dirty"},
            "color": "#9ecae1",
            "x": 1.35,
            "w": 2.25,
        },
        {
            "title": "2. One-hot decode",
            "subtitle": "r_reg -> w_reg",
            "rows": {"r", "w"},
            "color": "#fdae6b",
            "x": 3.95,
            "w": 1.75,
        },
        {
            "title": "3. Copy stage",
            "subtitle": "w_reg + dirty -> target",
            "rows": {"w", "dirty", "t"},
            "color": "#a1d99b",
            "x": 5.95,
            "w": 2.10,
        },
        {
            "title": "4. Uncompute flags",
            "subtitle": "r_reg + w_reg restored",
            "rows": {"r", "w"},
            "color": "#c4b5fd",
            "x": 8.35,
            "w": 1.90,
        },
        {
            "title": "5. MBU cleanup",
            "subtitle": "measure dirty, phase fix, reset",
            "rows": {"q", "dirty", "phase"},
            "color": "#f2b6c6",
            "x": 10.55,
            "w": 2.15,
        },
    ]

    fig, ax = plt.subplots(figsize=(21, 10.5))
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.87, bottom=0.08)
    ax.set_xlim(0, 13.3)
    ax.set_ylim(-1.0, max_y + 1.15)
    ax.axis("off")

    fig.suptitle(
        "Forked SelectCopy QROM Schematic",
        fontsize=21,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.93,
        "Example: N=16, bits=4, lambda=4. Five grouped operations are shown as single blocks on the affected registers.",
        ha="center",
        va="top",
        fontsize=11,
    )

    for idx, (label, _key) in enumerate(register_rows[::-1]):
        y = idx
        ax.hlines(y, 0.55, 12.85, color="#606060", linewidth=1.0, alpha=0.75)
        ax.text(
            0.30,
            y,
            label,
            ha="right",
            va="center",
            fontsize=11,
            family="monospace",
        )

    ax.hlines(max_y + 0.45, 1.15, 12.85, color="#888888", linewidth=1.2, alpha=0.8)

    def draw_block(block, idx):
        touched = [y_positions[row] for row in block["rows"]]
        y0 = min(touched) - 0.40
        y1 = max(touched) + 0.40
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
            y1 - 0.10,
            block["title"],
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            block["x"] + block["w"] / 2,
            (y0 + y1) / 2 - 0.03,
            textwrap.fill(block["subtitle"], width=26),
            ha="center",
            va="center",
            fontsize=10.5,
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

    ax.text(
        0.56,
        -0.65,
        "Forked layout: one SelectCopy QROM, five grouped steps, measurement-based cleanup.",
        fontsize=10.5,
        color="#333333",
        ha="left",
        va="center",
    )

    # Keep a reference to the actual circuit so the example is rooted in the
    # implemented SelectCopy QROM, even though the image is a grouped schematic.
    _ = circuit

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = draw_forked_grouped_selectcopy_schematic()
    print(f"Wrote {path}")
