import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from selectcopy import build_select_copy_qrom, pretty_print_select_copy_qrom


def main():
    """Print and render a small, labeled SelectCopy QROM example."""
    data = [1, 2, 3, 0]
    lambda_param = 2
    circuit = build_select_copy_qrom(
        data,
        lambda_param=lambda_param,
        use_measurement_uncompute=True,
        use_reverse_uncompute=False,
    )

    print(
        pretty_print_select_copy_qrom(
            data,
            lambda_param=lambda_param,
            use_measurement_uncompute=True,
            use_reverse_uncompute=False,
        )
    )

    pretty_text = pretty_print_select_copy_qrom(
        data,
        lambda_param=lambda_param,
        use_measurement_uncompute=True,
        use_reverse_uncompute=False,
    )

    lines = pretty_text.splitlines()
    width = 16
    height = max(8, 0.22 * len(lines) + 1.0)
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("white")
    fig.text(
        0.01,
        0.99,
        pretty_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    plt.axis("off")
    out_path = Path("selectcopy_pretty_print.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
