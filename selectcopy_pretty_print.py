import argparse
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from selectcopy import pretty_print_select_copy_qrom


def _parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Print and render a small SelectCopy QROM example."
    )
    parser.add_argument(
        "--data",
        default="1,2,3,0",
        help="Comma-separated classical table values.",
    )
    parser.add_argument(
        "--lambda-param",
        type=int,
        default=2,
        help="SelectCopy block size parameter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "figures",
        help="Directory for the rendered PNG.",
    )
    return parser


def main():
    """Print and render a small, labeled SelectCopy QROM example."""
    args = build_arg_parser().parse_args()
    data = _parse_csv_ints(args.data)
    lambda_param = args.lambda_param
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pretty_text = pretty_print_select_copy_qrom(
        data,
        lambda_param=lambda_param,
        use_measurement_uncompute=True,
        use_reverse_uncompute=False,
    )
    print(pretty_text)

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
    out_path = output_dir / "selectcopy_pretty_print.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
