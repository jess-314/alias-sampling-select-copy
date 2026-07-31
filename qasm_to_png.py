import argparse
import os
import re
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cirq
import matplotlib.pyplot as plt


class LabelGate(cirq.Gate):
    """Minimal labeled gate for circuit-diagram rendering."""

    def __init__(self, label, num_qubits=1):
        self._label = label
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    def _circuit_diagram_info_(self, args):
        if self._num_qubits == 1:
            return self._label
        return tuple(self._label for _ in range(self._num_qubits))


def _parse_qubit_names(qasm_text):
    match = re.search(r"// Qubits:\s*\[(.*)\]", qasm_text)
    if not match:
        raise ValueError("Could not find qubit name comment in QASM file.")
    names = [
        part.strip()
        for part in match.group(1).split(",")
        if part.strip()
    ]
    return [cirq.NamedQubit(name) for name in names]


def _parse_gate_invocation(text, qubits_by_index):
    text = text.strip().rstrip(";")

    if text.startswith("x "):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return cirq.X(qubits_by_index[idx])
    if text.startswith("h "):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return cirq.H(qubits_by_index[idx])
    if text.startswith("z "):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return cirq.Z(qubits_by_index[idx])
    if text.startswith("t "):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return cirq.T(qubits_by_index[idx])
    if text.startswith("tdg "):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return LabelGate("Tdg").on(qubits_by_index[idx])
    if text.startswith("cx "):
        a, b = [int(x) for x in re.findall(r"q\[(\d+)\]", text)]
        return cirq.CNOT(qubits_by_index[a], qubits_by_index[b])
    if text.startswith("ccx "):
        a, b, c = [int(x) for x in re.findall(r"q\[(\d+)\]", text)]
        return cirq.CCX(qubits_by_index[a], qubits_by_index[b], qubits_by_index[c])
    if text.startswith("reset "):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return cirq.reset(qubits_by_index[idx])
    if "measure" in text:
        key = re.match(r"([A-Za-z_]\w*)\[\d+\]\s*=\s*measure\s+q\[(\d+)\]", text)
        if not key:
            raise ValueError(f"Could not parse measurement: {text}")
        meas_key, idx = key.group(1), int(key.group(2))
        return cirq.measure(qubits_by_index[idx], key=meas_key)
    if text.startswith("rz("):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return LabelGate("Rz").on(qubits_by_index[idx])
    if text.startswith("u3("):
        idx = int(re.search(r"q\[(\d+)\]", text).group(1))
        return LabelGate("U3").on(qubits_by_index[idx])

    generic = re.match(r"([a-z][a-z0-9_]*)(?:\(.+\))?\s+q\[(\d+)\]$", text)
    if generic:
        gate_name, idx = generic.group(1), int(generic.group(2))
        return LabelGate(gate_name.upper()).on(qubits_by_index[idx])

    raise ValueError(f"Unsupported QASM operation: {text}")


def _line_qubit_names(line, qubits_by_index):
    return [
        str(qubits_by_index[int(idx)])
        for idx in re.findall(r"q\[(\d+)\]", line)
    ]


def parse_cirq_qasm3(qasm_text):
    """Parse the Cirq-generated QASM subset used by this project."""
    qubits = _parse_qubit_names(qasm_text)
    qubits_by_index = {i: q for i, q in enumerate(qubits)}
    circuit = cirq.Circuit()

    for raw_line in qasm_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("OPENQASM") or line.startswith("include "):
            continue
        if line.startswith("qubit[") or line.startswith("bit["):
            continue

        if line.startswith("if ("):
            match = re.match(r"if \(([^)]+)!=0\)\s+(.+);$", line)
            if not match:
                raise ValueError(f"Could not parse classical control: {line}")
            meas_key, op_text = match.group(1), match.group(2)
            op = _parse_gate_invocation(op_text + ";", qubits_by_index)
            circuit.append(op.with_classical_controls(meas_key))
            continue

        op = _parse_gate_invocation(line, qubits_by_index)
        circuit.append(op)

    return circuit


def _advance_functional_stage(line, phase, qubits_by_index):
    """Heuristically label the major sampler phases in the exported QASM."""
    names = _line_qubit_names(line, qubits_by_index)

    def has_prefix(prefix):
        return any(name == prefix or name.startswith(prefix + "_") for name in names)

    touches_comparator = has_prefix("u") or has_prefix("cmp_z")
    touches_alias = has_prefix("alias")
    touches_output = has_prefix("out") or has_prefix("sample_branch")

    if phase == "tail":
        return "tail"

    if phase in {None, "prelude"}:
        if touches_comparator:
            return "comparator"
        return "QROM_keep"

    if phase == "QROM_keep":
        if touches_comparator:
            return "comparator"
        return "QROM_keep"

    if phase == "comparator":
        if touches_alias and not touches_output:
            return "QROM_alias"
        return "comparator"

    if phase == "QROM_alias":
        if touches_output:
            return "tail"
        return "QROM_alias"

    return phase


def parse_cirq_qasm3_blocking(qasm_text):
    """Parse QASM while collapsing major sampler regions into labeled gates."""
    qubits = _parse_qubit_names(qasm_text)
    qubits_by_index = {i: q for i, q in enumerate(qubits)}
    moments = []

    active_stage = None
    active_qubits = set()
    active_phase = "prelude"

    def flush_active_stage():
        nonlocal active_stage, active_qubits
        if not active_stage or not active_qubits:
            active_stage = None
            active_qubits = set()
            return
        ordered = sorted(active_qubits, key=str)
        moments.append(cirq.Moment([LabelGate(active_stage, num_qubits=len(ordered)).on(*ordered)]))
        active_stage = None
        active_qubits = set()

    for raw_line in qasm_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("OPENQASM") or line.startswith("include "):
            continue
        if line.startswith("qubit[") or line.startswith("bit["):
            continue

        if line.startswith("if ("):
            match = re.match(r"if \(([^)]+)!=0\)\s+(.+);$", line)
            if not match:
                raise ValueError(f"Could not parse classical control: {line}")
            meas_key, op_text = match.group(1), match.group(2)
            op = _parse_gate_invocation(op_text + ";", qubits_by_index)
            op = op.with_classical_controls(meas_key)
        else:
            op = _parse_gate_invocation(line, qubits_by_index)

        next_phase = _advance_functional_stage(line, active_phase, qubits_by_index)

        if next_phase in {"QROM_keep", "comparator", "QROM_alias"}:
            if active_stage is not None and next_phase != active_stage:
                flush_active_stage()
            active_stage = next_phase
            active_qubits.update(op.qubits)
            active_phase = next_phase
            continue

        flush_active_stage()
        active_phase = next_phase

    flush_active_stage()
    return cirq.Circuit(*moments)


def circuit_to_png(circuit, png_path, fold=40):
    """Render a Cirq circuit to a PNG using folded text-diagram panels."""
    if fold < 1:
        fold = len(circuit) or 1

    chunks = [
        circuit[i : i + fold]
        for i in range(0, len(circuit), fold)
    ] or [circuit]

    fig_height = max(4, 1.8 * len(chunks))
    fig, axes = plt.subplots(
        len(chunks),
        1,
        figsize=(16, fig_height),
        constrained_layout=True,
    )
    if len(chunks) == 1:
        axes = [axes]

    for idx, (ax, chunk) in enumerate(zip(axes, chunks)):
        diagram = chunk.to_text_diagram(transpose=False)
        ax.set_axis_off()
        ax.text(
            0.01,
            0.99,
            diagram,
            fontfamily="monospace",
            fontsize=7,
            va="top",
            ha="left",
        )
        start = idx * fold
        end = min(len(circuit), (idx + 1) * fold)
        ax.set_title(f"Moments {start} - {end - 1}", loc="left", fontsize=9)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def render_qasm_to_png(qasm_path, png_path=None, fold=40, blocking=False):
    """Render a Cirq-generated QASM file to a PNG circuit diagram."""
    qasm_path = Path(qasm_path)
    if not qasm_path.exists():
        raise FileNotFoundError(f"QASM file not found: {qasm_path}")

    if png_path is None:
        png_path = qasm_path.with_name(qasm_path.stem + ("_blocking.png" if blocking else ".png"))
    else:
        png_path = Path(png_path)

    qasm_text = qasm_path.read_text(encoding="utf-8")
    if blocking:
        circuit = parse_cirq_qasm3_blocking(qasm_text)
        return circuit_to_png(circuit, png_path, fold=fold)
    circuit = parse_cirq_qasm3(qasm_text)
    return circuit_to_png(circuit, png_path, fold=fold)


def main():
    parser = argparse.ArgumentParser(
        description="Render a Cirq-generated QASM 3 file to a PNG diagram."
    )
    parser.add_argument("qasm_file", help="Path to the OpenQASM 3 file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Defaults to the input stem with .png.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=40,
        help="Unused compatibility option retained for CLI stability.",
    )
    parser.add_argument(
        "--blocking",
        action="store_true",
        help="Collapse QROM_keep, comparator, and QROM_alias into labeled functional-unit gates.",
    )
    args = parser.parse_args()

    png_path = render_qasm_to_png(
        args.qasm_file,
        png_path=args.output,
        fold=args.fold,
        blocking=args.blocking,
    )
    print(png_path)


if __name__ == "__main__":
    main()
