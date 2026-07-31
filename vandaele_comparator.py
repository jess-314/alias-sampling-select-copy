import argparse
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import cirq

from selectcopy import analyze_gate_metrics, format_compact_resource_report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build, verify, and export the Vandaele comparator circuit."
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=5,
        help="Bit-width of each comparator register.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "comparator",
        help="Directory for the generated QASM file.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the small basis-state verification pass.",
    )
    return parser


def _majority_step_ops(x_bit, y_bit, carry_bit):
    """One ripple-carry majority step.

    The chosen orientation matches the optimal one-helper comparator pattern:
    the forward ripple is followed by its Cirq inverse to restore the input
    registers while leaving the compare bit untouched.
    """
    return [
        cirq.CNOT(x_bit, y_bit),
        cirq.CNOT(x_bit, carry_bit),
        cirq.CCX(y_bit, carry_bit, x_bit),
    ]


def build_quantum_quantum_comparator(num_bits):
    """
    Build a reversible quantum-quantum comparator:

        |a> |b> |z> -> |a> |b> |z xor [a < b]>

    The circuit uses a single clean carry helper and restores all ancillas.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be at least 1.")

    a = [cirq.NamedQubit(f"a_{i}") for i in range(num_bits)]
    b = [cirq.NamedQubit(f"b_{i}") for i in range(num_bits)]
    z = cirq.NamedQubit("z")
    carry = cirq.NamedQubit("carry")

    circuit = cirq.Circuit()
    # The comparator uses `a` as the work register. We temporarily complement it
    # so the ripple chain computes the desired borrow/carry relation with a
    # single helper qubit.
    for qubit in a:
        circuit.append(cirq.X(qubit))

    forward_blocks = []
    first_block = _majority_step_ops(a[0], b[0], carry)
    forward_blocks.append(first_block)
    circuit.append(first_block)

    for i in range(1, num_bits):
        block = _majority_step_ops(a[i], b[i], a[i - 1])
        forward_blocks.append(block)
        circuit.append(block)

    circuit.append(cirq.CNOT(a[-1], z))

    # Restore the work register by replaying the exact inverse of the forward
    # ripple chain.
    forward_circuit = cirq.Circuit(forward_blocks)
    circuit += cirq.inverse(forward_circuit)

    for qubit in a:
        circuit.append(cirq.X(qubit))

    return circuit, a, b, z, carry


def count_operations(circuit):
    return sum(1 for _ in circuit.all_operations())


def sample_comparator_cases(num_bits):
    """Return a small set of basis-state comparisons for verification."""
    if num_bits <= 3:
        return [(a, b) for a in range(2**num_bits) for b in range(2**num_bits)]

    top = 2**num_bits - 1
    mid = 2 ** (num_bits - 1)
    return [
        (0, 0),
        (0, 1),
        (1, 0),
        (mid - 1, mid),
        (mid, mid - 1),
        (top, top),
    ]


def verify_small_instances(max_bits=4, cases_per_size=None):
    """Quick basis-state check for small sizes."""
    sim = cirq.Simulator()
    for n in range(1, max_bits + 1):
        circuit, a, b, z, carry = build_quantum_quantum_comparator(n)
        cases = cases_per_size(n) if cases_per_size is not None else sample_comparator_cases(n)

        for aval, bval in cases:
            prep = cirq.Circuit()
            for i in range(n):
                if (aval >> i) & 1:
                    prep.append(cirq.X(a[i]))
                if (bval >> i) & 1:
                    prep.append(cirq.X(b[i]))

            expected = 1 if aval < bval else 0
            test_circuit = prep + circuit + cirq.Circuit(
                cirq.measure(*a, key="a"),
                cirq.measure(*b, key="b"),
                cirq.measure(z, key="z"),
                cirq.measure(carry, key="carry"),
            )
            result = sim.run(test_circuit, repetitions=1)
            actual_z = int(result.measurements["z"][0][0])
            actual_a = [int(bit) for bit in result.measurements["a"][0]]
            actual_b = [int(bit) for bit in result.measurements["b"][0]]
            actual_carry = int(result.measurements["carry"][0][0])
            expected_a = [(aval >> i) & 1 for i in range(n)]
            expected_b = [(bval >> i) & 1 for i in range(n)]
            if actual_z != expected:
                raise AssertionError(
                    f"Comparator failed for n={n}, a={aval}, b={bval}"
                )
            if actual_a != expected_a or actual_b != expected_b:
                raise AssertionError(
                    f"Comparator corrupted inputs for n={n}, a={aval}, b={bval}"
                )
            if actual_carry != 0:
                raise AssertionError(
                    f"Comparator left ancillas dirty for n={n}, a={aval}, b={bval}"
                )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    num_bits = args.num_bits
    circuit, *_ = build_quantum_quantum_comparator(num_bits)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Vandaele Quantum-Quantum Comparator ===")
    print(f"Bits per register: {num_bits}")
    metrics = analyze_gate_metrics(circuit)
    print(format_compact_resource_report(metrics))
    if args.skip_verify:
        print("Verification: skipped.")
    else:
        verify_small_instances(max_bits=min(4, num_bits))
        print("Verification: passed on small basis-state instances.")

    qasm_path = output_dir / f"vandaele_comparator_n{num_bits}.qasm"
    qasm_str = cirq.qasm(circuit, args=cirq.QasmArgs(version="3.0"))
    qasm_path.write_text(qasm_str, encoding="utf-8")
    print(f"QASM written to {qasm_path}")
