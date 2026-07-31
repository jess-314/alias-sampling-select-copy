import argparse
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cirq

from selectcopy import (
    build_select_copy_qrom,
    analyze_gate_metrics,
    format_compact_resource_report,
    choose_lambda,
    qubits_with_prefix,
    verify_qrom_load,
)
from vandaele_comparator import build_quantum_quantum_comparator


def _parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build and export a quantum alias-sampling circuit."
    )
    parser.add_argument(
        "--alias-values",
        help="Comma-separated alias table values. Defaults to the built-in demo table.",
    )
    parser.add_argument(
        "--keep-values",
        help="Comma-separated keep table values. Defaults to the built-in demo table.",
    )
    parser.add_argument(
        "--keep-bits",
        type=int,
        default=None,
        help="Bit-width for keep values when you want to override inference.",
    )
    parser.add_argument(
        "--threshold-bits",
        type=int,
        default=None,
        help="Bit-width for the threshold register.",
    )
    parser.add_argument(
        "--alias-bits",
        type=int,
        default=None,
        help="Bit-width for the alias/output registers.",
    )
    parser.add_argument(
        "--lambda-param",
        type=int,
        default=None,
        help="SelectCopy block size parameter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "alias_sampler",
        help="Directory for the generated QASM file.",
    )
    parser.add_argument(
        "--clean-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse scratch qubits by uncomputing intermediate stages.",
    )
    parser.add_argument(
        "--measurement-uncompute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the measurement-based uncompute path for the QROMs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a small basis-state verification before exporting QASM.",
    )
    return parser


def int_to_bits(value, width):
    if value < 0:
        raise ValueError("Values must be non-negative.")
    if value >= 2**width:
        raise ValueError(f"Value {value} does not fit in {width} bits.")
    return [(value >> i) & 1 for i in range(width)]


def normalize_table(values, width):
    """Accept either integers or explicit bit lists and normalize to bits."""
    normalized = []
    for value in values:
        if isinstance(value, int):
            normalized.append(int_to_bits(value, width))
        else:
            bits = list(value)
            if len(bits) != width:
                raise ValueError("Bit list has the wrong width.")
            normalized.append([int(b) for b in bits])
    return normalized


def remap_circuit_qubits(circuit, prefix):
    """Prefix every qubit in a subcircuit so multiple copies stay disjoint."""
    mapping = {q: cirq.NamedQubit(f"{prefix}_{q}") for q in circuit.all_qubits()}
    return circuit.transform_qubits(mapping), mapping


def remap_stage_to_physical(circuit, stage_prefix, prefix_to_qubits):
    """Rename a stage, then map selected qubit groups onto physical qubits."""
    renamed, _ = remap_circuit_qubits(circuit, stage_prefix)
    mapping = {}
    for local_prefix, physical_qubits in prefix_to_qubits.items():
        stage_qubits = sorted(
            [q for q in renamed.all_qubits() if str(q).startswith(f"{stage_prefix}_{local_prefix}")],
            key=str,
        )
        if len(stage_qubits) != len(physical_qubits):
            raise ValueError(
                f"Stage {stage_prefix} prefix {local_prefix} expected "
                f"{len(physical_qubits)} qubits, found {len(stage_qubits)}."
            )
        mapping.update({q: physical_qubits[i] for i, q in enumerate(stage_qubits)})
    return renamed.transform_qubits(mapping)


def table_bit_width(values):
    max_value = 0
    for value in values:
        if isinstance(value, int):
            max_value = max(max_value, value)
        else:
            bits = [int(b) for b in value]
            max_value = max(
                max_value,
                int("".join(str(bit) for bit in reversed(bits)), 2),
            )
    return max(1, max_value.bit_length())


def analyze_dirty_ancilla_budget(
    keep_dirty_qubits,
    alias_dirty_qubits,
    comparator_dirty_qubits,
    comparison_padding_qubits=0,
    branch_qubits=1,
    clean_intermediates=True,
):
    """Summarize the reusable auxiliary-qubit footprint for the composed sampler."""
    comparator_stage = comparator_dirty_qubits + comparison_padding_qubits
    if clean_intermediates:
        peak_auxiliary = max(
            keep_dirty_qubits,
            comparator_stage,
            alias_dirty_qubits,
        )
    else:
        peak_auxiliary = keep_dirty_qubits + comparator_stage + alias_dirty_qubits
    result_latches = branch_qubits + 1
    return {
        "shared_dirty_scratch": peak_auxiliary,
        "keep_qrom_scratch": keep_dirty_qubits,
        "alias_qrom_scratch": alias_dirty_qubits,
        "comparator_scratch": comparator_dirty_qubits,
        "comparison_padding_scratch": comparison_padding_qubits,
        "comparator_stage_scratch": comparator_stage,
        "peak_auxiliary_scratch": peak_auxiliary,
        "result_latches": result_latches,
        "total_dirty_ancillas": peak_auxiliary + result_latches,
    }


def build_qrom(
    data,
    lambda_param,
    use_measurement_uncompute=True,
    measurement_key_prefix="",
):
    """Build a QROM circuit with either literal or measurement-based uncompute."""
    return build_select_copy_qrom(
        data,
        lambda_param=lambda_param,
        use_measurement_uncompute=use_measurement_uncompute,
        use_reverse_uncompute=not use_measurement_uncompute,
        measurement_key_prefix=measurement_key_prefix,
    )


def verify_keep_qrom_load(keep_values, sample_addresses=None):
    """Verify the QROM used to load keep values."""
    return verify_qrom_load(
        keep_values,
        lambda_param=len(keep_values),
        sample_addresses=sample_addresses,
    )


def verify_alias_qrom_load(alias_values, sample_addresses=None):
    """Verify the QROM used to load alias values."""
    return verify_qrom_load(
        alias_values,
        lambda_param=len(alias_values),
        sample_addresses=sample_addresses,
    )


def build_alias_sampler_circuit(
    alias_values,
    keep_values,
    keep_bits=None,
    threshold_bits=None,
    alias_bits=None,
    lambda_param=None,
    clean_intermediates=True,
    use_measurement_uncompute=True,
):
    """
    Build a quantum alias-sampling circuit.

    The circuit assumes `alias_values` and `keep_values` are classical alias
    table data supplied ahead of time. The address register is MSB-first to
    match the existing QROM implementation, while the threshold, keep, alias,
    and output registers are little-endian to match the comparator and loaded
    data. The alias register width is constrained to equal the address width so
    the output space always matches the table index space. The comparator uses
    the larger of the threshold and keep widths with zero-extension qubits that
    are recycled later in the circuit. When `use_measurement_uncompute` is set,
    the QROM dirty ancillas are cleared using the measurement-based path from
    the QROM paper, while the reversible cleanup path remains available for the
    fully literal circuit.

        |addr>|u>|0> -> |addr>|u>|sample(addr, u)>

    where sample(addr, u) = addr if u < keep[addr], otherwise alias[addr].
    """
    if len(alias_values) != len(keep_values):
        raise ValueError("alias_values and keep_values must have the same length.")

    n_entries = len(alias_values)
    if n_entries < 2:
        raise ValueError("Need at least two alias-table entries.")

    if keep_bits is None:
        keep_bits = table_bit_width(keep_values)
    if threshold_bits is None:
        threshold_bits = keep_bits
    if lambda_param is None:
        lambda_param = choose_lambda(n_entries)
    if lambda_param < 2 or lambda_param > n_entries:
        raise ValueError("lambda_param must be between 2 and the table size.")
    if lambda_param & (lambda_param - 1) != 0 and lambda_param != n_entries:
        raise ValueError(
            "lambda_param must be a power of two unless it is the full table size."
        )

    addr_bits = max(1, (n_entries - 1).bit_length())
    if lambda_param == n_entries:
        qrom_q_bits = 0
        qrom_r_bits = addr_bits
    else:
        qrom_r_bits = int(lambda_param.bit_length() - 1)
        qrom_q_bits = addr_bits - qrom_r_bits
    if qrom_q_bits < 0:
        raise ValueError("lambda_param is too large for the table address width.")
    if alias_bits is None:
        alias_bits = addr_bits
    if alias_bits != addr_bits:
        raise ValueError("alias_bits must equal the address width.")

    alias_bits_table = normalize_table(alias_values, alias_bits)
    keep_bits_table = normalize_table(keep_values, keep_bits)

    addr = [cirq.NamedQubit(f"addr_{i}") for i in range(addr_bits)]
    threshold = [cirq.NamedQubit(f"u_{i}") for i in range(threshold_bits)]
    keep_reg = [cirq.NamedQubit(f"keep_{i}") for i in range(keep_bits)]
    alias_reg = [cirq.NamedQubit(f"alias_{i}") for i in range(alias_bits)]
    out = [cirq.NamedQubit(f"out_{i}") for i in range(alias_bits)]
    branch = cirq.NamedQubit("sample_branch")
    compare_bits = max(keep_bits, threshold_bits)
    threshold_padding = compare_bits - threshold_bits
    keep_padding = compare_bits - keep_bits
    cmp_dirty = 1
    comparator_scratch = cmp_dirty + threshold_padding + keep_padding

    cmp_circuit, cmp_a, cmp_b, cmp_z, cmp_carry = build_quantum_quantum_comparator(
        compare_bits
    )
    qrom_q_reg = addr[:qrom_q_bits]
    qrom_r_reg = addr[qrom_q_bits:]
    # Measurement-based uncompute needs a non-trivial split-address QROM to be
    # useful. Degenerate single-block lookups fall back to the literal inverse
    # path, which is more robust for tiny tables.
    qrom_use_measurement_uncompute = use_measurement_uncompute and qrom_q_bits > 0

    keep_qrom = build_qrom(
        keep_bits_table,
        lambda_param=lambda_param,
        use_measurement_uncompute=qrom_use_measurement_uncompute,
        measurement_key_prefix="keep_",
    )

    alias_qrom = build_qrom(
        alias_bits_table,
        lambda_param=lambda_param,
        use_measurement_uncompute=qrom_use_measurement_uncompute,
        measurement_key_prefix="alias_",
    )

    keep_dirty = len(qubits_with_prefix(keep_qrom, "dirty_"))
    alias_dirty = len(qubits_with_prefix(alias_qrom, "dirty_"))
    if clean_intermediates:
        shared_scratch = [
            cirq.NamedQubit(f"shared_scratch_{i}")
            for i in range(max(keep_dirty + comparator_scratch, alias_dirty))
        ]
    else:
        shared_scratch = [
            cirq.NamedQubit(f"shared_scratch_{i}")
            for i in range(keep_dirty + comparator_scratch + alias_dirty)
        ]
    keep_dirty_qubits = shared_scratch[:keep_dirty]
    comparator_shared = shared_scratch[keep_dirty : keep_dirty + comparator_scratch]
    threshold_pad = comparator_shared[:threshold_padding]
    keep_pad = comparator_shared[threshold_padding : threshold_padding + keep_padding]
    cmp_carry_qubit = comparator_shared[threshold_padding + keep_padding]
    threshold_for_cmp = threshold + threshold_pad
    keep_for_cmp = keep_reg + keep_pad
    cmp_circuit = remap_stage_to_physical(
        cmp_circuit,
        "cmp",
        {
            "a_": threshold_for_cmp,
            "b_": keep_for_cmp,
            "z": [cirq.NamedQubit("cmp_z")],
            "carry": [cmp_carry_qubit],
        },
    )
    keep_qrom = remap_stage_to_physical(
        keep_qrom,
        "keep",
        {
            "q_": qrom_q_reg,
            "r_": qrom_r_reg,
            "t_": keep_reg,
            "dirty_": keep_dirty_qubits,
        },
    )
    if clean_intermediates:
        alias_dirty_qubits = shared_scratch[:alias_dirty]
    else:
        alias_dirty_qubits = shared_scratch[keep_dirty + comparator_scratch : keep_dirty + comparator_scratch + alias_dirty]
    alias_qrom = remap_stage_to_physical(
        alias_qrom,
        "alias",
        {
            "q_": qrom_q_reg,
            "r_": qrom_r_reg,
            "t_": alias_reg,
            "dirty_": alias_dirty_qubits,
        },
    )

    circuit = cirq.Circuit()

    # Keep and alias tables are loaded from the same address register.
    circuit += keep_qrom
    circuit += cmp_circuit
    circuit.append(cirq.CNOT(cirq.NamedQubit("cmp_z"), branch))

    if clean_intermediates:
        # Release comparator scratch before the alias lookup. This remains
        # available even when the QROM itself uses measurement-based uncompute.
        circuit += cirq.inverse(cmp_circuit)
        if not qrom_use_measurement_uncompute:
            circuit += cirq.inverse(keep_qrom)

    # Reuse the freed auxiliary pool for the alias lookup now that the earlier
    # stages have been uncomputed.
    circuit += alias_qrom

    # If z == 1, copy addr into out. The QROM address wires are MSB-first,
    # so we reverse them when writing the sampled output.
    for i in range(addr_bits):
        circuit.append(cirq.CCX(branch, addr[addr_bits - 1 - i], out[i]))

    # If z == 0, reuse the same control qubit by toggling it in place.
    circuit.append(cirq.X(branch))

    # If z == 0, copy alias[addr] into out.
    for i in range(addr_bits):
        circuit.append(cirq.CCX(branch, alias_reg[i], out[i]))

    circuit.append(cirq.X(branch))

    if clean_intermediates and not qrom_use_measurement_uncompute:
        circuit += cirq.inverse(alias_qrom)

    dirty_budget = analyze_dirty_ancilla_budget(
        keep_dirty_qubits=keep_dirty,
        alias_dirty_qubits=alias_dirty,
        comparator_dirty_qubits=cmp_dirty,
        comparison_padding_qubits=threshold_padding + keep_padding,
        branch_qubits=1,
        clean_intermediates=clean_intermediates,
    )
    gate_metrics = analyze_gate_metrics(circuit)

    return circuit, {
        "addr": addr,
        "threshold": threshold,
        "keep_reg": keep_reg,
        "alias_reg": alias_reg,
        "out": out,
        "cmp_z": branch,
        "clean_intermediates": clean_intermediates,
        "register_widths": {
            "address": addr_bits,
            "qrom_q": qrom_q_bits,
            "qrom_r": qrom_r_bits,
            "threshold": threshold_bits,
            "keep": keep_bits,
            "alias": alias_bits,
            "compare": compare_bits,
        },
        "qrom_lambda": lambda_param,
        "dirty_budget": dirty_budget,
        "gate_metrics": gate_metrics,
    }


def verify_alias_sampler(
    alias_values,
    keep_values,
    cases=None,
    keep_bits=None,
    threshold_bits=None,
    alias_bits=None,
    lambda_param=None,
    clean_intermediates=True,
    use_measurement_uncompute=True,
):
    """Check the sampler on a small set of basis inputs."""
    circuit, regs = build_alias_sampler_circuit(
        alias_values,
        keep_values,
        keep_bits=keep_bits,
        threshold_bits=threshold_bits,
        alias_bits=alias_bits,
        lambda_param=lambda_param,
        clean_intermediates=clean_intermediates,
        use_measurement_uncompute=use_measurement_uncompute,
    )
    addr = regs["addr"]
    threshold = regs["threshold"]
    out = regs["out"]
    n = len(addr)
    m = len(threshold)

    if cases is None:
        max_u = 2**m - 1
        last = len(alias_values) - 1
        cases = [(0, 0), (last, 0), (last, max_u)]

    for a, u in cases:
        prep = cirq.Circuit()
        for i in range(n):
            if (a >> (n - 1 - i)) & 1:
                prep.append(cirq.X(addr[i]))
        for i in range(m):
            if (u >> i) & 1:
                prep.append(cirq.X(threshold[i]))

        sim = cirq.Simulator()
        test = prep + circuit + cirq.Circuit(cirq.measure(*out, key="out"))
        result = sim.run(test, repetitions=1)
        observed_bits = [int(bit) for bit in result.records["out"][0][0]]
        observed = sum(observed_bits[i] << i for i in range(n))
        expected = a if u < keep_values[a] else alias_values[a]
        if observed != expected:
            raise AssertionError(
                f"Alias sampler failed for addr={a}, u={u}: "
                f"got {observed}, expected {expected}"
            )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.alias_values is None and args.keep_values is None:
        alias_values = [1, 0, 3, 2]
        keep_values = [2, 1, 3, 1]
    elif args.alias_values is not None and args.keep_values is not None:
        alias_values = _parse_csv_ints(args.alias_values)
        keep_values = _parse_csv_ints(args.keep_values)
    else:
        raise SystemExit("Provide both --alias-values and --keep-values, or neither.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit, regs = build_alias_sampler_circuit(
        alias_values,
        keep_values,
        keep_bits=args.keep_bits,
        threshold_bits=args.threshold_bits,
        alias_bits=args.alias_bits,
        lambda_param=args.lambda_param,
        clean_intermediates=args.clean_intermediates,
        use_measurement_uncompute=args.measurement_uncompute,
    )

    print("=== Quantum Alias Sampler ===")
    print(f"Entries: {len(alias_values)}")
    print(f"Address bits: {len(regs['addr'])}")
    print(f"Threshold bits: {len(regs['threshold'])}")
    print(f"Keep bits: {len(regs['keep_reg'])}")
    print(f"Alias bits: {len(regs['alias_reg'])}")
    metrics = regs["gate_metrics"]
    print(format_compact_resource_report(metrics))
    if args.verify:
        verify_alias_sampler(
            alias_values,
            keep_values,
            keep_bits=args.keep_bits,
            threshold_bits=args.threshold_bits,
            alias_bits=args.alias_bits,
            lambda_param=args.lambda_param,
            clean_intermediates=args.clean_intermediates,
            use_measurement_uncompute=args.measurement_uncompute,
        )
        print("Verification: passed on small basis-state cases.")
    else:
        print("Verification: skipped (use --verify to enable).")

    qasm_path = output_dir / f"alias_sampler_n{len(alias_values)}.qasm"
    qasm_str = cirq.qasm(circuit, args=cirq.QasmArgs(version="3.0"))
    qasm_path.write_text(qasm_str, encoding="utf-8")
    print(f"QASM written to {qasm_path}")
