import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cirq
import numpy as np

def controlled_on_val(qubits, val):
    """Utility to return X gates handling 0-controls for a specific value."""
    gates = []
    for i, q in enumerate(qubits):
        bit = (val >> (len(qubits) - 1 - i)) & 1
        if bit == 0:
            gates.append(cirq.X(q))
    return gates

def make_sample_database(num_entries, bitsize, seed=0, density=0.5):
    """Create a deterministic pseudo-random database for scaling studies."""
    rng = np.random.default_rng(seed)
    data = (rng.random((num_entries, bitsize)) < density).astype(int)
    return data.tolist()

def int_to_bits(value, width):
    """Convert a non-negative integer to a little-endian bit list."""
    if value < 0:
        raise ValueError("Values must be non-negative.")
    if value >= 2**width:
        raise ValueError(f"Value {value} does not fit in {width} bits.")
    return [(value >> i) & 1 for i in range(width)]

def normalize_table(values, width):
    """Accept integers or bit lists and normalize each entry to bits."""
    normalized = []
    for value in values:
        if isinstance(value, int):
            normalized.append(int_to_bits(value, width))
        else:
            bits = [int(b) for b in value]
            if len(bits) != width:
                raise ValueError("Bit list has the wrong width.")
            normalized.append(bits)
    return normalized

def count_operations(circuit):
    """Return the explicit operation count in a Cirq circuit."""
    return sum(1 for _ in circuit.all_operations())

def _unwrap_classically_controlled_operation(op):
    """Return the wrapped sub-operation if this is classically controlled."""
    if isinstance(op, cirq.ClassicallyControlledOperation):
        return getattr(op, "_sub_operation", None), True
    return op, False


def _is_toffoli_like_gate(gate):
    """Return True for Toffoli gates and their inverses."""
    return isinstance(gate, cirq.CCXPowGate)


def _toffoli_equivalent_count_for_gate(gate):
    """Return a Toffoli-equivalent count for explicit controlled-X gates."""
    if isinstance(gate, cirq.CCXPowGate):
        return 1
    if isinstance(gate, cirq.ControlledGate) and gate.sub_gate == cirq.X:
        controls = gate.num_controls()
        if controls >= 2:
            return controls - 1
    return 0


def count_toffolis(circuit):
    """Count explicit Toffoli-equivalent controlled-X work in a Cirq circuit."""
    return sum(
        _toffoli_equivalent_count_for_gate(getattr(op, "gate", None))
        for op in circuit.all_operations()
    )


def _is_t_gate(gate):
    """Return True for literal T or T^-1 gates."""
    return gate == cirq.T or gate == cirq.T**-1


def count_t_gates(circuit):
    """Count explicit T gates in a Cirq circuit."""
    return sum(1 for op in circuit.all_operations() if _is_t_gate(getattr(op, "gate", None)))


def decompose_to_clifford_t_circuit(circuit):
    """Decompose a circuit while preserving non-unitary operations."""
    return cirq.Circuit(cirq.decompose(circuit))


def count_clifford_t_explicit_t_gates(circuit):
    """Count explicit T gates after Clifford+T decomposition."""
    decomposed = decompose_to_clifford_t_circuit(circuit)
    return count_t_gates(decomposed)


def count_classically_controlled_toffoli_equivalents(circuit):
    """Count Toffoli-like gates that appear under classical control."""
    total = 0
    for op in circuit.all_operations():
        sub_op, is_classically_controlled = _unwrap_classically_controlled_operation(op)
        if is_classically_controlled:
            total += _toffoli_equivalent_count_for_gate(getattr(sub_op, "gate", None))
    return total

def analyze_gate_metrics(circuit):
    """Return literal gate counts for the emitted Toffoli-native circuit."""
    explicit_toffolis = count_toffolis(circuit)
    classical_toffoli_equivalents = count_classically_controlled_toffoli_equivalents(
        circuit
    )
    literal_toffolis = explicit_toffolis + classical_toffoli_equivalents
    return {
        "total_gate_count": count_operations(circuit),
        "explicit_toffoli_count": explicit_toffolis,
        "classically_controlled_toffoli_equiv_count": classical_toffoli_equivalents,
        "literal_toffoli_count": literal_toffolis,
        "toffoli_count": literal_toffolis,
    }


def analyze_clifford_t_metrics(circuit):
    """Return a Clifford+T comparison count for the same logical circuit."""
    decomposed = decompose_to_clifford_t_circuit(circuit)
    explicit_t_count = count_t_gates(decomposed)
    return {
        "decomposed_gate_count": count_operations(decomposed),
        "explicit_t_count": explicit_t_count,
        "t_count": explicit_t_count,
    }


def loglog_slope(xs, ys):
    """Fit a slope on a log-log plot using only positive samples."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
    return float(slope)

def format_resource_report(metrics, dirty_budget=None, moment_count=None):
    """Format a standardized resource summary for CLI output."""
    parts = [
        f"gates={metrics['total_gate_count']}",
        f"toffolis={metrics['toffoli_count']}",
        f"explicit_toffoli={metrics['explicit_toffoli_count']}",
        f"classical_toffoli={metrics['classically_controlled_toffoli_equiv_count']}",
    ]
    if dirty_budget is not None:
        parts.append(f"dirty={dirty_budget['total_dirty_ancillas']}")
    if moment_count is not None:
        parts.append(f"moments={moment_count}")
    return "  ".join(parts)


def format_compact_resource_report(metrics):
    """Format the compact report requested for console output and plot labels."""
    return (
        f"total_gate={metrics['total_gate_count']}  "
        f"toffoli={metrics['toffoli_count']}  "
        f"explicit_toffoli={metrics['explicit_toffoli_count']}  "
        f"classical_toffoli={metrics['classically_controlled_toffoli_equiv_count']}"
    )

def qubits_with_prefix(circuit, prefix):
    """Return qubits from a circuit whose names start with the given prefix."""
    return sorted(
        [q for q in circuit.all_qubits() if str(q).startswith(prefix)],
        key=str,
    )

def prepare_msb_basis_state(qubits, value):
    """Return a circuit that prepares `value` on `qubits` using MSB-first order."""
    circuit = cirq.Circuit()
    n = len(qubits)
    for i, q in enumerate(qubits):
        if (value >> (n - 1 - i)) & 1:
            circuit.append(cirq.X(q))
    return circuit

def size_one_qrom_phase_oracle_ops(q_reg, address, phase_work):
    """Return ops for a size-1 QROM phase correction on one address.

    This is the measurement-based uncompute optimization from the QROM paper:
    each measured dirty output bit classically selects a single-address phase
    correction, which is implemented as an exact-address phase oracle.
    """
    ops = controlled_on_val(q_reg, address)
    if not q_reg:
        return ops

    if len(q_reg) == 1:
        ops.append(cirq.Z(q_reg[0]))
    else:
        if len(phase_work) < len(q_reg) - 1:
            raise ValueError("phase_work does not have enough ancillas.")

        # Compute the conjunction of the address bits into phase_work[-1].
        ops.append(cirq.CCX(q_reg[0], q_reg[1], phase_work[0]))
        for i in range(2, len(q_reg)):
            ops.append(cirq.CCX(phase_work[i - 2], q_reg[i], phase_work[i - 1]))

        ops.append(cirq.Z(phase_work[len(q_reg) - 2]))

        # Uncompute the address match chain.
        for i in range(len(q_reg) - 1, 1, -1):
            ops.append(cirq.CCX(phase_work[i - 2], q_reg[i], phase_work[i - 1]))
        ops.append(cirq.CCX(q_reg[0], q_reg[1], phase_work[0]))

    ops.extend(controlled_on_val(q_reg, address))
    return ops

def address_phase_oracle_ops(q_reg, address, phase_work):
    """Backward-compatible alias for the size-1 QROM phase oracle helper."""
    return size_one_qrom_phase_oracle_ops(q_reg, address, phase_work)


def unary_flag_work_size(num_bits):
    """Number of clean work qubits needed for all unary flags of a bit register."""
    if num_bits <= 1:
        return 0
    return (1 << (num_bits + 1)) - 2


def append_unary_flag_compute(circuit, q_reg, flag_work):
    """Compute one-hot flags for q_reg and return (leaf_flags, compute_ops)."""
    n = len(q_reg)
    if n <= 1:
        return [], []
    required = unary_flag_work_size(n)
    if len(flag_work) < required:
        raise ValueError("flag_work does not have enough ancillas.")

    compute_ops = []

    def add(op):
        compute_ops.append(op)
        circuit.append(op)

    # First level: flags for q_0 == 0 and q_0 == 1 use Clifford gates only.
    add(cirq.X(flag_work[0]))
    add(cirq.CNOT(q_reg[0], flag_work[0]))
    add(cirq.CNOT(q_reg[0], flag_work[1]))
    level = [flag_work[0], flag_work[1]]
    next_idx = 2

    for bit in q_reg[1:]:
        next_level = []
        for parent in level:
            zero_child = flag_work[next_idx]
            one_child = flag_work[next_idx + 1]
            next_idx += 2
            add(cirq.CCX(parent, bit, one_child))
            add(cirq.X(bit))
            add(cirq.CCX(parent, bit, zero_child))
            add(cirq.X(bit))
            next_level.extend([zero_child, one_child])
        level = next_level

    return level, compute_ops


def append_unary_flag_uncompute(circuit, compute_ops):
    """Uncompute flags generated by append_unary_flag_compute."""
    circuit.append(cirq.inverse(cirq.Circuit(compute_ops)))

def append_measurement_based_uncompute(
    circuit,
    q_reg,
    dirty_ancillas,
    data,
    lambda_param,
    phase_work,
    measurement_key_prefix="",
):
    """
    Measure the dirty lookup outputs in the X basis and apply phase fixup
    on the address register conditioned on the measurement outcomes.
    """
    N = len(data)
    b = len(data[0])
    n_q = len(q_reg)
    measured_terms = []

    def append_new(op):
        circuit.append(op, strategy=cirq.InsertStrategy.NEW)

    for r in range(1, lambda_param):
        for i in range(b):
            dirty_qubit = dirty_ancillas[r - 1][i]
            meas_key = f"{measurement_key_prefix}mbu_dirty_{r}_{i}"

            append_new(cirq.H(dirty_qubit))
            append_new(cirq.measure(dirty_qubit, key=meas_key))
            append_new(cirq.reset(dirty_qubit))
            measured_terms.append((meas_key, r, i))

    if not q_reg:
        return

    if len(q_reg) == 1:
        q_flags = []
        compute_ops = []
    else:
        q_flags, compute_ops = append_unary_flag_compute(circuit, q_reg, phase_work)

    for q_val in range(2**n_q):
        addr = q_val * lambda_param
        if addr >= N:
            continue

        f_q_0 = data[addr]
        corrections = []
        for meas_key, r, i in measured_terms:
            idx = addr + r
            if idx >= N:
                continue
            f_q_r = data[idx]
            if f_q_r[i] ^ f_q_0[i] == 1:
                corrections.append(meas_key)

        if not corrections:
            continue

        if len(q_reg) == 1:
            pad = controlled_on_val(q_reg, q_val)
            append_new(pad)
            match_qubit = q_reg[0]
        else:
            match_qubit = q_flags[q_val]

        for meas_key in corrections:
            append_new(cirq.Z(match_qubit).with_classical_controls(meas_key))

        if len(q_reg) == 1:
            append_new(pad)

    if len(q_reg) > 1:
        append_unary_flag_uncompute(circuit, compute_ops)

def verify_qrom_load(table, lambda_param=None, sample_addresses=None):
    """
    Verify a small QROM load by simulating selected basis addresses.

    The circuit is built in reversible mode and the loaded target register is
    checked against the provided classical table.
    """
    if not table:
        raise ValueError("table must not be empty.")

    if isinstance(table[0], int):
        target_width = max(1, max(int(v) for v in table).bit_length())
        table = normalize_table(table, target_width)

    if lambda_param is None:
        lambda_param = len(table)
    if sample_addresses is None:
        if len(table) <= 8:
            sample_addresses = list(range(len(table)))
        else:
            sample_addresses = [0, len(table) // 2, len(table) - 1]

    circuit = build_select_copy_qrom(
        table,
        lambda_param=lambda_param,
    )
    selection_reg = qubits_with_prefix(circuit, "q_") + qubits_with_prefix(circuit, "r_")
    target_reg = qubits_with_prefix(circuit, "t_")
    expected_table = normalize_table(table, len(target_reg))

    sim = cirq.Simulator()
    for address in sample_addresses:
        prep = prepare_msb_basis_state(selection_reg, address)
        test = prep + circuit
        all_qubits = sorted(test.all_qubits(), key=str)
        result = sim.simulate(test, qubit_order=all_qubits)
        state = np.asarray(result.final_state_vector)
        basis_index = int(np.argmax(np.abs(state)))
        basis_bits = cirq.big_endian_int_to_bits(basis_index, bit_count=len(all_qubits))
        bit_by_qubit = {q: basis_bits[i] for i, q in enumerate(all_qubits)}
        observed_addr = [int(bit_by_qubit[q]) for q in selection_reg]
        observed = [int(bit_by_qubit[q]) for q in target_reg]
        expected_addr = [
            (address >> (len(selection_reg) - 1 - i)) & 1 for i in range(len(selection_reg))
        ]
        expected = expected_table[address]
        if observed_addr != expected_addr:
            raise AssertionError(
                f"QROM address changed for address={address}: "
                f"got {observed_addr}, expected {expected_addr}"
            )
        if observed != expected:
            raise AssertionError(
                f"QROM load failed for address={address}: got {observed}, expected {expected}"
            )

def build_select_copy_qrom(
    data,
    lambda_param,
    use_measurement_uncompute=True,
    use_reverse_uncompute=False,
    measurement_key_prefix="",
):
    """
    Constructs an explicit SelectCopy QROM circuit in Cirq.
    
    Args:
        data: List of lists/arrays, where data[x] is the b-bit string.
        lambda_param: The tuning parameter defining block sizing.
        use_measurement_uncompute: If True, measure and reset the auxiliary
            block-difference outputs and apply the corresponding address-phase
            fixups.
        use_reverse_uncompute: If True, coherently clear the auxiliary
            block-difference outputs by replaying the outer-address loads on
            those outputs only. This leaves the looked-up target value in place.
    """
    N = len(data)
    if N == 0:
        raise ValueError("data must not be empty.")
    if isinstance(data[0], int):
        b = max(1, max(int(v) for v in data).bit_length())
        data_bits = [int_to_bits(int(v), b) for v in data]
    else:
        b = len(data[0])
        data_bits = [[int(bit) for bit in row] for row in data]
    if lambda_param < 1:
        raise ValueError("lambda_param must be at least 1.")
    if lambda_param & (lambda_param - 1) != 0:
        raise ValueError("lambda_param must be a power of two.")
    if lambda_param > N:
        raise ValueError("lambda_param must not exceed the table size.")

    n_q = int(np.ceil(np.log2(N / lambda_param)))
    n_r = int(np.ceil(np.log2(lambda_param)))
    selection_bits = n_q + n_r
    addr_bits = max(1, selection_bits)

    padded_len = 1 << addr_bits
    if len(data_bits) < padded_len:
        data_bits = data_bits + [[0] * b for _ in range(padded_len - len(data_bits))]
    elif len(data_bits) > padded_len:
        raise ValueError("data is too large for the derived address register width.")

    q_reg = [cirq.NamedQubit(f"q_{i}") for i in range(n_q)]
    r_reg = [cirq.NamedQubit(f"r_{i}") for i in range(n_r)]
    target_reg = [cirq.NamedQubit(f"t_{i}") for i in range(b)]
    dirty_blocks = [
        [cirq.NamedQubit(f"dirty_{(r - 1) * b + i}") for i in range(b)]
        for r in range(1, lambda_param)
    ]
    phase_work = [
        cirq.NamedQubit(f"dirty_phase_{i}") for i in range(unary_flag_work_size(n_q))
    ]
    r_flag_work = [
        cirq.NamedQubit(f"dirty_rflag_{i}") for i in range(unary_flag_work_size(n_r))
    ]
    dirty_pad = max(0, lambda_param - 1) * b + len(phase_work) + len(r_flag_work)

    if n_q == 0:
        circuit = cirq.Circuit()
        for addr, bits in enumerate(data_bits[:N]):
            pad = controlled_on_val(r_reg, addr)
            circuit.append(pad)
            for bit in range(b):
                if bits[bit]:
                    circuit.append(cirq.X(target_reg[bit]).controlled_by(*r_reg))
            circuit.append(pad)
        for qubit in q_reg + r_reg + target_reg:
            circuit.append(cirq.I(qubit))
        for i in range(dirty_pad):
            circuit.append(cirq.I(cirq.NamedQubit(f"dirty_{i}")))
        return circuit

    def append_qrom_outer_load(circuit, include_target=True, include_dirty=True):
        """Load block baselines and within-block differences controlled by q."""
        if n_q == 1:
            q_flags = []
            compute_ops = []
        else:
            q_flags, compute_ops = append_unary_flag_compute(circuit, q_reg, phase_work)

        for q_val in range(2**n_q):
            base_idx = q_val * lambda_param
            base = data_bits[base_idx]

            if n_q == 1:
                pad_q = controlled_on_val(q_reg, q_val)
                circuit.append(pad_q)
                q_control = q_reg[0]
            else:
                pad_q = []
                q_control = q_flags[q_val]

            if include_target:
                for bit in range(b):
                    if base[bit]:
                        circuit.append(cirq.CNOT(q_control, target_reg[bit]))

            if include_dirty:
                for r in range(1, lambda_param):
                    row = data_bits[base_idx + r]
                    for bit in range(b):
                        if row[bit] ^ base[bit]:
                            circuit.append(cirq.CNOT(q_control, dirty_blocks[r - 1][bit]))

            circuit.append(pad_q)

        if n_q > 1:
            append_unary_flag_uncompute(circuit, compute_ops)

    def append_select_copy(circuit):
        """Copy the selected within-block difference into the target."""
        if n_r == 1:
            r_flags = []
            compute_ops = []
        else:
            r_flags, compute_ops = append_unary_flag_compute(circuit, r_reg, r_flag_work)

        for r in range(1, lambda_param):
            if n_r == 1:
                pad_r = controlled_on_val(r_reg, r)
                circuit.append(pad_r)
                r_control = r_reg[0]
            else:
                pad_r = []
                r_control = r_flags[r]
            for bit in range(b):
                circuit.append(cirq.CCX(r_control, dirty_blocks[r - 1][bit], target_reg[bit]))
            circuit.append(pad_r)

        if n_r > 1:
            append_unary_flag_uncompute(circuit, compute_ops)

    circuit = cirq.Circuit()
    append_qrom_outer_load(circuit, include_target=True, include_dirty=True)
    append_select_copy(circuit)

    if use_measurement_uncompute:
        append_measurement_based_uncompute(
            circuit,
            q_reg=q_reg,
            dirty_ancillas=dirty_blocks,
            data=data_bits,
            lambda_param=lambda_param,
            phase_work=phase_work,
            measurement_key_prefix=measurement_key_prefix,
        )
    elif use_reverse_uncompute:
        append_qrom_outer_load(circuit, include_target=False, include_dirty=True)

    for qubit in q_reg + r_reg + target_reg:
        circuit.append(cirq.I(qubit))
    for block in dirty_blocks:
        for qubit in block:
            circuit.append(cirq.I(qubit))
    for qubit in phase_work:
        circuit.append(cirq.I(qubit))
    for qubit in r_flag_work:
        circuit.append(cirq.I(qubit))

    return circuit

def build_exportable_select_copy_qrom(data, lambda_param):
    """Build a fully decomposed circuit that is safe to export as QASM 3."""
    return build_select_copy_qrom(data, lambda_param=lambda_param)

def choose_lambda(num_entries):
    """Pick a power-of-two block size for the measurement-cleaned SelectCopy QROM."""
    if num_entries < 2:
        raise ValueError("num_entries must be at least 2.")
    addr_bits = max(1, (num_entries - 1).bit_length())
    candidates = [1 << k for k in range(1, addr_bits + 1) if (1 << k) <= num_entries]
    if not candidates:
        return num_entries

    def proxy_cost(lam):
        q_blocks = max(1, num_entries // lam)
        q_bits = max(1, (q_blocks - 1).bit_length())
        r_bits = max(1, (lam - 1).bit_length())
        outer = q_blocks
        inner = lam
        copy = lam * addr_bits
        flag_overhead = q_bits * q_blocks + r_bits * lam
        return outer + inner + copy + flag_overhead

    return min(candidates, key=proxy_cost)


def run_stamp():
    """Return a compact timestamp suitable for run-unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def sweep_gate_scaling(num_entries_list, bitsize, seed=0, density=0.5):
    """Build QROMs across a range of sizes and collect gate counts."""
    results = []
    for num_entries in num_entries_list:
        lambda_param = choose_lambda(num_entries)
        database = make_sample_database(
            num_entries=num_entries,
            bitsize=bitsize,
            seed=seed + num_entries,
            density=density,
        )
        plot_circuit = build_select_copy_qrom(
            database,
            lambda_param=lambda_param,
            use_measurement_uncompute=True,
        )
        gate_metrics = analyze_gate_metrics(plot_circuit)
        results.append(
            {
                "N": num_entries,
                "lambda": lambda_param,
                "bitsize": bitsize,
                "gate_metrics": gate_metrics,
                "data": database,
            }
        )
    return results

def plot_gate_scaling(results, out_path=None):
    """Plot the measurement-based uncompute gate-count scaling on a log-log axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot gate scaling."
        ) from exc

    if out_path is None:
        out_path = Path("output") / "selectcopy" / "plots" / f"selectcopy_gate_scaling_{run_stamp()}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = [row["N"] for row in results]
    counts = [row["gate_metrics"]["total_gate_count"] for row in results]
    lambdas = [row["lambda"] for row in results]
    bitsize = results[0]["bitsize"] if results else None
    slope = loglog_slope(sizes, counts)

    def fmt_range(values):
        unique = sorted(set(values))
        if not unique:
            return "n/a"
        if len(unique) == 1:
            return str(unique[0])
        return f"{unique[0]}-{unique[-1]}"

    q_sizes = []
    r_sizes = []
    dirty_sizes = []
    phase_sizes = []
    for row in results:
        n_q = int(np.ceil(np.log2(row["N"] / row["lambda"])))
        n_r = int(np.ceil(np.log2(row["lambda"])))
        q_sizes.append(n_q)
        r_sizes.append(n_r)
        dirty_sizes.append(max(0, row["lambda"] - 1) * bitsize)
        phase_sizes.append(max(0, n_q - 1))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.loglog(
        sizes,
        counts,
        marker="o",
        linewidth=2,
        label="Measurement-based uncompute",
    )

    # Keep the plot readable by only marking the sweep line, not every point.
    for size, count, lam in zip(sizes, counts, lambdas):
        if size == sizes[-1] or size == sizes[0]:
            ax.annotate(
                f"λ={lam}",
                (size, count),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax.set_title(
        (
            f"SelectCopy QROM Gate Count Scaling\n"
            f"log-log slope={slope:.2f}; "
            f"q={fmt_range(q_sizes)} bits, "
            f"r={fmt_range(r_sizes)} bits, "
            f"t={bitsize} bits, "
            f"dirty={fmt_range(dirty_sizes)} qubits, "
            f"phase={fmt_range(phase_sizes)} qubits"
        ),
        fontsize=13,
    )
    ax.set_xlabel("Database size N", fontsize=12)
    ax.set_ylabel("Operation count", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def pretty_format_qasm3(qasm_str):
    """Group consecutive identical if-conditions into brace-delimited blocks."""
    lines = qasm_str.splitlines()
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("if (") and line.endswith(";"):
            cond = line.split(") ", 1)[0] + ")"
            block = [line[len(cond) + 1 :].strip()]
            j = i + 1
            while j < len(lines) and lines[j].startswith(f"{cond} "):
                block.append(lines[j][len(cond) + 1 :].strip())
                j += 1

            out.append(f"{cond} {{")
            out.extend(f"  {stmt}" for stmt in block)
            out.append("}")
            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if qasm_str.endswith("\n") else "")


def summarize_select_copy_layout(data, lambda_param):
    """Summarize the registers used by a SelectCopy QROM instance."""
    if not data:
        raise ValueError("data must not be empty.")

    if isinstance(data[0], int):
        b = max(1, max(int(v) for v in data).bit_length())
    else:
        b = len(data[0])

    N = len(data)
    n_q = int(np.ceil(np.log2(N / lambda_param)))
    n_r = int(np.ceil(np.log2(lambda_param)))

    return {
        "N": N,
        "bitsize": b,
        "lambda_param": lambda_param,
        "q_bits": n_q,
        "r_bits": n_r,
        "q_reg": [f"q_{i}" for i in range(n_q)],
        "r_reg": [f"r_{i}" for i in range(n_r)],
        "target_reg": [f"t_{i}" for i in range(b)],
        "dirty_ancillas": [
            [f"dirty_{r}_{i}" for i in range(b)]
            for r in range(1, lambda_param)
        ],
        "phase_work": [f"phase_{i}" for i in range(max(0, n_q - 1))],
        "w_reg": [f"w_{r}" for r in range(1, lambda_param)],
    }


def pretty_print_select_copy_qrom(
    data,
    lambda_param,
    use_measurement_uncompute=False,
    use_reverse_uncompute=True,
):
    """Return a labeled, human-readable summary of a SelectCopy QROM circuit."""
    layout = summarize_select_copy_layout(data, lambda_param)
    circuit = build_select_copy_qrom(
        data,
        lambda_param=lambda_param,
        use_measurement_uncompute=use_measurement_uncompute,
        use_reverse_uncompute=use_reverse_uncompute,
    )

    lines = []
    lines.append("SelectCopy QROM Circuit")
    lines.append("=" * 24)
    lines.append(
        f"Instance: N={layout['N']}, bits={layout['bitsize']}, "
        f"lambda={layout['lambda_param']}, q_bits={layout['q_bits']}, "
        f"r_bits={layout['r_bits']}"
    )
    lines.append("")
    lines.append("Registers")
    lines.append(f"  q_reg     : {', '.join(layout['q_reg']) or '(none)'}")
    lines.append(f"  r_reg     : {', '.join(layout['r_reg']) or '(none)'}")
    lines.append(f"  target    : {', '.join(layout['target_reg'])}")
    lines.append(
        f"  dirty     : {len(layout['dirty_ancillas'])} blocks, "
        f"{layout['bitsize']} bits each"
    )
    lines.append(
        f"  phase_work: {', '.join(layout['phase_work']) or '(none needed)'}"
    )
    lines.append(f"  w_reg     : {', '.join(layout['w_reg']) or '(none)'}")
    lines.append("")
    lines.append("Big Steps")
    lines.append(
        "  1. Sel1 lookup: q_reg selects one block; target gets f(q,0) and dirty "
        "blocks get f(q,r) xor f(q,0)."
    )
    lines.append(
        "     Uses: q_reg controls the table load; target_reg stores the clean "
        "baseline result; dirty_ancillas store block differences."
    )
    lines.append(
        "  2. One-hot decode: r_reg activates exactly one w_reg flag for the "
        "within-block offset."
    )
    lines.append(
        "     Uses: r_reg as control; w_reg as the one-hot workspace."
    )
    lines.append(
        "  3. Copy stage: w_reg and each dirty block copy the selected bits into "
        "the target register via Toffoli gates."
    )
    lines.append(
        "     Uses: w_reg, dirty_ancillas, and target_reg."
    )
    lines.append(
        "  4. Uncompute one-hot flags: the r_reg decoding is reversed so the "
        "index registers are restored."
    )
    lines.append(
        "     Uses: r_reg and w_reg."
    )
    if use_measurement_uncompute:
        lines.append(
            "  5. Measurement-based cleanup: q_reg-controlled phase work removes "
            "the dirty lookup outputs without reversing the full QROM."
        )
        lines.append(
            "     Uses: q_reg, dirty_ancillas, and phase_work."
        )
    elif use_reverse_uncompute:
        lines.append(
            "  5. Reverse lookup cleanup: the Sel1 lookup is run backwards to "
            "clear target and dirty_ancillas."
        )
        lines.append(
            "     Uses: q_reg and dirty_ancillas."
        )
    else:
        lines.append(
            "  5. Cleanup omitted: the circuit stops at the compute/copy boundary."
        )

    lines.append("")
    lines.append("Circuit Diagram")
    lines.append("-" * 14)
    lines.append(str(circuit))
    return "\n".join(lines)


def _parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Sweep SelectCopy QROM sizes and export QASM plus plots."
    )
    parser.add_argument(
        "--num-entries-list",
        default="4,5,8,16,32,64,128,256,512",
        help="Comma-separated database sizes to sweep.",
    )
    parser.add_argument(
        "--bitsize",
        type=int,
        default=4,
        help="Bit-width of each table entry.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for the deterministic sample databases.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Probability that each sampled bit is 1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output") / "selectcopy",
        help="Root directory for QASM and plot artifacts.",
    )
    return parser

# =========================================================================
# Execution Example
# =========================================================================
def main():
    args = build_arg_parser().parse_args()
    num_entries_list = _parse_csv_ints(args.num_entries_list)
    bitsize = args.bitsize
    seed = args.seed
    density = args.density
    output_root = args.output_root
    stamp = run_stamp()
    qasm_dir = output_root / "qasm"
    plot_dir = output_root / "plots"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = sweep_gate_scaling(
        num_entries_list=num_entries_list,
        bitsize=bitsize,
        seed=seed,
        density=density,
    )

    print("=== SelectCopy QROM Scaling Study ===")
    print("Using the Appendix C measurement-based uncompute path for scaling.")
    print("\n--- Uncompute circuit (Cirq text) ---")
    print(
        build_exportable_select_copy_qrom(
            results[0]["data"],
            lambda_param=results[0]["lambda"],
        )
    )
    for row in results:
        metrics = analyze_gate_metrics(
            build_exportable_select_copy_qrom(
                row["data"],
                lambda_param=row["lambda"],
            )
        )
        print(f"N={row['N']:>3}  lambda={row['lambda']:>2}  " + format_compact_resource_report(metrics))

    for row in results:
        qasm_path = qasm_dir / f"qrom_N{row['N']}_b{bitsize}_{stamp}.qasm"
        qasm_circuit = build_exportable_select_copy_qrom(
            row["data"],
            lambda_param=row["lambda"],
        )
        qasm_str = cirq.qasm(qasm_circuit, args=cirq.QasmArgs(version="3.0"))
        qasm_str = pretty_format_qasm3(qasm_str)
        qasm_path.write_text(qasm_str, encoding="utf-8")

    plot_path = plot_gate_scaling(
        results,
        out_path=plot_dir / f"selectcopy_gate_scaling_{stamp}.png",
    )
    print(f"\nWrote QASM files for {len(results)} QROM sizes into {qasm_dir}.")
    print(f"Saved scaling plot to {plot_path}.")


if __name__ == "__main__":
    main()
