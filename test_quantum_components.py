import unittest

import cirq

from alias_sampler_cirq import (
    build_alias_sampler_circuit,
    verify_alias_qrom_load,
    verify_alias_sampler,
    verify_keep_qrom_load,
)
from selectcopy import (
    analyze_clifford_t_metrics,
    count_clifford_t_explicit_t_gates,
    format_compact_resource_report,
    format_resource_report,
)
from vandaele_comparator import verify_small_instances


class QuantumComponentTests(unittest.TestCase):
    def test_keep_qrom_loads_small_table(self):
        keep_values = [2, 1, 3, 1]
        verify_keep_qrom_load(keep_values)

    def test_alias_qrom_loads_small_table(self):
        alias_values = [1, 0, 3, 2]
        verify_alias_qrom_load(alias_values)

    def test_quantum_comparator_small_cases(self):
        verify_small_instances(max_bits=3)

    def test_alias_sampler_end_to_end_small_cases(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        cases = [
            (0, 0),  # keep branch: 0 < keep[0]
            (1, 1),  # alias branch: 1 !< keep[1]
        ]
        verify_alias_sampler(alias_values, keep_values, cases=cases)

    def test_alias_sampler_reports_dirty_budget(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        _, regs = build_alias_sampler_circuit(
            alias_values,
            keep_values,
            keep_bits=2,
            threshold_bits=1,
            alias_bits=1,
        )
        budget = regs["dirty_budget"]
        self.assertEqual(
            budget["shared_dirty_scratch"],
            max(
                budget["keep_qrom_scratch"],
                budget["comparator_stage_scratch"],
                budget["alias_qrom_scratch"],
            ),
        )
        self.assertEqual(
            budget["comparator_stage_scratch"],
            budget["comparator_scratch"] + budget["comparison_padding_scratch"],
        )
        self.assertEqual(budget["comparator_scratch"], 1)
        self.assertEqual(
            budget["total_dirty_ancillas"],
            budget["shared_dirty_scratch"] + budget["result_latches"],
        )
        self.assertGreater(budget["total_dirty_ancillas"], 0)

    def test_alias_sampler_enforces_alias_width_equals_address_width(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        cases = [(0, 0), (1, 1)]
        verify_alias_sampler(
            alias_values,
            keep_values,
            cases=cases,
            keep_bits=2,
            threshold_bits=1,
            alias_bits=1,
        )

    def test_alias_sampler_rejects_alias_width_mismatch(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        with self.assertRaisesRegex(ValueError, "alias_bits must equal the address width"):
            build_alias_sampler_circuit(
                alias_values,
                keep_values,
                keep_bits=2,
                threshold_bits=1,
                alias_bits=2,
            )

    def test_alias_sampler_supports_dirty_intermediate_mode(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        cases = [
            (0, 0),
            (1, 1),
        ]
        verify_alias_sampler(
            alias_values,
            keep_values,
            cases=cases,
            keep_bits=2,
            threshold_bits=1,
            alias_bits=1,
            clean_intermediates=False,
        )

    def test_alias_sampler_supports_split_qrom_addresses(self):
        alias_values = [3, 2, 1, 0, 0, 1, 2, 3]
        keep_values = [1, 2, 1, 2, 1, 2, 1, 2]
        cases = [
            (0, 0),
            (3, 0),
            (5, 1),
            (7, 3),
        ]
        verify_alias_sampler(
            alias_values,
            keep_values,
            cases=cases,
            keep_bits=2,
            threshold_bits=2,
            alias_bits=3,
            lambda_param=2,
        )

    def test_alias_sampler_reports_gate_metrics(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        _, regs = build_alias_sampler_circuit(
            alias_values,
            keep_values,
            keep_bits=2,
            threshold_bits=1,
            alias_bits=1,
        )
        metrics = regs["gate_metrics"]
        self.assertGreater(metrics["total_gate_count"], 0)
        self.assertGreaterEqual(metrics["explicit_toffoli_count"], 0)
        self.assertGreaterEqual(metrics["classically_controlled_toffoli_equiv_count"], 0)
        self.assertGreater(metrics["toffoli_count"], 0)

    def test_resource_report_formatter(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        _, regs = build_alias_sampler_circuit(
            alias_values,
            keep_values,
            keep_bits=2,
            threshold_bits=1,
            alias_bits=1,
        )
        report = format_resource_report(
            regs["gate_metrics"],
            dirty_budget=regs["dirty_budget"],
            moment_count=3,
        )
        self.assertIn("gates=", report)
        self.assertIn("toffolis=", report)
        self.assertIn("explicit_toffoli=", report)
        self.assertIn("classical_toffoli=", report)
        self.assertIn("dirty=", report)
        self.assertIn("moments=", report)

    def test_compact_resource_report_formatter(self):
        alias_values = [1, 0]
        keep_values = [1, 1]
        _, regs = build_alias_sampler_circuit(
            alias_values,
            keep_values,
            keep_bits=2,
            threshold_bits=1,
            alias_bits=1,
        )
        report = format_compact_resource_report(regs["gate_metrics"])
        self.assertEqual(
            report,
            f"total_gate={regs['gate_metrics']['total_gate_count']}  "
            f"toffoli={regs['gate_metrics']['toffoli_count']}  "
            f"explicit_toffoli={regs['gate_metrics']['explicit_toffoli_count']}  "
            f"classical_toffoli={regs['gate_metrics']['classically_controlled_toffoli_equiv_count']}",
        )

    def test_clifford_t_toffoli_decomposition_counts_seven_t_gates(self):
        circuit = cirq.Circuit(cirq.CCX(*cirq.LineQubit.range(3)))
        self.assertEqual(count_clifford_t_explicit_t_gates(circuit), 7)
        metrics = analyze_clifford_t_metrics(circuit)
        self.assertEqual(metrics["explicit_t_count"], 7)


if __name__ == "__main__":
    unittest.main()
