# State Preparation Circuits

This repository contains small Cirq and Qualtran experiments for alias sampling
and QROM-based state preparation.

## What To Run

| Script | What it does | Main output |
| --- | --- | --- |
| `selectcopy.py` | Builds and analyzes a SelectCopy QROM example | QASM files and a gate-scaling plot |
| `alias_sampler_cirq.py` | Builds a full quantum alias sampler | Example QASM export |
| `full_alias_sampler_scaling.py` | Sweeps alias-sampler sizes and exports all artifacts | QASM bundle, log, plot |
| `alias_sampler_physical_scaling.py` | Converts logical counts into Qualtran physical-cost estimates | Plot and model note |
| `alias_sampler_uncompute_tradeoff.py` | Compares reverse vs measurement-based uncompute | Tradeoff plot |
| `alias_sampler_grouped_steps.py` | Draws a grouped alias-sampler schematic | PNG schematic |
| `selectcopy_pretty_print.py` | Prints and renders a small SelectCopy summary | PNG text figure |
| `selectcopy_pretty_print_forked.py` | Draws a block-diagram SelectCopy schematic | PNG schematic |
| `qasm_to_png.py` | Renders Cirq-generated QASM 3 into a PNG | PNG circuit diagram |
| `vandaele_comparator.py` | Builds and verifies the quantum-quantum comparator | Example QASM export |
| `test_quantum_components.py` | Unit tests for the core helpers | Test result only |

## Suggested Workflow

1. Run the tests first:

```bash
python3 -m unittest -q test_quantum_components.py
```

2. Generate the full alias-sampler sweep:

```bash
python3 full_alias_sampler_scaling.py
```

3. Inspect a rendered circuit or schematic:

```bash
python3 qasm_to_png.py output/full_alias_sampler/<stamp>/full_alias_sampler_*.qasm
```

4. Use `--help` on any runnable script to see its CLI flags. A few examples:

```bash
python3 selectcopy.py --num-entries-list 16,32,64 --bitsize 8
python3 alias_sampler_cirq.py --alias-values 1,0,3,2 --keep-values 2,1,3,1
python3 vandaele_comparator.py --num-bits 6
python3 alias_sampler_uncompute_tradeoff.py --num-entries-list 16,32,64
```

## Generated Output

Generated plots, QASM, and logs are written under `output/`. The repo now keeps
those artifacts out of the top-level directory so the workspace stays readable.
The main subfolders are:

- `output/full_alias_sampler/`
- `output/selectcopy/`
- `output/alias_sampler/`
- `output/comparator/`
- `output/figures/`

## Notes

- The scripts are intentionally runnable as plain `python3 script.py` commands.
- `qasm_to_png.py` accepts an input OpenQASM 3 file and writes a PNG next to it
  unless you pass `--output`.
