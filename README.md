# State Preparation Circuits

This repository contains small Cirq and Qualtran experiments for alias sampling
and QROM-based state preparation.

## Main scripts

- `selectcopy.py`: select-copy QROM construction, resource counting, and
  plotting helpers.
- `alias_sampler_cirq.py`: alias-sampler circuit builder and verification
  helpers.
- `full_alias_sampler_scaling.py`: end-to-end sweep and artifact export for the
  full alias sampler.
- `alias_sampler_physical_scaling.py`: logical-to-physical resource estimates.
- `test_quantum_components.py`: unit tests for the core helpers.

## Generated output

Running the scripts writes plots, QASM, and logs into `output/`. Those files are
treated as generated artifacts and are ignored by git.

## Quick start

```bash
python3 -m unittest -q test_quantum_components.py
python3 full_alias_sampler_scaling.py
```
