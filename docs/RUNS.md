# Runs and outputs

All scripts in `scripts/` generate raw simulation output files (CSV). Outputs are not tracked in git.

## Script list

- `scripts/genLattice-MPS-run-3m.py`
  - Purpose: TEBD evolution of the 3-meson initial state
  - Key parameters: L, chi, T, N, J, K, h, m
  - Output: `MPS_lattice_3m_L{L}_chi{chi}_T{T}_N{N}_J{J}_K{K}_m{m}_h{h}.csv

- `scripts/genLattice-MPS-run-tq.py`
  - Purpose: TEBD evolution of the tetraquark initial state
  - Key parameters: L, chi, T, N, J, K, h, m
  - Output: `MPS_lattice_tq_L{L}_chi{chi}_T{T}_N{N}_J{J}_K{K}_m{m}_h{h}.csv

- `scripts/genLattice-MPS-run-vac.py`
  - Purpose: TEBD evolution of the vacuum state
  - Key parameters: L, chi, T, N, J, K, h, m
  - Output: `MPS_lattice_vac_L{L}_chi{chi}_T{T}_N{N}_J{J}_K{K}_m{m}_h{h}.csv

- `scripts/LGT_Z2_effective.py`
  - Purpose: Effective model evolution for both 3-meson and tetraquark states
  - Key parameters: L, T, N, J, K, h
  - Output: numpy pickle files depending on initial state and parameters
