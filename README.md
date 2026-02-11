# Z2_LGT

This repository contains numerical simulation codes used to study real-time dynamics of a one-dimensional $\mathbb{Z}_2$ lattice gauge theory using matrix product state (MPS)–based time-evolving block decimation (TEBD).

The codes were used to generate the raw numerical data reported in:

S. Guha Roy and V. Sharma, K. Xu, U. Borla, J. C. Halimeh, K. R. A. Hazzard, *Repulsively bound hadronic states in a $\mathbb{Z}_2$ lattice gauge theory*, arXiv:2510.23618.

---

## Repository structure

Z2_LGT/
  scripts/
    TEBD_MPS_*.py
    effective_model_*.py
  docs/
    RUNS.md
    REPRODUCIBILITY.md
  README.md
  LICENSE


- `scripts/` contains the full simulation codes used to generate the raw time-evolution data.
- The scripts are parameterized and were run multiple times to generate the datasets used in the paper.
- Post-processing and plotting scripts are not included in this repository.

---

## Reproducibility

The scripts in `scripts/` were used to generate the raw MPS-based TEBD data reported in arXiv:2510.23618. These raw datasets are several gigabytes in size and are therefore not included in this repository.

Convergence checks with respect to bond dimension and Trotter time step were performed as described in the manuscript.

In addition, `scripts/` contains codes used for simulations of the effective model discussed in the paper.

Raw data and post-processing scripts used to generate the figures are available from the authors upon reasonable request.

---

## Requirements

The codes were developed and tested using Python 3.

Typical dependencies include:
- numpy
- scipy
- pandas

---

## Usage notes

- The scripts are intended to be run as standalone simulation jobs.
- Simulation parameters (e.g., system size, couplings, bond dimension, and time step) should be set directly in the scripts before execution.
- Simulations can be computationally intensive and may require substantial runtime and memory resources for larger system sizes or bond dimensions.

---

## License

This project is released under the MIT License. See the LICENSE file for details.

---

## Contact

For questions, requests for raw data, or additional information, please contact the authors.

