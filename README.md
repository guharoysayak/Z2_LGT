# Z2_LGT

This repository contains numerical simulation codes used to study real-time dynamics of a one-dimensional $\mathbb{Z}_2$ lattice gauge theory using matrix product state (MPS)–based time-evolving block decimation (TEBD). main_data folder contains data for Fig. 2, 3, S7 and S8 of the following paper. 

The codes were used to generate the raw numerical data reported in:

S. Guha Roy, V. Sharma, K. Xu, U. Borla, J. C. Halimeh, and K. R. A. Hazzard, *Repulsively Bound Hadrons in a $\mathbb{Z}_2$ Lattice Gauge Theory*, arxiv:2510.23618.

---

## Repository structure

Z2_LGT/
  scripts/
    *TEBD python scripts*.py
    *effective model script*.py
  main_data/
    *data for Figs 2, 3, S7, and S8*
  docs/
    RUNS.md
    REPRODUCIBILITY.md
  README.md
  LICENSE


- `scripts/` contains the full simulation codes used to generate the raw time-evolution data.
- The scripts are parameterized and were run multiple times to generate the datasets used in the paper.
- `main_data/` folder contains data for the main figures in the paper.

---

## Reproducibility

The scripts in `scripts/` were used to generate the raw MPS-based TEBD data reported in arXiv:2510.23618.

Convergence checks with respect to bond dimension and Trotter time step were performed as described in the manuscript.

In addition, `scripts/` contains codes used for simulations of the effective model discussed in the paper.

Data for Fig 2, 3, S7, and S8 are available in `main_data/`

## Data structure

`main_data/` contains 4 folders each corresponding Figs 2, 3, S7, and S8. The folders are named accordingly.
The files within each folder contains the x and y values of the each of the different curves in each Figure. Those files are also named accordingly. For instance, `Fig2_density_3m_m6_h6/` contains the data for Fig2 and each file in that folder correspond to each of the 6 panels of the main figure. Similarly, `Fig3_data/` folder has files named `panel_b_m3_m2.csv` or `panel_a_q4_m4.csv`. Here, m3 and q4 correspond to the total 3-meson number of the total tetraquark number respectively. The folder also contains data for the inset in Fig 3.  `FigS7_density_tq_m6_h6/` and `FigS8_data/` follow the same structure.


### Parameter specification

The scripts in `scripts/` are provided in the same form used to generate the raw data
reported in the paper. Simulation parameters such as bond dimension, system size,
total evolution time, number of Trotter steps and the Hamiltonian parameters are specified directly in the scripts
using placeholders (e.g., `#CHI#`, `#LL#`, `#TT#`, `#NN#`).

Users wishing to reproduce or extend the simulations should replace these placeholders
with appropriate numerical values.


---

## Requirements

The codes were written using Python 3.

Typical dependencies include:
- numpy
- scipy
- pandas

---

## Usage notes

- The scripts are intended to be run as standalone simulation jobs.
- Simulation parameters (e.g., system size, couplings, bond dimension, and time step) should be set directly in the scripts before execution.

---

## License

This project is released under the MIT License. See the LICENSE file for details.

---

## Contact

For questions, requests for raw data, or additional information, please contact the authors.

