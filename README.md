# Repulsively bound hadrons in a Z2 lattice gauge theory

In this repository, we share the Matrix Product State based Time Evolving Block Decimation (TEBD) code that we have used to generate the data for our repulsively bound hadrons paper. 

Here, there are 3 TEBD code files that are uploaded which are genLattice-MPS-run-tq.py, genLattice-MPS-run-3m.py and genLattice-MPS-run-vac.py which correspond to three different initial states, the 3-meson initial state, the tetraquark initial state and the vacuum state. Even though there are 3 different files, it is very easy to change the initial state in the code by changing the init_psi(L) function in the code.

Additionally, there is the code that simulates the Effective model (LGT_Z2_effective.py) described in the paper
