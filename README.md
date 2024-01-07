# GaugedEntanglement
transfer operator method for the calculation of entanglement entropies in (2+1)D LGTs
using Python 3
___________________________________________________

This project was developed for the numerical calculation of second normalized Renyi entanglement entropy S_2 for a Z2 lattice gauge theory. The method is based on a transfer operator approach using gauged projected entangled pair states in 2 spatial dimensions.
Paper: https://arxiv.org/abs/2401.01930

main files:
- Entropies_demonstration.ipynb: Jupyter notebook demonstrating the individual functions and calculations
- transfer_rows.py: contains functions calculating and evaluating all transfer matrices

evaluation files:
- Entropies_numerical_R2_dep.py: the dependence on the subsystem length R2 is analyzed
- Entropies_numerical_eigenvalues.py: the dependence of the dominant eigenvalues of E2 and E2p on R1 is analyzed
- Entropies_numerical_gamma_dep.py: the dependence on the deconfinement parameter gamma is analyzed
- Entropies_numerical_param_grid.py: the dependence of S_2 is analyzed in the gamma-delta parameter plane
