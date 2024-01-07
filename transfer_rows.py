""" Functions to calculate all relevant transfer row operators


"""

import numpy as np
from itertools import product



"""
Function calculating the norm of $\rho$ by exponentiating the single transfer row as
\begin{equation}
Tr_A[\rho_A] = Tr_A[Tr_B[\rho]] = Tr[\rho] = Tr[E^{(1) N_y}] ,
\end{equation}
where E$^{(1)}$ is the trace along a row, i.e.
\begin{equation}
E^{(1)} = Tr_{\text{row}}[\tau_0^{\otimes N_x}] = \sum_{\{\mu\}} (\lambda_{\mu_1} \cdot\ldots\cdot \lambda_{\mu_{N_x}}) Tr[M_{\mu_1} \cdot\ldots\cdot M_{\mu_{N_x}}] M_{\mu_1} \otimes\ldots\otimes M_{\mu_{N_x}},
\end{equation}
cf. eq. (75)
"""
def single_transfer_row(Nx, lambdas, M):
    print("... calculating E1")
    # for practical calculations, assume a thin lattice of dimensions Nx<9 << Ny~100
    E1 = np.zeros((2**Nx,2**Nx))

    # At every lattice site x along a row with length Nx,
    # the index mu can take effectively 3 different values mu=(0,1,2) [because lambda_4=0].
    # We sum up all the possible 3^Nx configurations contributing to E1:
    for config in product(range(3), repeat=Nx):
        mu_of_x = np.array(config) # = \mu(x)
        M_prod_mu = np.identity(2)
        lambda_prod_mu = 1.0

        # loop (product) along a row x:
        for x in range(Nx):
            lambda_prod_mu *= lambdas[mu_of_x[x]] # = \prod_x lambda_{\mu(x)}
            M_prod_mu = M_prod_mu @ M[mu_of_x[x]] # = \prod_x M_{\mu(x)}

        # calculate tensor product M_{\mu(x=1)} x...x M_{\mu(x=Nx)}
        # to get matrix form of E1:
        M_tensor_mu = np.kron(M[mu_of_x[0]], M[mu_of_x[1]])
        for x in range(2,Nx):
            M_tensor_mu = np.kron(M_tensor_mu, M[mu_of_x[x]])

        E1 = E1 + (lambda_prod_mu * np.trace(M_prod_mu) * M_tensor_mu)
    return E1

def norm_rho(Nx, Ny, lambdas, M):
    print("... calculating norm")
    # full state rho = E1^(Ny):
    E1 = single_transfer_row(Nx,lambdas,M)
    rho = np.linalg.matrix_power(E1,Ny)
    return np.trace(rho) # returns norm of state


"""
Calculate the "double transfer row" $E^{(2)}=E^{(1)} \otimes E^{(1)}$ explicitly with suitable (site-wise) tensor product structure:

\begin{align}
E^{(2)} = \sum_{\{\mu\},\{\nu\}} &(\lambda_{\mu_1} \cdot\ldots\cdot \lambda_{\mu_{N_x}}) (\lambda_{\nu_1} \cdot\ldots\cdot \lambda_{\nu_{N_x}}) Tr[M_{\mu_1} \cdot\ldots\cdot M_{\mu_{N_x}}] Tr[M_{\nu_1} \cdot\ldots\cdot M_{\nu_{N_x}}] \\
&(M_{\mu_1} \otimes M_{\nu_1}) \otimes\ldots\otimes (M_{\mu_{N_x}} \otimes M_{\nu_{N_x}})
\end{align}
"""
def double_transfer_row(Nx, lambdas, M):
    print("... calculating E2")
    E2 = np.zeros((2**(2*Nx),2**(2*Nx)))

    # At every lattice site x along the row with length Nx,
    # the indices mu,nu can take 3 different values mu,nu=(0,1,2).
    # We sum up all the possible (3*3)^Nx configurations contributing to E2:
    for config in product(range(3),range(3), repeat=Nx):
        munu_of_x = np.array(config) # contains mu(x) at index 2x, nu(x) at 2x+1
        lambda_prod_mu = 1.0
        lambda_prod_nu = 1.0
        M_prod_mu = np.identity(2)
        M_prod_nu = np.identity(2)

        # loop (product) along a row x:
        for x in range(Nx):
            lambda_prod_mu *= lambdas[munu_of_x[2*x]]   # = \prod_x lambda_{\mu(x)}
            lambda_prod_nu *= lambdas[munu_of_x[2*x+1]] # = \prod_x lambda_{\nu(x)}
            M_prod_mu = M_prod_mu @ M[munu_of_x[2*x]]   # = \prod_x M_{\mu(x)}
            M_prod_nu = M_prod_nu @ M[munu_of_x[2*x+1]] # = \prod_x M_{\nu(x)}

        # calculate tensor product  (M_{mu_1} x M_{nu_1}) x...x (M_{mu_Nx} x M_{nu_Nx})
        # to get matrix form of E2:
        M_tensor = np.kron(M[munu_of_x[0]], M[munu_of_x[1]])
        for x in range(1,Nx):
            M_tensor = np.kron(M_tensor, np.kron(M[munu_of_x[2*x]], M[munu_of_x[2*x+1]]))

        E2 = E2 + (lambda_prod_mu*lambda_prod_nu * np.trace(M_prod_mu)*np.trace(M_prod_nu) * M_tensor)
    return E2


"""
Calculate the boundary transfer row $E^{(2)}_{||}$ with two $X$ insertions:

\begin{align}
E^{(2)}_{||} = \sum_{\{\mu\},\{\nu\}} &(\lambda_{\mu_1} \cdot\ldots\cdot \lambda_{\mu_{N_x}}) (\lambda_{\nu_1} \cdot\ldots\cdot \lambda_{\nu_{N_x}})
Tr[X (M_{\mu_1} \otimes M_{\nu_1}) \cdot\ldots\cdot (M_{\mu_{R_1}} \otimes M_{\nu_{R_1}}) X (M_{\mu_{R_1+1}} \otimes M_{\nu_{R_1+1}}) \cdot\ldots\cdot (M_{\mu_{N_x}} \otimes M_{\nu_{N_x}})] \\
&(M_{\mu_1} \otimes M_{\nu_1}) \otimes\ldots\otimes (M_{\mu_{N_x}} \otimes M_{\nu_{N_x}})
\end{align}
"""
def boundary_transfer_row(Nx, R1, X, lambdas, M):
    print("... calculating E2p")
    if R1>=Nx:
        print("Error in def boundary_transfer_row(Nx, R1, lambdas, M): choose R1<Nx")
        return 0
    E2p = np.zeros((2**(2*Nx),2**(2*Nx)))

    # At every lattice site x along the row with length Nx,
    # the indices mu,nu can take 3 different values mu,nu=(0,1,2).
    # We sum up all the possible (3*3)^Nx configurations contributing to E2p:
    for config in product(range(3),range(3), repeat=Nx):
        munu_of_x = np.array(config) # contains mu(x) at index 2x, nu(x) at 2x+1
        lambda_prod_mu = 1.0
        lambda_prod_nu = 1.0
        XM_prod = X

        # loop (product) along a row x:
        for x in range(Nx):
            lambda_prod_mu *= lambdas[munu_of_x[2*x]]   # = \prod_x lambda_{\mu(x)}
            lambda_prod_nu *= lambdas[munu_of_x[2*x+1]] # = \prod_x lambda_{\nu(x)}
            # multiply iteratively all matrices along row:
            # XM_prod = X*(M_{mu_1} \otimes M_{nu_1})...(M_{mu_R1} \otimes M_{nu_R1})*X
            #           *(M_{mu_{R1+1}} \otimes M_{nu_{R1+1}})...(M_{mu_Nx} \otimes M_{nu_Nx})
            if x == R1-1:
                XM_prod = XM_prod @ np.kron(M[munu_of_x[2*x]], M[munu_of_x[2*x+1]]) @ X
            else:
                XM_prod = XM_prod @ np.kron(M[munu_of_x[2*x]], M[munu_of_x[2*x+1]])

        # calculate tensor product  (M_{mu_1} x M_{nu_1}) x...x (M_{mu_Nx} x M_{nu_Nx})
        # to get matrix form of E2p:
        M_tensor = np.kron(M[munu_of_x[0]], M[munu_of_x[1]])
        for x in range(1,Nx):
            M_tensor = np.kron(M_tensor, np.kron(M[munu_of_x[2*x]], M[munu_of_x[2*x+1]]))

        E2p = E2p + (lambda_prod_mu*lambda_prod_nu * np.trace(XM_prod) * M_tensor)
    return E2p


"""
Calculate the "bottom" and "top" boundary row
$$\mathcal X(R_1) = X(x=1) \otimes \ldots \otimes X(x=R_1) \otimes \mathbb{1}(x=R_1+1) \otimes \ldots \otimes \mathbb{1}(x=N_x)$$
"""
def boundary_operator(Nx, R1, X):
    print("... calculating XX")
    XX = X
    for x in range(1,Nx):
        if x < R1:
            XX = np.kron(XX, X)
        else:
            XX = np.kron(XX, np.identity(4))
    return XX


"""
Calculate the full normalized purity for a periodic system of dimensions $N_x \times N_y$ and a subsystem of size $R_1 \times R_2$:

\begin{equation}
\bar p_2 = \frac{Tr[\rho_A^2]}{Tr^2[\rho_A]} = \frac{Tr[\mathcal X(R_1) E^{(2)R_2}_{||}(R_1) \mathcal X(R_1) E^{(2) N_y-R_2}]}{Tr^2[E^{(1) N_y}]}
\end{equation}
"""
def purity(Nx, Ny, R1, R2, X, lambdas, M):
    print("... calculating purity")
    # matrix rows:
    nrm = norm_rho(Nx, Ny, lambdas, M)
    E2  = double_transfer_row(Nx, lambdas, M)
    E2p = boundary_transfer_row(Nx, R1, X, lambdas, M)
    XX  = boundary_operator(Nx, R1, X)

    # overall tiled matrix lattice:
    rho_A_2 = XX @ np.linalg.matrix_power(E2p,R2) @ XX @ np.linalg.matrix_power(E2,Ny-R2)

    # normalized purity:
    p2norm = np.trace(rho_A_2)/nrm**2

    return p2norm











