""" Numerical evaluation of second Renyi entropy in Z2 LGT using gauged PEPS transfer operator method

    In this file, the dependence of s2 is analyzed in the gamma-delta parameter plane.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import schur, eigvals, eig
from itertools import product
import pickle

from transfer_rows import purity



# Parameters:
Nx = 4
Ny = 100
R1 = 2
R2 = 60

alpha = 1.0
beta  = 1.0
gamma_vals = np.linspace(0,2,201)
delta_vals = np.linspace(0.5,1.5,101)

outputf = "s2_vs_delta_gamma_nonpert.p"


# Boundary operator X:
X_op = np.array([[1,0,0,0],
                [0,0,0,1],
                [0,0,1,0],
                [0,1,0,0]
               ])


# Variation of the deconfinement parameter gamma:
p2 = np.zeros((len(gamma_vals),len(delta_vals)))
s2 = np.zeros((len(gamma_vals),len(delta_vals)))
errors = []

for k in range(len(gamma_vals)):
    gamma = gamma_vals[k]

    for i in range(len(delta_vals)):
        delta = delta_vals[i]
        print("\ngamma,delta = ",gamma,delta)

        # Basic element: transfer operator matrix tau0
        tau0 = np.array([[np.abs(alpha)**2, np.abs(gamma)**2, 0, 0],
                         [np.abs(gamma)**2, np.abs(delta)**2, 0, 0],
                         [0               , 0               , np.abs(beta)**2, np.abs(beta)**2],
                         [0               , 0               , np.abs(beta)**2, np.abs(beta)**2]
                        ])

        # a Schur decomposition realizes (71) for a real block-diagonal tau0 into orthogonal V and diagonal L:
        L,V = schur(tau0)
        lambda_vals = np.diag(L)
        print("l1...l4 num:    ",lambda_vals)

        # We want the same ordering as in (126) and (127) to avoid expensive looping over lambda_4=0:
        l1 = 0.5*(np.abs(alpha)**2+np.abs(delta)**2 + np.sqrt((np.abs(alpha)**2-np.abs(delta)**2)**2+4*np.abs(gamma)**4))
        l2 = 0.5*(np.abs(alpha)**2+np.abs(delta)**2 - np.sqrt((np.abs(alpha)**2-np.abs(delta)**2)**2+4*np.abs(gamma)**4))
        l3 = 2*np.abs(beta)**2
        l4 = 0.0
        print("l1...l4 analyt: ",[l1,l2,l3,l4])
        cols = [list(V[:,i]) for i in range(4)] # all eigenvectors
        V3 = [0,0, 1/np.sqrt(2),1/np.sqrt(2)]
        V4 = [0,0,-1/np.sqrt(2),1/np.sqrt(2)]
        V3_ind = cols.index(V3) # identify correct indices of V3, V4 from known analytical form ...
        V4_ind = cols.index(V4) # ... (because l2==l4 might occur)
        perm = [np.where(np.isclose(lambda_vals,l1))[0], np.where(np.isclose(lambda_vals,l2))[0], V3_ind, V4_ind]
        print("perm possible: ",perm) # all possible permutations of eigenvalue/vector ordering
        V_ordered = V.copy()
        for j in range(2): # find correct eigenvector positions to first two eigenvalues
            if len(perm[j])>1:
                allowed_inds = [x for x in perm[j] if (x!=V3_ind and x!=V4_ind)] # exclude mu=3,4 indices
                if len(allowed_inds)>1: # take care of possible l1==l2
                    Vj_ind = allowed_inds[j]
                    V_ordered[:,j] = V[:,Vj_ind]
                    perm[j] = Vj_ind
                else:
                    Vj_ind = allowed_inds[0]
                    V_ordered[:,j] = V[:,Vj_ind]
                    perm[j] = Vj_ind
            else:
                Vj_ind = perm[j][0]
                V_ordered[:,j] = V[:,Vj_ind]
                perm[j] = Vj_ind
        V_ordered[:,2] = V3
        V_ordered[:,3] = V4
        print("perm taken:    ",perm)
        lambda_vals = np.array([l1,l2,l3,l4])
        L, V = np.diag(lambda_vals), V_ordered

        # Checks:
        check_perm = all(x in [0,1,2,3] for x in perm)
        check_schur = np.allclose(V @ L @ V.T, tau0)
        check_orth = np.allclose(V @ V.T,np.identity(4))
        check_order = V[2,0]==0 and V[2,1]==0 and V[2,2]==1/np.sqrt(2) and V[2,3]==-1/np.sqrt(2)
        print("all elems in perm:                ",check_perm)
        print("Schur decomposition tau0=V*L*V.T: ",check_schur)
        print("V is orthogonal:                  ",check_orth)
        print("V is correctly ordered:           ",check_order)
        if not (check_perm and check_schur and check_orth and check_order):
            errors.append((k,i))

        # M_mu matrices constructed from V:
        Pu = np.array([[1,0],[0,0]])
        Pd = np.array([[0,0],[0,1]])
        M1 = V[0,0]*Pu + V[1,0]*Pd
        M2 = V[0,1]*Pu + V[1,1]*Pd
        M3 = (1.0/np.sqrt(2))*np.array([[0, 1],[1,0]])
        M4 = (1.0/np.sqrt(2))*np.array([[0,-1],[1,0]])
        M_mats = [M1,M2,M3,M4]

        print("Tr[M1*M1.T] = ",np.trace(M1 @ M1.T)) # realizes (74)
        print("Tr[M1*M2.T] = ",np.trace(M1 @ M2.T))

        # calculation of normalized entropy from purity:
        p2[k,i] = purity(Nx,Ny,R1,R2,X_op,lambda_vals,M_mats)
        s2[k,i] = -np.log(p2[k,i])
        print("s2 = ",s2[k,i])


# Control:
print("\nerrors: ",errors)


# Write data to file:
output = {}
output['info'] = "[Nx,Ny,R1,R2,a,b] = "+str([Nx,Ny,R1,R2,alpha,beta])
output['gamma_vals'] = gamma_vals
output['delta_vals'] = delta_vals
output['p2'] = p2
output['s2'] = s2
file = open(outputf, "wb")
pickle.dump(output, file)
file.close()





























