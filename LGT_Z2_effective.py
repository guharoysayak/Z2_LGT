import numpy as np
from scipy.linalg import expm
import time

# Build the basis state structure
def basis_indices(L):
    basis_ind = []
    for i in range(L-1):
        if i == 0 or i == 1:
            for j in range(L-2):
                basis_ind.append([i,j-int((L-2)/2)])
        else:
            for j in range(L-i-1):
                basis_ind.append([i,j-int((L-i-1)/2)])
    return basis_ind

def hamiltonian(L, K, J, h):
    size = int((L**2 - L - 2) / 2)
    hamil = np.zeros((size, size), dtype=np.complex128)
    basis = basis_indices(L)
    J2_2h = J**2 / (2 * h)
    K2_4h = K**2 / (4 * h)
    for i in range(size):
        bi0, bi1 = basis[i]
        for j in range(i, size):
            bj0, bj1 = basis[j]
            # Diagonal elements
            if j == i:
                if bi0 == 1:
                    hamil[i, j] = -2 * J2_2h +2*K2_4h
                elif bi0 != 1 and bi0 != 0:# and bi0 != L-2:
                    hamil[i, j] = -4 * J2_2h +2*K2_4h + K2_4h
                elif bi0 == 0:
                    hamil[i, j] = 0
            # Off-diagonal elements
            elif bi0 == 0 and bj0 == 1 and bi1 == bj1:
                hamil[i, j] = K #+ #4*((J**2)*K)/(4*h**2)
            elif bi0 != 0 and bj0 == bi0 + 1:
                if bi0 % 2 != 0 and (bj1 == bi1 or bj1 == bi1 - 1):
                    hamil[i, j] = -J2_2h
                elif bi0 % 2 == 0 and (bj1 == bi1 or bj1 == bi1 + 1):
                    hamil[i, j] = -J2_2h
            hamil[j, i] = hamil[i, j]
    return hamil

def init_state(L):
    psi = np.zeros(int((L**2-L-2)/2),dtype=np.complex128)
    psi[int(L/2)-1] = 1.+0.*1j
    return psi

def init_state_tq(L):
    psi = np.zeros(int((L**2-L-2)/2),dtype=np.complex128)
    psi[int(L/2)-1+L-2] = 1.+0.*1j
    return psi

def tqe_measure(psi,L):
    n_3meson = 0
    for i in range(L-2,2*(L-2)):
        n_3meson += np.conjugate(psi[i])*psi[i]
    return n_3meson

def n_3meson(psi,L):
    n3 = 0
    for i in range(0,L-2):
        n3 += np.conjugate(psi[i])*psi[i]
    return n3

T = 100
N = 4000
L = 100
K = np.linspace(0.01,0.99,50)
J = 1
h = 10

tqe_arr = {}
n3_arr = {}

for k in K:
    t1 = time.time()
    H = hamiltonian(L,k,J,h)
    U = expm(-1j*(T/N)*H)
    psi = init_state(L)
    tqe_arr[f'K={k}'] = []
    tqe_arr[f'K={k}'].append(tqe_measure(psi, L))
    n3_arr[f'K={k}'] = []
    n3_arr[f'K={k}'].append(n_3meson(psi, L))
    for i in range(N):
    
        psi = np.dot(U,psi)
        tqe_arr[f'K={k}'].append(tqe_measure(psi, L))
        n3_arr[f'K={k}'].append(n_3meson(psi, L))
    t2 = time.time()
    print(f'K={k}, time={t2-t1}')

import pickle
with open(f'tqe_L100_J1_h10_tK_T{T}_N{N}_3m.pickle', 'wb') as f:
    pickle.dump(tqe_arr, f)
with open(f'n3_L100_J1_h10_tK_T{T}_N{N}_3m.pickle', 'wb') as f:
    pickle.dump(n3_arr, f)

