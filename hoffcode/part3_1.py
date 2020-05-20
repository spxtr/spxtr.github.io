#!/usr/bin/env python3

import io

import apng
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

π = np.pi


def Hamiltonian(N, α, ν, U):
    H = scipy.sparse.lil_matrix((N*N, N*N))
    for n in range(N):
        for m in range(N):
            H[(n*N + m + 1) % (N*N), n*N + m] = 1
            H[(n*N + m - 1) % (N*N), n*N + m] = 1
            H[((n + 1)*N + m) % (N*N), n*N + m] = 1
            H[((n - 1)*N + m) % (N*N), n*N + m] = 1
            H[n*N + m, n*N + m] = 2*np.cos(2*π*n*α - ν) + 2*np.cos(2*π*m*α - ν)
    for n in range(N):
        H[n*N + n, n*N + n] += U
    return H.tocsr()

if __name__ == '__main__':
    N = 200
    k = 10
    αs = np.linspace(0.0, 0.2, 41)
    νs = np.linspace(0, 2 * π, 1, endpoint=False)

    eigs = np.zeros((len(αs), len(νs), k))
    for i, α in enumerate(αs):
        for j, ν in enumerate(νs):
            eigs[i, j, :] = scipy.sparse.linalg.eigsh(Hamiltonian(N, α, ν, 0.5), k=k, tol=1e-2, which='SA')[0]
    eigs = eigs.reshape((len(αs), len(νs) * k))

    fig = plt.figure(figsize=(6, 4))
    plt.ylim(0, 0.2)
    plt.xlim(-8, -5)
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$\\alpha$')
    ys = np.linspace(0, 0.2, 101)
    def E(α, ν1, ν2, U):
        return 8 - U*np.sqrt(α) - 4*π*α*(ν1 + ν2 + 1) + 4*π*π*α*α*((2*ν1 + 1)**2 + (2*ν2 + 1)**2 + 2) / 16
    plt.plot(-E(ys, 0, 0, 0), ys, 'grey')
    plt.plot(-E(ys, 0, 0, 0.5), ys, 'r--')
    plt.plot(-E(ys, 1, 0, 0), ys, 'grey')
    plt.plot(-E(ys, 1, 0, 0.5), ys, 'r--')
    plt.plot(-E(ys, 1, 1, 0), ys, 'grey')
    plt.plot(-E(ys, 1, 1, 0.5), ys, 'r--')

    plt.plot(eigs, αs, 'ko', markersize=2)
    fig.patch.set_alpha(0.0)
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    plt.tight_layout()
    plt.savefig('hoffimg/3_1.png')
