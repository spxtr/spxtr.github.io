#!/usr/bin/env python3

import io

import apng
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def Hamiltonian(N, α, ν, U):
    H = np.zeros((N*N, N*N))
    for n in range(N):
        for m in range(N):
            H[(n*N + m + 1) % (N*N), n*N + m] = 1
            H[(n*N + m - 1) % (N*N), n*N + m] = 1
            H[((n + 1)*N + m) % (N*N), n*N + m] = 1
            H[((n - 1)*N + m) % (N*N), n*N + m] = 1
            H[n*N + m, n*N + m] = 2*np.cos(2*np.pi*n*α - ν) + 2*np.cos(2*np.pi*m*α - ν)
    for n in range(N):
        H[n*N + n, n*N + n] += U
    return H

def calc_hoffer(N, αs, νs, U):
    eigs = np.zeros((len(αs), len(νs), N * N))
    for i, α in enumerate(αs):
        for j, ν in enumerate(νs):
            eigs[i, j, :] = linalg.eigvalsh(Hamiltonian(N, α, ν, U))
    return eigs.reshape((len(αs), len(νs) * N * N))

if __name__ == '__main__':
    size = 40
    resα = 21
    resν = 1

    αs = np.linspace(0.0, 0.5, resα)
    νs = np.linspace(0, 2 * np.pi, resν, endpoint=False)

    imgs = []
    for U in np.linspace(0, 10, 21):
        bio = io.BytesIO()
        eigs = calc_hoffer(size, αs, νs, U)
        fig = plt.figure(figsize=(6, 4))
        plt.plot(eigs, αs, 'k,')
        plt.plot(eigs, 1 - αs, 'k,')
        plt.ylim(0, 1)
        plt.xlim(-8, 14)
        plt.xlabel('$\\epsilon$')
        plt.ylabel('$\\alpha$')
        plt.text(14, 0, f'U = {U:.1f}', ha='right', va='bottom')
        fig.patch.set_alpha(0.0)
        plt.gca().patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(bio)
        imgs.append(bio)
        plt.close(fig)
    for img in imgs[-2:0:-1]:
        imgs.append(io.BytesIO(img.getbuffer()))
    apng.APNG.from_files(imgs, delay=100).save('hoffimg/3_0.png')
