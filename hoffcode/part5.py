#!/usr/bin/env python3

import io

import apng
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

π = np.pi


def Hamiltonian(N, α, t):
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 2 * t * np.cos(2 * π * i * α)
        H[(i + 1) % N, i] = 1
        H[(i - 1) % N, i] = 1
    return H


def calc_eigs(N, αs, t):
    eigs = np.zeros((len(αs), N))
    for i, α in enumerate(αs):
        eigs[i, :] = linalg.eigvalsh(Hamiltonian(N, α, t))
    return eigs.reshape((len(αs), N))


def plot_butterfly(eigs, outfile, t):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(eigs, αs, 'k,')
    plt.plot(eigs, 1 - αs, 'k,')
    plt.ylim(0, 1)
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$\\alpha$')
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    fig.patch.set_alpha(0.0)
    plt.title(f'$t = {t:.1f}$')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


if __name__ == '__main__':
    N = 200
    αs = np.linspace(0.0, 0.5, 101)
    ts = np.linspace(3.0, 1.0, 21)
    imgs = []
    for t in ts:
        bio = io.BytesIO()
        plot_butterfly(calc_eigs(N, αs, t), bio, t)
        imgs.append(bio)
        imgs.insert(0, io.BytesIO(bio.getvalue()))
    apng.APNG.from_files(imgs, delay=200).save('hoffimg/5_0.png')
