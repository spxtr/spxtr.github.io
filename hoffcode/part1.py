#!/usr/bin/env python3

import io

import apng
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

π = np.pi


def Hamiltonian(N, α, ν):
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 2 * np.cos(2 * π * i * α - ν)
        H[(i + 1) % N, i] = 1
        H[(i - 1) % N, i] = 1
    return H


def calc_eigs(N, αs, νs):
    eigs = np.zeros((len(αs), len(νs), N))
    for i, α in enumerate(αs):
        for j, ν in enumerate(νs):
            eigs[i, j, :] = linalg.eigvalsh(Hamiltonian(N, α, ν))
    return eigs.reshape((len(αs), len(νs)*(N)))


def plot_butterfly(eigs, outfile):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(eigs, αs, 'k,')
    plt.plot(eigs, 1 - αs, 'k,')
    plt.xlim(-4, 4)
    plt.ylim(0, 1)
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$\\alpha$')
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


def plot_gaps(eigs, N, αs, νs, outfile):
    topn = np.argsort(np.diff(np.sort(eigs)))[:,-10:] / N / len(νs)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(topn, αs, 'k,')
    plt.plot(topn, 1 - αs, 'k,')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('$n$')
    plt.ylabel('$\\alpha$')
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


if __name__ == '__main__':
    N = 200
    αs = np.linspace(0.0, 0.5, 101)
    νs = np.linspace(0, 2 * π, 1, endpoint=False)

    eigs = calc_eigs(200, αs, νs)
    plot_butterfly(eigs, 'hoffimg/1_0.png')
    plot_gaps(eigs, 200, αs, νs, 'hoffimg/1_1.png')

    αs = np.linspace(0.0, 0.5, 201)
    νs = np.linspace(0, 2 * π, 20, endpoint=False)
    imgs = []
    for ν in νs:
        bio = io.BytesIO()
        plot_butterfly(calc_eigs(N, αs, [ν]), bio)
        imgs.append(bio)
    apng.APNG.from_files(imgs, delay=100).save('hoffimg/1_2.png')

    N = 10
    imgs = []
    for ν in νs:
        bio = io.BytesIO()
        plot_butterfly(calc_eigs(N, αs, [ν]), bio)
        imgs.append(bio)
    apng.APNG.from_files(imgs, delay=100).save('hoffimg/1_3.png')
