#!/usr/bin/env python3

import io

import apng
import numpy as np
import matplotlib.pyplot as plt

π = np.pi


def Hamiltonian(N, α, ν):
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 2 * np.cos(2 * π * i * α - ν)
        H[(i + 1) % N, i] = 1
        H[(i - 1) % N, i] = 1
    return H


if __name__ == '__main__':
    αs = []
    for q in [101]:
        for p in range(1, q // 2 + 1):
            αs.append((p / q, p, q))
    αs = sorted(αs, key=lambda α: α[0])
    εs = np.linspace(-4, 4, 1001)

    outs = np.zeros((len(αs), len(εs)))
    for i, (_, p, q) in enumerate(αs):
        for j, ε in enumerate(εs):
            m = Hamiltonian(q, p / q, 0) - ε * np.eye(q)
            outs[i, j] = np.linalg.slogdet(m)[1] / q
    imgs = []
    for trace in outs:
        bio = io.BytesIO()
        fig = plt.figure(figsize=(6, 4))
        plt.plot(εs, trace)
        plt.ylim(0, 1.2)
        plt.xlim(-4, 4)
        plt.xlabel('$\\epsilon$')
        plt.ylabel('logdet')
        fig.patch.set_alpha(0.0)
        plt.gca().patch.set_facecolor('w')
        plt.gca().patch.set_alpha(1.0)
        plt.tight_layout()
        plt.savefig(bio)
        plt.close(fig)
        imgs.append(bio)
    for img in imgs[-2:0:-1]:
        imgs.append(io.BytesIO(img.getbuffer()))
    apng.APNG.from_files(imgs, delay=100).save('hoffimg/2_0.png')

    fig = plt.figure(figsize=(6, 4))
    plt.pcolormesh(εs, np.hstack((np.array(αs)[:,0], 1 - np.array(αs)[:,0])), np.vstack((outs, np.flip(outs, axis=1))), cmap='Spectral')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$\\alpha$')
    fig.patch.set_alpha(0.0)
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    plt.tight_layout()
    plt.savefig('hoffimg/2_1.png')
