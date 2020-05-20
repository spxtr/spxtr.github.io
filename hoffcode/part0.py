#!/usr/bin/env python3

import io

import apng
import numpy as np
import matplotlib.pyplot as plt

π = np.pi


def A(ε, m, α, ν):
    return np.array([[ε - 2*np.cos(2*π*m*α - ν), -1], [1, 0]])


def primes(n):
    def sieve(ns):
        if len(ns) == 0:
            return ns
        car = ns[0]
        cdr = ns[1:]
        return [car] + sieve(list(filter(lambda x: x % car != 0, cdr)))
    return sieve(list(range(2, n + 1)))


def plot_butterfly(qmax, outfile):
    αs = []
    for q in primes(qmax):
        for p in range(1, q // 2 + 1):
            αs.append((p / q, p, q))
    αs = sorted(αs, key=lambda α: α[0])
    εs = np.linspace(-4, 4, 1001)


    trs = np.empty((len(αs), len(εs)))
    for i, (_, p, q) in enumerate(αs):
        for j, ε in enumerate(εs):
            m = np.eye(2)
            for k in range(q):
                m = A(ε, k, p/q, π/2/q) @ m
            trs[i, j] = np.abs(np.trace(m))


    xs = []
    ys = []
    for (α, _, _), tr in zip(αs, trs):
        for ε, t in zip(εs, tr):
            if t < 4:
                xs.extend([ε, ε])
                ys.extend([α, 1 - α])

    fig = plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, 'k,')
    plt.xlim(-4, 4)
    plt.ylim(0, 1)
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$\\alpha$')
    plt.text(4, 0, f'$q \\leq {qmax}$', ha='right', va='bottom')
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    plt.savefig(outfile)


if __name__ == '__main__':
    plot_butterfly(19, 'hoffimg/0_0.png')

    imgs = []
    for qmax in primes(29):
        bio = io.BytesIO()
        plot_butterfly(qmax, bio)
        imgs.append(bio)
    for img in imgs[-2:0:-1]:
        imgs.append(io.BytesIO(img.getbuffer()))
    apng.APNG.from_files(imgs, delay=500).save('hoffimg/0_1.png')
