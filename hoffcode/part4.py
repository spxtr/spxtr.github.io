#!/usr/bin/env python3

import io

import apng
import numpy as np
import matplotlib.pyplot as plt


def triangular(N):
    xs, ys = [], []
    for x in range(N):
        for y in range(N):
            xs.append(x)
            xs.append(x + 0.5)
            ys.append(np.sqrt(3) * y)
            ys.append(np.sqrt(3) * (y + 0.5))
    return np.array(xs), np.array(ys)


def honeycomb(N):
    xs, ys = triangular(N)
    return np.concatenate((xs, xs + 0.5)), np.concatenate((ys, ys + np.sqrt(3)/6))


def rotate(xs, ys, θ):
    cos = np.cos(θ)
    sin = np.sin(θ)
    return cos*xs - sin*ys, sin*xs + cos*ys


if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 4))
    xs, ys = triangular(30)
    plt.plot(xs, ys, 'k.', ms=2)
    xs, ys = triangular(30)
    plt.plot(xs + 0.5, ys + np.sqrt(3)/6, 'r.', ms=2)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.ylim(0, 20)
    plt.xlim(0, 20)
    plt.gca().patch.set_facecolor('w')
    plt.gca().patch.set_alpha(1.0)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    plt.savefig('hoffimg/4_0.png')

    imgs = []
    for size in np.linspace(1.01, 1.1, 21):
        fig = plt.figure(figsize=(4, 4))
        xs, ys = honeycomb(50)
        plt.plot(xs - 25, ys - 25, 'k.', ms=1)
        xs, ys = honeycomb(50)
        plt.plot((xs - 25)*size, (ys - 25)*size, 'r.', ms=1)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.ylim(-25, 25)
        plt.xlim(-25, 25)
        plt.gca().patch.set_facecolor('w')
        plt.gca().patch.set_alpha(1.0)
        fig.patch.set_alpha(0.0)
        plt.tight_layout()
        bio = io.BytesIO()
        plt.savefig(bio)
        plt.close(fig)
        imgs.append(bio)
    for img in imgs[-2:0:-1]:
        imgs.append(io.BytesIO(img.getbuffer()))
    apng.APNG.from_files(imgs, delay=100).save('hoffimg/4_1.png')

    imgs = []
    for angle in np.linspace(0, np.pi/3, 51, endpoint=False):
        fig = plt.figure(figsize=(4, 4))
        xs, ys = honeycomb(50)
        plt.plot(xs - 25, ys - 25, 'k.', ms=1)
        xs, ys = honeycomb(100)
        xs, ys = rotate(xs - 50, ys - 50, angle)
        plt.plot(xs, ys, 'r.', ms=1)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.ylim(-25, 25)
        plt.xlim(-25, 25)
        plt.gca().patch.set_facecolor('w')
        plt.gca().patch.set_alpha(1.0)
        fig.patch.set_alpha(0.0)
        plt.tight_layout()
        bio = io.BytesIO()
        plt.savefig(bio)
        plt.close(fig)
        imgs.append(bio)
    for img in imgs[-2:0:-1]:
        imgs.append(io.BytesIO(img.getbuffer()))
    apng.APNG.from_files(imgs, delay=100).save('hoffimg/4_2.png')
