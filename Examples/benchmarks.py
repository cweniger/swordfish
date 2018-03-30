#!/usr/bin/env python
# -*- coding: utf-8 -*-

import swordfish as sf
import numpy as np
import pylab as plt

def main():
    N = 10000
    P = np.random.random((N, 2))*20
    X = P
    SH = sf.SignalHandler(P, X)
    V = SH.volume(estimate_dim = False)
    b = SH.get_benchmarks()
    print V, len(b)
    plt.scatter(P[:,0], P[:,1])
    plt.scatter(P[b,0], P[b,1])
    plt.show()

if __name__ == "__main__":
    main()
