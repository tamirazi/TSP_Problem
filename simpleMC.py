"""
@author: ofersh@telhai.ac.il
"""
import numpy as np
import matplotlib.pyplot as plt
import calcTSP as TSP
import os

def simpleMonteCarlo(n , G, evals, func) :
    history = []
    min_perm = np.random.permutation(n)
    calc_min = func(min_perm , G)
    history.append(calc_min)
    for _ in range(evals) :
        perm = np.random.permutation(n)
        calc = func(perm , G)
        if calc < calc_min :
            min_perm = perm
            calc_min = calc
        history.append(calc_min)
    return min_perm,calc_min,history
#
if __name__ == "__main__" :
    dirname = ""
    fname = os.path.join(dirname, "hachula130.dat")
    data = []
    NTrials = 10 ** 6
    with open(fname) as f:
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.linalg.norm(
                np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
            G[j, i] = G[i, j]
    Nruns = 30
    fbest = []
    xbest = []
    for i in range(Nruns):
        xmin, fmin, history = simpleMonteCarlo(n, G, NTrials, TSP.computeTourLength)
        plt.plot(history)
        plt.show()
        print(i, ": minimal path found is ", fmin, "\n\n perm ", xmin ,"\n")
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", min(fbest), "x*=", xbest[fbest.index(min(fbest))])
