# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
Simulated Annealing on a continuous domain bounded within [lb,ub]**n
"""
import numpy as np
import matplotlib.pyplot as plt
import calcTSP as TSP
import os


def SimulatedAnnealing(n, G, max_evals, variation, func, seed=None):
    T_init = 100000
    T_min = 1e-4
    alpha = 0.999
    f_lower_bound = 6110
    eps_satisfactory = 1e-5
    max_internal_runs = 50
    local_state = np.random.RandomState(seed)
    history = []
    best_perm = min_perm = np.random.permutation(n)
    calc_best = calc_min = func(best_perm, G)
    eval_cntr = 0
    T = T_init
    history.append(calc_min)
    while (T > T_min) and eval_cntr < max_evals:
        for _ in range(max_internal_runs):
            perm = variation(min_perm, 2)
            calc = func(perm, G)
            eval_cntr += 1
            dE = calc - calc_min
            if dE <= 0 or local_state.uniform(size=1) < np.exp(-dE / T):
                min_perm = perm
                calc_min = calc
            if dE < 0:
                calc_best = calc_min
                best_perm = min_perm
            history.append(calc_min)
            if np.mod(eval_cntr, int(max_evals / 10)) == 0:
                print(eval_cntr, " evals: Best path found=", calc_min)
                print(T)
            if calc_best < f_lower_bound + eps_satisfactory:
                T = T_min
                break
        T *= alpha
    return best_perm, calc_best, history ,eval_cntr


#


def myswap(perm, number_of_swaps):
    temp = np.copy(perm)
    for _ in range(number_of_swaps):
        num1, num2 = np.random.choice(perm, 2)
        a = np.where(temp == num1)
        b = np.where(temp == num2)
        temp[b] = num1
        temp[a] = num2

    return temp

def findTemp():
    NTrials = 10 ** 6
    eval=0
    perm = np.random.permutation(n)
    for _ in range(NTrials):
        perm2 = myswap(perm, 2)
        calc = TSP.computeTourLength(perm2, G)
        eval += 1
        dE = calc - calc_min


if __name__ == "__main__":
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
    Nruns=30
    fbest = []
    xbest = []
    for i in range(Nruns) :
        xmin,fmin,history,eval = SimulatedAnnealing(n, G, NTrials, myswap, TSP.computeTourLength)
        plt.semilogy(history)
        plt.show()
        print(i,": minimal path found is ", fmin,"\n perm ", xmin , "\n evals: ",eval)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ",min(fbest),"x*=",xbest[fbest.index(min(fbest))])
