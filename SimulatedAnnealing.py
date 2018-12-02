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
    bad_choice = 0
    T_init = 35
    T_min = 1e-5
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
            perm = variation(min_perm , eval_cntr)
            calc = func(perm, G)
            eval_cntr += 1
            dE = calc - calc_min
            if dE <= 0 or local_state.uniform(size=1) < np.exp(-dE / T):
                if local_state.uniform(size=1) < np.exp(-dE / T):
                    bad_choice += 1
                min_perm = perm
                calc_min = calc
            if dE < 0:
                calc_best = calc_min
                best_perm = min_perm
            history.append(calc_min)
            if np.mod(eval_cntr, int(max_evals / 10)) == 0:
                print(eval_cntr, " evals: Best path found=", calc_min)
                print("Temperature is : ", T)
            if calc_best < f_lower_bound + eps_satisfactory:
                T = T_min
                break
        T *= alpha
    return best_perm, calc_best, history, eval_cntr, bad_choice

#

def myswap3(perm , iter):
    # this func choose an index (i) and size randomly
    # and reverse the permutation from this index by length -> size
    end = round(iter/100000) * 15
    temp = list(np.copy(perm))
    size = np.random.randint(0, len(perm) - end)
    i = np.random.randint(0, len(perm) - size)
    temp[i: i + size] = reversed(temp[i: i + size])
    return temp


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
    Nruns = 30
    fbest = []
    xbest = []
    for i in range(Nruns):
        xmin, fmin, history, eval, bad_choices = SimulatedAnnealing(n, G, NTrials, myswap3, TSP.computeTourLength)
        plt.ylim(6110, 30000)
        plt.plot(history)
        plt.show()
        print("bad : ", bad_choices)
        print(i, ": minimal path found is ", fmin, "\n perm ", xmin, "\n evals: ", eval)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", min(fbest), "x*=", xbest[fbest.index(min(fbest))])
