"""
Script for preliminary network of message passing until consensus in a lattice structured graph
    Simulation for given number of iterations and alpha and beta noise parameter settings 

Written by Jade Dubbeld
21/12/2023
"""

def tolerant_mean(arrs):
    """
    Function to compute mean of curves with differing lengths

    Parameters:
    - arrs: nested list with sublists of different lengths

    Returns:
    - arr.mean: computed mean over all sublists
    - arr.std: computed standard deviation over all sublists
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)



import timeit, numpy as np, random
import matplotlib.pyplot as plt
import networkx as nx

from lattice import init_lattice, simulate

# initials
dimensions = (5,5)
M = 0
# alpha = 1.0
beta = 0.05
n_iters = 10

for alpha in [1.0, 0.75, 0.5]:
    # for beta in [0.0, 0.25, 0.5]:

    start = timeit.default_timer()

    # simulate until consensus in network
    total_messages, all_similarities = simulate(dimensions,alpha,beta,n_iters)

    stop = timeit.default_timer()
    execution_time = stop - start

    print(f"execution time = {execution_time} in seconds")
    print(f"total messages = {total_messages}")

    # scatterplot all simulations
    fig = plt.figure()
    plt.scatter(range(n_iters),total_messages)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Total messages sent", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Lattice graph (dimensions={dimensions}; N={n_iters}; runtime={round(execution_time,5)} sec.)")
    plt.savefig(f"images/alpha={alpha}-beta={beta}-dim={dimensions}-simulate{n_iters}-lattice.png"
                , bbox_inches='tight')

    # convergence plot all simulations
    fig = plt.figure()

    for iter in range(n_iters):
        plt.plot(np.arange(len(all_similarities[iter]))+1, all_similarities[iter],label=f"Run{iter}")

    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Convergence per run on lattice {dimensions}; alpha={alpha}; beta={beta} (N={n_iters})")
    plt.legend()
    plt.savefig(f"images/alpha={alpha}-beta={beta}-dim={dimensions}-convergence{n_iters}-lattice.png")

    # mean convergence plot over all simulations
    fig = plt.figure()
    y, error = tolerant_mean(all_similarities)
    plt.plot(np.arange(len(y))+1, y, label=f"Mean of {n_iters} iterations")
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5, label=f"Std. dv. of {n_iters} runs")

    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Mean convergence on lattice {dimensions}; alpha={alpha}; beta={beta} (N={n_iters})")
    plt.legend()
    plt.savefig(f"images/alpha={alpha}-beta={beta}-dim={dimensions}-average-convergence{n_iters}-lattice.png")