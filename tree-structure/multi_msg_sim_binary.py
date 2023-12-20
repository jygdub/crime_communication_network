"""
Script for preliminary network of message passing until consensus in a binary tree graph
    Simulation for multiple messaging methods of n_iters on given tree depth

Written by Jade Dubbeld
13/12/2023
"""

import timeit, numpy as np
import matplotlib.pyplot as plt

from binary_tree import simulate

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

all_means = []
all_medians = []

# initials
n_iters = 100
depth = 3
convergence = plt.figure()

# differing messaging methods
for messaging in ['forward','random','efficient']:
        
    # simulate network
    start = timeit.default_timer()

    total_messages, all_similarities = simulate(depth=depth,n_iters=n_iters,messaging=messaging)

    stop = timeit.default_timer()
    execution_time = stop - start

    # scatterplot all simulations
    fig = plt.figure()
    plt.scatter(range(n_iters),total_messages)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Total messages sent", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Binary tree graph (depth={depth}; n_iters={n_iters}; runtime={round(execution_time,5)} sec.)")
    plt.savefig(f"images/preliminary-binary-tree/converged-messaging/{messaging}-depth{depth}-simulate{n_iters}-binomial.png"
                , bbox_inches='tight')
    
    # convergence plot all simulations of given depth for differing messagings
    fig = plt.figure()

    for iter in range(n_iters):
        plt.plot(np.arange(len(all_similarities[iter]))+1, all_similarities[iter],label=f"Iteration {iter}")

    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("Average string similarity (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Convergence of all simulations on binary tree - depth = {depth} (N={n_iters})")
    plt.legend()
    plt.savefig(f"images/preliminary-binary-tree/converged-messaging/{messaging}-depth{depth}-convergence{n_iters}-binomial.png")

    # mean convergence plot over all simulations of given depth for differing messagings
    plt.figure(convergence)
    y, error = tolerant_mean(all_similarities)
    plt.plot(np.arange(len(y))+1, y, label=f"Mean of {n_iters} iterations - {messaging}")
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5, label=f"Std. dv. of {n_iters} iterations - {messaging}")

plt.figure(convergence)
plt.xlabel("Rounds", fontsize=14)
plt.ylabel("Average string similarity (Hamming distance)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Mean convergence of all simulations on binary tree - depth = {depth} (N={n_iters})")
plt.legend()
plt.savefig(f"images/preliminary-binary-tree/converged-messaging/depth{depth}-average-convergence{n_iters}-binomial.png")
