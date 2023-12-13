"""
Script for preliminary network of message passing until consensus in a binary tree graph
    Simulation for multiple tree depth for N repeats, each simulation takes n_iters

Written by Jade Dubbeld
12/12/2023
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
n_iters = 10
repeats = 5
messaging = 'forward'

# differing binary tree depths
for depth in [3,5,10]:

    # N simulations
    for N in range(repeats):

        # simulate network
        start = timeit.default_timer()

        total_messages, all_similarities = simulate(depth=depth,n_iters=n_iters,messaging=messaging)

        stop = timeit.default_timer()
        execution_time = stop - start

        # compute mean and median total message sent
        print(total_messages)
        mean = np.mean(total_messages)
        all_means.append(mean)
        median = np.median(total_messages)
        all_medians.append(median)
        print(f"Mean total messages sent = {mean} | Median total messages sent = {median}\n")

        # scatterplot all simulations
        fig = plt.figure()
        plt.scatter(range(n_iters),total_messages)
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Total messages sent", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Binary tree graph (depth={depth}; n_iters={n_iters}; runtime={round(execution_time,5)} sec.) - mean={mean}; median={median}")
        plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-depth{depth}-simulate{n_iters}-binomial-repeat{N}.png"
                    , bbox_inches='tight')
        
        # convergence plot all simulations
        fig = plt.figure()

        for iter in range(n_iters):
            plt.plot(np.arange(len(all_similarities[iter]))+1, all_similarities[iter],label=f"Iteration {iter}")

        plt.xlabel("Rounds", fontsize=14)
        plt.ylabel("Average string similarity (Hamming distance)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Convergence of all simulations on binary tree - depth = {depth} (N={n_iters})")
        plt.legend()
        plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-depth{depth}-convergence{n_iters}-binomial-repeat{N}.png")

        # mean convergence plot over all simulations
        fig = plt.figure()
        y, error = tolerant_mean(all_similarities)
        plt.plot(np.arange(len(y))+1, y, label=f"Mean of {n_iters} iterations")
        plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5, label=f"Std. dv. of {n_iters} iterations")

        plt.xlabel("Rounds", fontsize=14)
        plt.ylabel("Average string similarity (Hamming distance)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Mean convergence of all simulations on binary tree - depth = {depth} (N={n_iters})")
        plt.legend()
        plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-depth{depth}-average-convergence{n_iters}-binomial-repeat{N}.png")

# plot mean nr. messages for multiple depths for n iterations
fig = plt.figure()
plt.scatter([3]*5 + [5]*5 + [10]*5, all_means)
plt.xlabel("Depth", fontsize=14)
plt.ylabel("Mean messages sent", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Binary tree graph - mean messages (depth=[3,5,10], n_iters={n_iters}, repeats={repeats})")
plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-means-depth-3-5-10.png", bbox_inches='tight')

# plot median nr. messages for multiple depths for n iterations
fig = plt.figure()
plt.scatter([3]*5 + [5]*5 + [10]*5, all_medians)
plt.xlabel("Depth", fontsize=14)
plt.ylabel("Median messages sent", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Binary tree graph - median messages (depth=[3,5,10], n_iters={n_iters}, repeats={repeats})")
plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-medians-depth-3-5-10.png", bbox_inches='tight')