"""
Script for preliminary network of message passing until consensus in a binary tree graph
    Simulation for multiple tree depth for N repeats, each simulation takes n_iters

Written by Jade Dubbeld
12/12/2023
"""

import timeit, numpy as np
import matplotlib.pyplot as plt

from binary_tree import simulate

all_means = []
all_medians = []

# differing binary tree depths
for depth in [3,5,10]:

    # N simulations
    for N in range(5):

        # initials
        n_iters = 100
        messaging = 'efficient'

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
        plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-depth{depth}-simulate{n_iters}-binomial-repeats{N}.png"
                    , bbox_inches='tight')

# plot mean for multiple depths for n iterations
fig = plt.figure()
plt.scatter([3]*5 + [5]*5 + [10]*5, all_means)
plt.xlabel("Depth", fontsize=14)
plt.ylabel("Mean messages sent", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Binary tree graph - mean messages (depth=[3,5,10], n_iters=5)")
plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-means-depth-3-5-10.png", bbox_inches='tight')

# plot median for multiple depths for n iterations
fig = plt.figure()
plt.scatter([3]*5 + [5]*5 + [10]*5, all_medians)
plt.xlabel("Depth", fontsize=14)
plt.ylabel("Median messages sent", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Binary tree graph - median messages (depth=[3,5,10], n_iters=5)")
plt.savefig(f"images/preliminary-binary-tree/multi-run/{messaging}-medians-depth-3-5-10.png", bbox_inches='tight')