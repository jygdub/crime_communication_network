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
        total_messages = []

        # simulate network
        start = timeit.default_timer()

        total_messages = simulate(depth=depth,n_iters=n_iters,total_messages=total_messages)

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
        plt.xlabel("Iterations")
        plt.ylabel("Total messages sent")
        plt.title(f"Binary tree graph (depth={depth}; n_iters={n_iters}; runtime={round(execution_time,5)} sec.) - mean={mean}; median={median}")
        plt.savefig(f"images/preliminary-binary-tree/simulate{n_iters}-binary-tree-depth{depth}({N}).png", bbox_inches='tight')

# plot mean for multiple depths for n iterations
fig = plt.figure()
plt.scatter([3]*5 + [5]*5 + [10]*5, all_means)
plt.xlabel("Depth")
plt.ylabel("Mean messages sent")
plt.title("Binary tree graph - mean messages (depth=[3,5,10], n_iters=5)")
plt.savefig(f"images/preliminary-binary-tree/means-depth-3-5-10.png", bbox_inches='tight')

# plot median for multiple depths for n iterations
fig = plt.figure()
plt.scatter([3]*5 + [5]*5 + [10]*5, all_medians)
plt.xlabel("Depth")
plt.ylabel("Median messages sent")
plt.title("Binary tree graph - median messages (depth=[3,5,10], n_iters=5)")
plt.savefig(f"images/preliminary-binary-tree/multi-run/medians-depth-3-5-10.png", bbox_inches='tight')