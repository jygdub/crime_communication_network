import timeit
import matplotlib.pyplot as plt

from binary_tree import simulate

# initials
depth = 3
n_iters = 100
total_messages = []

# simulate network
start = timeit.default_timer()

total_messages = simulate(depth=depth,n_iters=n_iters,total_messages=total_messages)

stop = timeit.default_timer()
execution_time = stop - start

# scatterplot all simulations
fig = plt.figure()
plt.scatter(range(n_iters),total_messages)
plt.xlabel("Iterations")
plt.ylabel("Total messages sent")
plt.title(f"Binary tree graph (depth={depth}; n_iters={n_iters}; runtime={round(execution_time,5)} sec.)")
plt.savefig(f"images/preliminary-binary-tree/single-run/simulate{n_iters}-binary-tree-depth{depth}.png", bbox_inches='tight')