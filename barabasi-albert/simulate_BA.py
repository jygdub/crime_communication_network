"""
Script for preliminary network of message passing until consensus in randomly generated Barabasi-Albert networks.

Using scalable noise parameters alpha and beta for sender bias and receiver bias, respectively.
- High alpha means higher probability of sending information the receiver does NOT know yet; 
    low alpha means higher probability of sending information the receiver ALREADY knows, prolonging the convergence.
- Low beta means lower probability of no receiving error and therefore no change in information;
    high beta means higher probability of interpreting message incorrectly and therefore change in information.

Written by Jade Dubbeld
16/01/2024
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

from BA_network import simulate

# initials
n = 10
m = 1
alpha = 1.0
beta = 0.0
n_iters = 10

combi = plt.figure()

for m in [1,2,3,4]:
    
    print(f"m = {m}")
    total_messages = []
    all_diffScores = []

    start = timeit.default_timer()

    for iter in range(n_iters):
        print(f"iter={iter}")

        # simulate until consensus in network
        M, similarity, G_init = simulate(n,m,alpha,beta,n_iters)

        total_messages.append(M)
        all_diffScores.append(list(similarity))

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
    plt.title(f"BA network (n={n}; m={m}; n_samples={n_iters}; runtime={round(execution_time,5)} sec.)")
    plt.savefig(f"images/alpha={alpha}-beta={beta}-n={n}-m={m}-simulate{n_iters}-BA.png",
                bbox_inches='tight')

    # convergence plot all simulations
    fig = plt.figure()

    for iter in range(n_iters):
        plt.plot(np.arange(len(all_diffScores[iter]))+1, all_diffScores[iter],label=f"Run{iter}")

    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Convergence per run on BA network - n={n}|m={m} (n_samples={n_iters})")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(f"images/alpha={alpha}-beta={beta}-n={n}-m={m}-convergence{n_iters}-BA.png"
                , bbox_inches='tight')

    # mean convergence plot over all simulations
    fig = plt.figure()
    y, error = tolerant_mean(all_diffScores)
    plt.plot(np.arange(len(y))+1, y, label=f"Mean of {n_iters} iterations")
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5, label=f"Std. dv. of {n_iters} runs")

    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Mean convergence on BA network - n={n}|m={m} (n_samples={n_iters})")
    plt.legend()
    plt.savefig(f"images/alpha={alpha}-beta={beta}-n={n}-m={m}-average-convergence{n_iters}-BA.png",
                bbox_inches='tight')

    # combi plot of convergence for all combinations of BA parameters n, m
    plt.figure(combi)
    plt.plot(np.arange(len(y))+1, y, label=f"Mean - m={m}")
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5, label=f"Std. dv. - alpha={alpha}")

plt.xlabel("Total messages sent", fontsize=14)
plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Mean convergence BA network (n={n}, alpha={alpha}, beta={beta}, n_samples={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/combiplot-alpha={alpha}-beta={beta}-n={n}-average-convergence{n_iters}-BA.png",
            bbox_inches='tight')