"""
Trial script to simulate Deffuant-like dynamics on LFR benchmark graphs.

Written by Jade Dubbeld
14/02/2024
"""

import networkx as nx, random, numpy as np, matplotlib.pyplot as plt, pickle, glob, re, timeit
from tqdm import tqdm
from datetime import datetime

from LFR_network import init, message, hamming_distance, simulate

exp_degree = 3.0
exp_community = 1.5

listFileNames = sorted(glob.glob(f'graphs/first-generation/tau1={exp_degree}-tau2={exp_community}-*.pickle'))

alpha = 1.0
beta = 0.0
# n_iters = 1

total_messages = []
all_diffScores = []

print(listFileNames[1])

for i, filename in enumerate(listFileNames[:1]):

    G_init = pickle.load(open(filename, 'rb'))

    G = init(G_init)

    # for iter in range(n_iters):
        # print(f"iter={iter}")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    start = timeit.default_timer()
    # simulate until consensus in network
    M, similarity = simulate(G,alpha,beta)

    total_messages.append(M)
    all_diffScores.append(list(similarity))
    print("Converged!")
    
    stop = timeit.default_timer()
    execution_time = stop - start

    print(f"execution time = {execution_time} in seconds")
    print(f"total messages = {total_messages}")

    # scatterplot all simulations
    fig_scatter = plt.figure()
    plt.scatter(range(len(total_messages)),total_messages, label=filename)

    # convergence plot all simulations
    fig_converge = plt.figure(figsize=(13,8))
    plt.plot(np.arange(0,len(all_diffScores[i])), np.reshape(all_diffScores,(-1,1)))#,label=filename)


plt.figure(fig_scatter)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Total messages sent", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"LFR; runtime={round(execution_time,5)} sec.")
plt.savefig(f"images/test-scatter.png",bbox_inches='tight')

plt.figure(fig_converge)
plt.xlabel("Total messages sent", fontsize=14)
plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Convergence on LFR")
# plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/test-convergence.png", bbox_inches='tight')