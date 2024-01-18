"""
Script for preliminary network of message passing until consensus in randomly generated Barabasi-Albert networks.

Using scalable noise parameters alpha and beta for sender bias and receiver bias, respectively.
- High alpha means higher probability of sending information the receiver does NOT know yet; 
    low alpha means higher probability of sending information the receiver ALREADY knows, prolonging the convergence.
- Low beta means lower probability of no receiving error and therefore no change in information;
    high beta means higher probability of interpreting message incorrectly and therefore change in information.

Correlate communication efficiency with structural efficiency.

Written by Jade Dubbeld
17/01/2024
"""

import timeit, numpy as np, random
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

from BA_network import simulate

# initials
n = 100
alpha = 1.0
beta = 0.0
n_iters = 10

betweenness = plt.figure()
degree = plt.figure()

for m in [1,2,3,4]:
    
    print(f"m = {m}")

    total_messages = []
    all_diffScores = []
    avgBetweenness = []
    avgDegree = []
    avgGlobal = []
    avgLocal = []

    start = timeit.default_timer()

    for iter in range(n_iters):
        print(f"iter={iter}")

        start2 = timeit.default_timer()

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

        # simulate until consensus in network or reach cut-off
        M, similarity, G_init = simulate(n,m,alpha,beta,n_iters)

        stop2 = timeit.default_timer()
        execution_time2 = stop2 - start2
        print(f"execution time = {execution_time2} in seconds")

        total_messages.append(M)
        all_diffScores.append(list(similarity))

        # compute network measures
        dictBetweenness = nx.betweenness_centrality(G_init)
        avgBetweenness.append(sum(dictBetweenness.values()) / len(dictBetweenness))

        dictDegree = nx.degree_centrality(G_init)
        avgDegree.append(sum(dictDegree.values()) / len(dictDegree))

        avgGlobal.append(nx.global_efficiency(G_init))
        avgLocal.append(nx.local_efficiency(G_init))

    stop = timeit.default_timer()
    execution_time = stop - start

    print(f"execution time = {execution_time} in seconds")
    print(f"total messages = {total_messages}")
    print(f"Average betweenness = {avgBetweenness}")
    print(f"Global efficiency = {avgGlobal} (mean = {np.mean(avgGlobal)})")
    print(f"Local efficiency = {avgLocal} (mean = {np.mean(avgLocal)})")

    # plot relation betweenness and communication efficiency for all combinations of BA parameters n, m
    plt.figure(betweenness)
    plt.scatter(avgBetweenness, total_messages, label=f"m={m}")

    # plot relation degree and communication efficiency for all combinations of BA parameters n, m
    plt.figure(degree)
    plt.scatter(avgDegree, total_messages, label=f"m={m}")

plt.figure(betweenness)
plt.xlabel("Average betweenness centrality", fontsize=14)
plt.ylabel("Total messages sent until convergence", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. betweenness (n={n}, alpha={alpha}, beta={beta}, n_samples={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/betweenness-efficiency{n_iters}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

plt.figure(degree)
plt.xlabel("Average degree centrality", fontsize=14)
plt.ylabel("Total messages sent until convergence", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. degree (n={n}, alpha={alpha}, beta={beta}, n_samples={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/degree-efficiency{n_iters}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')