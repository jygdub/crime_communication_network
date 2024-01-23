"""
Script for preliminary network of message passing until consensus in randomly generated Barabasi-Albert networks.

Using scalable noise parameters alpha and beta for sender bias and receiver bias, respectively.
- High alpha means higher probability of sending information the receiver does NOT know yet; 
    low alpha means higher probability of sending information the receiver ALREADY knows, prolonging the convergence.
- Low beta means lower probability of no receiving error and therefore no change in information;
    high beta means higher probability of interpreting message incorrectly and therefore change in information.

Correlate communication efficiency with structural efficiency.
- Link density
- Degree centrality
- Betweenness centrality
- Closeness centrality
- Clustering coefficient
- Transitivity
- Global efficiency
- Local efficiency

Written by Jade Dubbeld
17/01/2024
"""

import timeit, random, pickle
import matplotlib.pyplot as plt, numpy as np
import networkx as nx

from datetime import datetime

from BA_network import simulate, init_BA, visualize, generate

test = "test"

# initials
n = 100
alpha = 1.0
beta = 0.0
n_iters = 2

# initialize figures
link = plt.figure()
degree = plt.figure()
betweenness = plt.figure()
closeness = plt.figure()
clustering = plt.figure()
transitivity = plt.figure()
globalEfficiency = plt.figure()
localEfficiency = plt.figure()

# simulate for various m
for m in [1,2,3,4]:
    
    print(f"m = {m}")

    # set interval of graphs
    first = 1
    last = first + n_iters

    # set-up dataset
    data = np.zeros((n_iters,8))

    # initialze
    total_messages = []
    all_diffScores = []

    # some time indications
    startTime = timeit.default_timer()

    # simulate for a subset of pre-generated graphs
    for iter in range(first,last):
        print(f"iter={iter}")

        # some time indications
        startTime2 = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

        # load graph
        G_init = pickle.load(open(f"graphs/m={m}/graph{iter}-n={n}-m={m}.pickle", 'rb'))
        G_init = init_BA(G_init)

        # simulate until consensus in network or reach cut-off
        M, stringDiff = simulate(G_init,alpha,beta)

        # some time indications
        stopTime2 = timeit.default_timer()
        executionTime2 = stopTime2 - startTime2
        print(f"execution time = {executionTime2} in seconds")

        # save messages sent and string difference (Hamming distance)
        total_messages.append(M)
        all_diffScores.append(list(stringDiff))

    # retrieve corresponding network measures
    data = pickle.load(open(f"graphs/m={m}/measures-m={m}.pickle", 'rb'))
    avgLink = data[first-1:last-1,0]
    avgDegree = data[first-1:last-1,1]
    avgBetweenness = data[first-1:last-1,2]
    avgCloseness = data[first-1:last-1,3]
    avgClustering = data[first-1:last-1,4]
    avgTransitivity = data[first-1:last-1,5]
    avgGlobal = data[first-1:last-1,6]
    avgLocal = data[first-1:last-1,7]

    # some time indications
    stopTime = timeit.default_timer()
    executionTime = stopTime - startTime

    # print(f"execution time = {executionTime} in seconds")
    print(f"total messages = {total_messages}")

    # plot relation link density and communication efficiency for all combinations of BA parameters n, m
    plt.figure(link)
    plt.scatter(avgLink, total_messages, label=f"m={m}")

    # plot relation degree and communication efficiency for all combinations of BA parameters n, m
    plt.figure(degree)
    plt.scatter(avgDegree, total_messages, label=f"m={m}")

    # plot relation betweenness and communication efficiency for all combinations of BA parameters n, m
    plt.figure(betweenness)
    plt.scatter(avgBetweenness, total_messages, label=f"m={m}")

    # plot relation closeness and communication efficiency for all combinations of BA parameters n, m
    plt.figure(closeness)
    plt.scatter(avgCloseness, total_messages, label=f"m={m}")

    # plot relation clustering and communication efficiency for all combinations of BA parameters n, m
    plt.figure(clustering)
    plt.scatter(avgClustering, total_messages, label=f"m={m}")

    # plot relation transitivity and communication efficiency for all combinations of BA parameters n, m
    plt.figure(transitivity)
    plt.scatter(avgTransitivity, total_messages, label=f"m={m}")

    # plot relation global efficiency and communication efficiency for all combinations of BA parameters n, m
    plt.figure(globalEfficiency)
    plt.scatter(avgGlobal, total_messages, label=f"m={m}")

    # plot relation local efficiency and communication efficiency for all combinations of BA parameters n, m
    plt.figure(localEfficiency)
    plt.scatter(avgLocal, total_messages, label=f"m={m}")


# decorate link density plot
plt.figure(link)
plt.xlabel("Average link density", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. link density")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}link-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate degree centrality plot
plt.figure(degree)
plt.xlabel("Average degree centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. degree")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}degree-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate betweenness centrality plot
plt.figure(betweenness)
plt.xlabel("Average betweenness centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. betweenness")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}betweenness-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate closeness centrality plot
plt.figure(closeness)
plt.xlabel("Average closeness centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. closeness")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}closeness-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate clustering coefficient plot
plt.figure(clustering)
plt.xlabel("Average clustering centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. clustering")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}clustering-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate transitivity plot
plt.figure(transitivity)
plt.xlabel("Average transitivity", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. transitivity")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}transitivity-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate global efficiency plot
plt.figure(globalEfficiency)
plt.xlabel("Average global efficiency", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. global efficiency")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}global-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate local efficiency plot
plt.figure(localEfficiency)
plt.xlabel("Average local efficiency", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. local efficiency")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/efficiency/{test}local-efficiency-graphs{first}-{last}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')