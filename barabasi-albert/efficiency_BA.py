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

import timeit, random, pickle
import matplotlib.pyplot as plt, numpy as np
import networkx as nx

from datetime import datetime

from BA_network import simulate, init_BA, visualize, generate

test = ""
graph = 3
path = f"images/efficiency/graph{graph}/simulation"
# path = "images/efficiency/various-graphs1-20/simulation"

# initials
n = 100
alpha = 1.0
beta = 0.0
n_iters = 20

# initialize figures
link = plt.figure()
degree = plt.figure()
betweenness = plt.figure()
closeness = plt.figure()
clustering = plt.figure()
transitivity = plt.figure()
globalEfficiency = plt.figure()
localEfficiency = plt.figure()

combiMeanConvergence = plt.figure()

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

    # load graph; topology remains fixed over simulation
    G_init = pickle.load(open(f"graphs/m={m}/graph{graph}-n={n}-m={m}.pickle", 'rb'))

    # retrieve corresponding network measures
    data = pickle.load(open(f"graphs/m={m}/measures-m={m}.pickle", 'rb'))
    avgLink = [data[first-1,0]]*n_iters
    avgDegree = [data[first-1,1]]*n_iters
    avgBetweenness = [data[first-1,2]]*n_iters
    avgCloseness = [data[first-1,3]]*n_iters
    avgClustering = [data[first-1,4]]*n_iters
    avgTransitivity = [data[first-1,5]]*n_iters
    avgGlobal = [data[first-1,6]]*n_iters
    avgLocal = [data[first-1,7]]*n_iters

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

        # load graph per iteration
        # G_init = pickle.load(open(f"graphs/m={m}/graph{iter}-n={n}-m={m}.pickle", 'rb'))

        # initialize agents in network
        G_init = init_BA(G_init)

        # simulate until consensus in network or reach cut-off
        M, stringDiff = simulate(G_init,alpha,beta)

        # some time indications
        stopTime2 = timeit.default_timer()
        executionTime2 = stopTime2 - startTime2
        print(f"execution time = {executionTime2} in seconds")

        # for each state update, save messages sent and string difference (Hamming distance)
        total_messages.append(M)
        all_diffScores.append(list(stringDiff))

    # some time indications
    stopTime = timeit.default_timer()
    executionTime = stopTime - startTime

    # retrieve corresponding network measures
    # data = pickle.load(open(f"graphs/m={m}/measures-m={m}.pickle", 'rb'))
    # avgLink = data[first-1:last-1,0]
    # avgDegree = data[first-1:last-1,1]
    # avgBetweenness = data[first-1:last-1,2]
    # avgCloseness = data[first-1:last-1,3]
    # avgClustering = data[first-1:last-1,4]
    # avgTransitivity = data[first-1:last-1,5]
    # avgGlobal = data[first-1:last-1,6]
    # avgLocal = data[first-1:last-1,7]

    pickle.dump(total_messages, open(f"{path}/{test}consensus-rate-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-m={m}-BA.pickle",'wb'))

    # print(f"execution time = {executionTime} in seconds")
    print(f"total messages = {total_messages}")

    ##############
    ## PLOTTING ##
    ##############

    # convergence plot all simulations
    fig = plt.figure()

    for run in range(n_iters):
        plt.plot(np.arange(len(all_diffScores[run]))+1, all_diffScores[run],label=f"Run{run}")

    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Convergence per run on BA network - n={n}|m={m} (N={n_iters})")
    plt.legend(bbox_to_anchor=(1,1))
    # plt.savefig(f"{path}/{test}convergence-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-m={m}-BA.png"
    #             , bbox_inches='tight')
    plt.savefig(f"{path}/{test}convergence-graphs{first}-run{iter}-alpha={alpha}-beta={beta}-n={n}-m={m}-BA.png"
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
    plt.title(f"Mean convergence BA network - n={n}|m={m} (N={n_iters})")
    plt.legend()
    # plt.savefig(f"{path}/{test}avg-convergence-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-m={m}-BA.png",
    #             bbox_inches='tight')
    plt.savefig(f"{path}/{test}avg-convergence-graphs{first}-run{iter}-alpha={alpha}-beta={beta}-n={n}-m={m}-BA.png",
                bbox_inches='tight')
    
    # combi plot of convergence for all combinations of BA parameters n, m
    plt.figure(combiMeanConvergence)
    plt.plot(np.arange(len(y))+1, y, label=f"Mean - m={m}")
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5, label=f"Std. dv. - m={m}")

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


plt.figure(combiMeanConvergence)
plt.xlabel("Total messages sent", fontsize=14)
plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Mean convergence BA network (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}combi-avg-convergence-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}combi-avg-convergence-graph{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate link density plot
plt.figure(link)
plt.xlabel("Average link density", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. link density (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}link-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}link-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate degree centrality plot
plt.figure(degree)
plt.xlabel("Average degree centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. degree (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}degree-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}degree-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate betweenness centrality plot
plt.figure(betweenness)
plt.xlabel("Average betweenness centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. betweenness (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}betweenness-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}betweenness-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate closeness centrality plot
plt.figure(closeness)
plt.xlabel("Average closeness centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. closeness (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}closeness-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}closeness-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate clustering coefficient plot
plt.figure(clustering)
plt.xlabel("Average clustering centrality", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. clustering (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}clustering-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}clustering-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate transitivity plot
plt.figure(transitivity)
plt.xlabel("Average transitivity", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. transitivity (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}transitivity-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}transitivity-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate global efficiency plot
plt.figure(globalEfficiency)
plt.xlabel("Average global efficiency", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. global efficiency (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}global-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}global-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')

# decorate local efficiency plot
plt.figure(localEfficiency)
plt.xlabel("Average local efficiency", fontsize=14)
plt.ylabel("Total messages sent until consensus", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Correlation communication efficiency vs. local efficiency (N={n_iters})")
plt.legend(bbox_to_anchor=(1,1))
# plt.savefig(f"{path}/{test}local-efficiency-graphs{first}-{last-1}-alpha={alpha}-beta={beta}-n={n}-BA.png",
#             bbox_inches='tight')
plt.savefig(f"{path}/{test}local-efficiency-graphs{first}-alpha={alpha}-beta={beta}-n={n}-BA.png",
            bbox_inches='tight')