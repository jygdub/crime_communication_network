"""
Script to compute network measures of pre-generated Barabasí-Albert graphs
- Link density
- Degree centrality
- Betweenness centrality
- Closeness centrality
- Clustering coefficient
- Transitivity
- Global efficiency
- Local efficiency

Written by Jade Dubbeld
23/01/2024
"""

import networkx as nx, numpy as np, pickle
from BA_network import init_BA

n = 100

for m in [1,2,3,4]:
    
    # set-up dataset
    data = np.zeros((50,8))

    avgLink = []
    avgDegree = []
    avgBetweenness = []
    avgCloseness = []
    avgClustering = []
    avgTransitivity = []
    avgGlobal = []
    avgLocal = []

    for iter in range(1,51):
        G_init = pickle.load(open(f"graphs/m={m}/graph{iter}-n={n}-m={m}.pickle", 'rb'))
        G_init = init_BA(G_init)

        # compute average link density
        avgLink.append(nx.density(G_init))

        # compute average degree centrality
        dictDegree = nx.degree_centrality(G_init)
        avgDegree.append(sum(dictDegree.values()) / len(dictDegree))

        # compute average betweenness centrality
        dictBetweenness = nx.betweenness_centrality(G_init)
        avgBetweenness.append(sum(dictBetweenness.values()) / len(dictBetweenness))

        # compute average closeness centrality
        dictCloseness = nx.closeness_centrality(G_init)
        avgCloseness.append(sum(dictCloseness.values()) / len(dictCloseness))

        # compute average clustering coefficient
        avgClustering.append(nx.average_clustering(G_init))

        # compute transitivity
        avgTransitivity.append(nx.transitivity(G_init))    

        # compute global efficiency (Latora & Marchiori, 2001)
        avgGlobal.append(nx.global_efficiency(G_init))

        # compute local efficiency (Latora & Marchiori, 2001)
        avgLocal.append(nx.local_efficiency(G_init))

        # insert calculated measures in initialized dataset
        data[iter-1] = [avgLink[iter-1],
                        avgDegree[iter-1],
                        avgBetweenness[iter-1],
                        avgCloseness[iter-1],
                        avgClustering[iter-1],
                        avgTransitivity[iter-1],
                        avgGlobal[iter-1],
                        avgLocal[iter-1]]
        
    pickle.dump(data, open(f"graphs/m={m}/measures-m={m}.pickle", 'wb'))