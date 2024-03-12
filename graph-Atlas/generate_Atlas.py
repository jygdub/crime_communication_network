"""
Script to generate all known network structures up to 7 nodes using networkx.graph_atlas_g().
- Computing and storing characteristics of interest in TSV-file

Written by Jade Dubbeld
12/03/2024
"""

import networkx as nx, pandas as pd

ALL_G = nx.graph_atlas_g()

df = pd.DataFrame(columns=['index',
                           'nodes',
                           'edges',
                           'degree',
                           'betweenness',
                           'CFbetweenness',
                           'closeness',
                           'clustering',
                           'globalEff',
                           'localEff'])

index = list()
nodes = list()
edges = list()
degree = list()
betweenness = list()
# CFbetweenness = list()
closeness = list()
clustering = list()
globalEff = list()
localEff = list()

for i, G in enumerate(ALL_G):

    if i == 0:
        continue

    index.append(i)
    nodes.append(G.number_of_nodes())
    edges.append(G.number_of_edges())

    dictDegree = nx.degree_centrality(G)
    degree.append(sum(dictDegree.values()) / len(dictDegree))

    dictBetweenness = nx.betweenness_centrality(G)
    betweenness.append(sum(dictBetweenness.values()) / len(dictBetweenness))

    # dictCFBetweenness = nx.current_flow_betweenness_centrality(G)
    # CFbetweenness.append(sum(dictCFBetweenness.values()) / len(dictCFBetweenness))

    dictCloseness = nx.closeness_centrality(G)
    closeness.append(sum(dictCloseness.values()) / len(dictCloseness))   

    clustering.append(nx.average_clustering(G)) 

    globalEff.append(nx.global_efficiency(G))

    localEff.append(nx.local_efficiency(G))

df['index'] = index
df['nodes'] = nodes
df['edges'] = edges
df['degree'] = degree
df['betweenness'] = betweenness
# df['CFbetweenness'] = CFbetweenness
df['closeness'] = closeness
df['clustering'] = clustering
df['globalEff'] = globalEff
df['localEff'] = localEff

df.to_csv('data-GraphAtlas.tsv',sep='\t',index=False)