"""
Script to compute network measures for generated LFR benchmark graphs.

Written by Jade Dubbeld
12/02/2024
"""

import networkx as nx, numpy as np, pandas as pd, pickle, glob, re
from tqdm import tqdm

def find_substring(sub1, sub2):   
    start=str(re.escape(sub1)) 
    end=str(re.escape(sub2))
    
    # printing result
    result=re.findall(start+"(.*)"+end,filename)[0]

    return result

exp_degree = 3.0
exp_community = 1.5
min_community = 10

# get all relevant filenames
listFileNames = sorted(glob.glob(f'graphs/tau1={exp_degree}-tau2={exp_community}-*.pickle'))

# create initial dataframe
data = pd.DataFrame(None, index=np.arange(len(listFileNames)),
                     columns=['tau1',
                             'tau2',
                             'mu',
                             'average_degree',
                             'min_community',
                             'seed',
                             'degreeCentrality',
                             'betweennessCentrality',
                             'currentFlowBetweenness',
                             'closenessCentrality',
                             'clusteringCoefficient'])

# fill in fixed values
data['tau1'] = exp_degree
data['tau2'] = exp_community
data['min_community'] = min_community

for i, filename in tqdm(enumerate(listFileNames)):

    # retrieve values from filename
    mu = (find_substring('mu=','-avg_deg'))
    avg_deg = (find_substring('avg_deg=','-min_comm'))
    seed = (find_substring('seed=','.pickle'))

    # load in graph
    G = pickle.load(open(filename, 'rb'))

    # compute average degree centrality
    dictDegree = nx.degree_centrality(G)
    degree = (sum(dictDegree.values()) / len(dictDegree))

    # compute average betweenness centrality
    dictBetweenness = nx.betweenness_centrality(G)
    betweenness = (sum(dictBetweenness.values()) / len(dictBetweenness))

    # compute average betweenness centrality
    dictCFBetweenness = nx.current_flow_betweenness_centrality(G)
    CFbetweenness = (sum(dictCFBetweenness.values()) / len(dictCFBetweenness))

    # compute average closeness centrality
    dictCloseness = nx.closeness_centrality(G)
    closeness = (sum(dictCloseness.values()) / len(dictCloseness))

    # compute average clustering coefficient
    clustering = (nx.average_clustering(G))

    # insert row
    data.iloc[i,[2,3,5,6,7,8,9,10]] = [mu, avg_deg, seed, degree, betweenness, CFbetweenness, closeness, clustering]

print(data)

data.to_csv(f'data/all-measures.tsv',sep='\t',index=False)