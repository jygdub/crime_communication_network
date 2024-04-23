"""
Script to check transition possibility.
- Only allowing single edge transitions

Written by Jade Dubbeld
23/04/2024
"""

import multiprocessing as mp
import pandas as pd, numpy as np, networkx as nx, pickle
from itertools import product

def run(x):

    i,j = x

    if j > i:
        return None

    graph1 = 'G' + str(i)
    file1 = f'graphs/{graph1}.pickle'

    G1 = pickle.load(open(file1,'rb'))

    graph2 = 'G' + str(j)
    file2 = f'graphs/{graph2}.pickle'

    G2 = pickle.load(open(file2,'rb'))

    if nx.graph_edit_distance(G1,G2) == 1.0:
        return i, j
    

if __name__ == "__main__":

    n=7
    
    df = pd.read_csv('data-GraphAtlas.tsv',usecols=['index','nodes'],sep='\t')
    df = df[df['nodes']==n]

    df = df.reindex(index=df.index[::-1])

    from_graph = []
    to_graph = []

    with mp.Pool(processes=mp.cpu_count()) as p: # NOTE: remove -1 from cpu_count for simulation on Snellius
        for result in p.map(run, product(df['index'],df['index'])):

            if result != None:
                i,j = result

                from_graph.append(i)
                to_graph.append(j)

    np.savetxt(f'from_graph_n={n}.tsv',
        from_graph,
        delimiter ="\t",
        fmt ='% i')

    np.savetxt(f'to_graph_n={n}.tsv',
        to_graph,
        delimiter ="\t",
        fmt ='% i')