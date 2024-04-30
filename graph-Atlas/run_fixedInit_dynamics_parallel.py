"""
Script to run communication dynamics on networkx's Graph Atlas networks.
- Saving intermediate results
- Using vectorized implementation

Written by Jade Dubbeld
30/04/2024
"""

from fixedInit_dynamics_vectorized import simulate

from itertools import product
import multiprocessing as mp, numpy as np, pandas as pd

def run(x):
    graph, i = x

    alpha = 1.00
    beta = 0.00
    nbits = 3

    totalMessages, graph = simulate(graph=graph, alpha=alpha, beta=beta, nbits=nbits)
    
    return totalMessages, graph, i


if __name__ == "__main__":

    path = "results/fixed-alpha1_00-beta0_00"

    # # setting for local simulation
    # df = pd.read_csv('data/data-GraphAtlas.tsv',sep='\t')

    # setting for Snellius simulation
    df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')


    # only select graph sizes n=7
    df = df[df['nodes']==7]

    graphs = []

    for j in range(len(df)):

        # generate graph ID
        name = 'G' + str(df['index'].iloc[j])

        graphs.append(name)

    with mp.Pool(processes=mp.cpu_count()) as p: # NOTE: remove -1 from cpu_count for simulation on Snellius
        for result in p.imap_unordered(run, product(graphs,np.arange(100))):
            M, graph, i = result

            df_convergence = pd.DataFrame(
                {'graph': graph,
                'nMessages': M},
                index=[0]
            )

            df_convergence.to_csv(f'{path}/convergence-{graph}-run{i}.tsv',sep='\t',index=False)
