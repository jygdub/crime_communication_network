"""
Script to run communication dynamics on networkx's Graph Atlas networks.
- Saving intermediate results
- Using vectorized implementation

Specific use for parameter setting alpha=0.50 and beta=0.50

Adapted from run_vectorized_parallel_Atlas.py
Adjusted by Jade Dubbeld
24/04/2024
"""

from adjusted_dynamics_vectorized import simulate

from itertools import product
import multiprocessing as mp, numpy as np, pandas as pd

def run(x):

    graph, i = x

    alpha = 0.50
    beta = 0.50

    totalMessages, graph = simulate(graph=graph, alpha=alpha, beta=beta)
    
    return totalMessages, graph, i


if __name__ == "__main__":

    path = "results/alpha0_50-beta0_50"

    df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

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
