"""
Script to run efficient sampling communication dynamics on networkx's Graph Atlas networks.
- Saving intermediate results
- Using vectorized implementation

Adapted from run_vectorized_parallel_Atlas.py
Adjusted by Jade Dubbeld
25/04/2024
"""

from efficient_dynamics import simulate

from itertools import product
import multiprocessing as mp, numpy as np, pandas as pd

def run(x):
    graph, i = x

    alpha = 1.00
    beta = 0.00
    nbits = 3

    totalMessages, meanHammingDistance, statesTrajectories, graph = simulate(graph=graph, alpha=alpha, beta=beta, nbits=nbits)
    
    return totalMessages, meanHammingDistance, statesTrajectories, graph, i


if __name__ == "__main__":

    path = "results/efficient-alpha1_00-beta0_00"

    # # setting for local simulation
    # df = pd.read_csv('data/data-GraphAtlas.tsv',sep='\t')

    # setting for Snellius simulation
    df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

    graphs = []

    for j in range(len(df)):

        # generate graph ID
        name = 'G' + str(df['index'].iloc[j])

        graphs.append(name)

    with mp.Pool(processes=mp.cpu_count()) as p: # NOTE: remove -1 from cpu_count for simulation on Snellius
        for result in p.imap_unordered(run, product(graphs,np.arange(100))):
            M, meanHammingDistance, states_trajectory, graph, i = result

            df_states = pd.DataFrame(states_trajectory)
            df_states.to_csv(f'{path}/states-{graph}-run{i}.tsv',sep='\t',index=False)

            df_convergence = pd.DataFrame(
                {'graph': graph,
                'nMessages': M,
                'meanHammingDist': [meanHammingDistance]}
            )

            df_convergence.to_csv(f'{path}/convergence-{graph}-run{i}.tsv',sep='\t',index=False)
