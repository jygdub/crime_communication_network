"""
Script to run communication dynamics on networkx's Graph Atlas networks.
- Saving intermediate results
- Using vectorized implementation

Written by Jade Dubbeld
19/03/2024
"""

from dynamics_vectorized_Atlas import simulate

from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import multiprocessing as mp, numpy as np, pandas as pd, glob, pickle

def run(x):
    # print(mp.current_process())

    graph, i = x

    alpha = 1.0
    beta = 0.0

    totalMessages, meanHammingDistance, statesTrajectories, graph = simulate(graph=graph, alpha=alpha, beta=beta)
    
    return totalMessages, meanHammingDistance, statesTrajectories, graph, i


if __name__ == "__main__":

    df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

    graphs = []

    for j in range(len(df)-900):

        # generate graph ID
        name = 'G' + str(df['index'].iloc[j])

        graphs.append(name)

    with mp.Pool(processes=mp.cpu_count()-1) as p: # NOTE: remove -1 from cpu_count for simulation on Snellius
        for result in p.imap_unordered(run, product(graphs,np.arange(10))):
            M, meanHammingDistance, states_trajectory, graph, i = result

            df_states = pd.DataFrame(states_trajectory)
            df_states.to_csv(f'test/states-{graph}-run{i}.tsv',sep='\t',index=False)

            df_convergence = pd.DataFrame(
                {'graph': graph,
                'nMessages': M,
                'meanHammingDist': [meanHammingDistance]}
            )

            df_convergence.to_csv(f'test/convergence-{graph}-run{i}.tsv',sep='\t',index=False)
