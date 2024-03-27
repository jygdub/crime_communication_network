"""
Script to run communication dynamics on networkx's Graph Atlas networks.
- Saving intermediate results
- Using vectorized implementation

Written by Jade Dubbeld
26/03/2024
"""

from dynamics_vectorized_Atlas import simulate, init
import pickle, networkx as nx, matplotlib.pyplot as plt, numpy as np, pandas as pd, glob
from tqdm import tqdm
from itertools import product

df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')
graphs = []

for i in range(len(df)):

    # generate graph ID
    name = 'G' + str(df['index'].iloc[i])

    graphs.append(name)

for alpha, beta in product([.75],[0.0,0.25,0.50]):

    a = '1_00'
    b = '0_00'

    if alpha == 0.75:
        a = '0_75'
    elif alpha == 0.5:
        a = '0_50'
    
    if beta == 0.25:
        b = '0_25'
    elif beta == 0.5:
        b = '0_50'

    path = f"results/alpha{a}-beta{b}"

    # listFileNames = sorted(glob.glob(f'graphs/*.pickle'))
    # df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

    for j in tqdm(range(len(df))):

        # # generate filename
        # name = 'G' + str(df['index'].iloc[j])
        # file = f'graphs/{name}.pickle'

        # # load graph from file
        # G = pickle.load(open(file,'rb'))

        graph = graphs[j]

        total_messages = []
        allDifferences = []

        # simulate 100 iterations
        for i in range(100):
            # states = []
            # G_init = init(G)

            M, meanHammingDistance, states_trajectory, graph = simulate(graph=graph, alpha=alpha, beta=beta)

            # save state diversity to DataFrame
            df_states = pd.DataFrame(states_trajectory)
            df_states.to_csv(f'{path}/states-{graph}-run{i}.tsv',sep='\t',index=False)

            total_messages.append(M)
            allDifferences.append(list(meanHammingDistance))

        # save convergence rate to DataFrame
        df_convergence = pd.DataFrame(
            {'nMessages': total_messages,
            'meanHammingDist': allDifferences}
        )
        df_convergence.to_csv(f'{path}/convergence-{graph}.tsv',sep='\t',index=False)