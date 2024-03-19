"""
Script to run communication dynamics on networkx's Graph Atlas networks.
- Saving intermediate results

Written by Jade Dubbeld
12/03/2024
"""

from dynamics_nonvectorized_Atlas import simulate, init
import pickle, networkx as nx, matplotlib.pyplot as plt, numpy as np, pandas as pd, glob
from tqdm import tqdm

def visualize(G):
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G: networkx graph to display
    """

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))
    plt.show()

listFileNames = sorted(glob.glob(f'graphs/*.pickle'))
df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

for j in tqdm(range(len(df))):

    # generate filename
    name = 'G' + str(df['index'].iloc[j])
    file = f'graphs/{name}.pickle'

    # load graph from file
    G = pickle.load(open(file,'rb'))

    total_messages = []
    allDifferences = []

    # simulate 100 iterations
    for i in range(100):
        states = []
        G_init = init(G)

        M, meanStringDifference,states_trajectory = simulate(G_init)

        # save state diversity to DataFrame
        df_states = pd.DataFrame(states_trajectory)
        df_states.to_csv(f'results/states-{name}-run{i}.tsv',sep='\t',index=False)

        total_messages.append(M)
        allDifferences.append(list(meanStringDifference))

    # save convergence rate to DataFrame
    df_convergence = pd.DataFrame(
        {'nMessages': total_messages,
         'meanHammingDist': allDifferences}
    )
    df_convergence.to_csv(f'results/convergence-{name}.tsv',sep='\t',index=False)