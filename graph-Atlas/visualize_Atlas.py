"""
Script to visualize networkx's Graph Atlas networks.

Written by Jade Dubbeld
19/03/2024
"""

import pandas as pd, matplotlib.pyplot as plt, networkx as nx, pickle
from tqdm import tqdm

def visualize(G: nx.classes.graph.Graph):
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G (nx.classes.graph.Graph): networkx graph to display
    """

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    return nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))

df = pd.read_csv('data-GraphAtlas.tsv',usecols=['index'],sep='\t')

for i in tqdm(df['index']):
    graph = 'G' + str(i)
    file = f'graphs/{graph}.pickle'

    G = pickle.load(open(file,'rb'))
    
    fig = visualize(G)
    plt.savefig(f'graphs/{graph}.png')
    plt.close(fig)