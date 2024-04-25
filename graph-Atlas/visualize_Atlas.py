"""
Script to visualize networkx's Graph Atlas networks.

Written by Jade Dubbeld
19/03/2024
"""

import pandas as pd, matplotlib.pyplot as plt, networkx as nx, pickle
from tqdm import tqdm

def visualize(G: nx.classes.graph.Graph) -> None:
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G (nx.classes.graph.Graph): networkx graph to display

    Returns:
    - Figure of graph
    """

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    return nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))

def saveGraph(data: pd.DataFrame):
    """
    Function to save all graph images.

    Parameters:
    - data (pd.DataFrame): Subset of graph IDs to generate in figure.
    """

    for i in tqdm(data):
        graph = 'G' + str(i)
        file = f'graphs/{graph}.pickle'

        G = pickle.load(open(file,'rb'))
        
        fig = visualize(G)
        
        # plt.savefig(f'graphs/{graph}.png')
        plt.close(fig)

if __name__ == "__main__":

    # load data
    df = pd.read_csv('data/data-GraphAtlas.tsv',usecols=['index'],sep='\t')

    # # save graph image
    # saveGraph(data=df['index'])
