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

# def showSideBySide(graphID1: int, graphID2: int):
def showSideBySide(graphIDs: list):
    """
    Function to visualize two graphs side-by-side.

    Parameters:
    - graphIDs (list): Contains graph IDs to visualize/compare
    """

    fig, axs = plt.subplots(nrows=1,ncols=len(graphIDs),figsize=(20,5))

    for index,id in enumerate(graphIDs):
        G = nx.graph_atlas(id)
        nx.draw(G=G,pos=nx.kamada_kawai_layout(G),ax=axs[index],node_size=50)
        axs[index].set_title(f"G{id}")

    plt.show()
    fig.savefig("images/test-figure.png",bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":

    # load data
    df = pd.read_csv('data/data-GraphAtlas.tsv',usecols=['index'],sep='\t')

    # # save graph image
    # saveGraph(data=df['index'])

    # showSideBySide(graphID1=286,graphID2=433)
    showSideBySide(graphIDs=[340,348,351,353,337,338,336,350,349,286])
