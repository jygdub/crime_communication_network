"""
Script for preliminary network of message passing until consensus in a lattice structured graph

Written by Jade Dubbeld
20/12/2023
"""

# import packages
import numpy as np, matplotlib.pyplot as plt
import random
import networkx as nx

def init_lattice(dim=(5,5)):
    """
    Build a lattice structured network with given dimensions.
    Initialize each node with random 3-bit strings.

    Parameters:
    - dim: dimensions of lattice

    Returns:
    - G: networkx graph
    """

    # set up bounded lattice with given dimensions
    G = nx.grid_graph(dim=dim)

    # opt for rounded edges
    for i in range(dim[1]):
        G.add_edge((0,i),(dim[0]-1,i))   

    for j in range(dim[0]):
        G.add_edge((j,0),(j,dim[1]-1))    

    # initialize all nodes with 3-bit string state
    for node in G.nodes:
        binary_string = f'{random.getrandbits(3):=03b}'     # generate random 3-bit string
        G.nodes[node]['state'] = binary_string

    # show initial configuration
    print(G.nodes(data=True))
    return G


def visualize(G):
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G: networkx graph
    """
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))
    plt.show()

G = init_lattice()
visualize(G)