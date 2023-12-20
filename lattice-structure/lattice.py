"""
Script for preliminary network of message passing until consensus in a lattice structured graph

Written by Jade Dubbeld
20/12/2023
"""

# import packages
import numpy as np, matplotlib.pyplot as plt
import random
import networkx as nx

def init_lattice(dimensions=(5,5)):
    """
    Build a lattice structured network with given dimensions.
    Initialize each node with random 3-bit strings.

    Parameters:
    - dimensions: dimensions of lattice

    Returns:
    - G: networkx graph
    """

    # set up bounded lattice with given dimensions
    G = nx.grid_graph(dim=dimensions)

    # opt for rounded edges (row-wise and column-wise respectively)
    for i in range(dimensions[1]):
        G.add_edge((0,i),(dimensions[0]-1,i))   

    for j in range(dimensions[0]):
        G.add_edge((j,0),(j,dimensions[1]-1))    

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


def message(G, source, destination, alpha=1.0, beta=0.0):
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - source
    - destination
    - alpha
    - beta

    """

    match = []

    # find correct bits in source node's state compared to destination node's state
    for position, bit in enumerate(G.nodes[source]['state']):
        if bit == G.nodes[destination]['state'][position]:
            match.append(position)

    mismatch = [i for i in [0,1,2] if i not in match]

    # generate random float representing matching/sender bias
    P_Matching = random.random()

    # random pick if no bits in common
    # pick from correct list with probability alpha (incorrect with probability 1 - alpha)
    if mismatch == []:
        index = random.choice([0,1,2])
    elif P_Matching > alpha:
        index = random.choice(match)
    elif P_Matching <= alpha:
        index = random.choice(mismatch)

    # generate message
    message = G.nodes[source]['state'][index]

    print(f"index = {index}, message = {message}")

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == '0':
            message = '1'
        elif message == '1':
            message = '0'

    # get current state of selected downstream neighbor
    current_state = G.nodes[destination]['state']

    # copy received bit at given position (redundant if bit is already agreed)
    new_state = current_state[:index] + message + current_state[index + 1:]
    G.nodes[destination]['state'] = new_state

    return G


def hamming_distance(string1, string2):
    """
    Function to compute string similarity using Hamming distance.

    Parameters:
    - string1: First string in comparison
    - string2: Second string in comparison

    Returns:
    - distance: number differing characters between string1 and string2
    """

    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance


def simulate(G, M, alpha=1.0, beta=0.0):

    attributes = nx.get_node_attributes(G, "state")

    # converge when all nodes agree on state
    while np.unique(list(attributes.values())).size > 1:
        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))
        print(f"{source} -> {destination}")

        G = message(G=G,source=source,destination=destination,alpha=alpha,beta=beta)

        print(G.nodes(data=True))

        M += 1
        attributes = nx.get_node_attributes(G, "state")   

    return G, M