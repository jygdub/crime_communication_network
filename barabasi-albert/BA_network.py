"""
Script for preliminary network of message passing until consensus in randomly generated Barabasi-Albert networks
    Simulation for given number of iterations and alpha and beta noise parameter settings

Written by Jade Dubbeld
16/01/2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np

def generate(n,m):
    """
    Function to generate a random Barabasi-Albert network (a.k.a. scale-free network).

    Parameters:
    - n: number of nodes in network
    - m: number of edges to add between new node and existing nodes in network

    Returns:
    - G: Randomly generated Barabasi-Albert network with given parameters.
    """

    G = nx.barabasi_albert_graph(n,m)

    return G

def init_BA(G):
    """
    Function to initialize a Barabasi-Albert network with initial 3-bit string states.

    Parameters:
    - G: Randomly generated Barabasi-Albert network.

    Returns:
    - G: Randomly generated Barabasi-Albert network with initial states.

    """

    # initialize all nodes with 3-bit string state
    for node in G.nodes:
        binary_string = f'{random.getrandbits(3):=03b}'     # generate random 3-bit string
        G.nodes[node]['state'] = binary_string

    return G

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


def message(G, source, destination, alpha=1.0, beta=0.0):
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - G: current state of networkx graph
    - source: selected sender node in network
    - destination: selected receiver node in network
    - alpha: probability of sender bias (sending match or mismatch bits)
    - beta: probability of receiver bias (flipping message or not)

    Returns:
    - G: new state of networkx graph

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
    if P_Matching > alpha and match != []:
        index = random.choice(match)
    elif P_Matching <= alpha and mismatch != []:
        index = random.choice(mismatch)
    else:
        index = random.choice([0,1,2])

    # generate message
    message = G.nodes[source]['state'][index]

    # print(f"index = {index}, message = {message}")

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


def simulate(n, m, alpha=1.0, beta=0.0, n_iters=10):
    """
    Function to run a simulation for n_iters on a lattice.

    Parameters:
    - n: number of nodes in network
    - m: number of edges to add between new node and existing nodes in network
    - alpha: probability of sender bias (sending match or mismatch bits)
    - beta: probability of receiver bias (flipping message or not)
    - n_iters: number of iterations to simulate for

    Returns:
    - total_messages: list of total messages sent in all simulations
    - all_diffScores: list of all string difference scores in all simulations
    """

    similarity = []

    G_init = init_BA(n,m)
    G = G_init
    M = 0
    attributes = nx.get_node_attributes(G, "state")

    # converge when all nodes agree on state
    while (np.unique(list(attributes.values())).size > 1 and M < 100000):
        
        S = []

        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))
        # print(f"{source} -> {destination}")

        G = message(G=G,source=source,destination=destination,alpha=alpha,beta=beta)

        # print(G.nodes(data=True))

        M += 1
        attributes = nx.get_node_attributes(G, "state")   

        # string similarity using Hamming distance    
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 == node2:
                    continue

                distance = hamming_distance(attributes[node1],attributes[node2])
                S.append(distance)

        similarity.append(np.mean(S))
    
    return M, similarity, G_init