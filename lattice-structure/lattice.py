"""
Script for preliminary network of message passing until consensus in a lattice structured graph.

Using scalable noise parameters alpha and beta for sender bias and receiver bias, respectively.
- High alpha means higher probability of sending information the receiver does NOT know yet; 
    low alpha means higher probability of sending information the receiver ALREADY knows, prolonging the convergence.
- Low beta means lower probability of no receiving error and therefore no change in information;
    high beta means higher probability of interpreting message incorrectly and therefore change in information.

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
    - G: initialized networkx graph
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


def simulate(dimensions, alpha=1.0, beta=0.0, n_iters=10):
    """
    Function to run a simulation for n_iters on a lattice.

    Parameters:
    - dimensions: dimensions of lattice
    - alpha: probability of sender bias (sending match or mismatch bits)
    - beta: probability of receiver bias (flipping message or not)
    - n_iters: number of iterations to simulate for

    Returns:
    - total_messages: list of total messages sent in all simulations
    - all_diffScores: list of all string difference scores in all simulations
    """

    # initials
    total_messages = []
    all_diffScores = []

    # simulate for n_iters
    for iteration in range(n_iters):
        print(f"iter={iter}")
        similarity = []

        G = init_lattice(dimensions)
        M = 0
        attributes = nx.get_node_attributes(G, "state")

        # converge when all nodes agree on state
        while np.unique(list(attributes.values())).size > 1:
            
            S = []

            source = random.choice(list(G.nodes))
            destination = random.choice(list(G.neighbors(source)))
            print(f"{source} -> {destination}")

            G = message(G=G,source=source,destination=destination,alpha=alpha,beta=beta)

            print(G.nodes(data=True))

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

        total_messages.append(M)
        all_diffScores.append(list(similarity))
    
    return total_messages, all_diffScores