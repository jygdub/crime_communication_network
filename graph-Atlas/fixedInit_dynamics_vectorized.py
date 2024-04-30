"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Vectorized hamming distance computation

Specific use for fixed state initialization for graph size n=7.

Adapted from dynamics_vectorized_Atlas.py
Adjusted by Jade Dubbeld
30/04/2024
"""

import numpy as np, pickle


def fixedInit(n: int) -> np.ndarray:
    """
    Function to initialize agents in a network with fixed initial bit strings.
    - For graph size n=7

    Parameters:
    - n (int): number agents in network

    Returns:
    - _ (np.ndarray): array containing states of all agents
    """

    states = None
    
    if n == 7:
        states = np.array(([0,0,0],[1,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0]))
    elif n == 6:
        states = np.array(([1,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0]))
    elif n == 5:
        states = np.array(([1,0,1],[0,0,1],[0,1,0],[1,0,0],[0,1,1]))
    elif n == 4:
        states = np.array(([1,0,1],[0,0,1],[0,1,0],[1,0,0]))
    elif n == 3:
        states = np.array(([1,0,1],[0,1,0],[1,0,0]))

    return states

def message_update(states: np.ndarray, source: int, destination: int, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - states (np.ndarray): states of all nodes in network
    - source (int): selected sender node in network
    - destination (int): selected receiver node in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - states (np.ndarray): updated states of all nodes in network
    """

    match = []

    # find correct bits in source node's state compared to destination node's state
    for position, bit in enumerate(states[source]):
        if bit == states[destination][position]:
            match.append(position)

    mismatch = [i for i in [0,1,2] if i not in match]

    # generate random float representing matching/sender bias
    P_Matching = np.random.random()

    # random pick if no bits in common
    # pick from correct list with probability alpha (incorrect with probability 1 - alpha)
    if P_Matching > alpha and match != []:
        index = np.random.choice(match)
    elif P_Matching <= alpha and mismatch != []:
        index = np.random.choice(mismatch)
    else:
        index = np.random.choice([0,1,2])

    # generate message
    message = states[source][index]

    # generate random float representing miscommunication/receiver bias
    P_Miss = np.random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == 0:
            message = 1
        elif message == 1:
            message = 0

    # update downstream neighbors' state according to message (redundant if bit is already agreed)
    states[destination][index] = message

    return states

def hamming_vector(states: np.ndarray, agents: slice) -> np.ndarray:
    """
    Function to compute Hamming distance in vectorized fashion.

    Parameters:
    - states (np.ndarray): states of all nodes in network
    - agents (slice): slice of agents to compare in Hamming distance calculation

    Returns:
    - _ (np.ndarray): Average pairwise Hamming distances
    """

    return (states[agents,np.newaxis] != states[np.newaxis,:]).mean(-1)

def simulate(graph: str, alpha: float = 1.0, beta: float = 0.0) -> int | list:
    """
    Function to run simulation of communication dynamics on network.

    Parameters:
    - G (nx.classes.graph.Graph): generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    file = f'graphs/{graph}.pickle'
    G = pickle.load(open(file,'rb'))

    nodes = list(G.nodes())

    states = fixedInit(n=len(nodes))

    M = 0

    hammingDistance = hamming_vector(states,range(len(states)))

    # initialize neighbor dictionary
    dictNeighbors = {key: [] for key in nodes}

    # find neighbors of all nodes and store in dictionary
    for node in nodes:
        neighbors = G.neighbors(node)
        
        for nb in neighbors:
            dictNeighbors[node].append(nb)

    # converge when all nodes agree on state
    while (hammingDistance.mean() != 0.0):
        source = np.random.choice(nodes)
        destination = np.random.choice(dictNeighbors[source])

        states = message_update(states, source, destination, alpha=alpha, beta=beta)

        M += 1

        # re-calculate normalized hamming distance for all pair combinations for node update
        hammingDistance = hamming_vector(states, destination)

    return M, graph
