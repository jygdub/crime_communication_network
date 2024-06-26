"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Vectorized hamming distance computation

Specific use for parameter setting alpha=0.50 and beta=0.50

Adapted from dynamics_vectorized_Atlas.py
Adjusted by Jade Dubbeld
24/04/2024
"""

import numpy as np, pickle


def init(n: int, nbits: int) -> np.ndarray:
    """
    Function to initialize agents in a network with a random bit string.

    Parameters:
    - n (int): number of agents in network
    - nbits (int): length of bit string

    Returns:
    - _ (np.ndarray): array containing states of all agents
    """

    return np.random.randint(0, 2, size=(n, nbits))

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

    states = init(n=G.number_of_nodes(),nbits=3)

    M = 0

    hammingDistance = hamming_vector(states,range(len(states)))

    nodes = list(G.nodes())

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
