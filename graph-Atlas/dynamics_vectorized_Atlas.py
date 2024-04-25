"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Vectorized hamming distance computation

Written by Jade Dubbeld (with contribution of Casper van Elteren)
13/03/2024
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

def count_states(states: np.ndarray, states_trajectory: dict) -> dict:

    counts = {'000': 0,
              '001': 0,
              '010': 0,
              '011': 0,
              '100': 0,
              '101': 0,
              '110': 0,
              '111': 0}

    for s in states:
        if (s==np.array([0,0,0])).all():
            counts['000'] += 1
        elif (s==np.array([0,0,1])).all():
            counts['001'] += 1
        elif (s==np.array([0,1,0])).all():
            counts['010'] += 1
        elif (s==np.array([0,1,1])).all():
            counts['011'] += 1
        elif (s==np.array([1,0,0])).all():
            counts['100'] += 1
        elif (s==np.array([1,0,1])).all():
            counts['101'] += 1
        elif (s==np.array([1,1,0])).all():
            counts['110'] += 1
        elif (s==np.array([1,1,1])).all():
            counts['111'] += 1

    for state in states_trajectory.keys():

        if counts[state] != 0:
            states_trajectory[state].append(counts[state])
        else:
            states_trajectory[state].append(0)
 
    return states_trajectory


def simulate(graph: str, alpha: float = 1.0, beta: float = 0.0, nbits: int =3) -> int | list:
    """
    Function to run simulation of communication dynamics on network.

    Parameters:
    - G (nx.classes.graph.Graph): generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)
    - nbits (int): length of agent's state

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    file = f'graphs/{graph}.pickle'
    G = pickle.load(open(file,'rb'))

    states = init(n=G.number_of_nodes(),nbits=nbits)

    meanHammingDistance = []
    M = 0

    states_trajectory = {'000':[],
                        '001':[],
                        '010':[],
                        '011':[],
                        '100':[],
                        '101':[],
                        '110':[],
                        '111':[]}

    states_trajectory = count_states(states,states_trajectory)

    hammingDistance = hamming_vector(states,range(len(states)))
    meanHD = hammingDistance.mean()
    meanHammingDistance.append(meanHD)

    # print(M, meanHammingDistance[-1])

    nodes = list(G.nodes())

    # initialize neighbor dictionary
    dictNeighbors = {key: [] for key in nodes}

    # find neighbors of all nodes and store in dictionary
    for node in nodes:
        neighbors = G.neighbors(node)
        
        for nb in neighbors:
            dictNeighbors[node].append(nb)

    # converge when all nodes agree on state
    while (meanHD != 0.0):
        source = np.random.choice(nodes)
        destination = np.random.choice(dictNeighbors[source])

        states = message_update(states, source, destination, alpha=alpha, beta=beta)
        states_trajectory = count_states(states,states_trajectory)

        M += 1

        # if M % 1000 == 0:
        #     print(M, meanHammingDistance[-1])

        # re-calculate normalized hamming distance for all pair combinations for node update
        hammingDistance = hamming_vector(states, destination)
        meanHD = hammingDistance.mean()
        meanHammingDistance.append(meanHD)

    return M, meanHammingDistance, states_trajectory, graph
