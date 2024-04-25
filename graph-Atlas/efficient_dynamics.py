"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Vectorized hamming distance computation
- NOTE: Change random dynamics to efficient dynamics 
  (i.e. if agent is congruent with its neighborhood, do not sample for messaging)

Use functions adapted from dynamics_vectorized_Atlas.py

Function simulate editted by Jade Dubbeld 
25/04/2024
"""

from dynamics_vectorized_Atlas import init, message_update, hamming_vector, count_states
import numpy as np, pickle, random


def simulate(graph: str, alpha: float = 1.0, beta: float = 0.0, nbits: int = 3) -> int | list:
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

    # file path
    file = f'graphs/{graph}.pickle'

    # retrieve graph and number of nodes
    G = pickle.load(open(file,'rb'))
    n = G.number_of_nodes()

    # initialize graph
    states = init(n=n,nbits=nbits)

    # initials
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

    # update states diversity trajectory
    states_trajectory = count_states(states,states_trajectory)

    # compute Hamming distance and store value
    hammingDistance = hamming_vector(states,range(len(states)))
    meanHD = hammingDistance.mean()
    meanHammingDistance.append(meanHD)

    # list node IDs
    nodes = list(G.nodes())

    # initialize neighbor dictionary
    dictNeighbors = {key: [] for key in nodes}

    # find neighbors of all nodes and store in dictionary
    for node in nodes:
        neighbors = G.neighbors(node)
        
        for nb in neighbors:
            dictNeighbors[node].append(nb)

    # converge when all agents agree on state
    while (meanHD != 0.0):

        # initialize list to store candidate agent pairs
        selection = []

        # check congruency within direct neighborhood
        for (node, neighbors) in dictNeighbors.items():
            for nb in neighbors:
                if all([a == b for a, b in zip(states[node], states[nb])]) == False:
                    selection.append((node,nb))
        
        # only select from NOT congruent pair
        randomPair = random.choice(selection)
        source = randomPair[0]
        destination = randomPair[1]

        # message from source to destination agent
        states = message_update(states, source, destination, alpha=alpha, beta=beta)

        # update states diversity trajectory
        states_trajectory = count_states(states,states_trajectory)

        M += 1

        # re-calculate normalized hamming distance for all pair combinations for node update
        hammingDistance = hamming_vector(states, destination)
        meanHD = hammingDistance.mean()
        meanHammingDistance.append(meanHD)

    return M, meanHammingDistance, states_trajectory, graph