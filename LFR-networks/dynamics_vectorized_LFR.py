"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Vectorized hamming distance computation

Written by Jade Dubbeld (with contribution of Casper van Elteren)
13/03/2024
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx, timeit, pickle
from tqdm import tqdm
from itertools import product
from typing import Tuple


def init(n: int, nbits: int) -> Tuple[np.ndarray, np.ndarray]:
    bits = np.random.randint(0, 2, size=(n, nbits))
    # string_bits = np.array(["".join(str(j) for j in i) for i in bits])
    return bits #, string_bits

def message_update(states: np.ndarray, source: int, destination: int, alpha: float = 1.0, beta: float = 0.0):
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
    - bits (np.ndarray): states of all nodes in network

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

def hamming_vector(states: np.ndarray) -> np.ndarray:
    return (states[:,np.newaxis] != states[np.newaxis,:]).sum(-1)


def simulate(G: nx.classes.graph.Graph, states: np.ndarray, nbits: int, alpha: float = 1.0, beta: float = 0.0) -> Tuple[int, list]:
    """
    Function to run a simulation for n_iters on a lattice.
    Parameters:
    - G (nx.classes.graph.Graph): generated networkx graph
    - states (np.ndarray): states of all nodes in network
    - nbits (int): length of bit string
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    meanHammingDistance = []
    M = 0

    hammingDistance = hamming_vector(states)
    print(hammingDistance.shape)
    print(np.mean(hammingDistance)/nbits)
    meanHammingDistance.append(np.mean(hammingDistance/nbits))

    print(M, meanHammingDistance[-1])

    nodes = list(G.nodes())

    # converge when all nodes agree on state
    while (meanHammingDistance[-1] != 0.0):
        source = np.random.choice(nodes)
        destination = np.random.choice(list(G.neighbors(source)))

        states = message_update(states, source, destination, alpha=alpha, beta=beta)

        M += 1

        if M % 100 == 0:
            print(M, meanHammingDistance[-1])

        # re-calculate normalized hamming distance for all pair combinations for node update
        hammingDistance = hamming_vector(states)
        meanHammingDistance.append(np.mean(hammingDistance/nbits))

    return M, meanHammingDistance

if __name__ == "__main__":
    filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
    G = pickle.load(open(filename, 'rb'))
    n = len(list(G))
    nbits = 3
    states = init(n, nbits)

    M, meanStringDifference = simulate(G, states, nbits, alpha = 1.0, beta = 0.0)
    print(M)