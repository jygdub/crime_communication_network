import networkx as nx, random, numpy as np, time
from collections import Counter

def init(G):
    """
    Function to initialize a network with initial 3-bit string states.
    Parameters:
    - G: Randomly generated network.
    Returns:
    - G: Randomly generated network with initial states.
    """
    # initialize all nodes with 3-bit string state
    for i, node in enumerate(G.nodes):
        binary_string = f'{random.getrandbits(3):=03b}'     # generate random 3-bit string
        # bits = bytes(binary_string, 'utf-8')                # convert to bytes
        G.nodes[node]['state'] = binary_string
    return G

def message_update(G, source: int, destination: int, attributes: dict, alpha: float = 1.0, beta: float = 0.0):
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - G: current state of networkx graph
    - source (int): selected sender node in network
    - destination (int): selected receiver node in network
    - attributes (dict): states of all nodes in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - G: new state of networkx graph
    - attributes (dict): states of all nodes in network

    """

    match = []

    # find correct bits in source node's state compared to destination node's state
    for position, bit in enumerate(attributes[source]):
        if bit == attributes[destination][position]:
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
    message = attributes[source][index:index+1]

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == '0':
            message = '1'
        elif message == '1':
            message = '0'

    # get current state of selected downstream neighbor
    current_state = attributes[destination]

    # copy received bit at given position (redundant if bit is already agreed)
    new_state = current_state[:index] + message + current_state[index + 1:]
    attributes[destination] = new_state

    return G, attributes


def hamming_distance(string1: str, string2: str) -> int:
    """
    Function to compute string difference using Hamming distance.
    Parameters:
    - string1 (str): First string in comparison
    - string2 (str): Second string in comparison
    Returns:
    - distance (int): number differing characters between string1 and string2
    """
    distance = 0
    L = len(string1)
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


def count_states(attributes: dict, states_trajectory: dict) -> dict:
    listAttributes = list(attributes.values())
    occurrences = Counter(listAttributes)

    for state in states_trajectory.keys():

        if occurrences[state]:
            states_trajectory[state].append(occurrences[state])
        else:
            states_trajectory[state].append(0)
    
    return states_trajectory

def simulate(G, alpha: float =1.0, beta: float = 0.0) -> int | list:
    """
    Function to run a simulation for n_iters on a lattice.
    Parameters:
    - G: generated networkx graph with initial states
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)
    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    N = len(G.nodes())
    nPairs = (((N*N)-N)/2)
    meanStringDifference = []
    stringDifference = np.zeros((N,N))
    M = 0
    states_trajectory = {'000':[],
                         '001':[],
                         '010':[],
                         '011':[],
                         '100':[],
                         '101':[],
                         '110':[],
                         '111':[]}
    
    attributes = nx.get_node_attributes(G, "state")
    states_trajectory = count_states(attributes,states_trajectory)

    # for each node pair (no redundant calculations)
    for index1, node1 in enumerate(G.nodes()):
        for index2, node2 in enumerate(G.nodes()):
            if node1 >= node2:
                continue

            # compute hamming distance for initial configuration
            hammingDistance = hamming_distance(attributes[node1],attributes[node2]) / 3 # regular

            # fill in normalized hamming distance array
            stringDifference[index1,index2] = hammingDistance
    meanStringDifference.append(stringDifference.sum()/nPairs)

    # converge when all nodes agree on state
    while (np.unique(list(attributes.values())).size > 1):
        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))

        # G = message(G=G,source=source,destination=destination,alpha=alpha,beta=beta)
        G,attributes = message_update(G=G,source=source,destination=destination,attributes=attributes,alpha=alpha,beta=beta)
        states_trajectory = count_states(attributes,states_trajectory)
        M += 1

        # re-calculate normalized hamming distance for all pair combinations for node update
        for node in G.nodes(): 
            if destination == node:
                continue

            hammingDistance = hamming_distance(attributes[destination],attributes[node]) / 3 # regular

            # fill in normalized hamming distance array
            if node < destination:
                stringDifference[node,destination] = hammingDistance
            elif node > destination:
                stringDifference[destination,node] = hammingDistance

        meanStringDifference.append(stringDifference.sum()/nPairs)

    return M, meanStringDifference, states_trajectory