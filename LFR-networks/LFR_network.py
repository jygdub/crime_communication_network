"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.

Written by Jade Dubbeld
17/02/2024
"""

import networkx as nx, random, numpy as np, pickle, time

def init(G: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
    """
    Function to initialize a network with initial 3-bit string states.

    Parameters:
    - G: Randomly generated network.

    Returns:
    - G: Randomly generated network with initial states.

    """

    # generate random integers for all agents in network
    randNums = np.random.randint(0, 8, size=(len(list(G)),1), dtype = np.uint8)

    # convert integers to array of bits
    bits = np.unpackbits(randNums, axis=1)

    # convert to dictionary to assign states in node attributes
    states  = dict(list(enumerate(bits)))
    nx.set_node_attributes(G, states,  'state')

    return G

def message(G: nx.classes.graph.Graph, source: int, destination: int, alpha: float = 1.0, beta: float = 0.0) -> nx.classes.graph.Graph:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - G: current state of networkx graph
    - source (int): selected sender node in network
    - destination (int): selected receiver node in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

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
    message = G.nodes[source]['state'][index:index+1]

    # print(f"index = {index}, message = {message}")

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == b'0':
            message = b'1'
        elif message == b'1':
            message = b'0'

    # get current state of selected downstream neighbor
    current_state = G.nodes[destination]['state']

    # copy received bit at given position (redundant if bit is already agreed)
    new_state = current_state[:index] + message + current_state[index + 1:]
    G.nodes[destination]['state'] = new_state

    return G


def message_update(source: np.ndarray, destination: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - source (numpy.ndarray): bit state of sender node
    - destination (numpy.ndarray): bit state of receiver node
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - destination (numpy.ndarray): updated state of receiver node

    """

    # compensate for 8-bit positions while only using last 3 bit positions
    onset = 5

    # compare source and destination states
    comparison = np.bitwise_xor(source[onset:],destination[onset:])
    
    # extract matching and mismatching position
    match = np.where(comparison==0)[0] + onset
    mismatch = np.where(comparison==1)[0] + onset

    # generate random float representing matching/sender bias
    P_Matching = random.random()

    # random pick if no bits in common
    # pick from correct list with probability alpha (incorrect with probability 1 - alpha)
    if P_Matching > alpha and not match.size == 0:
        index = random.choice(match)
    elif P_Matching <= alpha and not mismatch.size ==0:
        index = random.choice(mismatch)
    else:
        index = random.choice([5,6,7])

    # generate message
    message = source[index]

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == 0:
            message = 1
        elif message == 1:
            message = 0

    # copy received bit at given position (NOTE: redundant if bit is already agreed)
    destination[index] = message

    return destination


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


def hamming_distance_vector(arr: np.ndarray) -> np.ndarray:
    """
    Function to compute string difference using bitwise XOR (analogous to Hamming distance).

    Parameters:
    - arr (numpy.ndarray): array containing all agents' bit states

    Returns: 
    - hammDist (numpy.ndarray): array of pairwise hamming distances (integer values)
    """

    # convert array of bits to integers
    arr = np.packbits(arr,axis=1)

    # representation bit shifts instead of strings (using mask)
    # numpy broadcast np.bitwise_xor(array[None], array[:,None])

    # compute pairwise hamming distance and return neater shaped array
    hammDist = np.bitwise_xor(arr[np.newaxis,:], arr[:,np.newaxis])
    return np.reshape(hammDist, (hammDist.shape[0]**2,1))

def count_ones_vectorized(arr: np.ndarray) -> int:
    """
    Function to count number of differing positions across all agents in a network.

    Parameters:
    - arr (numpy.ndarray): array of pairwise hamming distances (integer values)

    Returns: 
    - ones_count (int): number of differing positions in the entire network
    """

    # convert the array to binary representation
    binary_arr = np.unpackbits(arr.view(np.uint8),axis=1)

    # count the number of ones in each binary representation
    ones_count = binary_arr.sum()
    return ones_count

def simulate(G: nx.classes.graph.Graph, alpha: float = 1.0, beta: float = 0.0) -> int | list:
    """
    Function to run a single simulation.

    Parameters:
    - G (networkx.classes.graph.Graph): generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    N = len(G.nodes())
    # nPairs = (N*N/2-(N/2))
    meanStringDifference = []
    M = 0
    
    attributes = nx.get_node_attributes(G, "state")
    attributes = np.array(list(attributes.values()))

    # compute hamming distance (vectorized)
    hammingDistance = hamming_distance_vector(attributes)
    meanHammDist = count_ones_vectorized(hammingDistance)/3/(N*N)
    # print(meanHammDist)
    meanStringDifference.append(meanHammDist)

    # converge when all nodes agree on state
    while ((attributes==attributes[0]).all()==False):
        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))

        attributes[destination] = message_update(source=attributes[source],destination=attributes[destination],alpha=alpha,beta=beta)

        M += 1

        if M % 2000 == 0:
            print(M, meanStringDifference[-1])

        # compute hamming distance (vectorized)
        hammingDistance = hamming_distance_vector(attributes)
        meanHammDist = count_ones_vectorized(hammingDistance)/3/(N*N)
        meanStringDifference.append(meanHammDist)
    
    return M, meanStringDifference

def simulate_parallel(G: nx.classes.graph.Graph, proc: int, return_dict: dict, alpha: float = 1.0, beta: float = 0.0) -> dict:
    """
    Function to run a single simulation in parallel.

    Parameters:
    - G: generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    print(f"Starting Process {proc}!")

    N = len(G.nodes())
    nPairs = (N*N/2-(N/2))
    meanStringDifference = []
    stringDifference = np.zeros((N,N))
    M = 0
    attributes = nx.get_node_attributes(G, "state")

    # for each node pair (no redundant calculations)
    for index1, node1 in enumerate(G.nodes()):
        for index2, node2 in enumerate(G.nodes()):
            if node1 >= node2:
                continue
            
            # compute hamming distance for initial configuration
            hammingDistance = hamming_distance(attributes[node1],attributes[node2]) / 3 # regular
            # hammingDistance = sum(hamming_distance_vector(attributes[node1],attributes[node2])) / 3 # vectorized

            # fill in normalized hamming distance array
            stringDifference[index1,index2] = hammingDistance

    meanStringDifference.append(stringDifference.sum()/nPairs)

    # converge when all nodes agree on state
    while (np.unique(list(attributes.values())).size > 1):

        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))

        G,attributes = message_update(G=G,source=source,destination=destination,attributes=attributes,alpha=alpha,beta=beta)

        M += 1
        """ Uncomment when using message() and remove attributes in line 237, not for message_update()"""
        # attributes = nx.get_node_attributes(G, "state") 

        if M % 2000 == 0:
            print(M, meanStringDifference[-1])

        # re-calculate normalized hamming distance for all pair combinations for node update
        for node in G.nodes(): 
            if destination == node:
                continue

            hammingDistance = hamming_distance(attributes[destination],attributes[node]) / 3 # regular
            # hammingDistance = sum(hamming_distance_vector(attributes[destination],attributes[node])) / 3 # vectorized

            # fill in normalized hamming distance array
            if node < destination:
                stringDifference[node,destination] = hammingDistance
            elif node > destination:
                stringDifference[destination,node] = hammingDistance

        meanStringDifference.append(stringDifference.sum()/nPairs)
    
    return_dict[proc] = [M, meanStringDifference]