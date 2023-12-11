"""
Script for preliminary network of message passing until consensus in a binary tree graph

Written by Jade Dubbeld
07/12/2023
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def init_binary(depth=3):
    """
    Structure binary tree with given depth/height.
    Initialize each node with random 3-bit strings.

    Parameters:
    - depth: depth (or height) of binary tree
    
    Returns:
    - G: networkx graph
    """

    # set up binary tree with given depth/height (h) [and branching factor (r) = 2]
    G = nx.balanced_tree(r=2,h=depth)

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

def simulate_forward(G,total_messages):
    """
    Function to forward root node's message throughout the entire network.
    Every downstream node copies root node's message into position.

    Parameters:
    - G: networkx graph
    - total_messages: counter for total messages send in network simulation

    Returns:
    - total_messages: counter for total messages send in network simulation
    """

    # pick random bit from state
    index = random.choice([0,1,2])
    print(index)

    message = G.nodes[0]['state'][index]
    
    # forward root node's message to all downstream neighbors
    for node in G.nodes:
        for neighbor in G.neighbors(node):

            # only propagate downstream
            if neighbor < node:
                continue

            # get current state of selected neighbor
            current_state = G.nodes[neighbor]['state']

            # copy received bit at given position (redundant if bit is already agreed)
            new_state = current_state[:index] + message + current_state[index + 1:]
            G.nodes[neighbor]['state'] = new_state
            total_messages += 1
    
    return total_messages

def simulate_random(G,total_messages):
    """
    Function to send messages containing a random bit through the entire network.
    Every node copies the bit from message into position.
    Then randomly sends a bit from its own state to its downstream neighbors, until leaf nodes are reached.

    Parameters:
    - G: networkx graph
    - total_messages: counter for total messages send in network simulation

    Returns:
    - total_messages: counter for total messages send in network simulation
    """
    
    # randomly generate node's message
    for node in G.nodes:

        # pick random bit from state
        index = random.choice([0,1,2])
        print(f"Bit-string index = {index} | Node = {node}")
        
        message = G.nodes[0]['state'][index]

        # send message to its downstream neigbors
        for neighbor in G.neighbors(node):
            if neighbor < node:
                continue

            # get current state of selected neighbor
            current_state = G.nodes[neighbor]['state']

            # copy received bit at given position (redundant if bit is already agreed)
            new_state = current_state[:index] + message + current_state[index + 1:]
            G.nodes[neighbor]['state'] = new_state
            total_messages += 1

        print(f"{G.nodes(data=True)}\n")
    
    return total_messages


if __name__ == '__main__':
    # initialize binary tree
    G = init_binary(depth=3)

    # visualize initial state of tree
    # visualize(G)

    # initial settings
    total_messages = 0
    attributes = nx.get_node_attributes(G, "state")

    # converge when all nodes agree on state
    while np.unique(list(attributes.values())).size > 1:

        # propagate root node's message through network
        # total_messages = simulate_forward(G=G, total_messages=total_messages)

        # simulate network
        total_messages = simulate_random(G=G, total_messages=total_messages)

        # update attributes dictionary
        attributes = nx.get_node_attributes(G, "state")

    print(f"Number of messages send until consensus = {total_messages}")