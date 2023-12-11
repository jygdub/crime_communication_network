import random
import matplotlib.pyplot as plt
import networkx as nx

def init_binary(depth=3):
    """
    Structure binary tree with given depth/height.
    Initialize each node with random 3-bit strings.
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
    """
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))
    plt.show()

if __name__ == '__main__':
    # initialize binary tree
    G = init_binary(depth=3)

    # visualize initial state of tree
    visualize(G)