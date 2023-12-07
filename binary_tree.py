import random
import matplotlib.pyplot as plt
import networkx as nx

# set up binary tree with given depth/height (h) and branching factor (r)
G = nx.balanced_tree(r=2,h=3)

# initialize all nodes with 3-bit string state
for node in G.nodes:
    binary_string = f'{random.getrandbits(3):=03b}'     # generate random 3-bit string
    G.nodes[node]['state'] = binary_string

print(G.nodes(data=True))

# visualize binary tree network
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))
plt.show()