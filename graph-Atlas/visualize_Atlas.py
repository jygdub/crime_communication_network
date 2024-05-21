"""
Script to visualize networkx's Graph Atlas networks.

Written by Jade Dubbeld
19/03/2024
"""

import pandas as pd, matplotlib.pyplot as plt, networkx as nx, pickle, json
from tqdm import tqdm

def visualize(G: nx.classes.graph.Graph) -> None:
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G (nx.classes.graph.Graph): networkx graph to display

    Returns:
    - Figure of graph
    """

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    return nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))

def saveGraph(data: pd.DataFrame):
    """
    Function to save all graph images.

    Parameters:
    - data (pd.DataFrame): Subset of graph IDs to generate in figure.
    """

    for i in tqdm(data):
        graph = 'G' + str(i)
        file = f'graphs/{graph}.pickle'

        G = pickle.load(open(file,'rb'))
        
        fig = visualize(G)
        
        # plt.savefig(f'graphs/{graph}.png')
        plt.close(fig)


def graphsSideBySide(graphIDs: list):
    """
    Function to visualize two graphs side-by-side.

    Parameters:
    - graphIDs (list): Contains graph IDs to visualize/compare
    """

    # set initials
    fig, axs = plt.subplots(nrows=1,ncols=len(graphIDs),figsize=(20,5))
    color = 'tab:blue'

    # plot every provided graph
    for index,id in enumerate(graphIDs):

        # change color of starting graph
        if index == 0:
            color = 'tab:green'
        else:
            color = 'tab:blue'

        # generate graph
        G = nx.graph_atlas(id)

        # draw graph
        nx.draw(G=G,pos=nx.kamada_kawai_layout(G),ax=axs[index],node_size=50,node_color=color)
        axs[index].set_title(f"G{id}")

    plt.show()
    fig.savefig(f"images/transitions/transitions-startG{graphIDs[0]}.png",bbox_inches='tight')
    plt.close(fig)



def identicalLayout(G_layout: nx.classes.graph.Graph, graphIDs: list):

    fig, axs = plt.subplots(nrows=1,ncols=len(graphIDs),figsize=(13,8))

    pos = nx.kamada_kawai_layout(G_layout)

    # edgesLayout = G_layout.edges()
    # print(edgesLayout)

    # degreeLayout = {}
    
    # for node in G_layout.nodes():
    #     degreeLayout[node] = G_layout.degree(node)

    # print(degreeLayout)

    # print("---------------")

    for index,id in enumerate(graphIDs):

        G = nx.graph_atlas(id)

        # degreeDict = {}
        
        # for node in G.nodes():
        #     degreeDict[node] = G.degree(node)
        
        # print(degreeDict)


        # if id == 271:
        #     G = nx.relabel_nodes(G, {0: 2, 1: 1, 2: 0, 3: 3, 4: 4, 5: 5, 6: 6})

        # edgesG = G.edges()
        # print(edgesG)

        # if id == 271:
        #     G = nx.relabel_nodes(G, {0: 3, 1: 6, 2: 0, 3: 2, 4: 4, 5: 5, 6: 1})
        # if id == 270:
        #     G = nx.relabel_nodes(G, {0: 6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0})

        # if nx.is_isomorphic(G,G_layout):
        #     nx.draw(G=G_layout,pos=nx.kamada_kawai_layout(G_layout),ax=axs[index],node_color="tab:green",with_labels=True)
        # else:
        #     nx.draw(G=G,pos=pos,ax=axs[index],node_color="tab:blue",with_labels=True)


        # G = nx.relabel_nodes(G, {0: -, 1: -, 2: -, 3: -, 4: -, 5: -, 6: -})
        # G = nx.relabel_nodes(G, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6})
        if graphIDs[0] == 317 and id == 276:
            G = nx.relabel_nodes(G, {0: 2, 1: 4, 2: 0, 3: 5, 4: 1, 5: 3, 6: 6})
        elif graphIDs[0] == 324 and id == 276:
            G = nx.relabel_nodes(G, {0: 6, 1: 2, 2: 1, 3: 4, 4: 5, 5: 3, 6: 0})
        elif graphIDs[0] == 331 and id == 279:
            G = nx.relabel_nodes(G, {0: 5, 1: 0, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6})
        elif graphIDs[0] == 383 and id == 320:
            G = nx.relabel_nodes(G, {0: 0, 1: 2, 2: 5, 3: 6, 4: 3, 5: 4, 6: 1})
        elif graphIDs[0] == 416 and id == 416:
            G = nx.relabel_nodes(G, {0: 1, 1: 6, 2: 3, 3: 5, 4: 2, 5: 4, 6: 0})
        elif graphIDs[0] == 440 and id == 337:
            G = nx.relabel_nodes(G, {0: 0, 1: 4, 2: 2, 3: 3, 4: 6, 5: 5, 6: 1})
        elif graphIDs[0] == 486 and id == 411:
            G = nx.relabel_nodes(G, {0: 3, 1: 5, 2: 0, 3: 1, 4: 2, 5: 4, 6: 6})
        elif graphIDs[0] == 561 and id == 432:
            G = nx.relabel_nodes(G, {0: 0, 1: 4, 2: 3, 3: 5, 4: 1, 5: 2, 6: 6})
        elif graphIDs[0] == 620 and id == 489:
            G = nx.relabel_nodes(G, {0: 4, 1: 0, 2: 1, 3: 5, 4: 3, 5: 6, 6: 2})
        elif graphIDs[0] == 636 and id == 526:
            G = nx.relabel_nodes(G, {0: 0, 1: 2, 2: 5, 3: 1, 4: 3, 5: 6, 6: 4})
        elif graphIDs[0] == 639 and id == 525:
            G = nx.relabel_nodes(G, {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6})

        nx.draw(G=G,pos=pos,ax=axs[index],node_color="tab:blue")#,with_labels=True)
        axs[index].set_title(f"G{id}")

    plt.show()

    fig.savefig(fname=f"images/transitions/n=7/successfulTransitions/structuredLayout/G{graphIDs[0]}-G{graphIDs[1]}.png",bbox_inches='tight')

    plt.close(fig)

def compareSuccessTransitions(G_layout: nx.classes.graph.Graph, graphIDs: list):
    
    fig, axs = plt.subplots(nrows=1,ncols=len(graphIDs),figsize=(13,8))

    pos = nx.kamada_kawai_layout(G_layout)
    

if __name__ == "__main__":

    # # load data
    # df = pd.read_csv('data/data-GraphAtlas.tsv',usecols=['index'],sep='\t')

    alpha = '1_00'
    beta = '0_00'
    n = 7

    data = json.load(open(f"data/graphTransitions-PairedMaxima-alpha{alpha}-beta{beta}-n={n}.json"))

    # for ID in tqdm(list(data.keys())):
    ID = '639'
    transition = [int(ID)] + data[ID]
    identicalLayout(G_layout=nx.graph_atlas(int(ID) ), graphIDs=transition)

    # # save graph image
    # saveGraph(data=df['index'])

    # visualize graph transitions per starting point
    # graphsSideBySide(graphIDs=[1151,1098,1097,1079,1078,1068,1047,1031]) # starting graph G1151 @ index 100 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[1150,1091,1081,1078,1071,1046]) # starting graph G1150 @ index 101 in unique set of 'index_graph1'
    
    # graphsSideBySide(graphIDs=[516,434,425,421,410,400,398,394,383]) # starting graph G516 @ index 700 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[515,432,424,422,403,400,399,394]) # starting graph G515 @ index 701 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[514,429,419,403,389,382]) # starting graph G514 @ index 702 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[513,428,419,405,402,392,390]) # starting graph G513 @ index 703 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[512,421,401,390,389]) # starting graph G512 @ index 704 in unique set of 'index_graph1'
    
    # graphsSideBySide(graphIDs=[388,318,314]) # starting graph G388 @ index 800 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[386,329,325,321]) # starting graph G386 @ index 801 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[385,328,327,320]) # starting graph G385 @ index 802 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[384,329,324,322,317,316]) # starting graph G384 @ index 803 in unique set of 'index_graph1'
    # graphsSideBySide(graphIDs=[383,327,320,317]) # starting graph G383 @ index 804 in unique set of 'index_graph1'
