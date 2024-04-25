"""
Script to analyze effect of graph transitions by edge removal on consensus formation.

Written by Jade Dubbeld
12/04/2024
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, networkx as nx, pickle
from tqdm import tqdm
from itertools import product


def hellinger(p: np.ndarray, q: np.ndarray) -> np.float64:
    """
    Function to compute Hellinger distance between two probability distributions.

    Parameters:
    - p (np.ndarray): First probability distribution in comparison
    - q (np.ndarray): Second probability distribution in comparison

    Returns:
    - (np.float64): Computed Hellinger distance
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def pairwise_comparison(alpha: str, beta: str, from_graph: list, to_graph: list, n: int, plots: bool = False):
    """
    Function to pairwise compare a graph transition to compute Hellinger distance (and save computations).
    - Optional to generate corresponding distribution plots for visual comparison.
    
    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - from_graph (list): List of graph IDs BEFORE transition
    - to_graph (list): List of graph IDs AFTER transition
    - n (int): Graph size
    - plots (bool): True to indicate to generate plots; False if not
    """

    # set paths
    settings = f'alpha{alpha}-beta{beta}'      

    # load graph data    
    data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')

    # initialize DataFrame to store Hellinger distance data
    data_hellinger = pd.DataFrame(data=None,index=range(len(from_graph)),columns=["index_graph1","index_graph2","GE_graph1","GE_graph2","Hellinger"])
    
    for i,(graph1,graph2) in tqdm(enumerate(zip(from_graph,to_graph))):

        # get corresponding convergence distribution for respective graphs
        data1 = data[data['index']==graph1]
        data2 = data[data['index']==graph2]

        # setup comparison data
        compare = data1['nMessages'].to_frame().rename(columns={"nMessages": f"G{graph1}"})
        compare[f"G{graph2}"] = list(data2['nMessages'])

        fig, (ax0,ax1) = plt.subplots(1,2,figsize=(13,8))
        x, bins, _ = ax0.hist(x=compare,bins=10,label=[f"G{graph1}: GE={round(data1['globalEff'].unique()[0],4)}",f"G{graph2}: GE={round(data2['globalEff'].unique()[0],4)}"])
        bin_centers = 0.5*(bins[1:]+bins[:-1])

        # compute probabilities
        p_graph1 = x[0]/100
        p_graph2 = x[1]/100

        # compute Hellinger distance
        dist_hellinger = hellinger(p_graph1,p_graph2)

        # construct Hellinger data
        data_hellinger.iloc[i] = pd.Series(data={"index_graph1": int(graph1),
                                  "index_graph2": int(graph2),
                                  "GE_graph1": data1['globalEff'].unique()[0],
                                  "GE_graph2": data2['globalEff'].unique()[0],
                                  "Hellinger": dist_hellinger})

        # plot figure, if indicated
        if plots:
            ax0.legend(fontsize=16)
            ax0.set_xlabel("Number of messages",fontsize=16)
            ax0.set_ylabel("Frequency",fontsize=16)
            ax0.set_title(f"Hellinger={round(dist_hellinger,3)}; n_agents={n}; n_repeats=100",fontsize=16)
            ax0.tick_params(axis="both",which="major",labelsize=16)

            ax1.plot(bin_centers,x[0],label=f"G{graph1}: GE={round(data1['globalEff'].unique()[0],4)}")
            ax1.plot(bin_centers,x[1],label=f"G{graph2}: GE={round(data2['globalEff'].unique()[0],4)}")
            ax1.legend(fontsize=16)
            ax1.set_xlabel("Number of messages",fontsize=16)
            ax1.set_ylabel("Frequency",fontsize=16)
            ax1.set_title(f"Hellinger={round(dist_hellinger,3)}; n_agents={n}; n_repeats=100",fontsize=16)
            ax1.tick_params(axis="both",which="major",labelsize=16)

            # plt.show()
            fig.savefig(f"images/transitions/G{graph1}-G{graph2}.png",bbox_inches='tight')

        plt.close(fig)

    # add column GE_difference
    data_hellinger["GE_difference"] = data_hellinger["GE_graph1"] - data_hellinger["GE_graph2"]

    # save Hellinger distance data
    data_hellinger.to_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t',index=False)


def distribution_hellinger(alpha: str, beta: str, n: int):
    """
    Function to visualize Hellinger distance distribution for given parameter settings.

    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - n (int): Graph size
    """

    # load data
    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t')

    # plot and save histogram distribution of Hellinger distances
    fig,ax = plt.subplots()
    ax.hist(data_hellinger["Hellinger"],bins=10)
    ax.set_xlabel("Hellinger distance", fontsize=16)
    ax.set_ylabel("Frequency",fontsize=16)
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=16)
    
    fig.savefig(f"images/transitions/HellingerDistribution-alpha={alpha}-beta={beta}-n={n}.png",bbox_inches='tight')
    plt.close(fig)


def correlate_hellinger_globalEff(alpha: str, beta: str, n: int):
    """
    Function to show relation between operational efficiency (Hellinger distance) and structural efficiency
    (global efficiency) by comparing single edge transitions between graphs.
    
    i.e. graph1 can transition into graph2 by only removing a single edge.

    Parameters:
    - alpha (float): Setting value of alpha noise 
    - beta (float): Setting value of beta noise
    - n (int): Graph size
    
    Returns:
    - None
    """

    # load data
    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t')

    # plot Hellinger distance against global efficiency difference between each transition pair
    fig, ax = plt.subplots()
    ax.scatter(x=data_hellinger["GE_difference"],y=data_hellinger["Hellinger"])
    ax.set_xlabel("Difference in global efficiency",fontsize=16)
    ax.set_ylabel("Hellinger distance",fontsize=16)
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=16)
    plt.show()

    # fig.savefig(f"images/transitions/GE-Hellinger-correlation-alpha={alpha}-beta={beta}-n={n}.png",bbox_inches='tight')
    plt.close(fig)


def investigate_intervention(alpha: str, beta: str, n: int):
    """
    Function to investigate difference in efficiency, both structural and operational, 
    after single edge transition (i.e., intervention) per starting graph

    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - n (int): Graph size
    """

    # load data
    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t')
    fig, ax = plt.subplots()

    startGraphs = data_hellinger['index_graph1'].unique()[100:105]

    for s in startGraphs:
        subset = data_hellinger[data_hellinger['index_graph1']==s]
        print(subset)

        # TODO

    plt.close(fig)


def showSideBySide(graphIDs: list):
    """
    Function to visualize two graphs side-by-side.

    Parameters:
    - graphIDs (list): Contains graph IDs to visualize/compare
    """

    fig, axs = plt.subplots(nrows=1,ncols=len(graphIDs),figsize=(20,5))

    for index,id in enumerate(graphIDs):
        G = nx.graph_atlas(id)
        nx.draw(G=G,pos=nx.kamada_kawai_layout(G),ax=axs[index],node_size=50)
        axs[index].set_title(f"G{id}")

    plt.show()
    fig.savefig("images/test-figure.png",bbox_inches='tight')
    plt.close(fig)


def possible_transitions(n: int) -> None:
    """
    Function to check possible transitions between graphs, given parameter settings.

    Parameters:
    - n (int): Graph size
    """
    
    # load raw data
    df = pd.read_csv('data/data-GraphAtlas.tsv',usecols=['index','nodes'],sep='\t')

    # select data given graph size
    df = df[df['nodes']==n]
    df = df.reindex(index=df.index[::-1])

    # initials
    from_graph = []
    to_graph = []

    # compare all graphs within the same graph size class
    for i,j in tqdm(product(df['index'],df['index'])):
        if j > i:
            continue

        # load graphs
        graph1 = 'G' + str(i)
        file1 = f'graphs/{graph1}.pickle'
        G1 = pickle.load(open(file1,'rb'))

        graph2 = 'G' + str(j)
        file2 = f'graphs/{graph2}.pickle'
        G2 = pickle.load(open(file2,'rb'))

        # only account for single edge transitions
        if nx.graph_edit_distance(G1,G2) == 1.0:
            from_graph.append(i)
            to_graph.append(j)
    
    # save transitions in order
    np.savetxt(f"data/from_graph_n={n}.tsv",
        from_graph,
        delimiter ="\t",
        fmt ='% i')

    np.savetxt(f"data/to_graph_n={n}.tsv",
        to_graph,
        delimiter ="\t",
        fmt ='% i')
    

def analyze_graphPairs(alpha: str, beta: str, n: int):
    """
    Function to obtain/show top 10 highest and top 10 lowest for both Hellinger distance and global efficiency.
    
    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - n (int): Graph size
    """

    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t')

    # find top 10 graph pairs with highest Hellinger distance
    sortedHellinger = data_hellinger.sort_values(by=['Hellinger'],axis=0,ascending=False)
    print("Top 10 HIGHEST Helliger distance")
    print(sortedHellinger.head(10))
    print()

    # find top 10 graph pairs with highest global efficiency difference
    sortedGE = data_hellinger.sort_values(by=['GE_difference'],axis=0,ascending=False)
    print("Top 10 HIGHEST global efficiency difference")
    print(sortedGE.head(10))
    print()

    # find top 10 graph pairs with highest Hellinger distance
    sortedHellinger = data_hellinger.sort_values(by=['Hellinger'],axis=0,ascending=False)
    print("Top 10 LOWEST Helliger distance")
    print(sortedHellinger.tail(10))

    # find top 10 graph pairs with highest global efficiency difference
    sortedGE = data_hellinger.sort_values(by=['GE_difference'],axis=0,ascending=False)
    print("Top 10 LOWEST global efficiency difference")
    print(sortedGE.tail(10))

if __name__ == "__main__":


    # NOTE: CHOOSE DESIRED SETTINGS
    alpha = "1_00"
    beta = "0_00"

    alphas = ['1_00','0_75','0_50']
    betas = ['0_00','0_25', '0_50']    

    n = 7
    ###############################

    # # check possible single edge transitions within graph size
    # possible_transitions(n=n)

    ### NOTE: CHOOSE GRAPHS TO COMPARE IN TRANSITIONS ###
    # automatically checked using networkx module graph_edit_distance
    from_graph = list(map(int, np.loadtxt(f"data/from_graph_n={n}.tsv",delimiter='\t')))
    to_graph = list(map(int, np.loadtxt(f"data/to_graph_n={n}.tsv",delimiter='\t')))

    # # manual selection
    # from_graph = [556,714,572,850,348,340,334,433,443,348]
    # to_graph = [394,497,432,609,279,286,280,349,349,286]
    #################################

    # for alpha, beta in product(alphas,betas): 
    #     if beta == '0_50' and alpha == '0_50':
    #         continue        
        
    #     print(f'alpha={alpha} & beta={beta}')

    #     # NOTE
    #     # NOTE: CHOOSE FUNCTION TO RUN
    #     # NOTE

    #     # run comparison and generate Hellinger distance data
    #     pairwise_comparison(alpha=alpha,
    #                         beta=beta,
    #                         from_graph=from_graph,
    #                         to_graph=to_graph,
    #                         n=n,
    #                         plots=False)

    # # plot histogram distribution of Hellinger distance
    # distribution_hellinger(alpha,
    #                        beta,
    #                        n=n)

    # # scatterplot relation between difference in global efficiency and difference in Helling distance
    # correlate_hellinger_globalEff(alpha=alpha,
    #                               beta=beta,
    #                               n=n)

    # # investigate graph pairs
    # analyze_graphPairs(alpha=alpha, 
    #                    beta=beta, 
    #                    n=n)

    # investigate intervention effectiveness
    investigate_intervention(alpha=alpha,
                             beta=beta,
                             n=n) 
    
    # showSideBySide(graphID1=286,graphID2=433)
    showSideBySide(graphIDs=[340,348,351,353,337,338,336,350,349,286])
