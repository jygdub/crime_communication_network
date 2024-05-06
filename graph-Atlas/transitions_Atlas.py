"""
Script to analyze effect of graph transitions by edge removal on consensus formation.

Written by Jade Dubbeld
12/04/2024
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, networkx as nx, pickle, json
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


def pairwise_comparison(alpha: str, beta: str, from_graph: list, to_graph: list, n: int, plots: bool = False, efficient: bool = False, saveData: bool = False):
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
    - efficient (bool): False for random dynamics; True for efficient dynamics
    - saveData (bool): False if not writing stored data to file; True if writing stored data to file
    """

    # set paths
    settings = f"alpha{alpha}-beta{beta}"  

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}" 

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
    if saveData:
        data_hellinger.to_csv(f"data/Hellinger-data-{settings}-n={n}.tsv",sep='\t',index=False)


def distribution_hellinger(alpha: str, beta: str, n: int, efficient: bool = False):
    """
    Function to visualize Hellinger distance distribution for given parameter settings.

    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - n (int): Graph size
    - efficient (bool): False for random dynamics; True for efficient dynamics
    """

    # set paths
    settings = f"alpha{alpha}-beta{beta}"  

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}" 

    # load data
    data_hellinger = pd.read_csv(f"data/Hellinger-data-{settings}-n={n}.tsv",sep='\t')

    # plot and save histogram distribution of Hellinger distances
    fig,ax = plt.subplots()
    ax.hist(data_hellinger["Hellinger"],bins=10)
    ax.set_xlabel("Hellinger distance", fontsize=16)
    ax.set_ylabel("Frequency",fontsize=16)
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=16)
    
    fig.savefig(f"images/transitions/n={n}/HellingerDistribution-{settings}-n={n}.png",bbox_inches='tight')

    plt.close(fig)


def correlate_hellinger_globalEff(alpha: str, beta: str, n: int, startGraph: bool = False, efficient: bool = False):
    """
    Function to show relation between operational efficiency (Hellinger distance) and structural efficiency
    (global efficiency) by comparing single edge transitions between graphs.
    
    i.e. graph1 can transition into graph2 by only removing a single edge.

    Parameters:
    - alpha (float): Setting value of alpha noise 
    - beta (float): Setting value of beta noise
    - n (int): Graph size
    - startGraph (bool): Including all transitions in analysis (False) or 
                         only transitions from specific starting graph (True) 
    - efficient (bool): False for random dynamics; True for efficient dynamics
    
    Returns:
    - None
    """

    # set paths
    settings = f"alpha{alpha}-beta{beta}"  

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}" 

    # load Hellinger data for given graph size
    data_hellinger = pd.read_csv(f"data/Hellinger-data-{settings}-n={n}.tsv",sep='\t')

    if not startGraph:
        # plot Hellinger distance against global efficiency difference between each transition pair
        fig, ax = plt.subplots()
        ax.scatter(x=data_hellinger["GE_difference"],y=data_hellinger["Hellinger"])
        ax.set_xlabel("Difference in global efficiency",fontsize=16)
        ax.set_ylabel("Hellinger distance",fontsize=16)
        ax.tick_params(axis="both",which="major",labelsize=16)
        ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=16)
        plt.show()

        fig.savefig(f"images/transitions/n={n}/GE-Hellinger-correlation-{settings}-n={n}.png",bbox_inches='tight')
        plt.close(fig)

    elif startGraph:

        startGraphs = data_hellinger['index_graph1'].unique()[800:805] # NOTE: POSSIBLE TO ADJUST RANGE

        for s in startGraphs:
            print(s)

            subset = data_hellinger[data_hellinger['index_graph1']==s]
            l = list(map(int,subset['index_graph2']))
            x = list(subset['GE_difference'])
            y = list(subset['Hellinger'])
            print(subset)

            fig, ax = plt.subplots()
            ax.scatter(x=x,y=y)
            for i, txt in enumerate(l):
                ax.annotate(txt, (x[i], y[i]))
            ax.set_xlabel("GE difference")
            ax.set_ylabel("Hellinger distance")
            ax.set_title(f"G{int(s)}->G{l}")
            plt.show()
            
            # fig.savefig(f"images/transitions/startG{int(s)}-GE-Hellinger-correlation-{settings}-n={n}.png",bbox_inches='tight')
            plt.close(fig)


def addlabels(x: list, y: list):
    """
    Function to add value labels.

    Parameters:
    - x (list): x-values
    - y (list): y-values
    """

    for i in range(len(x)):
        plt.text(i, y[i]+5, y[i], ha = 'center', fontsize=14)

def successTransitions(alpha: str, beta: str, n: int, efficient: bool = False):
    """
    Function to investigate difference in efficiency, both structural and operational, 
    after single edge transition (i.e., intervention) per starting graph

    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - n (int): Graph size
    - efficient (bool): False for random dynamics; True for efficient dynamics
    """

    # set paths
    settings = f"alpha{alpha}-beta{beta}"  

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}" 

    # load data
    data_hellinger = pd.read_csv(f"data/Hellinger-data-{settings}-n={n}.tsv",sep='\t')
    
    # find all start graphs
    startGraphs = data_hellinger['index_graph1'].unique()
    bothMaximum = {}
    maxProbs = {}

    print(len(startGraphs))

    for s in startGraphs:

        subset = data_hellinger[data_hellinger['index_graph1']==s]
        # print(subset)

        # find maximum values for Hellinger distance and global efficiency difference
        maxHellinger = max(subset['Hellinger'])
        maxGE = max(subset['GE_difference'])

        for index in subset.index:

            # get graph IDs from transition pair
            graphID1 = int(subset['index_graph1'][index])
            graphID2 = int(subset['index_graph2'][index])
            
            # append final graph ID to dictionary if both Hellinger distance and global efficiency difference is maximum
            if subset['Hellinger'][index] == maxHellinger and subset['GE_difference'][index] == maxGE:
                if graphID1 in bothMaximum:
                    bothMaximum.append(graphID2)
                else:
                    bothMaximum[graphID1] = [graphID2]

        # compute probabilities of finding maximum values for both measures per start graph
        if graphID1 in bothMaximum:
            maxProbs[graphID1] = len(bothMaximum[graphID1]) / len(subset)
        else:
            maxProbs[graphID1] = 0.0

    print(len(bothMaximum))

    # serialize data into file:
    json.dump(maxProbs, open(f"data/probabilities-PairedMaxima-alpha{alpha}-beta{beta}-n={n}.json", 'w')) # NOTE: read JSON-file -> data = json.load( open( "file_name.json" ) )
    json.dump(bothMaximum, open(f"data/graphTransitions-PairedMaxima-alpha{alpha}-beta{beta}-n={n}.json", 'w'))

    x = ['Maximum effect','Not maximum effect']
    y = [len(bothMaximum),len(startGraphs)-len(bothMaximum)]
    
    # barplot ratio success/fail maxGE-maxHellinger from start graphs
    fig, ax = plt.subplots(figsize=(5,5))
    ax.bar(x=x,
           height=y,
           color=['green','tab:red'])

    addlabels(x, y)

    ax.set_ylabel("Frequency",fontsize=14)
    ax.set_ylim(0,550)
    ax.tick_params(axis="both",which="major",labelsize=14)
    # ax.tick_params(axis='x', labelrotation=90)

    plt.show()
    fig.savefig(f"images/transitions/n={n}/binomialDistribution-{settings}-n={n}.png",bbox_inches='tight')
    plt.close(fig)

    # # plot distribution of maxGE-maxHellinger probability per start graph
    # fig,ax = plt.subplots()
    # ax.hist(maxProbs.values())
    # ax.set_xlabel("Fraction of paired maximum measures", fontsize=16)
    # ax.set_ylabel("Frequency",fontsize=16)
    # ax.tick_params(axis="both",which="major",labelsize=16)
    # ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n} ({round(len(maxProbs.values())/len(data_hellinger),3)*100}%)",fontsize=16)
    # plt.show()
    # fig.savefig(f"images/transitions/n={n}/freqPairedMaximum-{settings}-n={n}.png",bbox_inches='tight')
    # plt.close(fig)

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
    

def analyze_graphPairs(alpha: str, beta: str, n: int, efficient: bool = False):
    """
    Function to obtain/show top 10 highest and top 10 lowest for both Hellinger distance and global efficiency.
    
    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - n (int): Graph size
    - efficient (bool): False for random dynamics; True for efficient dynamics
    """

    # set paths
    settings = f"alpha{alpha}-beta{beta}"  

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}" 

    data_hellinger = pd.read_csv(f"data/Hellinger-data-{settings}-n={n}.tsv",sep='\t')

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


def examineProbs_PairedMaxima():
    """
    Function that retrieves graph pairs conform given probability.
    """

    data = json.load(open("data/probabilities-PairedMaxima-alpha1_00-beta0_00-n=7.json") )
    data = {int(k):v for k,v in data.items()} # convert key strings back to ints

    res=dict()
    x=list(data.values())    
    y=list(set(x))
    for i in y:
        res[i]=x.count(i)
    
    print(res[1.0])

    p = 1.0

    for key, value in data.items():
        if value == p:
            print(key)

def findCycles(graphs: list):

    G = nx.graph_atlas(graphs[0])
    print(id,nx.cycle_basis(G))
    
    for id in graphs[1:]:
        G = nx.graph_atlas(id)
        print(id, nx.cycle_basis(G))


        

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

    # manual selection
    # from_graph = [316,316]
    # to_graph = [278,272]
    # from_graph = [556,714,572,850,348,340,334,433,443,348]
    # to_graph = [394,497,432,609,279,286,280,349,349,286]
    #################################

    # for alpha, beta in product(alphas,betas): 
        # if beta == '0_50' and alpha == '0_50':
        #     continue        
        
        # print(f'alpha={alpha} & beta={beta}')

    #     # NOTE
    #     # NOTE: CHOOSE FUNCTION TO RUN
    #     # NOTE

    # # run comparison and generate Hellinger distance data
    # pairwise_comparison(alpha=alpha,
    #                     beta=beta,
    #                     from_graph=from_graph,
    #                     to_graph=to_graph,
    #                     n=n,
    #                     plots=True,
    #                     efficient=False,
    #                     saveData=False)

    # # plot histogram distribution of Hellinger distance
    # distribution_hellinger(alpha=alpha,
    #                        beta=beta,
    #                        n=n,
    #                        efficient=False)

    # # scatterplot relation between difference in global efficiency and difference in Hellinger distance
    # correlate_hellinger_globalEff(alpha=alpha,
    #                               beta=beta,
    #                               n=n,
    #                               startGraph=False,
    #                               efficient=False) # if TRUE, adjust range as desired

    # # investigate graph pairs
    # analyze_graphPairs(alpha=alpha, 
    #                    beta=beta, 
    #                    n=n,
    #                    efficient=False)

    # # investigate intervention effectiveness
    # successTransitions(alpha=alpha,
    #                    beta=beta,
    #                    n=n,
    #                    efficient=False) 

    # examineProbs_PairedMaxima()

    # graphs = [730,550,573,578]
    # findCycles(graphs=graphs)



