"""
Script to analyze effect of graph transitions by edge removal on consensus formation.

Written by Jade Dubbeld
12/04/2024
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, networkx as nx, pickle, json
from tqdm import tqdm
from itertools import product
from typing import Tuple


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
        fig, ax = plt.subplots(figsize=(13,8))
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
    maxStructuralChange = {}

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
            if subset['GE_difference'][index] == maxGE:

                if graphID1 in maxStructuralChange:
                    maxStructuralChange[graphID1].append(graphID2)
                else:
                    maxStructuralChange[graphID1] = [graphID2]

                if subset['Hellinger'][index] == maxHellinger:
                    if graphID1 in bothMaximum:
                        bothMaximum[graphID1].append(graphID2)
                    else:
                        bothMaximum[graphID1] = [graphID2]

        # compute probabilities of finding maximum values for both measures per start graph
        if graphID1 in bothMaximum:
            maxProbs[graphID1] = len(bothMaximum[graphID1]) / len(subset)
        else:
            maxProbs[graphID1] = 0.0

    print(len(bothMaximum))
    print(len(maxStructuralChange))

    # serialize data into file:
    json.dump(maxProbs, open(f"data/probabilities-PairedMaxima-alpha{alpha}-beta{beta}-n={n}.json", 'w')) # NOTE: read JSON-file -> data = json.load( open( "file_name.json" ) )
    json.dump(bothMaximum, open(f"data/graphTransitions-PairedMaxima-alpha{alpha}-beta{beta}-n={n}.json", 'w'))
    json.dump(maxStructuralChange, open(f"data/graphTransitions-maxStructural-alpha{alpha}-beta{beta}-n={n}.json", 'w'))

    # x = ['Maximal','Not maximal']
    # y = [len(bothMaximum),len(startGraphs)-len(bothMaximum)]
    
    # # barplot ratio success/fail maxGE-maxHellinger from start graphs
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.bar(x=x,
    #        height=y,
    #        color=['green','tab:red'])

    # addlabels(x, y)

    # ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=14)
    # ax.set_ylabel("Frequency",fontsize=14)
    # ax.set_ylim(0,550)
    # ax.tick_params(axis="both",which="major",labelsize=14)
    # # ax.tick_params(axis='x', labelrotation=90)

    # plt.show()
    # fig.savefig(f"images/transitions/n={n}/binomialDistribution-{settings}-n={n}.png",bbox_inches='tight')
    # plt.close(fig)

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

def possible_transitions(n: int, intervention: str) -> None:
    """
    Function to check possible transitions between graphs, given parameter settings.
    - Transition is valid when final graph remains connected.
    - Transition is valid when final graph remains connected.

    Parameters:
    - n (int): Graph size
    - intervention (str): Indicate type of transition ('single_node','single_edge')
    """
    
    # load raw data
    df = pd.read_csv('data/data-GraphAtlas.tsv',usecols=['index','nodes'],sep='\t')

    # select data given graph size
    df = df[df['nodes']==n]
    df = df.reindex(index=df.index[::-1])

    # initials
    from_graph = []
    to_graph = []
    removeNode = []

    if intervention == 'single_edge':

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
        np.savetxt(f"data/singleEdge_from_graph_n={n}.tsv",
            from_graph,
            delimiter ="\t",
            fmt ='% i')

        np.savetxt(f"data/singleEdge_to_graph_n={n}.tsv",
            to_graph,
            delimiter ="\t",
            fmt ='% i')

    elif intervention == 'single_node':
        for id in tqdm(df['index']):
            # print(id)
            
            G = nx.graph_atlas(id)

            for n in G.nodes:
                # print(n)
                G_copy = G.copy()
                G_copy.remove_node(n)

                if nx.is_connected(G_copy):
                    from_graph.append(id) 
                    removeNode.append(n)

                    # print(nx.is_connected(G_copy))

        # print(from_graph)
        # print(removeNode)
        print(len(from_graph))

        # save transitions in order
        np.savetxt(f"data/singleNode_from_graph_n={n}.tsv",
            from_graph,
            delimiter ="\t",
            fmt ='% i')
                

        # save transitions in order
        np.savetxt(f"data/singleNode_remove_node_n={n}.tsv",
            removeNode,
            delimiter ="\t",
            fmt ='% i') 

def top10_graphPairs(alpha: str, beta: str, n: int, efficient: bool = False):
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

    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha1_00-beta0_00-n=7.tsv",sep='\t')
    print(data_hellinger)

    res=dict()
    x=list(data.values())    
    y=list(set(x))
    for i in y:
        res[i]=x.count(i)
    
    print(res[1.0])

    p = 1.0

    for key, value in data.items():
        if value == p:
            transition = data_hellinger[['index_graph1','index_graph2']][data_hellinger['index_graph1']==key]
            print(transition)

def countDegreeFreq(distr: list) -> int:
    degreeFreq = {0: 0,
                  1: 0,
                  2: 0,
                  3: 0,
                  4: 0,
                  5: 0,
                  6: 0}

    for d in distr:
        degreeFreq[d[1]] += 1

    return degreeFreq

def countCycles(G: nx.classes.graph.Graph, n: int) -> int:
    """
    Function to count number of cycles up to size n.

    Parameters:
    - G (nx.classes.graph.Graph): Networkx graph object
    - n (int): Maximum size of cycles

    Returns:
    - nCycles (int): Number of cycles with maximum size up to n
    """
    cycles = nx.simple_cycles(G=G,length_bound=n)

    nCycle = 0

    for c in cycles:
        nCycle += 1
    
    return nCycle

def graphProperties(G: nx.classes.graph.Graph) -> Tuple[dict, int, int, int, int, int]:
    """
    Function to retrieve properties from given graph G.
    - Degree distribution
    - Number cycles (size up to 7)

    Parameters:
    - G (nx.classes.graph.Graph): Networkx graph object

    Returns:
    - degreeFreq (dict): Mapping of degree distribution (k,v) -> (degree, frequency)
    - n3Cycles (int): Number of cycles size 3
    - n4Cycles (int): Number of cycles size 4
    - n5Cycles (int): Number of cycles size 5
    - n6Cycles (int): Number of cycles size 6
    - n7Cycles (int): Number of cycles size 7
    """
 
    
    # get degree distribution
    nodeDegree = G.degree
    degreeFreq = countDegreeFreq(nodeDegree)

    # get number of cycles of size 3
    n3Cycles = countCycles(G=G,n=3)  

    n4Cycles = countCycles(G=G,n=4)
    n4Cycles -= n3Cycles

    n5Cycles = countCycles(G=G,n=5)
    n5Cycles -= (n4Cycles+n3Cycles)

    n6Cycles = countCycles(G=G,n=6)
    n6Cycles -= (n5Cycles+n4Cycles+n3Cycles)

    n7Cycles = countCycles(G=G,n=7)
    n7Cycles -= (n6Cycles+n5Cycles+n4Cycles+n3Cycles)

    return degreeFreq, n3Cycles, n4Cycles, n5Cycles, n6Cycles, n7Cycles

def findPropertyChange(from_graph: list, to_graph: list):
    """
    Function to analyze a few property changes due to single edge removal.
    - Increase in number of isolated agents.
    - Number of broken cycles of sizes 3 and 4.

    Parameters:
    - from_graph (list): List of graph IDs of representing graphs before transition
    - to_graph (list): List of graph IDs of representing graphs after transition
    """

    # initialize counter variables
    total3Cycles = 0
    total4Cycles = 0
    total5Cycles = 0
    total6Cycles = 0
    total7Cycles = 0

    counterIsolated = 0
    counterDecreaseMaxDegree = 0
    counterChangeMaxDegree = 0
    counterForced3Cycles = 0
    counter3Cycles = 0
    counter4Cycles = 0
    counter5Cycles = 0
    counter6Cycles = 0
    counter7Cycles = 0
    
    for id1, id2 in tqdm(zip(from_graph,to_graph)):

        # operations applied on start graph
        G = nx.graph_atlas(int(id1))

        degreeFreq1, n3Cycles1, n4Cycles1, n5Cycles1, n6Cycles1, n7Cycles1 = graphProperties(G=G)

        if n3Cycles1 != 0:
            total3Cycles += 1
        # else:
        #     print(f"ID without 3-cycle: {id1}")

        if n3Cycles1 == 1 and n4Cycles1 == 0 and n5Cycles1 == 0 and n6Cycles1 == 0 and n7Cycles1 == 0:
            counterForced3Cycles += 1
            # print(f"ID with only 1x 3-cycle: {id1}")

        if n4Cycles1 != 0:
            total4Cycles += 1

        if n5Cycles1 != 0:
            total5Cycles += 1

        if n6Cycles1 != 0:
            total6Cycles += 1

        if n7Cycles1 != 0:
            total7Cycles += 1

        # operations applied on final graph
        G = nx.graph_atlas(int(id2))

        degreeFreq2, n3Cycles2, n4Cycles2, n5Cycles2, n6Cycles2, n7Cycles2 = graphProperties(G=G)

        # count number of additional isolated agents from transition
        if degreeFreq2[1] > degreeFreq1[1]:
            counterIsolated += 1

        maxDegree1 = None
        maxDegree2 = None

        # search number maximum degree before and after transition and get corresponding frequency
        for d in reversed(range(7)):

            if degreeFreq1[d] != 0 and maxDegree1 == None:
                maxDegree1 = d
                            
            if degreeFreq2[d] != 0 and maxDegree2 == None:
                maxDegree2 = d

        # count frequency of decrease in maximum degree
        if maxDegree1 > maxDegree2:
            counterDecreaseMaxDegree += 1
        
        # count frequency of edge removal from agent with maximum degree (before transition)
        if degreeFreq1[maxDegree1] > degreeFreq2[maxDegree1]:
            counterChangeMaxDegree +=1

        # count number broken cycles of size 3
        if n3Cycles1 > n3Cycles2:
            counter3Cycles += 1
        
        # count number broken cycles of size 4
        if n4Cycles1 > n4Cycles2:
            counter4Cycles += 1

        # count number broken cycles of size 5
        if n5Cycles1 > n5Cycles2:
            counter5Cycles += 1

        # count number broken cycles of size 6
        if n6Cycles1 > n6Cycles2:
            counter6Cycles += 1
    
        # count number broken cycles of size 7
        if n7Cycles1 > n7Cycles2:
            counter7Cycles += 1

    print(f"Additional isolated agent: {counterIsolated} ({round(counterIsolated/len(from_graph)*100,3)}% of all transition pairs)")
    print(f"Decrease maximum degree: {counterDecreaseMaxDegree} ({round(counterDecreaseMaxDegree/len(from_graph)*100,3)}% of all transition pairs)")
    print(f"Remove from agent with maximum degree: {counterChangeMaxDegree} ({round(counterChangeMaxDegree/len(from_graph)*100,3)}% of all transition pairs)")
    print(f"Broken cycle of size 3: {counter3Cycles} ({round(counter3Cycles/total3Cycles*100,3)}% of initial graphs with this cycle size)")
    print(f"Broken cycle of size 4: {counter4Cycles} ({round(counter4Cycles/total4Cycles*100,3)}% of initial graphs with this cycle size)")
    print(f"Broken cycle of size 5: {counter5Cycles} ({round(counter5Cycles/total5Cycles*100,3)}% of initial graphs with this cycle size)")
    print(f"Broken cycle of size 6: {counter6Cycles} ({round(counter6Cycles/total6Cycles*100,3)}% of initial graphs with this cycle size)")
    print(f"Broken cycle of size 7: {counter7Cycles} ({round(counter7Cycles/total7Cycles*100,3)}% of initial graphs with this cycle size)")
    print(total3Cycles)
    print(counterForced3Cycles)








if __name__ == "__main__":


    # NOTE: CHOOSE DESIRED SETTINGS
    alpha = "1_00"
    beta = "0_00"

    alphas = ['1_00','0_75','0_50']
    betas = ['0_00','0_25', '0_50']    

    n = 7
    ###############################

    # # check possible single edge transitions within graph size
    # possible_transitions(n=n, intervention='single_node')

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
    #                             beta=beta,
    #                             n=n,
    #                             startGraph=False,
    #                             efficient=False) # if TRUE, adjust range as desired

    # # investigate graph pairs
    # top10_graphPairs(alpha=alpha, 
    #                    beta=beta, 
    #                    n=n,
    #                    efficient=False)

    # # investigate intervention effectiveness
    # successTransitions(alpha=alpha,
    #                    beta=beta,
    #                    n=n,
    #                    efficient=False) 

    # examineProbs_PairedMaxima()

    # # find properties in successful transitions
    # successful = json.load(open("data/graphTransitions-PairedMaxima-alpha1_00-beta0_00-n=7.json") )

    # from_graph = list(successful.keys())
    # to_graph = []

    # temp = list(successful.values())
    
    # for d in temp:
    #     to_graph.append(d[0])

    # findPropertyChange(from_graph=from_graph, to_graph=to_graph)

    # # find properties in transitions resulting in maximum structural difference
    # maxStruct = json.load(open("data/graphTransitions-maxStructural-alpha1_00-beta0_00-n=7.json") )

    # from_graph = []
    # to_graph = []

    # for k,v in maxStruct.items():
    #     from_graph = from_graph +([k]*len(v))
    #     to_graph = to_graph + v

    # findPropertyChange(from_graph=from_graph,to_graph=to_graph)

    # # find properties in all transitions
    # findPropertyChange(from_graph=from_graph,to_graph=to_graph)
