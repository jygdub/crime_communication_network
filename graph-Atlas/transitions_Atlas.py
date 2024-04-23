"""
Script to analyze effect of graph transitions by edge removal on consensus formation.

Written by Jade Dubbeld
12/04/2024
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, networkx as nx, pickle
from tqdm import tqdm
from itertools import product

def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def pairwise_comparison(alpha,beta,from_graph,to_graph,n):

    settings = f'alpha{alpha}-beta{beta}'          
    data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')

    # NOTE: DROP FIRST 100 ROWS IF EXCLUDING N=2 GRAPH SIZE
    data = data.drop(range(0,100))

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
        # bin_centers = 0.5*(bins[1:]+bins[:-1])

        # compute probabilities
        p_graph1 = x[0]/100
        p_graph2 = x[1]/100

        # compute Hellinger distance
        dist_hellinger = hellinger(p_graph1,p_graph2)

        data_hellinger.iloc[i] = pd.Series(data={"index_graph1": int(graph1),
                                  "index_graph2": int(graph2),
                                  "GE_graph1": data1['globalEff'].unique()[0],
                                  "GE_graph2": data2['globalEff'].unique()[0],
                                  "Hellinger": dist_hellinger})

        # ax0.legend(fontsize=16)
        # ax0.set_xlabel("Number of messages",fontsize=16)
        # ax0.set_ylabel("Frequency",fontsize=16)
        # ax0.set_title(f"Hellinger={round(dist_hellinger,3)}; n_agents={n}; n_repeats=100",fontsize=16)
        # ax0.tick_params(axis="both",which="major",labelsize=16)

        # ax1.plot(bin_centers,n[0],label=f"G{graph1}: GE={round(data1['globalEff'].unique()[0],4)}")
        # ax1.plot(bin_centers,n[1],label=f"G{graph2}: GE={round(data2['globalEff'].unique()[0],4)}")
        # ax1.legend(fontsize=16)
        # ax1.set_xlabel("Number of messages",fontsize=16)
        # ax1.set_ylabel("Frequency",fontsize=16)
        # ax1.set_title(f"Hellinger={round(dist_hellinger,3)}; n_agents={n}; n_repeats=100",fontsize=16)
        # ax1.tick_params(axis="both",which="major",labelsize=16)

        # # plt.show()
        # fig.savefig(f"images/transitions/G{graph1}-G{graph2}.png",bbox_inches='tight')
        plt.close(fig)

    data_hellinger.to_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t',index=False)

def distribution_hellinger(alpha,beta,n):

    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t')

    # print(data_hellinger)

    fig,ax = plt.subplots()

    ax.hist(data_hellinger["Hellinger"],bins=10)
    
    ax.set_xlabel("Hellinger distance", fontsize=16)
    ax.set_ylabel("Frequency",fontsize=16)
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=16)
    
    fig.savefig(f"images/transitions/HellingerDistribution-alpha={alpha}-beta={beta}-n={n}.png",bbox_inches='tight')

    plt.close(fig)

def relate_structure_operation(alpha,beta,n):
    data_hellinger = pd.read_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}-n={n}.tsv",sep='\t')

    fig, ax = plt.subplots()
    ax.scatter(x=data_hellinger["GE_graph1"]-data_hellinger["GE_graph2"],y=data_hellinger["Hellinger"])
    ax.set_xlabel("Difference in global efficiency",fontsize=16)
    ax.set_ylabel("Hellinger distance",fontsize=16)
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}",fontsize=16)
    plt.show()

    fig.savefig(f"images/transitions/GE-Hellinger-correlation-alpha={alpha}-beta={beta}-n={n}.png",bbox_inches='tight')

    plt.close(fig)

def possible_transitions(n):
    
    df = pd.read_csv('data/data-GraphAtlas.tsv',usecols=['index','nodes'],sep='\t')
    df = df[df['nodes']==n]

    df = df.reindex(index=df.index[::-1])

    from_graph = []
    to_graph = []

    for i,j in tqdm(product(df['index'],df['index'])):
        if j > i:
            continue
        # print(i)
        graph1 = 'G' + str(i)
        file1 = f'graphs/{graph1}.pickle'

        G1 = pickle.load(open(file1,'rb'))

        graph2 = 'G' + str(j)
        file2 = f'graphs/{graph2}.pickle'

        G2 = pickle.load(open(file2,'rb'))

        if nx.graph_edit_distance(G1,G2) == 1.0:
            # print(f"G{i}->G{j}")
            from_graph.append(i)
            to_graph.append(j)
    
    np.savetxt(f"data/from_graph_n={n}.tsv",
        from_graph,
        delimiter ="\t",
        fmt ='% i')

    np.savetxt(f"data/to_graph_n={n}.tsv",
        to_graph,
        delimiter ="\t",
        fmt ='% i')

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

    # NOTE: CHOOSE GRAPHS TO COMPARE IN TRANSITIONS (below for n=5, manual analysis)
    from_graph = list(map(int, np.loadtxt(f"data/from_graph_n={n}.tsv",delimiter='\t')))
    to_graph = list(map(int, np.loadtxt(f"data/to_graph_n={n}.tsv",delimiter='\t')))
    #################################

    # for alpha, beta in product(alphas,betas): 
    #     if beta == '0_50' and alpha in ['0_75','0_50']:
    #         continue        
        
    print(f'alpha={alpha} & beta={beta}')

    # NOTE
    # NOTE: CHOOSE FUNCTION TO RUN
    # NOTE

    # # run comparison and generate Hellinger distance data
    # pairwise_comparison(alpha=alpha,beta=beta,from_graph=from_graph,to_graph=to_graph,n=n)

    # # plot histogram distribution of Hellinger distance
    # distribution_hellinger(alpha,beta,n=n)

    # scatterplot relation between difference in global efficiency and difference in Helling distance
    relate_structure_operation(alpha,beta,n=n)
