"""
Script to analyze effect of graph transitions by edge removal on consensus formation.

Written by Jade Dubbeld
12/04/2024
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

if __name__ == "__main__":
    # NOTE: CHOOSE DESIRED SETTINGS
    alpha = "0_50"
    beta = "0_00"

    settings = f'alpha{alpha}-beta{beta}'          
    data = pd.read_csv(f'data/relationData-withoutN=2-{settings}-Atlas.tsv', sep='\t')

    # NOTE: CHOOSE GRAPHS TO COMPARE IN TRANSITIONS
    from_graph = [52,
                  51,51,
                  50,50,
                  49,49,49,49,
                  48,48,48,
                  47,47,47,47,
                  46,
                  45,45,
                  44,
                  43,43,43,43,
                  42,42,
                  41,41,
                  40,40,
                  38,
                  37,
                  36,36,
                  35,35,
                  34]
    
    to_graph = [51,
                50,49,
                48,47,
                48,47,46,45,
                44,43,41,
                43,42,41,40,
                44,
                41,40,
                37,
                38,37,36,35,
                36,34,
                36,35,
                35,34,
                31,
                31,
                31,30,
                31,30,
                29]
    
    data_hellinger = pd.DataFrame(data=None,index=range(37),columns=["index_graph1","index_graph2","GE_graph1","GE_graph2","Hellinger"])
    
    for i,(graph1,graph2) in tqdm(enumerate(zip(from_graph,to_graph))):

        # get corresponding convergence distribution for respective graphs
        data1 = data[data['index']==graph1]
        data2 = data[data['index']==graph2]

        # setup comparison data
        compare = data1['nMessages'].to_frame().rename(columns={"nMessages": f"G{graph1}"})
        compare[f"G{graph2}"] = list(data2['nMessages'])
        # print(compare)

        fig, (ax0,ax1) = plt.subplots(1,2,figsize=(13,8))
        n, bins, _ = ax0.hist(x=compare,bins=10,label=[f"G{graph1}: GE={round(data1['globalEff'].unique()[0],4)}",f"G{graph2}: GE={round(data2['globalEff'].unique()[0],4)}"])
        # print(n)
        # print(bins)
        bin_centers = 0.5*(bins[1:]+bins[:-1])


        # compute probabilities
        p_graph1 = n[0]/100
        p_graph2 = n[1]/100

        # compute Hellinger distance
        dist_hellinger = hellinger(p_graph1,p_graph2)
        # print(f"Hellinger: {dist_hellinger}")

        data_hellinger.iloc[i] = pd.Series(data={"index_graph1": int(graph1),
                                  "index_graph2": int(graph2),
                                  "GE_graph1": data1['globalEff'].unique()[0],
                                  "GE_graph2": data2['globalEff'].unique()[0],
                                  "Hellinger": dist_hellinger})

        ax0.legend(fontsize=16)
        ax0.set_xlabel("Number of messages",fontsize=16)
        ax0.set_ylabel("Frequency",fontsize=16)
        ax0.set_title(f"Hellinger={round(dist_hellinger,3)}; n_agents=5; n_repeats=100",fontsize=16)
        ax0.tick_params(axis="both",which="major",labelsize=16)

        ax1.plot(bin_centers,n[0],label=f"G{graph1}: GE={round(data1['globalEff'].unique()[0],4)}")
        ax1.plot(bin_centers,n[1],label=f"G{graph2}: GE={round(data2['globalEff'].unique()[0],4)}")
        ax1.legend(fontsize=16)
        ax1.set_xlabel("Number of messages",fontsize=16)
        ax1.set_ylabel("Frequency",fontsize=16)
        ax1.set_title(f"Hellinger={round(dist_hellinger,3)}; n_agents=5; n_repeats=100",fontsize=16)
        ax1.tick_params(axis="both",which="major",labelsize=16)
        # compare.hist(n_bins=10,density=True, histtype='bar')


        # plt.show()
        fig.savefig(f"images/transitions/G{graph1}-G{graph2}.png",bbox_inches='tight')
        plt.close(fig)

    data_hellinger.to_csv(f"data/Hellinger-data-alpha={alpha}-beta={beta}.tsv",sep='\t',index=False)
