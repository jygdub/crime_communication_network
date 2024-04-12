"""
Script to analyze effect of graph transitions by edge removal on consensus formation.

Written by Jade Dubbeld
12/04/2024
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt

def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

if __name__ == "__main__":
    # NOTE: CHOOSE DESIRED SETTINGS
    alpha = "1_00"
    beta = "0_00"

    settings = f'alpha{alpha}-beta{beta}'          
    data = pd.read_csv(f'data/relationData-withoutN=2-{settings}-Atlas.tsv', sep='\t')

    print(data)

    # NOTE: CHOOSE GRAPHS TO COMPARE IN TRANSITIONS
    GRAPH1 = 35
    GRAPH2 = 30

    # get corresponding convergence
    data1 = data[data['index']==GRAPH1]
    data2 = data[data['index']==GRAPH2]

    # setup comparison data
    compare = data1['nMessages'].to_frame().rename(columns={"nMessages": f"G{GRAPH1}"})
    compare[f"G{GRAPH2}"] = list(data2['nMessages'])
    print(compare)

    fig, (ax0,ax1) = plt.subplots(1,2)
    n, bins, _ = ax0.hist(x=compare,bins=10,label=[f"G{GRAPH1}: GE={round(data1['globalEff'].unique()[0],4)}",f"G{GRAPH2}: GE={round(data2['globalEff'].unique()[0],4)}"])
    print(n)
    print(bins)
    bin_centers = 0.5*(bins[1:]+bins[:-1])

    ax0.legend()
    ax0.set_xlabel("Number of messages")
    ax0.set_ylabel("Frequency")
    ax0.set_title("n_agents=5; n_repeats=100")

    ax1.plot(bin_centers,n[0],label=f"G{GRAPH1}: GE={round(data1['globalEff'].unique()[0],4)}")
    ax1.plot(bin_centers,n[1],label=f"G{GRAPH2}: GE={round(data2['globalEff'].unique()[0],4)}")
    ax1.legend()
    ax1.set_xlabel("Number of messages")
    ax1.set_ylabel("Frequency")
    ax1.set_title("n_agents=5; n_repeats=100")
    # compare.hist(n_bins=10,density=True, histtype='bar')

    # compute Hellinger distance
    # print(hellinger(data1,data2))

    plt.show()
