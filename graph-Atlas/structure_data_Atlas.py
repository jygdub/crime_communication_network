"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024

Function mergeData() added from separate script.
    Script to merge convergence results per Graph Atlas network.
    - Convergence results from Snellius were dumped per run
    - For analysis purposes, all convergence results should be clustered per graph.

    Written by Jade Dubbeld
    27/03/2024
"""

import numpy as np, pandas as pd
from tqdm import tqdm

def mergeData(alpha: str, beta: str, efficient: bool = False, fixed: bool = False):
    """
    Function to merge separate convergence result files from Snellius server (parallel simulation).
    
    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - efficient (bool): False for random dynamics; True for efficient dynamics
    - fixed (bool): False for random states initialization; True for fixed states initialization
    """

    settings = f"alpha{alpha}-beta{beta}"
    df = pd.read_csv('data/data-GraphAtlas.tsv',sep='\t')

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}"
    
    if fixed:
        settings = f"fixed-alpha{alpha}-beta{beta}"

    graphs = []

    for j in range(len(df)):

        # generate graph ID
        name = 'G' + str(df['index'].iloc[j])
        graphs.append(name)

    for graph in tqdm(graphs):

        data = pd.DataFrame(index=range(100),columns=['nMessages','meanHammingDist'])

        for i in range(100):
            run = pd.read_csv(f'results/{settings}/convergence-{graph}-run{i}.tsv', sep='\t', usecols=['nMessages'])#,'meanHammingDist'])
            data.iloc[i] = run.iloc[0]

        data.to_csv(f'results/{settings}/merged/convergence-{graph}.tsv', sep='\t',index=False)   


def combineData(alpha: str, beta: str, efficient: bool = False, fixed: bool = False):
    """
    Function to combine structural metrics and number of messages.
    
    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - efficient (bool): False for random dynamics; True for efficient dynamics
    - fixed (bool): False for random states initialization; True for fixed states initialization
    """

    metrics = pd.read_csv('data/data-GraphAtlas.tsv', sep='\t')

    if fixed:
        metrics = metrics[metrics['nodes']==7]


    complete = pd.DataFrame(data=None, index=np.arange(len(metrics)*100), 
                            columns=['index',
                                    'nodes',
                                    'degree',
                                    'betweenness',
                                    'CFbetweenness',
                                    'closeness',
                                    'clustering',
                                    'globalEff',
                                    'localEff',
                                    'nMessages'])


    settings = f"alpha{alpha}-beta{beta}"

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}"  

    if fixed:
        settings = f"fixed-alpha{alpha}-beta{beta}"                     

    for idx in tqdm(range(len(metrics))):

        n100 = idx*100

        # generate filename and load corresponding convergence data
        name = 'G' + str(metrics['index'].iloc[idx])
        convergence = pd.read_csv(f'results/{settings}/convergence-{name}.tsv', usecols=['nMessages'], sep='\t')

        # insert convergence data into DataFrame with average number of messages 
        complete['index'].iloc[n100:n100+100] = metrics['index'].iloc[idx]
        complete['nodes'].iloc[n100:n100+100] = metrics['nodes'].iloc[idx]
        complete['degree'].iloc[n100:n100+100] = metrics['degree'].iloc[idx]
        complete['betweenness'].iloc[n100:n100+100] = metrics['betweenness'].iloc[idx]

        if name == 'G3':
            complete['CFbetweenness'].iloc[n100:n100+100] = 0
        else:
            complete['CFbetweenness'].iloc[n100:n100+100] = metrics['CFbetweenness'].iloc[idx]

        complete['closeness'].iloc[n100:n100+100] = metrics['closeness'].iloc[idx]
        complete['clustering'].iloc[n100:n100+100] = metrics['clustering'].iloc[idx]
        complete['globalEff'].iloc[n100:n100+100] = metrics['globalEff'].iloc[idx]
        complete['localEff'].iloc[n100:n100+100] = metrics['localEff'].iloc[idx]
        complete['nMessages'].iloc[n100:n100+100] = convergence['nMessages']

    print(complete)

    complete.to_csv(f'data/relationData-{settings}-Atlas.tsv',sep='\t',index=False)

def meanData(alpha: str, beta: str, efficient: bool = False):
    """
    Function to average number of messages per graph.

    Parameters:
    - alpha (str): Setting value of alpha noise 
    - beta (str): Setting value of beta noise
    - efficient (bool): False for random dynamics; True for efficient dynamics
    """

    settings = f"alpha{alpha}-beta{beta}"

    if efficient:
        settings = f"efficient-alpha{alpha}-beta{beta}"    

    data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')

    meanData = data.groupby('index').agg({'index':'mean',
                                        'nodes':'mean',
                                        'degree':'mean',
                                        'betweenness':'mean',
                                        'CFbetweenness': 'mean',
                                        'closeness':'mean',
                                        'clustering':'mean',
                                        'globalEff':'mean',
                                        'localEff':'mean',
                                        'nMessages':'mean'})   

    meanData.to_csv(f'data/meanRelationData-{settings}-Atlas.tsv',sep='\t',index=False) 

if __name__ == "__main__":
    alpha = '1_00'
    beta = '0_00'
    efficient = True
    fixed = False

    # mergeData(alpha=alpha,beta=beta,efficient=efficient,fixed=fixed)
    # combineData(alpha=alpha,beta=beta,efficient=efficient,fixed=fixed)