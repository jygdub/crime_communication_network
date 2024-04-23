"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

import numpy as np, pandas as pd
from tqdm import tqdm

def combineData(alpha: str, beta: str):
    """Combine structural metrics and number of messages"""
    metrics = pd.read_csv('data/data-GraphAtlas.tsv', sep='\t')

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


    settings = f'alpha{alpha}-beta{beta}'                       

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

def meanData(alpha: str, beta: str):
    """Average number of messages per graph"""
    settings = f'alpha{alpha}-beta{beta}'
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

    data.to_csv(f'data/meanRelationData-{settings}-Atlas.tsv',sep='\t',index=False) 

if __name__ == "__main__":
    alpha = '0_50'
    beta = '0_25'
    combineData(alpha=alpha,beta=beta)