"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

import numpy as np, pandas as pd
from tqdm import tqdm

"""Combine structural metrics and number of messages"""
metrics = pd.read_csv('data/data-GraphAtlas.tsv', sep='\t')

without2 = metrics[metrics['nodes'] != 2]

#############################################################################
# NOTE: CHOOSE DATA CORRECTLY                                               #
# (METRICS FOR GRAPH SIZES N=2 TO N=7; WITHOUT2 FOR GRAPH SIZES N=3 TO N=7) #
graphData = without2
#############################################################################

print(graphData)

complete = pd.DataFrame(data=None, index=np.arange(len(graphData)*100), 
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

#######################################################
# NOTE: Set simulation settings to save appropriately #
settings = 'alpha1_00-beta0_50'                       #
#######################################################

for idx in tqdm(range(len(graphData))):

    if graphData['nodes'].iloc[idx] == 2:
        continue

    n100 = idx*100

    # generate filename and load corresponding convergence data
    name = 'G' + str(graphData['index'].iloc[idx])
    convergence = pd.read_csv(f'results/{settings}/convergence-{name}.tsv', usecols=['nMessages'], sep='\t')

    # insert convergence data into DataFrame with average number of messages 
    complete['index'].iloc[n100:n100+100] = graphData['index'].iloc[idx]
    complete['nodes'].iloc[n100:n100+100] = graphData['nodes'].iloc[idx]
    complete['degree'].iloc[n100:n100+100] = graphData['degree'].iloc[idx]
    complete['betweenness'].iloc[n100:n100+100] = graphData['betweenness'].iloc[idx]
    
    if name == 'G3':
        complete['CFbetweenness'].iloc[n100:n100+100] = 0
    else:
        complete['CFbetweenness'].iloc[n100:n100+100] = graphData['CFbetweenness'].iloc[idx]

    complete['CFbetweenness'].iloc[n100:n100+100] = graphData['CFbetweenness'].iloc[idx]
    complete['closeness'].iloc[n100:n100+100] = graphData['closeness'].iloc[idx]
    complete['clustering'].iloc[n100:n100+100] = graphData['clustering'].iloc[idx]
    complete['globalEff'].iloc[n100:n100+100] = graphData['globalEff'].iloc[idx]
    complete['localEff'].iloc[n100:n100+100] = graphData['localEff'].iloc[idx]
    complete['nMessages'].iloc[n100:n100+100] = convergence['nMessages']

print(complete)
complete.to_csv(f'data/relationData-withoutN=2-{settings}-Atlas.tsv',sep='\t',index=False)
"""Combine structural metrics and number of messages"""

"""Average number of messages per graph"""
# settings = 'alpha0_50-beta0_00'
# data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')

# meanData = data.groupby('index').agg({'index':'mean',
#                                     'nodes':'mean',
#                                     'degree':'mean',
#                                     'betweenness':'mean',
#                                     'CFbetweenness': 'mean',
#                                     'closeness':'mean',
#                                     'clustering':'mean',
#                                     'globalEff':'mean',
#                                     'localEff':'mean',
#                                     'nMessages':'mean'})   

# data.to_csv(f'data/meanRelationData-{settings}-Atlas.tsv',sep='\t',index=False) 
"""Average number of messages per graph"""