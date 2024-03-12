"""
Script to analyze variation in network characteristics of networkx's Graph Atlas.

Written by Jade Dubbeld
12/03/2024
"""

import pandas as pd, matplotlib.pyplot as plt, numpy as np, pickle

df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

""" 2D relationships """
# columns = ['degree',
#            'betweenness',
#         #    'CFbetweenness',
#            'closeness',
#            'clustering',
#            'globalEff',
#            'localEff']

# metrics = ['Degree centrality',
#            'Betweenness centrality',
#         #    'CF betweenness centrality',
#            'Closeness centrality',
#            'Clustering coefficient',
#            'Global efficiency',
#            'Local efficiency']

# for col,metric in zip(columns,metrics):

#     variation = plt.figure()

#     for n in range(2,8):
#         subset = df.iloc[np.where(df.nodes == n)[0]]

#         plt.figure(variation)
#         plt.scatter(range(0,len(subset)),subset[col],label=f'n={n}')

#     plt.figure(variation)
#     plt.xlabel('Graph ID')
#     plt.ylabel(metric)
#     plt.title(f'Distribution Graph Atlas - {metric}')
#     plt.legend(bbox_to_anchor=(1,1))
#     plt.savefig(f'images/measure-variation/variation-{col}.png',bbox_inches='tight')
""" 2D relationships """

""" 3D relationships """
xyz = plt.figure()
ax_xyz = plt.axes(projection ="3d")

for n in range(2,8):
    
    subset = df.iloc[np.where(df.nodes == n)[0]]

    plt.figure(xyz)
    ax_xyz.scatter3D(subset.degree,subset.localEff,subset.globalEff,label=f'n={n}')

plt.figure(xyz)
ax_xyz.set_xlabel('Degree centrality')
ax_xyz.set_ylabel('Local efficiency')
ax_xyz.set_zlabel('Global efficiency')
# pickle.dump((xyz, ax_xyz), open('images/3D-degree-closeness-global.pickle', 'wb'))
plt.show()
# plt.savefig(f"images/3D-degree-closeness-global.png")
""" 3D relationships """