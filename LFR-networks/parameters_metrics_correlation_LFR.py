"""
Script to correlate LFR benchmark graph parameters with various network measures.

Written by Jade Dubbeld
12/02/2024
"""

import networkx as nx, numpy as np, matplotlib.pyplot as plt, pandas as pd, pickle
import statsmodels.formula.api as smf 

data = pd.read_csv('data/all-measures.tsv',sep='\t')
print(data)

muDegree = plt.figure()
muClustering = plt.figure()
muBetweenness = plt.figure()
muCFBetweenness = plt.figure()
muCloseness = plt.figure()
avg_degDegree = plt.figure()
avg_degClustering = plt.figure()
avg_degBetweenness = plt.figure()
avg_degCFBetweenness = plt.figure()
avg_degCloseness = plt.figure()

"""Uncomment to initialize 3D plot"""
# degree = plt.figure()
# ax = plt.axes(projection ="3d")
# clustering = plt.figure()
# ax = plt.axes(projection ="3d")
"""Uncomment until here"""

cm = ['chartreuse','teal','royalblue','deeppink','aquamarine','black','firebrick','darkkhaki','chocolate','forestgreen','orange','darkviolet','red','deepskyblue','fuchsia']

count = 0

for avg_deg in [5,10,15,20,25]:
    for mu in [0.75,0.5,0.25]:
        indices = np.where((data['mu'] == mu) & (data['average_degree'] == avg_deg))[0]
        subset = data.iloc[indices]

        plt.figure(muDegree)
        plt.scatter(subset.mu,subset.degreeCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(muBetweenness)
        plt.scatter(subset.mu,subset.betweennessCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(muCFBetweenness)
        plt.scatter(subset.mu,subset.currentFlowBetweenness,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(muCloseness)
        plt.scatter(subset.mu,subset.closenessCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(muClustering)
        plt.scatter(subset.mu,subset.clusteringCoefficient,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(avg_degDegree)
        plt.scatter(subset.average_degree,subset.degreeCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(avg_degBetweenness)
        plt.scatter(subset.average_degree,subset.betweennessCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(avg_degCFBetweenness)
        plt.scatter(subset.average_degree,subset.currentFlowBetweenness,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(avg_degCloseness)
        plt.scatter(subset.average_degree,subset.closenessCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        plt.figure(avg_degClustering)
        plt.scatter(subset.average_degree,subset.clusteringCoefficient,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        """Uncomment to plt in 3D"""
        # plt.figure(degree)
        # ax.scatter3D(subset.mu,subset.average_degree,subset.degreeCentrality,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')

        # plt.figure(clustering)
        # ax.scatter3D(subset.mu,subset.average_degree,subset.clusteringCoefficient,color=cm[count],label=f'avg_deg={avg_deg} & mu={mu}')
        """Uncomment until heere"""

        count += 1

plt.figure(muDegree)
plt.xlabel('Mu')
plt.ylabel('Degree centrality')
plt.title('Degree centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-degree.png",bbox_inches='tight')

plt.figure(muBetweenness)
plt.xlabel('Mu')
plt.ylabel('Betweenness centrality')
plt.title('Betweenness centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-betweenness.png",bbox_inches='tight')

plt.figure(muCFBetweenness)
plt.xlabel('Mu')
plt.ylabel('Current flow betweenness centrality')
plt.title('Current flow betweenness centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-CFbetweenness.png",bbox_inches='tight')

plt.figure(muCloseness)
plt.xlabel('Mu')
plt.ylabel('Closeness centrality')
plt.title('Closeness centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-closeness.png",bbox_inches='tight')

plt.figure(muClustering)
plt.xlabel('Mu')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-clustering.png",bbox_inches='tight')

plt.figure(avg_degDegree)
plt.xlabel('Average degree (as LFR parameter)')
plt.ylabel('Degree centrality')
plt.title('Degree centrality against average degree-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-degree.png",bbox_inches='tight')

plt.figure(avg_degBetweenness)
plt.xlabel('Average degree (as LFR parameter)')
plt.ylabel('Betweenness centrality')
plt.title('Betweenness centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-betweenness.png",bbox_inches='tight')

plt.figure(avg_degCFBetweenness)
plt.xlabel('Average degree (as LFR parameter)')
plt.ylabel('Current flow betweenness centrality')
plt.title('Current flow betweenness centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-CFbetweenness.png",bbox_inches='tight')

plt.figure(avg_degCloseness)
plt.xlabel('Average degree (as LFR parameter)')
plt.ylabel('Closeness centrality')
plt.title('Closeness centrality against µ-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-closeness.png",bbox_inches='tight')

plt.figure(avg_degClustering)
plt.xlabel('Average degree (as LFR parameter)')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient against average degree-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-clustering.png",bbox_inches='tight')

"""Uncomment to decorate 3D plot"""
# plt.figure(degree)
# ax.set_xlabel('Mu')
# ax.set_ylabel('Average degree (as LFR parameter)')
# ax.set_zlabel('Degree centrality')
# pickle.dump((degree, ax), open('images/3D-degreeCentrality.pickle', 'wb'))

# plt.figure(clustering)
# ax.set_xlabel('Mu')
# ax.set_ylabel('Average degree (as LFR parameter)')
# ax.set_zlabel('Clustering coefficient')
# pickle.dump((clustering, ax), open('images/3D-clusteringCoefficient.pickle', 'wb'))
""""Uncomment until here"""

# multiple linear regression
model_degree = smf.ols(f'degreeCentrality ~  mu + average_degree', data=data).fit()
print(model_degree.summary())

model_clustering = smf.ols(f'clusteringCoefficient ~  mu + average_degree', data=data).fit()
print(model_clustering.summary())