"""
Script to correlate LFR benchmark graph parameters with various network measures.

Written by Jade Dubbeld
12/02/2024
"""

import networkx as nx, numpy as np, matplotlib.pyplot as plt, pandas as pd, pickle

data = pd.read_csv('data/all-measures.tsv',sep='\t')
print(data)

muDegree = plt.figure()
muClustering = plt.figure()
muBetweenness = plt.figure()
muCFBetweenness = plt.figure()
muCloseness = plt.figure()
muGlobEff = plt.figure()
muLocEff = plt.figure()
avg_degDegree = plt.figure()
avg_degClustering = plt.figure()
avg_degBetweenness = plt.figure()
avg_degCFBetweenness = plt.figure()
avg_degCloseness = plt.figure()
avg_degGlobEff = plt.figure()
avg_degLocEff = plt.figure()

clustering_currentFlow = plt.figure()
clustering_degree = plt.figure()
degree_currentFlow = plt.figure()
degree_betweenness = plt.figure()

"""Uncomment to initialize 3D plot"""
degree = plt.figure()
ax_degree = plt.axes(projection ="3d")

betweenness = plt.figure()
ax_betweenness = plt.axes(projection ="3d")

CFbetweenness = plt.figure()
ax_CFbetweenness = plt.axes(projection ="3d")

closeness = plt.figure()
ax_closeness = plt.axes(projection ="3d")

clustering = plt.figure()
ax_clustering = plt.axes(projection ="3d")

globEff = plt.figure()
ax_globEff = plt.axes(projection ="3d")

locEff = plt.figure()
ax_locEff = plt.axes(projection ="3d")
"""Uncomment until here"""

cm = ['chartreuse','teal','royalblue','deeppink','aquamarine','black','firebrick','darkkhaki','chocolate','forestgreen','orange','darkviolet','red','deepskyblue','fuchsia']

count = 0

for avg_deg in [5,10,15,20,25]:
    for mu in [0.75,0.5,0.25]:
        indices = np.where((data['mu'] == mu) & (data['average_degree'] == avg_deg))[0]
        subset = data.iloc[indices]

        plt.figure(muDegree)
        plt.scatter(subset.mu,subset.degreeCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(muBetweenness)
        plt.scatter(subset.mu,subset.betweennessCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(muCFBetweenness)
        plt.scatter(subset.mu,subset.currentFlowBetweenness,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(muCloseness)
        plt.scatter(subset.mu,subset.closenessCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(muClustering)
        plt.scatter(subset.mu,subset.clusteringCoefficient,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(muGlobEff)
        plt.scatter(subset.mu,subset.globalEfficiency,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(muLocEff)
        plt.scatter(subset.mu,subset.localEfficiency,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degDegree)
        plt.scatter(subset.average_degree,subset.degreeCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degBetweenness)
        plt.scatter(subset.average_degree,subset.betweennessCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degCFBetweenness)
        plt.scatter(subset.average_degree,subset.currentFlowBetweenness,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degCloseness)
        plt.scatter(subset.average_degree,subset.closenessCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degClustering)
        plt.scatter(subset.average_degree,subset.clusteringCoefficient,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degGlobEff)
        plt.scatter(subset.average_degree,subset.globalEfficiency,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(avg_degLocEff)
        plt.scatter(subset.average_degree,subset.localEfficiency,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(clustering_currentFlow)
        plt.scatter(subset.currentFlowBetweenness,subset.clusteringCoefficient,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(clustering_degree)
        plt.scatter(subset.degreeCentrality,subset.clusteringCoefficient,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(degree_currentFlow)
        plt.scatter(subset.currentFlowBetweenness,subset.degreeCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(degree_betweenness)
        plt.scatter(subset.betweennessCentrality,subset.degreeCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        """Uncomment to plt in 3D"""
        plt.figure(degree)
        ax_degree.scatter3D(subset.mu,subset.average_degree,subset.degreeCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(betweenness)
        ax_betweenness.scatter3D(subset.mu,subset.average_degree,subset.betweennessCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(CFbetweenness)
        ax_CFbetweenness.scatter3D(subset.mu,subset.average_degree,subset.currentFlowBetweenness,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(closeness)
        ax_closeness.scatter3D(subset.mu,subset.average_degree,subset.closenessCentrality,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(clustering)
        ax_clustering.scatter3D(subset.mu,subset.average_degree,subset.clusteringCoefficient,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(globEff)
        ax_globEff.scatter3D(subset.mu,subset.average_degree,subset.globalEfficiency,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')

        plt.figure(locEff)
        ax_locEff.scatter3D(subset.mu,subset.average_degree,subset.localEfficiency,color=cm[count],label=fr'$\langle k \rangle = {avg_deg}, \mu = {mu}$')


        """Uncomment until heere"""

        count += 1

plt.figure(muDegree)
plt.xlabel(fr'$\mu$')
plt.ylabel('Degree centrality')
plt.title(fr'Degree centrality against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-degree.png",bbox_inches='tight')

plt.figure(muBetweenness)
plt.xlabel(fr'$\mu$')
plt.ylabel('Betweenness centrality')
plt.title(fr'Betweenness centrality against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-betweenness.png",bbox_inches='tight')

plt.figure(muCFBetweenness)
plt.xlabel(fr'$\mu$')
plt.ylabel('Current flow betweenness centrality')
plt.title(fr'Current flow betweenness centrality against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-CFbetweenness.png",bbox_inches='tight')

plt.figure(muCloseness)
plt.xlabel(fr'$\mu$')
plt.ylabel('Closeness centrality')
plt.title(fr'Closeness centrality against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-closeness.png",bbox_inches='tight')

plt.figure(muClustering)
plt.xlabel(fr'$\mu$')
plt.ylabel('Clustering coefficient')
plt.title(fr'Clustering coefficient against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-clustering.png",bbox_inches='tight')

plt.figure(muGlobEff)
plt.xlabel(fr'$\mu$')
plt.ylabel('Global efficiency')
plt.title(fr'Global efficiency against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-globalEfficiency.png",bbox_inches='tight')

plt.figure(muLocEff)
plt.xlabel(fr'$\mu$')
plt.ylabel('Local efficiency')
plt.title(fr'Local efficiency against $\mu$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/mu-localEfficiency.png",bbox_inches='tight')

plt.figure(avg_degDegree)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Degree centrality')
plt.title(fr'Degree centrality against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-degree.png",bbox_inches='tight')

plt.figure(avg_degBetweenness)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Betweenness centrality')
plt.title(fr'Betweenness centrality against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-betweenness.png",bbox_inches='tight')

plt.figure(avg_degCFBetweenness)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Current flow betweenness centrality')
plt.title(fr'Current flow betweenness centrality against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-CFbetweenness.png",bbox_inches='tight')

plt.figure(avg_degCloseness)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Closeness centrality')
plt.title(fr'Closeness centrality against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-closeness.png",bbox_inches='tight')

plt.figure(avg_degClustering)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Clustering coefficient')
plt.title(fr'Clustering coefficient against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-clustering.png",bbox_inches='tight')

plt.figure(avg_degGlobEff)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Global efficiency')
plt.title(fr'Global efficiency against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-globalEfficiency.png",bbox_inches='tight')

plt.figure(avg_degLocEff)
plt.xlabel(fr'$\langle k \rangle$')
plt.ylabel('Local efficiency')
plt.title(fr'Local efficiency against $\langle k \rangle$-parameter')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/avg_deg-localEfficiency.png",bbox_inches='tight')

plt.figure(clustering_currentFlow)
plt.xlabel('Current flow betweenness')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient against current flow betweenness')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/CFbetweenness-clustering.png",bbox_inches='tight')

plt.figure(clustering_degree)
plt.xlabel('Degree centrality')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient against degree centrality')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/degree-clustering.png",bbox_inches='tight')

plt.figure(degree_currentFlow)
plt.xlabel('Current flow betweenness')
plt.ylabel('Degree centrality')
plt.title('Degree centrality against current flow betweenness')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/CFbetweenness-degree.png",bbox_inches='tight')

plt.figure(degree_betweenness)
plt.xlabel('Betweenness centrality')
plt.ylabel('Degree centrality')
plt.title('Degree centrality against betweenness centrality')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"images/betweenness-degree.png",bbox_inches='tight')

"""Uncomment to decorate 3D plot"""
plt.figure(degree)
ax_degree.set_xlabel(fr'$\mu$')
ax_degree.set_ylabel(fr'$\langle k \rangle$')
ax_degree.set_zlabel('Degree centrality')
pickle.dump((degree, ax_degree), open('images/3D-degreeCentrality.pickle', 'wb'))
plt.savefig(f"images/3D-degreeCentrality.png")

plt.figure(betweenness)
ax_betweenness.set_xlabel(fr'$\mu$')
ax_betweenness.set_ylabel(fr'$\langle k \rangle$')
ax_betweenness.set_zlabel('Betweenness centrality')
pickle.dump((betweenness, ax_betweenness), open('images/3D-betweennessCentrality.pickle', 'wb'))
plt.savefig(f"images/3D-betweennessCentrality.png")

plt.figure(CFbetweenness)
ax_CFbetweenness.set_xlabel(fr'$\mu$')
ax_CFbetweenness.set_ylabel(fr'$\langle k \rangle$')
ax_CFbetweenness.set_zlabel('Current flow betweenness centrality')
pickle.dump((CFbetweenness, ax_CFbetweenness), open('images/3D-CFbetweennessCentrality.pickle', 'wb'))
plt.savefig(f"images/3D-CFbetweennessCentrality.png")

plt.figure(closeness)
ax_closeness.set_xlabel(fr'$\mu$')
ax_closeness.set_ylabel(fr'$\langle k \rangle$')
ax_closeness.set_zlabel('Closeness centrality')
pickle.dump((closeness, ax_closeness), open('images/3D-closenessCentrality.pickle', 'wb'))
plt.savefig(f"images/3D-closenessCentrality.png")

plt.figure(clustering)
ax_clustering.set_xlabel(fr'$\mu$')
ax_clustering.set_ylabel(fr'$\langle k \rangle$')
ax_clustering.set_zlabel('Clustering coefficient')
pickle.dump((clustering, ax_clustering), open('images/3D-clusteringCoefficient.pickle', 'wb'))
plt.savefig(f"images/3D-clusteringCoefficient.png")

plt.figure(globEff)
ax_globEff.set_xlabel(fr'$\mu$')
ax_globEff.set_ylabel(fr'$\langle k \rangle$')
ax_globEff.set_zlabel('Global efficiency')
pickle.dump((globEff, ax_globEff), open('images/3D-globalEfficiency.pickle', 'wb'))
plt.savefig(f"images/3D-globalEfficiency.png")

plt.figure(locEff)
ax_locEff.set_xlabel(fr'$\mu$')
ax_locEff.set_ylabel(fr'$\langle k \rangle$')
ax_locEff.set_zlabel('Local efficiency')
pickle.dump((locEff, ax_locEff), open('images/3D-localEfficiency.pickle', 'wb'))
plt.savefig(f"images/3D-localEfficiency.png")
""""Uncomment until here"""