"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import numpy as np, pandas as pd, seaborn as sns


if __name__ == "__main__":

    """Scatterplot all datapoints """
    # efficiency = 'global'
    # fig,ax = plt.subplots()

    # if efficiency == 'global':
    #     column = 'globalEff'
    # elif efficiency == 'local':
    #     column = 'localEff'

    # data = pd.read_csv('relationData-complete-Atlas.tsv', sep='\t')

    # for n in reversed(data['nodes'].unique()):
    #     # pre-determine colormap
    #     if  n == 2:
    #         color = "tab:blue"
    #     elif n == 3:
    #         color = "tab:orange"
    #     elif n == 4:
    #         color = "tab:green"
    #     elif n == 5:
    #         color = "tab:red"
    #     elif n == 6:
    #         color = "tab:purple"
    #     elif n == 7:
    #         color = "tab:pink"

    #     indices = np.where(data['nodes'] == n)[0]
    #     ax.scatter(data[column].iloc[indices],data['nMessages'].iloc[indices],color=color,alpha=0.3)

    #     # if n != 2:
    #     #     p = np.poly1d(np.polyfit(data[column].iloc[indices],data['nMessages'].iloc[indices],3))
    #     #     t = np.linspace(min(data[column]), max(data[column]), 250)
    #     #     print(p)
    #     #     ax.plot(t,p(t),color)


    # handles = [
    #     plt.scatter([], [], color=c, label=l)
    #     for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
    # ]

    # ax.legend(handles=handles)
    # ax.set_xlabel(f"{efficiency.capitalize()} efficiency (metric)")
    # ax.set_ylabel("Convergence rate (number of messages)")
    # ax.set_title(f"Relation between structural and operational efficiency ({efficiency})")
    # plt.show()

    # fig.savefig(f"images/relations/relation-{efficiency}-convergence-mean.png",bbox_inches='tight')
    # plt.close(fig)
    """Scatterplot all datapoints"""


    """Scatterplot mean of datapoints per graph"""
    # efficiency = 'local'
    # fig,ax = plt.subplots()

    # if efficiency == 'global':
    #     column = 'globalEff'
    # elif efficiency == 'local':
    #     column = 'localEff'

    # data = pd.read_csv('relationData-complete-Atlas.tsv', sep='\t')

    # meanData = data.groupby('index').agg({'index':'mean',
    #                                       'nodes':'mean',
    #                                       'degree':'mean',
    #                                       'betweenness':'mean',
    #                                       'closeness':'mean',
    #                                       'clustering':'mean',
    #                                       'globalEff':'mean',
    #                                       'localEff':'mean',
    #                                       'nMessages':'mean'})
    
    # for i, index in enumerate(reversed(data['index'].unique())):
    #     n = data['nodes'].iloc[len(data)-i*100-1]

    #     # pre-determine colormap
    #     if  n == 2:
    #         color = "tab:blue"
    #     elif n == 3:
    #         color = "tab:orange"
    #     elif n == 4:
    #         color = "tab:green"
    #     elif n == 5:
    #         color = "tab:red"
    #     elif n == 6:
    #         color = "tab:purple"
    #     elif n == 7:
    #         color = "tab:pink"

    #     indices = np.where(data['index'] == index)[0]

    #     ax.scatter(np.mean(data[column].iloc[indices]),np.mean(data['nMessages'].iloc[indices]),color=color, alpha=0.5)

    # for n in reversed(data['nodes'].unique()):
        
    #     # pre-determine colormap
    #     if  n == 2:
    #         color = "tab:blue"
    #     elif n == 3:
    #         color = "tab:orange"
    #     elif n == 4:
    #         color = "tab:green"
    #     elif n == 5:
    #         color = "tab:red"
    #     elif n == 6:
    #         color = "tab:purple"
    #     elif n == 7:
    #         color = "tab:pink"

    #     indices = np.where(meanData['nodes'] == n)[0]

    #     # """Uncomment for polynomial"""
    #     # if n != 2:
    #     #     p = np.poly1d(np.polyfit(meanData[column].iloc[indices],meanData['nMessages'].iloc[indices],3))
    #     #     t = np.linspace(min(meanData[column]), max(meanData[column]), 250)
    #     #     print(p)
    #     #     ax.plot(t,p(t),color)
    #     # """Uncomment for polynomial"""
    
    # handles = [
    #     plt.scatter([], [], color=c, label=l)
    #     for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
    # ]

    # ax.legend(handles=handles)
    # ax.set_xlabel(f"{efficiency.capitalize()} efficiency")
    # # ax.set_xlabel(f"Clustering coefficient")
    # ax.set_ylabel("Convergence rate (number of messages)")
    # ax.set_title(f"Relation between structural and operational efficiency ({efficiency})")
    # # ax.set_title(f"Relation between clustering coefficient and consensus formation")
    # plt.show()

    # fig.savefig(f"images/relations/relation-{efficiency}-convergence-mean.png",bbox_inches='tight')
    # # fig.savefig(f"images/relations/relation-clustering-convergence-mean.png",bbox_inches='tight')
    # plt.close(fig)
    """Scatterplot mean of datapoints per graph"""

    """Violinplot"""
    data = pd.read_csv('relationData-complete-Atlas.tsv', sep='\t')
    sns.violinplot(data=data,x=data['globalEff'],y=data['nMessages'])
    plt.show()
    """Violinplot"""
