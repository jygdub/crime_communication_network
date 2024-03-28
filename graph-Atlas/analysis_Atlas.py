"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np, pandas as pd, seaborn as sns


if __name__ == "__main__":

    """Scatterplot all datapoints """

    # draw_polynomial = True
    # metric = 'CFbetweenness'
    # settings = 'alpha1_00-beta0_00'

    # efficiency = False
    # coefficient = False

    # if metric == 'global':
    #     column = 'globalEff'
    #     efficiency = True
    # elif metric == 'local':
    #     column = 'localEff'
    #     efficiency = True
    # elif metric == 'clustering':
    #     column = 'clustering'
    #     coefficient = True
    # else:
    #     column = metric

    # fig,ax = plt.subplots()

    # data = pd.read_csv(f'data/meanRelationData-{settings}-Atlas.tsv', sep='\t')

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

    #     if n != 2 and draw_polynomial:
    #         p = np.poly1d(np.polyfit(data[column].iloc[indices],data['nMessages'].iloc[indices],3))
    #         t = np.linspace(min(data[column]), max(data[column]), 250)
    #         print(p)
    #         ax.plot(t,p(t),color)

    # handles = [
    #     plt.scatter([], [], color=c, label=l)
    #     for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
    # ]


    # ax.legend(handles=handles)
    # if efficiency:
    #     ax.set_xlabel(f"{metric.capitalize()} efficiency")
    #     ax.set_title(f"Relation between structural and operational efficiency ({metric})")
    # elif coefficient:
    #     ax.set_xlabel(f"{metric.capitalize()} coefficient")
    #     ax.set_title(f"Relation between {metric} coefficient and consensus formation")
    # else:
    #     ax.set_xlabel(f"{metric.capitalize()} centrality")
    #     ax.set_title(f"Relation between {metric} centrality and consensus formation")
    # ax.set_ylabel("Convergence rate (number of messages)")
    
    # # plt.show()

    # if draw_polynomial:
    #     fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-polynomial.png",bbox_inches='tight')
    # else:
    #     fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-scatter.png",bbox_inches='tight')

    # plt.close(fig)
    """Scatterplot all datapoints"""


    """Scatterplot mean of datapoints per graph"""
    # #######################################################
    # # NOTE: Set simulation settings to save appropriately #
    # draw_polynomial = True                                
    # metric = 'betweenness'                                      
    # settings = 'alpha0_50-beta0_00'                       
    # images_path = f'images/relations/{settings}'          
    # #######################################################

    # for metric in ['degree','betweenness','CFbetweenness','closeness','clustering','global','local']:
    #     efficiency = False
    #     coefficient = False
    #     fig,ax = plt.subplots()

    #     if metric == 'global':
    #         column = 'globalEff'
    #         efficiency = True
    #     elif metric == 'local':
    #         column = 'localEff'
    #         efficiency = True
    #     elif metric == 'clustering':
    #         column = 'clustering'
    #         coefficient = True
    #     else:
    #         column = metric

    #     data = pd.read_csv(f'data/meanRelationData-{settings}-Atlas.tsv', sep='\t')
    #     # print(data)
        
    #     for i, index in enumerate(reversed(data['index'].unique())):
    #         n = data['nodes'].iloc[len(data)-i*100-1]

    #         # pre-determine colormap
    #         if  n == 2:
    #             color = "tab:blue"
    #         elif n == 3:
    #             color = "tab:orange"
    #         elif n == 4:
    #             color = "tab:green"
    #         elif n == 5:
    #             color = "tab:red"
    #         elif n == 6:
    #             color = "tab:purple"
    #         elif n == 7:
    #             color = "tab:pink"

    #         indices = np.where(data['index'] == index)[0]

    #         ax.scatter(np.mean(data[column].iloc[indices]),np.mean(data['nMessages'].iloc[indices]),color=color, alpha=0.5)

    #     for n in reversed(data['nodes'].unique()):
            
    #         # pre-determine colormap
    #         if  n == 2:
    #             color = "tab:blue"
    #         elif n == 3:
    #             color = "tab:orange"
    #         elif n == 4:
    #             color = "tab:green"
    #         elif n == 5:
    #             color = "tab:red"
    #         elif n == 6:
    #             color = "tab:purple"
    #         elif n == 7:
    #             color = "tab:pink"

    #         indices = np.where(data['nodes'] == n)[0]

    #         # fit polynomial if desired
    #         if n != 2 and draw_polynomial:
    #             p = np.poly1d(np.polyfit(data[column].iloc[indices],data['nMessages'].iloc[indices],3))
    #             t = np.linspace(min(data[column]), max(data[column]), 250)
    #             ax.plot(t,p(t),color)
        
    #     handles = [
    #         plt.scatter([], [], color=c, label=l)
    #         for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
    #     ]

    #     ax.legend(handles=handles)
    #     if efficiency:
    #         ax.set_xlabel(f"{metric.capitalize()} efficiency")
    #         ax.set_title(f"Relation between structural and operational efficiency ({metric})")
    #     elif coefficient:
    #         ax.set_xlabel(f"{metric.capitalize()} coefficient")
    #         ax.set_title(f"Relation between {metric} coefficient and consensus formation")
    #     else:
    #         ax.set_xlabel(f"{metric.capitalize()} centrality")
    #         ax.set_title(f"Relation between {metric} centrality and consensus formation")
    #     ax.set_ylabel("Convergence rate (number of messages)")
        
    #     # plt.show()

    #     if draw_polynomial:
    #         fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-polynomial.png",bbox_inches='tight')
    #     else:
    #         fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-scatter.png",bbox_inches='tight')
    #     plt.close(fig)
    """Scatterplot mean of datapoints per graph"""

    """Comparing model parameter settings using polynomial fit through data"""

    # # NOTE: Set according to desired output figure #
    # changing = 'beta'
    # ################################################

    # for metric,n in product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7]):
    #     efficiency = False
    #     coefficient = False

    #     if metric == 'global':
    #         column = 'globalEff'
    #         efficiency = True
    #     elif metric == 'local':
    #         column = 'localEff'
    #         efficiency = True
    #     elif metric == 'clustering':
    #         column = 'clustering'
    #         coefficient = True
    #     else:
    #         column = metric

    #     fig,ax = plt.subplots()

    #     if changing == 'beta':
    #         values = ['00','25','50']
    #     elif changing == 'alpha':
    #         values = ['1_00','0_75','0_50']

    #     for x in values:

    #         if changing == 'beta':
    #             data = pd.read_csv(f'data/meanRelationData-alpha1_00-beta0_{x}-Atlas.tsv', sep='\t')
    #         elif changing == 'alpha':
    #             data = pd.read_csv(f'data/meanRelationData-alpha{x}-beta0_00-Atlas.tsv', sep='\t')

    #         # fit polynomial if desired
    #         p = np.poly1d(np.polyfit(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n],3))
    #         t = np.linspace(min(data[column]), max(data[column]), 250)

    #         if changing == 'beta':
    #             ax.plot(t,p(t),label=fr"$\beta$={x.replace('_','.')}")
    #         elif changing == 'alpha':
    #             ax.plot(t,p(t),label=fr"$\alpha$={x.replace('_','.')}")

    #     ax.legend(bbox_to_anchor=(1,1))

    #     if efficiency:
    #         ax.set_xlabel(f"{metric.capitalize()} efficiency")
    #         ax.set_title(f"Relation between structural and operational efficiency ({metric})")
    #     elif coefficient:
    #         ax.set_xlabel(f"{metric.capitalize()} coefficient")
    #         ax.set_title(f"Relation between {metric} coefficient and consensus formation")
    #     else:
    #         ax.set_xlabel(f"{metric.capitalize()} centrality")
    #         ax.set_title(f"Relation between {metric} centrality and consensus formation")
    #     ax.set_ylabel("Convergence rate (number of messages)")

    #     # plt.show()

    #     if changing == 'beta':
    #         ax.set_title(fr'$\alpha$=1.00 & n={n}')

    #         fig.savefig(f"images/relations/varyingBeta-alpha1_00-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
       
    #     elif changing == 'alpha':
    #         ax.set_title(fr'$\beta$=0.00 & n={n}')

    #         fig.savefig(f"images/relations/varyingAlpha-beta0_00-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
        
    #     plt.close(fig)
    """Comparing model parameter settings using polynomial fit through data"""

    """Combined comparison for all model parameter settings"""
    # for metric,n in product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7]):

    #     efficiency = False
    #     coefficient = False

    #     if metric == 'global':
    #         column = 'globalEff'
    #         efficiency = True
    #     elif metric == 'local':
    #         column = 'localEff'
    #         efficiency = True
    #     elif metric == 'clustering':
    #         column = 'clustering'
    #         coefficient = True
    #     else:
    #         column = metric

    #     fig,ax = plt.subplots()

    #     for alpha, beta in product(['1_00','0_75','0_50'],['0_00','0_25','0_50']):
    #         if alpha == '0_75' and (beta == '0_25' or beta == '0_50'):
    #             continue
    #         elif alpha == '0_50' and (beta == '0_25' or beta == '0_50'):
    #             continue

    #         print(alpha,beta)

    #         data = pd.read_csv(f'data/meanRelationData-alpha{alpha}-beta{beta}-Atlas.tsv', sep='\t')

    #         # fit polynomial if desired
    #         p = np.poly1d(np.polyfit(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n],3))
    #         t = np.linspace(min(data[column]), max(data[column]), 250)

    #         ax.plot(t,p(t),label=fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}")

    #     ax.legend(bbox_to_anchor=(1,1))

    #     if efficiency:
    #         ax.set_xlabel(f"{metric.capitalize()} efficiency")
    #     elif coefficient:
    #         ax.set_xlabel(f"{metric.capitalize()} coefficient")
    #     else:
    #         ax.set_xlabel(f"{metric.capitalize()} centrality")

    #     ax.set_ylabel("Convergence rate (number of messages)")
    #     ax.set_title(f"Varying alpha and beta parameters (n={n})")

    #     fig.savefig(f"images/relations/combined-parameters-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
    
    #     plt.close(fig)
    """Combined comparison for all model parameter settings"""


    """Violinplot"""
    # n = 7
    # metric = 'globalEff'
    # data = pd.read_csv(f'data/meanRelationData-alpha1_00-beta0_00-Atlas.tsv', sep='\t')

    # subset = data[data['nodes']==n]

    # sns.violinplot(data=subset,x=subset[metric],y=subset['nMessages'])
    # plt.show()

    
    # subset = data[['globalEff','nMessages']][data['nodes']==7][data['globalEff']<0.7][data['globalEff']>0.6]

    # lower_bound = 0.5
    # upper_bound = 0.6

    # while upper_bound < 0.7:
    #     subset = data[['globalEff','nMessages']][data['nodes']==n][data['globalEff']<upper_bound][data['globalEff']>lower_bound]
    #     sns.violinplot(data=subset,x=subset['globalEff'],y=subset['nMessages'])
    #     lower_bound += 0.1
    #     upper_bound += 0.1
    # plt.show()
    """Violinplot"""
