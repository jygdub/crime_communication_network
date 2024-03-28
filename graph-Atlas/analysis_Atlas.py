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

    # draw_polynomial = False
    # alpha = '1_00'
    # beta = '0_00'
    # settings = f'alpha{alpha}-beta{beta}'

    # for metric in tqdm(['degree','betweenness','CFbetweenness','closeness','clustering','global','local']):
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

    #     data = pd.read_csv(f'data/meanRelationData-{settings}-Atlas.tsv', sep='\t')

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
    #         ax.scatter(data[column].iloc[indices],data['nMessages'].iloc[indices],color=color,alpha=0.3)

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
    #     elif coefficient:
    #         ax.set_xlabel(f"{metric.capitalize()} coefficient")
    #     else:
    #         ax.set_xlabel(f"{metric.capitalize()} centrality")

    #     ax.set_ylabel("Convergence rate (number of messages)")
    #     ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}')
    #     # plt.show()

    #     if draw_polynomial:
    #         fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-polynomial.png",bbox_inches='tight')
    #     else:
    #         fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-scatter.png",bbox_inches='tight')

    #     plt.close(fig)

    """Scatterplot all datapoints"""


    """Scatterplot mean of datapoints per graph"""

    # #######################################################
    # # NOTE: Set simulation settings to save appropriately #
    # draw_polynomial = True   
    # alpha = '1_00'
    # beta = '0_50'                                                                     
    # #######################################################

    # settings = f'alpha{alpha}-beta{beta}'                       
    # images_path = f'images/relations/{settings}'  

    # # plots for all 7 metrics
    # for metric in tqdm(['degree','betweenness','CFbetweenness','closeness','clustering','global','local']):
        
    #     # set initial booleans
    #     efficiency = False
    #     coefficient = False

    #     # open figure
    #     fig,ax = plt.subplots()

    #     # conditional column assignment and boolean flip
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

    #     # read data for corresponding parameter settings
    #     data = pd.read_csv(f'data/meanRelationData-{settings}-Atlas.tsv', sep='\t')
    #     # print(data)
        
    #     # scatterplot all mean convergences per graph size (from n=7 to n=2)
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


    #     # fit 3rd polynomial to data per graph size (from n=7 to n=2)
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
    #     elif coefficient:
    #         ax.set_xlabel(f"{metric.capitalize()} coefficient")
    #     elif metric == 'CFbetweenness':
    #         ax.set_xlabel(f"Current flow betweenness centrality")
    #     else:
    #         ax.set_xlabel(f"{metric.capitalize()} centrality")
    #     ax.set_ylabel("Convergence rate (number of messages)")
    #     ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}')
        
    #     # plt.show()

    #     if draw_polynomial:
    #         fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-polynomial.png",bbox_inches='tight')
    #     else:
    #         fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-scatter.png",bbox_inches='tight')
    #     plt.close(fig)

    """Scatterplot mean of datapoints per graph"""


    """Comparing model parameter settings using polynomial fit through data"""

    # # NOTE: Set according to desired output figure #
    # changing = 'alpha'
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
    #     elif metric == 'CFbetweenness':
    #         ax.set_xlabel(f"Current flow betweenness centrality")
    #         ax.set_title(f"Relation between current flow betweenness centrality and consensus formation")
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

    #         # print(alpha,beta)

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
    #     elif metric == 'CFbetweenness':
    #         ax.set_xlabel(f"Current flow betweenness centrality")
    #     else:
    #         ax.set_xlabel(f"{metric.capitalize()} centrality")

    #     ax.set_ylabel("Convergence rate (number of messages)")
    #     ax.set_title(f"Varying alpha and beta parameters (n={n})")

    #     fig.savefig(f"images/relations/combined-parameters-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
    
    #     plt.close(fig)

    """Combined comparison for all model parameter settings"""


    """Violinplot"""
    #######################################################
    # NOTE: Set simulation settings to save appropriately #
    alpha = '1_00'
    beta = '0_50'                                                                     
    #######################################################

    settings = f'alpha{alpha}-beta{beta}'                       
    images_path = f'images/relations/{settings}/all-datapoints'  

    data = pd.read_csv(f'data/relationData-alpha{alpha}-beta{beta}-Atlas.tsv', sep='\t')

    # for metric,n in product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7]):
    for metric in ['degree','betweenness','CFbetweenness','closeness','clustering','global','local']:
        efficiency = False
        coefficient = False

        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        nData = data #[data['nodes']==n]

        fig,ax = plt.subplots()

        for TH in np.arange(0.0,1.0,0.1):
            subset = nData[nData[column]<=TH+0.1][nData[column]>TH]

            if not subset.empty:
                ax.scatter(subset[column],subset['nMessages'],color='lightgrey',alpha=0.1)
                ax.violinplot(subset['nMessages'],positions=[TH+0.05],widths=[0.1])
            

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency")
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient")
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality")
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality")

        ax.set_ylabel("Convergence rate (number of messages)")
        # ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}')
        ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} for all n's")

        # plt.show()

        # fig.savefig(f"{images_path}/distribution-n={n}-convergence-per-{metric}-violin.png",bbox_inches='tight')
        fig.savefig(f"{images_path}/distribution-allN-convergence-per-{metric}-violin.png",bbox_inches='tight')
        plt.close(fig)

    """Violinplot"""
