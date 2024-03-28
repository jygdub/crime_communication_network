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
    # ax.set_xlabel(f"{efficiency.capitalize()} efficiency (metric)")
    # ax.set_ylabel("Convergence rate (number of messages)")
    # ax.set_title(f"Relation between structural and operational efficiency ({efficiency})")
    # plt.show()

    # fig.savefig(f"images/relations/relation-{efficiency}-convergence-mean.png",bbox_inches='tight')
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

    # for metric in ['degree','betweenness','closeness','clustering','global','local']:
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

    #         indices = np.where(meanData['nodes'] == n)[0]

    #         # fit polynomial if desired
    #         if n != 2 and draw_polynomial:
    #             p = np.poly1d(np.polyfit(meanData[column].iloc[indices],meanData['nMessages'].iloc[indices],3))
    #             t = np.linspace(min(meanData[column]), max(meanData[column]), 250)
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
    #         fig.savefig(f"{images_path}/relation-{metric}-convergence-polynomial.png",bbox_inches='tight')
    #     else:
    #         fig.savefig(f"{images_path}/relation-{metric}-convergence-mean.png",bbox_inches='tight')
    #     plt.close(fig)
    """Scatterplot mean of datapoints per graph"""

    """Comparing model parameter settings """

    # NOTE: Set according to desired output figure #
    draw_polynomial = True
    # metric = 'global'
    # n = 3
    changing = 'alpha'
    ################################################

    for metric,n in product(['degree','betweenness','closeness','clustering','global','local'],[3,4,5,6,7]):
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

        fig,ax = plt.subplots()

        if changing == 'beta':
            values = ['00','25','50']
        elif changing == 'alpha':
            values = ['1_00','0_75','0_50']

        for x in values:
            # # pre-determine colormap
            # if  x == '00':
            #     color = "tab:blue"
            # elif x == '25':
            #     color = "tab:orange"
            # elif x == '50':
            #     color = "tab:green"

            if changing == 'beta':
                data = pd.read_csv(f'data/meanRelationData-alpha1_00-beta0_{x}-Atlas.tsv', sep='\t')
            elif changing == 'alpha':
                data = pd.read_csv(f'data/meanRelationData-alpha{x}-beta0_00-Atlas.tsv', sep='\t')

            # fit polynomial if desired
            p = np.poly1d(np.polyfit(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n],3))
            t = np.linspace(min(data[column]), max(data[column]), 250)

            if changing == 'beta':
                ax.plot(t,p(t),label=fr"$\beta$={x.replace('_','.')}")
                # ax.scatter(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n],alpha=0.2, label=fr'$\beta$=0.{x}')
            elif changing == 'alpha':
                ax.plot(t,p(t),label=fr"$\alpha$={x.replace('_','.')}")
                # ax.scatter(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n], alpha=0.2, label=fr'$\alpha$={x.replace('_','.')}')

        # handles = [
        #     plt.scatter([], [], color=c, label=l)
        #     for c, l in zip("tab:blue tab:orange tab:green".split(), fr"$\beta$=0.00 $\beta$=0.25 $\beta$=0.50".split())
        # ]

        ax.legend(bbox_to_anchor=(1,1))

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency")
            ax.set_title(f"Relation between structural and operational efficiency ({metric})")
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient")
            ax.set_title(f"Relation between {metric} coefficient and consensus formation")
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality")
            ax.set_title(f"Relation between {metric} centrality and consensus formation")
        ax.set_ylabel("Convergence rate (number of messages)")

        # plt.show()

        if changing == 'beta':
            ax.set_title(fr'$\alpha$=1.00 & n={n}')

            if draw_polynomial:
                fig.savefig(f"images/relations/varyingBeta-alpha1_00-{metric}-n={n}-convergence-polynomial.png",bbox_inches='tight')
            else:
                fig.savefig(f"images/relations/varyingBeta-alpha1_00-{metric}-n={n}-convergence-mean.png",bbox_inches='tight')
        
        elif changing == 'alpha':
            ax.set_title(fr'$\beta$=0.00 & n={n}')

            if draw_polynomial:
                fig.savefig(f"images/relations/varyingAlpha-beta0_00-{metric}-n={n}-convergence-polynomial.png",bbox_inches='tight')
            else:
                fig.savefig(f"images/relations/varyingAlpha-beat0_00-{metric}-n={n}-convergence-mean.png",bbox_inches='tight')
        
        plt.close(fig)


    """Comparing model parameter settings """

    """Violinplot"""
    # data = pd.read_csv('relationData-complete-Atlas.tsv', sep='\t')
    # sns.violinplot(data=data,x=data['globalEff'],y=data['nMessages'])
    # plt.show()
    """Violinplot"""
