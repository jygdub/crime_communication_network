"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np, pandas as pd, seaborn as sns


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
# images_path = f'images/relations/{settings}/averaged-convergence'  

# # plots for all 7 metrics
# for metric in ['CFbetweenness']: #tqdm(['degree','betweenness','CFbetweenness','closeness','clustering','global','local']):
    
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


def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

"""Violinplot per parameter configuration per metric (per graph size OR all sizes)"""
def violin_per_params(alpha: float, beta: float):
    """ 
    Violinplot distribution of convergence rates per metric per bin (size=0.1).

    Parameters:
    - alpha (float): alpha noise
    - beta (float): beta noise
    """

    settings = f'alpha{alpha}-beta{beta}'                       
    images_path = f'images/relations/{settings}/all-datapoints'  

    #######################################################
    # NOTE: CHOOSE CORRECT DATA
    data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')
    # NOTE: DROP FIRST 100 ROWS IF EXCLUDING N=2 GRAPH SIZE
    data = data.drop(range(0,100))
    #######################################################

    #######################################################
    # NOTE: USE CORRECT FOR-LOOP
    # for metric,n in tqdm(product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7])):
    for metric in tqdm(['degree','betweenness','CFbetweenness','closeness','clustering','global','local']):
    #######################################################
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

        #######################################################
        # NOTE: USE CORRECT DATA (all n's comment squared brackets)
        nData = data
        # nData = data[data['nodes']==n] 
        #######################################################

        fig,ax = plt.subplots(figsize=(13,8))
        MAX_DENSITY = []

        for TH in np.arange(0.0,1.0,0.1):

            if TH == 0.9:
                subset = nData[nData[column]<=TH+0.1][nData[column]>=TH]
            else:
                subset = nData[nData[column]<TH+0.1][nData[column]>=TH]

            if not subset.empty:

                # keep track of higest probability density
                PDF = plt.figure()
                density = subset['nMessages'].plot.kde().get_lines()[0].get_xydata()
                MAX_DENSITY.append(density[np.argmax(density[:,1])][0])
                plt.close(PDF)

                ax.scatter(subset[column],subset['nMessages'],color='lightgrey',alpha=0.1)
                plots = ax.violinplot(subset['nMessages'],positions=[TH+0.05],widths=[0.1])

                for vp in plots['bodies']:
                    vp.set_facecolor("tab:blue")
                    vp.set_edgecolor("black")
                
                for i, partname in enumerate(('cbars', 'cmins', 'cmaxes')): 
                    vp = plots[partname]
                    vp.set_edgecolor("tab:blue")
                    vp.set_linewidth(1)


        # define x-axis
        MIN_TH = my_floor(min(nData[column]),1)
        Xaxis = np.linspace(MIN_TH+0.05,MIN_TH+0.05+(len(MAX_DENSITY)-1.)*0.1,len(MAX_DENSITY))

        #######################################################
        # NOTE: FIT LINEAR LINE THROUGH MAXIMUM PROBABILITY DENSITY
        # coef_lin = np.polyfit(Xaxis, MAX_DENSITY, 1)
        # poly1d_fn_lin = np.poly1d(coef_lin) 

        # plot maximum probability density
        # ax.plot(Xaxis, MAX_DENSITY, 'ro', label='Maximum probability density')
        # ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), poly1d_fn_lin(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05)), 'g--', label=f'{round(coef_lin[0],2)} x + {round(coef_lin[1],2)}')
        
        # """NOTE: FITTING EXPONENTIAL"""
        coef_exp = np.polyfit(Xaxis, np.log(MAX_DENSITY), 1)
        poly1d_fn_exp = np.poly1d(coef_exp) 

        # plot maximum probability density
        ax.plot(Xaxis, MAX_DENSITY, 'ro', label='Maximum probability density')
        ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), 
                np.exp(poly1d_fn_exp(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05))), 
                'k--', 
                label=f'ln(C)={round(coef_exp[0],2)} * GE + {round(coef_exp[1],2)}')
        #######################################################

        plt.legend(bbox_to_anchor=(1,1),fontsize=14)

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency",fontsize=16)
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient",fontsize=16)
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality",fontsize=16)
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality",fontsize=16)

        ax.set_ylabel("Convergence rate (number of messages)",fontsize=16)

        #######################################################
        # NOTE: SETTING CORRECT TITLE 
        # ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}',fontsize=16)
        ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}}",fontsize=16)
        #######################################################

        ax.set_yscale("log") # NOTE: SETTING LOG SCALE -> CHANGE FILENAME
        plt.tick_params(axis='both', which='major', labelsize=16)

        # plt.show()
        #######################################################
        # NOTE: CHANGE FILENAME ACCORDINGLY 
        # (add LOG if log-scale; add lineFit for linear line or expFit for exponential
        # change allN to withoutN=2 if applied)
        # fig.savefig(f"{images_path}/expFit-LOGdistribution-n={n}-convergence-per-{metric}-violin.png",bbox_inches='tight')
        fig.savefig(f"{images_path}/expFit-LOGdistribution-withoutN=2-convergence-per-{metric}-violin.png",bbox_inches='tight')
        #######################################################

        plt.close(fig)
"""Violinplot per parameter configuration per metric (per graph size OR all sizes)"""

"""Histogram distribution of a single violin"""
def hist_per_violin(alpha: float, beta: float):
    """ 
    Histogram distribution of convergence rates per metric per bin/violin (size=0.1).

    Parameters:
    - alpha (float): alpha noise
    - beta (float): beta noise
    """

    settings = f'alpha{alpha}-beta{beta}'                       
    images_path = f'images/relations/{settings}/all-datapoints'  

    data = pd.read_csv(f'data/relationData-alpha{alpha}-beta{beta}-Atlas.tsv', sep='\t')
    ######################################################
    # NOTE: USE CORRECT FOR-LOOP
    # for metric,n in product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7]):
    for metric in ['degree','betweenness','CFbetweenness','closeness','clustering','global','local']:
    #######################################################
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

        #######################################################
        # NOTE: USE CORRECT DATA
        nData = data
        # nData = data[data['nodes']==n] 
        #######################################################

        counter = 0

        for TH in np.arange(0.0,1.0,0.1):
            if TH == 0.9:
                subset = nData[nData[column]<=TH+0.1][nData[column]>=TH]
            else:
                subset = nData[nData[column]<TH+0.1][nData[column]>=TH]

                if subset.empty:
                    continue

                counter += 1
                fig,ax = plt.subplots(figsize=(13,8))
                ax.hist(subset['nMessages'],bins=100)

                if efficiency:
                    ax.set_xlabel(f"{metric.capitalize()} efficiency",fontsize=16)
                elif coefficient:
                    ax.set_xlabel(f"{metric.capitalize()} coefficient",fontsize=16)
                elif metric == 'CFbetweenness':
                    ax.set_xlabel(f"Current flow betweenness centrality",fontsize=16)
                else:
                    ax.set_xlabel(f"{metric.capitalize()} centrality",fontsize=16)

                ax.set_ylabel("Convergence rate (number of messages)",fontsize=16)

                #######################################################
                # NOTE: SETTING CORRECT TITLE
                # ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n} | violin {counter}',fontsize=16)
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} for all n's | violin {counter}",fontsize=16)
                #######################################################
                        
                plt.tick_params(axis="both",which="major",labelsize=16)
                # plt.show()

                #######################################################
                # NOTE: CHANGE FILENAME ACCORDINGLY
                # fig.savefig(f"{images_path}/histograms-per-violin/histDistribution-n={n}-convergence-per-{metric}-violin{counter}.png",bbox_inches='tight')
                fig.savefig(f"{images_path}/histograms-per-violin/histDistribution-allN-convergence-per-{metric}-violin{counter}.png",bbox_inches='tight')
                #######################################################

                plt.close(fig)
"""Histogram distribution of a single violin"""


"""Violinplot to compare effect of noise (per bin)"""
def violin_noiseEffect(fixed_param: str, varying_param: list, A: bool, metric: str):
    """ 
    Violinplot distribution of convergence rates per metric per bin (size=0.1).

    Parameters:
    - fixed_param (str): value of fixed parameter (alpha or beta)
    - varying_param (list): list of values of varying paramter (alpha or beta)
    - A (bool): True indicates varying alpha; False indicates varying beta
    - metric (str): network metric
    """

    alphas = []
    betas = []
    alpha = ''
    beta = ''

    if A:
        alphas = varying_param
        beta = fixed_param
    else:
        betas = varying_param
        alpha = fixed_param


    print(metric)
    thresholds = []

    if metric == 'degree':
        thresholds = np.linspace(0.2,0.9,8)  # degree
    elif metric == 'betweenness' or metric == 'CFbetweenness':
        thresholds = np.linspace(0.0,0.3,4)  # betweenness & CF betweenness | NOTE NOTE CFBETWEENNESS NOT WORKING ???
    elif metric == 'closeness':   
        thresholds = np.linspace(0.3,0.9,7)  # closeness
    elif metric == 'global':
        thresholds = np.linspace(0.5,0.9,5) # global efficiency
    elif metric == 'local' or metric == 'clustering':
        thresholds = np.linspace(0.0,0.9,10)  # clustering & local efficiency

    images_path = 'images/relations'

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

    counter = 0

    for TH in tqdm(thresholds):

        fig,ax = plt.subplots(figsize=(13,8))
        counter += 1

        MAX_DENSITY = []

        ####################################################### 
        # NOTE NOTE: CHOOSE CORRECT FOR-LOOP
        # for i, beta in enumerate(betas):
        for i, alpha in enumerate(alphas):
        #######################################################

            settings = f'alpha{alpha}-beta{beta}'                       

            #######################################################
            # NOTE: CHOOSE CORRECT DATA
            data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')
            # NOTE: DROP FIRST 100 ROWS IF EXCLUDING N=2 GRAPH SIZE
            data = data.drop(range(0,100))
            #######################################################

            #######################################################
            # NOTE: USE CORRECT DATA (ALL VS. PER N)
            nData = data
            # nData = data[data['nodes']==n] 
            #######################################################

            if TH == 0.9:
                subset = nData[nData[column]<=TH+0.1][nData[column]>=TH]
            else:
                subset = nData[nData[column]<TH+0.1][nData[column]>=TH]
        
            # keep track of higest probability density
            PDF = plt.figure()
            density = subset['nMessages'].plot.kde().get_lines()[0].get_xydata()
            MAX_DENSITY.append(density[np.argmax(density[:,1])][0])
            # plt.show()
            plt.close(PDF)

            ax.violinplot(subset['nMessages'],positions=[np.linspace(0.,4.,5)[i+1]])

        labels = []

        # for varying alphas, else varying betas
        if A:

            labels = [fr"$\alpha$=1.00", fr"$\alpha$=0.75", fr"$\alpha$=0.50"]

            if efficiency:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | {metric.capitalize()} efficiency {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif coefficient:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | {metric.capitalize()} coefficient {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif metric == 'CFbetweenness':
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | Current flow betweenness centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            else:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | {metric.capitalize()} centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
        else:

            labels = [fr"$\beta$=0.00", fr"$\beta$=0.25", fr"$\beta$=0.50"]        

            if efficiency:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | {metric.capitalize()} efficiency {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif coefficient:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | {metric.capitalize()} coefficient {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif metric == 'CFbetweenness':
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | Current flow betweenness centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            else:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | {metric.capitalize()} centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)

        Xaxis = np.arange(1, len(labels) + 1)

        #######################################################
        # # NOTE: FIT LINEAR LINE THROUGH MAXIMUM PROBABILITY DENSITY
        # coef = np.polyfit(Xaxis, MAX_DENSITY, 1)
        # poly1d_fn = np.poly1d(coef) 

        # # plot maximum probability density
        # ax.plot(Xaxis, MAX_DENSITY, 'ro', label='Maximum probability density')
        # ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), poly1d_fn(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05)), 'g--', label=f'{round(coef[0],2)} x + {round(coef[1],2)}')


        # """NOTE: FITTING EXPONENTIAL"""
        coef_exp = np.polyfit(Xaxis, np.log(MAX_DENSITY), 1)
        poly1d_fn_exp = np.poly1d(coef_exp) 

        # plot maximum probability density
        ax.plot(Xaxis, MAX_DENSITY, 'ro', label='Maximum probability density')
        ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), np.exp(poly1d_fn_exp(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05))), 'k--', label=f'ln(y)={round(coef_exp[0],2)} x + {round(coef_exp[1],2)}')
        #######################################################

        plt.legend(bbox_to_anchor=(1,1),fontsize=14)

        ax.set_xticks(Xaxis, labels=labels)
        ax.set_xlim(0., len(labels) + 1.)

        ax.set_ylabel("Convergence rate (number of messages)",fontsize=16)
        ax.set_yscale("log")

        plt.tick_params(axis="both",which="major",labelsize=16)

        plt.show()
        #######################################################
        # NOTE: add LOG for regular log scale; add linePlot for linear fit; change allN to withoutN=2 if applied
        if A:
            # fig.savefig(f"{images_path}/linePlot-noiseEffect-varyingAlpha-allN-{metric}-violin{counter}.png",bbox_inches='tight')
            fig.savefig(f"{images_path}/expPlot-noiseEffect-varyingAlpha-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')

        else:
            # fig.savefig(f"{images_path}/linePlot-noiseEffect-varyingBeta-allN-{metric}-violin{counter}.png",bbox_inches='tight')
            fig.savefig(f"{images_path}/expPlot-noiseEffect-varyingBeta-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')
        #######################################################

        plt.close(fig)
"""Violinplot to compare effect of noise (per bin)"""


if __name__ == "__main__":
    #######################################################
    # NOTE: Set simulation settings to save appropriately #
    alpha = '1_00'
    beta = '0_00'      

    alphas = ['1_00','0_75','0_50']
    betas = ['0_00','0_25', '0_50']                                                               
    #######################################################

    # NOTE NOTE: RUN SCRIPT USING -W "ignore" :NOTE NOTE #
    
    for alpha, beta in product(alphas,betas):
        if beta == '0_50' and alpha in ['0_75','0_50']:
            continue
        print(f'alpha={alpha} & beta={beta}')
        violin_per_params(alpha=alpha,beta=beta) # NOTE: CHANGE PLOT SETTINGS AS DESIRED (in function!)
    # hist_per_violin(alphas=alphas,betas=betas) # NOTE: CHANGE PLOT SETTINGS AS DESIRED (in function!)
    # violin_noiseEffect(fixed_param=beta,varying_param=alphas,A=True,metric='global') # NOTE: CHANGE PLOT SETTINGS AS DESIRED (in function!)