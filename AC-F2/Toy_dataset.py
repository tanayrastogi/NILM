#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:57:41 2018

@author: tanay
"""


import random 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import pandas as pd
from scipy.stats import multivariate_normal 
plt.style.use('ggplot')
from sklearn.metrics import confusion_matrix
import itertools


def bulb(Amplitude_real_power = 200, Amplitude_reac_power = 5, lineplot = False, scatterplot = False):
    
    # Parameters
    noise = 100
    Signal_Length = 150
    
    
    # Define 
    bulb = pd.DataFrame(np.zeros((Signal_Length,2)), columns=['power', 'reacPower'])
    bulb['Label'] = pd.Series(['bulb']*bulb.shape[0])
    bulb['Label'] = bulb['Label'].astype('str')
    
    
    # Watt and VAR of Bulb
    bulb['power'] += Amplitude_real_power
    bulb['reacPower'] += Amplitude_reac_power
    
    # Add noise
    bulb.loc[:,'power'] += [noise*random.random() for _ in range(0, len(bulb))]
    bulb.loc[:,'reacPower'] += [(1/10)*noise*random.random() for _ in range(0, len(bulb))]
     
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(bulb.index.values, bulb['power'])
        line_ax[1].plot(bulb.index.values, bulb['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()
        
    
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(bulb['power'], bulb['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()


    return bulb


def fsm(lineplot = False, scatterplot = False):
    
    # Parameters for the fsm
    noise = 10
    Amplitude_real_power = 800
    Amplitude_reac_power = 80
    periodic_signal = [0,10,10,0]*5
    
    # Define the fsm signal
    fsm = pd.DataFrame(np.zeros((150,2)), columns=['power', 'reacPower'])
    fsm['Label'] = pd.Series(['fsm']*fsm.shape[0])
    fsm['Label'] = fsm['Label'].astype('str')
    
    
    # Real Power
    fsm['power'] += Amplitude_real_power
    fsm.loc[0:len(periodic_signal) - 1, 'power'] = periodic_signal
    fsm.loc[0:len(periodic_signal) - 1, 'power'] *= 200 
    fsm.loc[60:60+len(periodic_signal) - 1, 'power'] = periodic_signal
    fsm.loc[60:60+len(periodic_signal) - 1, 'power'] *= 200 
    fsm.loc[120:120+len(periodic_signal) - 1, 'power'] = periodic_signal
    fsm.loc[120:120+len(periodic_signal) - 1, 'power'] *= 200 
 
    # Reactive Power
    fsm['reacPower'] += Amplitude_reac_power
    fsm.loc[0:len(periodic_signal) - 1, 'reacPower'] += 20 
    fsm.loc[60:60+len(periodic_signal) - 1, 'reacPower'] += 20
    fsm.loc[120:120+len(periodic_signal) - 1, 'reacPower'] += 20
    
    
    fsm.loc[:,'power'] += [20*noise*random.random() for _ in range(0, len(fsm))]
    fsm.loc[:,'reacPower'] += [0.9*noise*random.random() for _ in range(0, len(fsm))]
    
    
  
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(fsm.index.values, fsm['power'])
        line_ax[1].plot(fsm.index.values, fsm['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
                        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()

    
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(fsm['power'], fsm['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()
        scatter_ax.set_xlim(0, 2500)
        scatter_ax.set_ylim(0, 150)
    return fsm






def tsm(lineplot = False, scatterplot = False):
    
    noise = 10
    
    tsm = pd.DataFrame(np.zeros((150,2)), columns=['power', 'reacPower'])
    tsm['Label'] = pd.Series(['tsm']*tsm.shape[0])
    tsm['Label'] = tsm['Label'].astype('str')
    
    tsm.loc[0:40, 'power'] += 500
    tsm.loc[40:150, 'power'] += 1000
    
    tsm.loc[0:40, 'reacPower'] += 100
    tsm.loc[40:150, 'reacPower'] += 50 

    
    tsm.loc[:,'power'] += [20*noise*random.random() for _ in range(0, len(tsm))]
    tsm.loc[:,'reacPower'] += [0.9*noise*random.random() for _ in range(0, len(tsm))]
    
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(tsm.index.values, tsm['power'])
        line_ax[1].plot(tsm.index.values, tsm['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()

        
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(tsm['power'], tsm['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()

    
    return tsm


def aggregated_signal(bulb, fsm, tsm, lineplot = False, scatterplot = False, ind_lineplot = False, ind_scatterplot = False):
    
    # Aggregated signal
    agg = pd.DataFrame(np.zeros((1500,2)), columns=['power', 'reacPower'])
    agg['Label'] = pd.Series(['GND']*agg.shape[0])
    agg['Label'] = agg['Label'].astype('str')
    
    # Interval 1: Only Bulb
    agg.loc[0:len(bulb)-1, :] = bulb.as_matrix()
    
    # Interval 2: Only fms
    agg.loc[200: 200 +len(fsm)-1, :]  = fsm.as_matrix()
    
    # Interval 3: Only tms
    agg.loc[450: 450 +len(tsm)-1, :]  = tsm.as_matrix()
    
    # Interval 4: Bulb + fsm
    agg.loc[650: 650 +len(fsm)-1, :]  = bulb.add(fsm, axis='index', fill_value=True).as_matrix()
    
    # Interval 5: Bulb + tsm
    agg.loc[850: 850 +len(bulb)-1, :]  = bulb.add(tsm, axis='index', fill_value=True).as_matrix()
    
    # Interval 6: fsm + tsm
    agg.loc[1050: 1050 +len(fsm)-1, :]  = fsm.add(tsm, axis='index', fill_value=True).as_matrix()
    
    # Interval 7: bulb + fsm + tsm
    temp = fsm.add(tsm, axis='index', fill_value=True)
    agg.loc[1250: 1250 +len(fsm)-1, :]  = bulb.add(temp, axis='index', fill_value=True).as_matrix()
    
    
    
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(agg.index.values, agg['power'])
        line_ax[1].plot(agg.index.values, agg['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()
        
        line_ax[0].set_title('Plot for aggregated signal')
    
    
    # Convert the Dtype to category
    agg['Label'] = agg['Label'].astype('category')
    
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(agg['power'], agg['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()
        scatter_ax.set_title('Plot for Aggregated Signal')
        
        
    if(ind_lineplot):
        _, line_Ax = plt.subplots(2,1, sharex=True) 
    if(ind_scatterplot):
        _,  scatter_Ax = plt.subplots()
    for sng in [bulb, fsm, tsm]:
        # Plotting
        if(ind_lineplot):
                   
            line_Ax[0].plot(sng.index.values, sng['power'])
            line_Ax[1].plot(sng.index.values, sng['reacPower'])
            
            line_Ax[0].set_ylabel('Real Power [Watt]')
            
            line_Ax[1].set_xlabel('Time [sec]')
            line_Ax[1].set_ylabel('Reactive Power [Var]')
            line_Ax[1].legend()
        
        if(ind_scatterplot):
            scatter_Ax.scatter(sng['power'], sng['reacPower'], s=10)
            scatter_Ax.set_xlabel('Real Power [Watt]')
            scatter_Ax.set_ylabel('Reactive Power [Var]')
            
            
            
    # Make a dictonaru for the signals
    Sng = {}
    Sng['bulb'] = bulb
    Sng['fsm'] = fsm
    Sng['tsm'] = tsm
    
    
    
    return Sng, agg






def gaussian_model_estimation(sng, name, plot=False):
    
    sng = sng.as_matrix(columns=['power', 'reacPower'])
    
    # Finding the best number of clusters
    cv_types = ['spherical', 'tied', 'diag', 'full'] 
    
    if(plot):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    opt_cluster = []
    for cv in cv_types:
        n_estimator = np.arange(1, 10)
        clfs = [GaussianMixture(n_components = n, covariance_type=cv, init_params='kmeans', random_state = 5).fit(sng) for n in n_estimator]
        bics = pd.DataFrame([clf.bic(sng) for clf in clfs])
        # Finding the optimum clusters
        bics_shift = bics.shift(1) - bics
        opt_cluster.append(int(bics_shift.idxmax()) + 1)
        
    
        if(plot):
            # Plot    
            ax.plot(n_estimator, bics, label = '{}'.format(cv))
    
    if(plot):
        ax.set_title('GMM Model Estimation for {}'.format(name))
        ax.legend()
        ax.set_xlabel('Number of Clusters')
    
    return opt_cluster







def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw x at means
    ax.scatter(position[0], position[1], s = 60, marker = '+', c='r', zorder = 3)
        
    # Draw the Ellipse
    for nsig in range(1, 5):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, lw = 0.5, ec = 'k', fc='none'))


def plot_gmm(gmm, X, label=True, ax=None):
       
    ax = ax or plt.gca()
    
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
    
    
    w_factor = 0.6 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax = ax, alpha=w * w_factor)
    
    
def Gaussian_clustering(sng, name, components, verbose = True, plot=False):
    
    # Converting dataframe to numpy array for running GMM
    sng = sng.as_matrix(columns=['power', 'reacPower'])
    
    # Gaussing Mixture Model
    gmm = GaussianMixture(n_components = components, covariance_type='full', init_params='kmeans', random_state = 5).fit(sng)
    
    if(verbose):
        # Print Summry of the GMM
        print('Gaussian Cluster for {}'.format(name))
        print('Mean: ',gmm.means_)
        print('Variance: ',gmm.covariances_)
        print('Weights: ', gmm.weights_)
        print('\n')
    
    # Plotting
    if (plot):
        fig, cluster_ax = plt.subplots() 
        plot_gmm(gmm, sng, label=True, ax=cluster_ax)  
        cluster_ax.set_ylim(0, 160)
        cluster_ax.set_xlim(-200, 2500)
        cluster_ax.set_title('Gaussian Clusters for {}'.format(name))
        cluster_ax.set_xlabel('Real Power [Watt]')
        cluster_ax.set_ylabel('Reactive Power [Var]')
        
      
    
    return gmm, gmm.means_, gmm.covariances_, gmm.weights_






def merge_clusters(Models, unq, merge_level,  plot = False):
    ## Merging Clusters to make Super clusters
    merge_mean = []
    merge_var = []
    merge_wght = []
    merge_name = []
    
    
    # Merge Level 2
    if(merge_level > 1):
        for sng1 in range(0, len(unq)):
            for sng2 in range(sng1+1, len(unq)):                                        
                for i in range(0, Models[unq[sng1]]['Mean'].shape[0]):                                    
                    for j in range(0, Models[unq[sng2]]['Mean'].shape[0]):
                        merge_mean.append(Models[unq[sng1]]['Mean'][i] + Models[unq[sng2]]['Mean'][j])
                        merge_var.append(Models[unq[sng1]]['Vars'][i] + Models[unq[sng2]]['Vars'][j])
                        merge_wght.append(Models[unq[sng1]]['Wght'][i] * Models[unq[sng2]]['Wght'][j])
#                        merge_name.append(Models[unq[sng1]]['Label'] + 'Clt{}'.format(i) + ' + ' + Models[unq[sng2]]['Label'] + 'Clt{}'.format(j) )     
                        merge_name.append(Models[unq[sng1]]['Label'] + Models[unq[sng2]]['Label'])     
                        
    # Merge Level 3
    if(merge_level > 2):
        for sng1 in range(0, len(unq)):
            for sng2 in range(sng1+1, len(unq)):
                for sng3 in range(sng2+1, len(unq)):                    
                    for i in range(0, Models[unq[sng1]]['Mean'].shape[0]):
                        for j in range(0, Models[unq[sng2]]['Mean'].shape[0]):
                            for k in range(0, Models[unq[sng3]]['Mean'].shape[0]):
                                merge_mean.append(Models[unq[sng1]]['Mean'][i] + Models[unq[sng2]]['Mean'][j] + Models[unq[sng3]]['Mean'][k])
                                merge_var.append(Models[unq[sng1]]['Vars'][i] + Models[unq[sng2]]['Vars'][j] + Models[unq[sng3]]['Vars'][k])
                                merge_wght.append(Models[unq[sng1]]['Wght'][i] * Models[unq[sng2]]['Wght'][j] * Models[unq[sng3]]['Wght'][k])
#                                merge_name.append(Models[unq[sng1]]['Label'] + 'Clt{}'.format(i) +' + '+ Models[unq[sng2]]['Label'] + 'Clt{}'.format(j) + ' + ' + Models[unq[sng3]]['Label'] + 'Clt{}'.format(k))
                                merge_name.append(Models[unq[sng1]]['Label'] + Models[unq[sng2]]['Label'] + Models[unq[sng3]]['Label'])
    
    
    # Merge Level 4
    if(merge_level > 3):
        for sng1 in range(0, len(unq)):
            for sng2 in range(sng1+1, len(unq)):
                for sng3 in range(sng2+1, len(unq)):
                    for sng4 in range(sng3+1, len(unq)):
                        for i in range(0, Models[unq[sng1]]['Mean'].shape[0]):
                            for j in range(0, Models[unq[sng2]]['Mean'].shape[0]):
                                for k in range(0, Models[unq[sng3]]['Mean'].shape[0]):
                                    for l in range(0, Models[unq[sng4]]['Mean'].shape[0]):
                                        merge_mean.append(Models[unq[sng1]]['Mean'][i] + Models[unq[sng2]]['Mean'][j] + Models[unq[sng3]]['Mean'][k] + Models[unq[sng4]]['Mean'][l])
                                        merge_var.append(Models[unq[sng1]]['Vars'][i] + Models[unq[sng2]]['Vars'][j] + Models[unq[sng3]]['Vars'][k] + Models[unq[sng4]]['Vars'][l])
                                        merge_wght.append(Models[unq[sng1]]['Wght'][i] * Models[unq[sng2]]['Wght'][j] * Models[unq[sng3]]['Wght'][k] * Models[unq[sng4]]['Wght'][l])
#                                        merge_name.append(Models[unq[sng1]]['Label'] + 'Clt{}'.format(i) + ' + ' + Models[unq[sng2]]['Label'] + 'Clt{}'.format(j) +' + '+ Models[unq[sng3]]['Label'] + 'Clt{}'.format(k) +' + '+ Models[unq[sng4]]['Label'] + 'Clt{}'.format(l))
                                        merge_name.append(Models[unq[sng1]]['Label'] + Models[unq[sng2]]['Label'] + Models[unq[sng3]]['Label'] + Models[unq[sng4]]['Label'])
    
    
    
    
    # Add original cluster to the lists
    for i in unq:
        for j in range(0, Models[i]['Mean'].shape[0]):
            merge_mean.append(Models[i]['Mean'][j])
            merge_var.append(Models[i]['Vars'][j])
            merge_wght.append(Models[i]['Wght'][j])
            merge_name.append(i)
    
    
    # Convert to Numpy array just for plotting
    merge_mean = np.array(merge_mean)
    merge_var = np.array(merge_var)
    merge_wght = np.array(merge_wght) 
    merge_name = np.array(merge_name)
    
    number_of_cluster = len(merge_mean)
    
    if(plot):        
        ## Plotting the super clusters
        fig, ax = plt.subplots() 
        w_factor = 0.6 / merge_wght.max()
        for pos, covar, w, name in zip(merge_mean, merge_var, merge_wght, merge_name):
            draw_ellipse(pos, covar, ax = ax, alpha=w * w_factor)
            ax.text(pos[0], pos[1], name, fontsize=10, color='blue')
        ax.set_title('Number of Super Clusters: {}'.format(number_of_cluster)) 
        ax.set_xlabel('Real Power [Watt]')
        ax.set_ylabel('Reactive Power [Var]')
        ax.set_ylim(0, 300)
        ax.set_xlim(-200, 4000)
    
    
    # Pass the merged clusters as a dictonary
    merged_Model = {}
    for i in range(0, len(merge_mean)):
        merged_Model[i] = {'Mean': merge_mean[i], 'Vars': merge_var[i], 'Wght' : merge_wght[i] , 'Label' : merge_name[i]}
    
    
       
    # Add GND as a model in the clusters
    merged_Model[len(merged_Model)] = {'Mean': np.ones((1,2)).flatten(), 'Vars': np.eye(2), 'Wght' : 1.0 , 'Label' : 'GND'}
    
    
    
    return merged_Model




def inference(Agg, Merged_Model):
    
    # Extract only the power data fromm the Dataframe
    X = Agg.as_matrix(columns=['power', 'reacPower'])
    
    # List variable for the identified names from AGG     
    signal_identified = []
    
    for x in X:
         pdf = []
         for i in Merged_Model:
             m = Merged_Model[i]['Mean']
             c = Merged_Model[i]['Vars']
             pdf.append(multivariate_normal.pdf(x, mean=m, cov=c))
    
         signal_identified.append(Merged_Model[pdf.index(max(pdf))]['Label'])
    
    Y_pred = pd.Series(signal_identified, dtype='category')
    
    return Y_pred







def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
 
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    plt.xticks(rotation=90)
    




def f_score(Y_true, Y_pred, unq):
    for sng in unq:
        true = [1 if sng in i else 0 for i in Y_true]
        pred = [1 if sng in i else 0 for i in Y_pred]
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f_score = 2*precision*recall/(precision+recall)
        print('F_score for {} : {}'.format(sng, f_score))








#%% Generate signal
bulb = bulb()
fsm = fsm()
tsm = tsm()
Data, Agg = aggregated_signal(bulb, fsm, tsm, lineplot = False, scatterplot = False, ind_lineplot = False, ind_scatterplot = False)

#%% Clustering
# Signal names
unq = list(Data.keys())
unq.sort()

opt_cluster = {"bulb":1 ,"fsm":3, "tsm": 2}

# Mean, Variance and Weights for all the clusters
Models = {}
for i in unq:  
    # Running Gaussian Mixutre Model for each appliance
#    opt_cluster = gaussian_model_estimation(Data[i], i, plot=True)
    GMM, Mean, Vars, Wght = Gaussian_clustering(Data[i], i, opt_cluster[i] , verbose = False, plot=False)
    Models[i] ={'Mean': Mean, 'Vars': Vars , 'Wght': Wght, 'Label': i}

#%% Creating Super Clusters
aggregation_level = 3  # Defines how many clusters will be added together to create super cluster
Merged_Model = merge_clusters(Models, unq, aggregation_level, plot=True)
plt.scatter(Agg['power'], Agg['reacPower'], s=10, c='yellow')

#%% Inference
# Take only the power data from the Agg
Y_true = pd.Series(Agg['Label'])
Y_pred = inference(Agg, Merged_Model)

#%% Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred, labels=Y_true.cat.categories)
classes = list(Y_true.cat.categories)
plot_confusion_matrix(cm, classes)

#%% Accuracy Metric: F-score
f_score(Y_true, Y_pred, unq)


plt.show()