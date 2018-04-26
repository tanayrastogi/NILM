#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:41:38 2018

@author: tanay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:04:32 2018

@author: tanay
"""

from os import listdir
import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

import random 
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import itertools

plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#%% --------------------------------- Short Names for the appliances -------------------------#
short_names = {'Coffee machines': 'Coffee,',
               'Computers stations (with monitors)': 'Computer,',
               'Fans': 'Fan,',
               'Fridges and freezers' : 'Fridge,',
               'Hi-Fi systems (with CD players)': 'HiFi,',
               'Kettles': 'Kettle,',
               'Lamps (compact fluorescent)': 'Lamp1,',
               'Lamps (incandescent)': 'Lamp2,',
               'Laptops (via chargers)': 'Laptop,',
               'Microwave ovens': 'Microwave,',
               'Mobile phones (via chargers)': 'Mobile,',
               'Monitors' :'Monitor,',
               'Printers' : 'Printer,',
               'Shavers (via chargers)': 'Shaver,',
               'Televisions (LCD or LED)': 'Tele,'}


#%% --------------------------------- Read labels -------------------------#
def read_labels(path):
    # Ccreate a dictonary to save all the appliances and their instances
    
    Appliance = {}
    
    
    # Find all the appliances name in the dataset
    Appliances_category = listdir(path = path)
    Appliances_category.sort()
    
    # For each appliance, find the all instances of appliance
    for app in Appliances_category:
        app_path = path + '/'+ app
        
        # Create a list of all the appliance by sessions
        Session = {}
        
        # Running for the session 1 and 2 in the range function
        for i in range(1,3):
            App = []
            os.chdir(app_path)
            for file in glob.glob('*a{}.xml'.format(i)):
                App.append(file)
            App.sort()
            Session[i] = App
    
        # Update the Dict for all the appliances and instances with sessions
        Appliance[app] = Session
    
    return Appliance




#%% --------------------------------- Read Data (traning) -------------------------#
def read_appliance_specific_data(path, Labels,  app, session, instance_range, filter_data = False, plot = False):
    
    # Empty variable for data
    data = pd.DataFrame()
    
    # Features to drop
    drop_features = ['phAngle','rmsVolt','time','freq']
    
    if(plot):
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.suptitle("{}".format(app), fontsize=16)
    
    # Append all the data for all the instances
    for instance in instance_range:
        
        # File path to be read
        path_instance = path + '/' + app + '/' + Labels[app][session][instance]
        
        # Check if the path exits
        if(os.path.exists(path_instance)):
            
            print('Reading Instance {} for {}'.format(instance, app))
            
            # Data at the instance
            instance_data = pd.DataFrame()
            
            # Read the xml file
            tree = ET.parse(path_instance)
            root = tree.getroot()
        
    #        # Extract Session information 
    #        acquisitionContext = root[0].attrib
    #        session = acquisitionContext['session']
            
            # Extract Data for the appliance
            signalCurve = root[5]
            for signalPoint in signalCurve:
                point_data = pd.DataFrame.from_dict(signalPoint.attrib, orient='index').transpose()
                instance_data = instance_data.append(point_data)
            
            # Reset the index
            instance_data.reset_index(drop=True, inplace=True)
            # Convert the data type for the coloumns
            instance_data = instance_data.convert_objects(convert_dates=True, convert_numeric=True)
            # Drop the features that are not required
            instance_data = instance_data.drop(drop_features, axis = 1)
            
            
            if(plot):
                # Plotting
                ax[0].plot(instance_data.index.values, instance_data['power'], label = 'App{}'.format(instance))
                ax[0].set_ylabel('P [W]')
                
                ax[1].plot(instance_data.index.values, instance_data['reacPower'], label = 'App{}'.format(instance))
                ax[1].set_ylabel('Q [Var]')
                           
                ax[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.0), shadow=True)
                
                
            data = data.append(instance_data)
            data.reset_index(drop=True, inplace=True)
            
            # Introduce labels to the data
            data['Label'] = pd.Series([short_names[app]]*data.shape[0])
            data['Label'] = data['Label'].astype('str')
       
        # If not raise error
        else:
            raise ValueError('The path does not exit')
    
    
    if(filter_data):
        # Cutoff the GND values data
        print('')
        print('Running filter for {}'.format(app))
        print('Current Cutoff: {}'.format(data['rmsCur'].mean()))
        print('\n')
        
        data = data[data['rmsCur'] > data['rmsCur'].mean()]
        
        # Median Filter
        data = data.rolling(window=3).median()
        data = data.dropna(axis = 0, how = 'any')

        
        # Plotting
        if(plot):
            data.plot(x=data.index.values, y=['power','reacPower', 'rmsCur'], subplots=True, sharex=True, title = 'Data without GND')
    
    data.reset_index(drop=True, inplace=True)
    
    return data



def read_data(path, Labels, unq_appliance, app_to_read, session, instance_range, filter_data = False, linePlot=False, scatterPlot = False):
    Data = {}
    
    # Print Summry
    print('Reading data with filter {}'.format(filter_data))
    print('\n')
    
    for app in unq_appliance.loc[app_to_read, 'Appliances']:
        temp = read_appliance_specific_data(path, Labels, app, session, instance_range, filter_data, linePlot)
        Data[short_names[app]] = temp
    
    if(scatterPlot):        
        keys = list(Data.keys())
        keys.sort()
        fig, ax = plt.subplots()
        for app in keys:
            ax.scatter(Data[app]['power'], Data[app]['reacPower'], label = app, s = 10)
        ax.legend()
        ax.set_title('Scatter plot for Appliances')
        ax.set_xlim([-100, 2500])
        ax.set_ylim([-100,800])
        ax.set_xlabel('Real Power [W]')
        ax.set_ylabel('Reactive Power [Var]')

        
#    if(scatterPlot3d):
#        keys = list(Data.keys())
#        keys.sort()
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        for app in keys:
#            ax.scatter(Data[app]['power'], Data[app]['reacPower'], Data[app]['phAngle'],  label = app, s = 10)
#        ax.legend()
#        ax.set_xlim([-100, 2500])
#        ax.set_ylim([-100,800])
#        ax.set_zlim([0, 360])
#        ax.set_xlabel('[W]')
#        ax.set_ylabel('[Var]')
#        ax.set_zlabel('[degree]')
    
    return Data







#%% --------------------------------- Create Aggregated Data (testing) -------------------------#
def single_data(unq, Data):
    rand = random.choice(unq)
    print(rand)
    df = Data[rand]
    df = df[df['rmsCur'] > df['rmsCur'].mean()]
    return df

def double_data(unq, Data):
    rand = random.sample(unq,2)
    print(rand)
    df = Data[rand[0]][:358].add(Data[rand[1]][:358], axis='index', fill_value=True)
    df = df[df['rmsCur'] > df['rmsCur'].mean()]
    return df
    
def triple_data(unq, Data):
    rand = random.sample(unq,3)
    print(rand)
    temp = Data[rand[0]][:358].add(Data[rand[1]][:358], axis='index', fill_value=True)
    df =  Data[rand[2]][:358].add(temp, axis='index', fill_value=True)
    df = df[df['rmsCur'] > df['rmsCur'].mean()]
    return df

def GND(length):
    GND = pd.DataFrame(np.zeros((length,3)), columns=['rmsCur','power', 'reacPower'])
    GND['Label'] = pd.Series(['GND,']*GND.shape[0])
    GND['Label'] = GND['Label'].astype('str')
    return GND

def create_agg_data(unq, Data, lineplot = False, scatterplot = False):
    
    # Initial variable for the aggregated data
    agg = pd.DataFrame(np.zeros((1,3)), columns=['rmsCur', 'power', 'reacPower'])
    agg['Label'] = pd.Series(['GND,']*agg.shape[0])
    agg['Label'] = agg['Label'].astype('str')
    
    # Single Data
    agg = pd.concat([agg, single_data(unq, Data)])
    # Add GND
    agg = pd.concat([agg, GND(50)])
    # Double Data    
    agg = pd.concat([agg, double_data(unq, Data)])
    # Add GND
    agg = pd.concat([agg, GND(50)])
    # Triple Data
    agg = pd.concat([agg, triple_data(unq, Data)])
    
    # Single Data
    agg = pd.concat([agg, single_data(unq, Data)])
    # Add GND
    agg = pd.concat([agg, GND(50)])
    # Double Data    
    agg = pd.concat([agg, double_data(unq, Data)])
    # Add GND
    agg = pd.concat([agg, GND(50)])
    # Triple Data
    agg = pd.concat([agg, triple_data(unq, Data)])
    # Add GND
    agg = pd.concat([agg, GND(50)])
    
    
    # Reset Index
    agg.reset_index(drop=True, inplace=True)    
    agg['Label'] = agg['Label'].astype('category')
    
    
    # Ploting the aggregated data
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(agg.index.values, agg['power'], 'r')
        line_ax[1].plot(agg.index.values, agg['reacPower'], 'b')
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()
        
        line_ax[0].set_title('Plot for aggregated signal')
    
       
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(agg['power'], agg['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.set_title('Plot for Aggregated Signal')
    
    
    
    return agg
    
    
    
    
    
#%% --------------------------------- Clustering -------------------------#

def gaussian_model_estimation(sng,feature_to_use, name, plot=False):
    
    sng = sng.as_matrix(columns=feature_to_use)
    
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
    for nsig in range(1, 2):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, lw = 0.5, ec = 'k', fc='none', ls = 'dashed'))


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
    
    
def Gaussian_clustering(sng,feature_to_use, name, components, verbose = True, plot=False):
    
    # Converting dataframe to numpy array for running GMM
    sng = sng.as_matrix(columns=feature_to_use)
    
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
        cluster_ax.set_ylim(-200, 800)
        cluster_ax.set_xlim(-200, 2500)
        cluster_ax.set_title('Gaussian Clusters for {}'.format(name))
        cluster_ax.set_xlabel('Real Power [Watt]')
        cluster_ax.set_ylabel('Reactive Power [Var]')
        
      
    
    return gmm, gmm.means_, gmm.covariances_, gmm.weights_



#%% --------------------------------- Super Clusters -------------------------#

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
            ax.text(pos[0], pos[1], name, fontsize=6, color='blue')
        ax.set_title('Number of Super Clusters: {}'.format(number_of_cluster)) 
        ax.set_xlabel('Real Power [Watt]')
        ax.set_ylabel('Reactive Power [Var]')
        ax.set_ylim(-100, 600)
        ax.set_xlim(-100, 4000)
    
    
    # Pass the merged clusters as a dictonary
    merged_Model = {}
    for i in range(0, len(merge_mean)):
        merged_Model[i] = {'Mean': merge_mean[i], 'Vars': merge_var[i], 'Wght' : merge_wght[i] , 'Label' : merge_name[i]}
    
    
       
    # Add GND as a model in the clusters
    merged_Model[len(merged_Model)] = {'Mean': np.ones((1,2)).flatten(), 'Vars': np.eye(2), 'Wght' : 1.0 , 'Label' : 'GND,'}
    
    
    
    return merged_Model




#%% --------------------------------- Inference on Agg data -------------------------#

def split(x):
    x = x.split(',')
    x.sort()
    return ''.join(x)

def true(Agg):
    x = [split(pd.Series(Agg['Label'])[i]) for i in pd.Series(Agg['Label']).index]
    return pd.Series(x, dtype = 'category')


def prediction(Agg, Merged_Model):
    
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
    
         signal_identified.append(split(Merged_Model[pdf.index(max(pdf))]['Label']))
    
    Y_pred = pd.Series(signal_identified, dtype='category')
    
    return Y_pred



#%% --------------------------------- Confusion Matrix -------------------------#
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


#%% --------------------------------- Accuracy Metrics -------------------------#
def f_score(Y_true, Y_pred, unq):
    unq = [split(unq[i]) for i in range(0,len(unq))]
    for sng in unq:
        true = [1 if sng in i else 0 for i in Y_true]
        pred = [1 if sng in i else 0 for i in Y_pred]
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        score = 2*precision*recall/(precision+recall)
        
        
#        print('True positive for {} : {}'.format(sng, tp))
#        print('True negative for {} : {}'.format(sng, tn))
#        print('False positive for {} : {}'.format(sng, fp))
#        print('False negative for {} : {}'.format(sng, fn))
#        print('Precision for {} : {}'.format(sng, precision))
#        print('Recall for {} : {}'.format(sng, recall))
        print('F_score for {} : {}'.format(sng, score))


#def plot_3D_scatter(data, colors = None, alpha = 0.5, app=False):
#    colors = ['r', 'g', 'b', 'y', 'c']
#    i = 0
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    for app in Data:
#        ax.scatter(d['P'], d['Q'], d['S'], marker = '.', c =colors[i] , alpha = alpha, label=app)
#        i += 1 
#    
#    ax.set_xlabel('Real Power')
#    ax.set_ylabel('Reactive Power')
#    ax.set_zlabel('Aparent Power')
#
#    ax.set_xlim(0, 3000)    
#    ax.set_ylim(0, 600)
#    ax.set_zlim(0, 3000)
#    
#    plt.legend()
