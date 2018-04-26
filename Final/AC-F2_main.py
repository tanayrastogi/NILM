#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:40:59 2018

@author: tanay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:04:47 2018

@author: tanay
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn')
import numpy as np

from ACF2_funcs import read_labels, read_data, gaussian_model_estimation
from ACF2_funcs import Gaussian_clustering, merge_clusters, create_agg_data
from ACF2_funcs import true, prediction, confusion_matrix, plot_confusion_matrix
from ACF2_funcs import f_score, split


#%% Read Labels
# Location of the dataset
path = '/home/tanay/_Thesis/Datasets/ACS-F2/Data'
Labels = read_labels(path) 

# List of unique appliances
unq_appliance = pd.DataFrame(list(Labels.keys()), columns=['Appliances'])
unq_appliance.sort_values(by=['Appliances'], inplace=True)
unq_appliance.reset_index(drop=True, inplace=True)

print(unq_appliance)
print('Number of Sessions:', 2)
print('\n')

#%% Read Traning Data 
print('-------------------Reading Traning Data---------------------------')
# Reading data
session = 1
# Until Max 14
instance_range = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
app_to_read = [0,1,5,3,9]
Traning_Data  = read_data(path, Labels, unq_appliance, app_to_read, session, instance_range, filter_data=True, linePlot=False, scatterPlot=False)

unq_train = list(Traning_Data.keys())
unq_train.sort()


#%% Create Agg data
print('-------------------Reading Test Data---------------------------')
# Reading data
session = 1
# Until Max 14
instance_range = [2]
app_to_read = [0,1,5,3,9]
Test_Data = read_data(path, Labels, unq_appliance, app_to_read, session, instance_range, filter_data=False, linePlot=False, scatterPlot=False)
unq_test = list(Test_Data.keys())
unq_test.sort()
Agg = create_agg_data(unq_test, Test_Data, lineplot = False, scatterplot = False)


#%% Clustering
# Mean, Variance and Weights for all the clusters
print('-------------------Clustering---------------------------')
Models = {}
for i in unq_train:  
    # Running Gaussian Mixutre Model for each appliance
    feature_to_use = ['power', 'reacPower']
#    opt_cluster = gaussian_model_estimation(Data[i],feature_to_use, i, plot=True)
    GMM, Mean, Vars, Wght = Gaussian_clustering(Traning_Data[i],feature_to_use, i, 1, verbose = False, plot=False)
    Models[i] ={'Mean': Mean, 'Vars': Vars , 'Wght': Wght, 'Label': i}


#%% Creating Super Clusters
aggregation_level = 3  # Defines how many clusters will be added together to create super cluster
Merged_Model = merge_clusters(Models, unq_train, aggregation_level, plot=True)
for app in Agg['Label'].cat.categories:
    plt.scatter(Agg['power'][Agg['Label'] == app], Agg['reacPower'][Agg['Label'] == app], s=10, label=app)
plt.legend()


#%% Inference
print('-------------------Inferece---------------------------')
Y_true = true(Agg)
Y_pred = prediction(Agg, Merged_Model)
classes = [split(Merged_Model[i]['Label']) for i in Merged_Model.keys()]
classes.sort()
  
#%% Confusion Matrix
print('-------------------Accuracy_Metric---------------------------')
cm = confusion_matrix(Y_true, Y_pred, labels=classes)
plot_confusion_matrix(cm, classes)

#%% Accuracy Metric: F-score
f_score(Y_true, Y_pred, unq_test)

#%% Animated plot
fig, ax = plt.subplots()
ax.set(xlim=(-10, len(Agg)+10), ylim=(-100, max(Agg['power']) + 200))

# Line plot
ax.plot(Agg.index.values, Agg['power'], 'r')
ax.set_ylabel('Real Power [Watt]')
ax.set_xlabel('Time [sec]')

# Scatter point
point = ax.scatter(Agg.index.values[0], Agg['power'][0], s=50, c='b')

def animate(i):  
    ax.set_title('True: {} | Pred: {}'.format(Y_true[i], Y_pred[i]))
    point.set_offsets((Agg.index.values[i], Agg['power'][i])) 
anim = FuncAnimation(fig, animate, interval=70, frames=len(Agg.index.values)-1)

plt.draw()
plt.show()