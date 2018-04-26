#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:47:28 2018

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

plt.style.use('seaborn')


from ACF2_funcs import read_labels, read_data, gaussian_model_estimation
from ACF2_funcs import Gaussian_clustering, merge_clusters


import AC-F2_main



short_names = {'Coffee machines': 'CfM',
               'Computers stations (with monitors)': 'CS',
               'Fans': 'F',
               'Fridges and freezers' : 'F&F',
               'Hi-Fi systems (with CD players)': 'HiFi',
               'Kettles': 'K',
               'Lamps (compact fluorescent)': 'LF',
               'Lamps (incandescent)': 'LI',
               'Laptops (via chargers)': 'LPT',
               'Microwave ovens': 'MO',
               'Mobile phones (via chargers)': 'MP',
               'Monitors' :'M',
               'Printers' : 'P',
               'Shavers (via chargers)': 'S',
               'Televisions (LCD or LED)': 'TELE'}

#%% Read Labels and Data 
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


# Reading data
session = 2
# Until Max 14
instance_range = [4,5,6]
app_to_read = [0,1,5,3,9]
Data = read_data(path, Labels, unq_appliance, app_to_read, session, instance_range, filter_data=False, linePlot=True, scatterPlot=False)

unq = list(Data.keys())
unq.sort()



#%% Aggregated data
