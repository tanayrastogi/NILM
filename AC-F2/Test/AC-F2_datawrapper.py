#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:35:13 2018

@author: tanay
"""
from os import listdir
import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def read_labels(path):
    # Ccreate a dictonary to save all the appliances and their instances
    Appliance = {}
    
    # Find all the appliances name in the dataset
    Appliances_category = listdir(path)
    Appliances_category.sort()
    
    # For each appliance, find the all instances of appliance
    for app in Appliances_category:
        app_path = path + '/'+ app
        
        # Create a list of all the appliance by sessions
        Session = {}
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




def read_appliance_specific_data(path, Labels,  app, session, instance_range, plot):
    
    # Empty variable for data
    data = pd.DataFrame()
    
    if(plot):
        fig, ax = plt.subplots(nrows=3,ncols=1, sharex=True)
        fig.suptitle("{}".format(app), fontsize=16)
    
    # Append all the data for all the instances
    for instance in instance_range:
        
        # File path to be read
        path_instance = path + '/' + app + '/' + Labels[app][session][instance]
        
        # Check if the path exits
        if(os.path.exists(path_instance)):
            
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
            
            if(plot):
                # Plotting
                ax[0].plot(instance_data.index.values, instance_data['power'], label = 'App{}'.format(instance))
                ax[0].set_ylabel('Real Power [W]')
                
                ax[1].plot(instance_data.index.values, instance_data['reacPower'], label = 'App{}'.format(instance))
                ax[1].set_ylabel('Reactive Power [Var]')
                
                ax[2].plot(instance_data.index.values, instance_data['rmsCur'], label = 'App{}'.format(instance))
                ax[2].set_ylabel('RMS Current [A]')
                ax[2].set_xlabel('Time [sec]')
                
                ax[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.0), shadow=True)
                
                
            data = data.append(instance_data)
            data.reset_index(drop=True, inplace=True)
                
        # If not raise error
        else:
            raise ValueError('The path does not exit')
    
        
    return data



def read_data(path, Labels, unq_appliance,  app_to_read, session, instance_range, linePlot=False, scatterPlot = False):
    Data = {}
    for app in unq_appliance.loc[app_to_read, 'Appliances']:
        temp = read_appliance_specific_data(path, Labels, app, session, instance_range, linePlot)
        Data[app] = temp
    
    if(scatterPlot):        
        keys = list(Data.keys())
        keys.sort()
    
        for app in keys:
            plt.scatter(Data[app]['power'], Data[app]['reacPower'], label = app, s = 10)
            plt.legend()
            plt.xlim([0, 2500])
            plt.xlabel('Real Power [W]')
            plt.ylabel('Reactive Power [Var]')
    
    return Data
    







#%% Reading Metadata
# Location of the dataset
path = '/home/tanay/_Thesis/Datasets/ACS-F2/Data'
Labels = read_labels(path) 

# List of unique appliances
unq_appliance = pd.DataFrame(list(Labels.keys()), columns=['Appliances'])
unq_appliance.sort_values(by=['Appliances'], inplace=True)
unq_appliance.reset_index(drop=True, inplace=True)

print(unq_appliance)
print('Number of Sessions:', 2)
#%% Reading data
session = 1
instance_range = [0]
app_to_read = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

Data = read_data(path, Labels, unq_appliance, app_to_read, session, instance_range, linePlot=False, scatterPlot=True)
   
    
plt.show()


