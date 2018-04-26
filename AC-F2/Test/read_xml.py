#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:52:09 2018

@author: tanay
"""

import xml.etree.ElementTree as ET
import pandas as pd

path = '/home/tanay/_Thesis/Datasets/ACS-F2/Data/Coffee machines/jh_coffeeMachine_Nespresso_TurMixTx100_a1.xml'
tree = ET.parse(path)
root = tree.getroot()

# Session information 
acquisitionContext = root[0].attrib
session = acquisitionContext['session']

# Data for the appliance
signalCurve = root[5]
data = pd.DataFrame()
for signalPoint in signalCurve:
    data = data.append(pd.DataFrame.from_dict(signalPoint.attrib, orient='index').transpose())