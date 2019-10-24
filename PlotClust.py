#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:58:04 2019

@author: amin
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math


X = pd.read_csv('150Cluster_Labels_DEC.txt', header=None).values
y_pred = X

b = 111*111
range_n_clusters = 100

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(5, 7)
fig.set_size_inches(12, 7)
fig.subplots_adjust(hspace=0.4, wspace=0.2)

for i in range(34):
    a = y_pred[i*b:(i+1)*b,]
    a = a.reshape(111, 111)
    a = np.rot90(a, 1)
      
    axes[int(4-math.floor(i/7)), 6-int(math.fmod(i, 7))].imshow(a, cmap='Paired')
    
    Snum = "slice {}".format(34-i)
    axes[int(4-math.floor(i/7)), 6-int(math.fmod(i, 7))].set_title(Snum)
    


plt.suptitle(("KMeans clustering on midbrain data "
              "with range_n_clusters = %d" % range_n_clusters),
             fontsize=14, fontweight='bold')

plt.show()

c = np.zeros((34,111,111))

for i in range(34):
    a = y_pred[i*b:(i+1)*b,]
    a = a.reshape(111, 111)
    a = np.rot90(a, 1)
    c[i,:,:] = a
    


# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(7, 8)
fig.set_size_inches(12, 7)
fig.subplots_adjust(hspace=0.4, wspace=0.2)

for i in range(55):
      
    axes[int(6-math.floor(i/8)), 7-int(math.fmod(i, 8))].imshow(np.rot90(np.rot90(c[:,:,i],1), 1), cmap='Paired')
    
    Snum = "slice {}".format(55-i)
    axes[int(6-math.floor(i/8)), 7-int(math.fmod(i, 8))].set_title(Snum)
    


plt.suptitle(("KMeans clustering on midbrain data "
              "with range_n_clusters = %d" % range_n_clusters),
             fontsize=14, fontweight='bold')

plt.show()
      