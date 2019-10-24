#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:56:46 2019

@author: amin
"""
    
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

subid1 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_Kmeans_notinf1.csv'  
subid3 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_Kmeans_inf1.csv'  
#subid2 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_basic_sym_no-height_neworient.csv'
perm1 = pd.read_csv(subid1, header=None, sep=' ').values
perm3 = pd.read_csv(subid3, header=None, sep=' ').values

fluc_ni = np.zeros((38,))
fluc_i = np.zeros((38,))

grad_ni = np.zeros((37,))
grad_i = np.zeros((37,))

notinfo = perm1[~np.all(perm1 == 0, axis=1)]
notinfo = np.mean(notinfo, axis = 1)

for i in range(1,39):
    fluc_ni[i-1] = (notinfo[i] - notinfo[i-1])*100/notinfo[i]
    
for i in range(1,37):
    grad_ni[i-1] = (notinfo[i+1] - notinfo[i-1])*100/notinfo[i]

info = perm3[~np.all(perm3 == 0, axis=1)]
info = np.mean(info, axis = 1)

for i in range(1,39):
    fluc_i[i-1] = (info[i] - info[i-1])*100/info[i]
    
for i in range(1,37):
    grad_i[i-1] = (info[i+1] - info[i-1])*100/info[i]

l = np.asarray(list(range(10,201,5)))

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(2, 1)
#fig.set_size_inches(12, 7)
fig.subplots_adjust(hspace=0.8, wspace=0.2)


axes[0].scatter(l, notinfo)
axes[0].plot(l, np.repeat(np.median(notinfo),39), '-')
axes[0].set_xlabel('Number of clusters(k)')
axes[0].set_ylabel('NMI')
axes[0].set_title('KMeans NMI vs. k')
   
axes[1].plot(l[1:38], grad_ni)
axes[1].plot(l[1:38], np.zeros((37,)), '-')
axes[1].set_xlabel('Number of clusters(k)')
axes[1].set_ylabel('NMI slope')
axes[1].set_title("KMeans NMI gradient vs. k")
    


#plt.suptitle(("KMeans NMI vs. k"),
#             fontsize=14, fontweight='bold')

plt.show()


#plt.plot(l[1:], fluc_ni)
##plt.plot(l[1:], fluc_i)
#plt.plot(l[1:], np.zeros((38,)), '-')
#plt.show()
#
#plt.plot(l[1:38], grad_ni)
##plt.plot(l[1:], grad_i)
#plt.plot(l[1:38], np.zeros((37,)), '-')
#plt.show()