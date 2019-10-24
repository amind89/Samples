# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:18:11 2019

@author: dadas
"""

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

X1 = pd.read_csv('Data_MNI_EntireBox_noT1_spatiallyinformed_VoxelLoc_MainGrad_neworient.csv', header=None).values
#X2 = pd.read_csv('Data_basic_sym_no-height_neworient.csv', header=None).as_matrix()

perm1 = np.zeros((201,10))
#perm2 = np.zeros((201,10))
perm3 = np.zeros((201,10))

Y1OneOut = np.empty((10, len(X1), 144))
#Y2OneOut = np.empty((10, len(X2), 108))
Y1Indi = np.empty((10, len(X1), 16))
Y1inf = np.empty((10, len(X1), 17))
#Y2Indi = np.empty((10, len(X2), 13))

subid1 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_Kmeans_notinf.csv'  
subid3 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_Kmeans_inf.csv'  
#subid2 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_basic_sym_no-height_neworient.csv'
perm1 = pd.read_csv(subid1, header=None, sep=' ').values
perm3 = pd.read_csv(subid3, header=None, sep=' ').values

subid1 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_Kmeans_notinf1.csv'  
subid3 = '/p/himmelbach/amin/9T_MNI/NMIvsCluster_Kmeans_inf1.csv' 

    
for i in range(10):

    count1 = 0
    count2 = 0
    for j in range(len(X1[1,:])):
        if (int(math.fmod(j-i+1,10)) != 0):
            Y1OneOut[i, :, count1] = X1[:,j]            
            count1 += 1
        else:
            Y1Indi[i, :, count2] = X1[:,j]
            count2 += 1 
             
#    count1 = 0
#    count2 = 0        
#    for j in range(len(X2[1,:])):
#        if (int(math.fmod(j-i+1,10)) != 0):            
#            Y2OneOut[i, :, count1] = X2[:,j]
#            count1 += 1
#        else:            
#            Y2Indi[i, :, count2] = X2[:,j]
#            count2 += 1   

for i in range(185, 201, 5):
    print(i)
    range_n_clusters = i
        
    clusterer = KMeans(n_clusters=range_n_clusters, random_state=20, n_init= 40, max_iter=500, tol=1e-6, n_jobs=-1)
    cluster_labels1 = clusterer.fit_predict(X1)
#    cluster_labels2 = clusterer.fit_predict(X2)
    
    Y1inf[:, :, :16] = Y1Indi;
    
#    Y1Indi[:, :, 16] = np.tile(cluster_labels1, (10, 1))
#    Y2Indi[:, :, 12] = np.tile(cluster_labels2, (10, 1))
    
    for j in range(10):
        labels1OneOut = clusterer.fit_predict(Y1OneOut[j, :, :])
#        labels2OneOut = clusterer.fit_predict(Y1OneOut[j, :, :])
        Y1inf[:, :, 16] = np.tile(labels1OneOut, (10, 1))
        
        labels1Indi = clusterer.fit_predict(Y1Indi[j, :, :])
        labels1inf = clusterer.fit_predict(Y1inf[j, :, :])
#        labels2Indi = clusterer.fit_predict(Y2Indi[j, :, :])
#        
        perm1[i-1,j] = normalized_mutual_info_score(labels1Indi, labels1OneOut, average_method='arithmetic')
        perm3[i-1,j] = normalized_mutual_info_score(labels1inf, labels1OneOut, average_method='arithmetic')
#        perm2[i-1,j] = normalized_mutual_info_score(labels2Indi, labels2OneOut)

    np.savetxt(subid1, perm1)
    #np.savetxt(subid2, perm2)
    np.savetxt(subid3, perm3)
    

np.savetxt(subid1, perm1)
#np.savetxt(subid2, perm2)
np.savetxt(subid3, perm3)

    